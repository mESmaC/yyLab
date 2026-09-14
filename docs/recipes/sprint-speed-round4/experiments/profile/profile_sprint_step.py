"""Kernel-level profile of complete Sprint optimizer updates. Disposable diagnostic.

Builds the exact case an official precision recipe describes (same model,
precision adapter, SprintStepEngine, compiled CPU optimizer), runs a few
untimed warmup updates, then records one or more updates with torch.profiler
and attributes every GPU kernel, memcpy and memset to a phase (forward,
backward, gradient clipping, optimizer/transfer) and a kernel family (FP4 /
FP8 / BF16 GEMM, torchao casts, inductor fused kernels, attention, Liger,
eager elementwise, copies). It also records Dynamo graph-break counters for
the warmup, so the summary says whether the NVFP4/MXFP8 linears run inside
compiled regions or fall out to eager execution.

Profiler synchronization changes overlap, so the update times reported here
are diagnostics, not throughput results, and must not be entered in a
speed table. The script never touches the ordinary trainer, checkpoints or
resume contract; it only reads a recipe and writes to --output.

    .venv/Scripts/python.exe docs/recipes/sprint-speed-round4/experiments/profile/profile_sprint_step.py \
        --config docs/recipes/sprint-speed-round3/baseline-b32.json --output artifacts/profile-1b-b32

    .venv/Scripts/python.exe .../profile_sprint_step.py --analyze-only artifacts/profile-1b-b32/trace.json.gz
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import logging
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "yylab").is_dir():
            return candidate
    raise SystemExit("run this script from inside the yyLab repository (pyproject.toml + yylab/ not found)")


ROOT = _find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(ROOT))

MARK_UPDATE = "sprint.update"
MARK_FORWARD = "sprint.forward"
MARK_CLIP = "sprint.clip_grad_norm"
MARK_OPTIMIZER = "sprint.optimizer_step"

GEMM_OPS = {"aten::_scaled_mm", "aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::matmul",
            "aten::linear", "aten::_scaled_mm.out", "aten::mm.out", "aten::addmm.out"}
COMPILED_REGION = re.compile(r"CompiledFunction|Torch-Compiled Region|compiled_fn|inductor", re.I)
BACKWARD_HINT = re.compile(r"Backward|autograd::engine|AccumulateGrad", re.I)

# Whitepaper dense tensor rates for the GeForce RTX 5070 (TFLOP/s), used only
# to express achieved GEMM throughput as a fraction of peak. Source: NVIDIA
# RTX Blackwell whitepaper table as reproduced by Guru3D.
PEAK_TFLOPS = {"fp4": 493.9, "fp8_fp16acc": 246.9, "fp8_fp32acc": 123.5, "bf16": 61.7}

FAMILY_RULES = (
    (re.compile(r"^Memcpy|memcpy", re.I), "memcpy"),
    (re.compile(r"^Memset|memset", re.I), "memset"),
    (re.compile(r"triton_(poi|red|per|tem|for)_"), "inductor fused kernel"),
    (re.compile(r"mxfp8|to_mx|mx_fp8|_mx_dim|e8m0", re.I), "torchao MXFP8 cast (Triton)"),
    (re.compile(r"nvfp4|fp4|f4_|e2m1|uint4", re.I), "NVFP4 quantization kernel"),
    (re.compile(r"cudnn|fmha|sdpa|attention|flash", re.I), "attention"),
    (re.compile(r"liger|cross_entropy|element_mul|softmax", re.I), "Liger / loss"),
    (re.compile(r"foreach|multi_tensor", re.I), "foreach (clip/optimizer)"),
    (re.compile(r"elementwise_kernel|vectorized_elementwise|unrolled_elementwise|index_elementwise|fill", re.I), "aten eager elementwise"),
    (re.compile(r"reduce_kernel|reduction|amax|norm", re.I), "aten eager reduction"),
    (re.compile(r"nvjet|cutlass|gemm|xmma|cublas|Cijk|sm120|sm_120|sm90|sm100|ampere|wgrad|dgrad|scaled_mm", re.I), "gemm (no aten parent)"),
    (re.compile(r"cat|copy|clone|contiguous|transpose|permute|pad", re.I), "copy-like"),
)


# --------------------------------------------------------------------------- run


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_versions() -> dict:
    from importlib import metadata

    import torch

    versions = {"torch": str(torch.__version__), "python": platform.python_version()}
    for name, distributions in (("torchao", ("torchao",)), ("transformers", ("transformers",)),
                                ("triton", ("triton", "triton-windows")), ("liger", ("liger-kernel",))):
        versions[name] = None
        for distribution in distributions:
            try:
                versions[name] = metadata.version(distribution)
                break
            except metadata.PackageNotFoundError:
                continue
    return versions


def _select_case(cfg: dict, args) -> tuple[str, int, str]:
    precision = args.precision or cfg.get("precisions", ["bf16"])[0]
    batch = args.batch or int(cfg.get("batch_sizes", [8])[0])
    mode = args.compile_mode or cfg.get("compile_modes", ["default"])[0]
    return precision, batch, mode


def _install_markers(engine, torch):
    """Mark forward, clipping and optimizer regions without changing their work."""
    from torch.profiler import record_function

    class MarkedForward(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.calls = 0

        def forward(self, *args, **kwargs):
            index = self.calls
            self.calls += 1
            with record_function(f"{MARK_FORWARD}[mb={index}]"):
                return self.inner(*args, **kwargs)

    marked = MarkedForward(engine.execution_model)
    engine.execution_model = marked

    original_step = engine.optimizer.step

    def marked_step(*args, **kwargs):
        with record_function(MARK_OPTIMIZER):
            return original_step(*args, **kwargs)

    engine.optimizer.step = marked_step

    original_clip = torch.nn.utils.clip_grad_norm_

    def marked_clip(*args, **kwargs):
        with record_function(MARK_CLIP):
            return original_clip(*args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = marked_clip
    return marked


def _dynamo_counters() -> dict:
    try:
        from torch._dynamo.utils import counters
    except Exception as exc:  # pragma: no cover - version drift
        return {"error": f"{type(exc).__name__}: {exc}"}
    result = {}
    for key, value in counters.items():
        if isinstance(value, dict):
            result[key] = {str(name): int(count) if isinstance(count, int) else str(count)
                           for name, count in value.items()}
        else:
            result[key] = str(value)
    return result


def run_profile(args) -> Path:
    import torch

    from yylab.models.sprint import SprintConfig, SprintForCausalLM, estimate_parameter_count, plan_sprint_model
    from yylab.train import benchmark_sprint_precision as harness
    from yylab.train.sprint import SprintStepEngine
    from yylab.train.sprint_precision import apply_sprint_precision

    cfg = json.loads(Path(args.config).read_text("utf-8"))
    original_threads = torch.get_num_threads()
    for key, default in (("effective_batch", 8), ("seq_len", 512), ("offload_bucket_mb", 32),
                         ("cpu_threads", original_threads)):
        cfg[key] = harness._positive_int(cfg.get(key, default), key)
    cfg["seed"] = int(cfg.get("seed", 42))
    cfg["lr"] = float(cfg.get("lr", 3e-4))
    cfg.setdefault("cpu_optimizer_kernel", "auto")
    cfg.setdefault("attention_backend", "auto")
    cfg.setdefault("gradient_checkpointing", False)
    precision, batch, mode = _select_case(cfg, args)
    if cfg["effective_batch"] % batch:
        raise ValueError("microbatch must divide effective_batch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.get("device", "auto") == "auto" \
        else torch.device(cfg["device"])
    config = SprintConfig(**cfg["model_config"]) if cfg.get("model_config") else plan_sprint_model(
        harness._positive_int(cfg.get("target_parameters", 50_000_000), "target_parameters"),
        harness._positive_int(cfg.get("vocab_size", 8192), "vocab_size"),
        max_position_embeddings=int(cfg.get("max_position_embeddings", max(2048, cfg["seq_len"]))),
        profile=cfg.get("shape_profile", "throughput"))
    config.validate()
    rd = Path(args.output)
    rd.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    inputs = torch.randint(config.vocab_size, (cfg["effective_batch"], cfg["seq_len"]), generator=generator)
    if device.type == "cuda":
        inputs = inputs.pin_memory()
    batches = [{"input_ids": part, "labels": part} for part in inputs.split(batch)]

    receipt = {
        "schema": "yylab-sprint-step-profile-v1", "experimental": True, "diagnostic_only": True,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "notice": "Profiler instrumentation and per-update synchronization change overlap; timings here are diagnostics and are not throughput results.",
        "config_path": str(Path(args.config)), "config_sha256": _sha256(Path(args.config)),
        "source_sha256": {**harness._source_sha256(), "profile_sprint_step.py": _sha256(Path(__file__).resolve())},
        "runtime_versions": _runtime_versions(), "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "host_processor": platform.processor(), "host_logical_cores": os.cpu_count(),
        "case": {"precision": precision, "micro_batch": batch, "grad_accum": cfg["effective_batch"] // batch,
                 "compile_mode": mode, "torch_compile": mode != "eager"},
        "model_config": config.to_dict(), "planned_parameters": estimate_parameter_count(config),
        "sequence_length": cfg["seq_len"], "effective_batch": cfg["effective_batch"],
        "warmup_steps": args.warmup_steps, "profiled_steps": args.profile_steps, "status": "running",
    }
    harness.write_json(rd / "receipt.json", receipt)

    # Dynamo diagnostics for the warmup (compilation) phase. set_logs resets
    # torch's handlers, so attach the file handler afterwards, directly to the
    # artifact loggers (they do not propagate to the root torch logger).
    log_path = rd / "dynamo.log"
    try:
        torch._logging.set_logs(graph_breaks=True, recompiles=True)
    except Exception as exc:
        receipt["dynamo_logging_error"] = f"{type(exc).__name__}: {exc}"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    dynamo_loggers = [logging.getLogger(name) for name in (
        "torch", "torch._dynamo", "torch._dynamo.__graph_breaks", "torch._dynamo.__recompiles",
        "torch._dynamo.convert_frame", "torch._dynamo.symbolic_convert", "torch._inductor")]
    for logger in dynamo_loggers:
        logger.addHandler(handler)
    try:
        from torch._dynamo.utils import counters as dynamo_counters
        dynamo_counters.clear()
    except Exception:
        pass

    torch.set_num_threads(cfg["cpu_threads"])
    try:
        torch.manual_seed(cfg["seed"])
        started = time.perf_counter()
        model = SprintForCausalLM(SprintConfig(**config.to_dict())).to(device=device, dtype=torch.bfloat16)
        receipt["precision_metadata"] = json.loads(json.dumps(dict(apply_sprint_precision(model, precision, device)),
                                                              allow_nan=False))
        step_cfg = {**cfg, "precision": precision, "micro_batch": batch, "grad_accum": cfg["effective_batch"] // batch,
                    "torch_compile": mode != "eager", "compile_mode": "default" if mode == "eager" else mode,
                    "parameter_dtype": "bfloat16", "optimizer_state_offload": "cpu",
                    "max_steps": 1_000_000, "warmup_steps": 0}
        engine = SprintStepEngine(model, step_cfg, device=device)
        threshold_runtime = harness._configure_cpu_adam_threshold(engine, step_cfg)
        receipt["runtime"] = {
            "parameter_dtype": engine.parameter_dtype, "compute_dtype": engine.compute_dtype,
            "optimizer": type(engine.optimizer).__name__, "cpu_optimizer_kernel": getattr(engine, "cpu_optimizer_kernel", None),
            "cpu_optimizer_kernel_reason": getattr(engine.optimizer, "cpu_kernel_reason", None),
            "cpu_threads": torch.get_num_threads(), "offload_bucket_mb": cfg["offload_bucket_mb"],
            "attention_backend": engine.attention_backend, "experimental_cpu_adam_threshold": threshold_runtime,
        }
        marked = _install_markers(engine, torch)
        harness._sync(device)
        receipt["initialization_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        for step in range(args.warmup_steps):
            marked.calls = 0
            result = engine.train_step(batches, step=step)
        harness._sync(device)
        receipt["warmup_and_compile_seconds"] = time.perf_counter() - started
        receipt["warmup_last_loss"] = float(result.loss)
        receipt["dynamo_counters"] = _dynamo_counters()
        harness.write_json(rd / "dynamo_counters.json", receipt["dynamo_counters"])
        if device.type == "cuda":
            receipt["cuda_memory_after_warmup"] = {
                "allocated": torch.cuda.memory_allocated(device), "reserved": torch.cuda.memory_reserved(device),
                "max_allocated": torch.cuda.max_memory_allocated(device), "max_reserved": torch.cuda.max_memory_reserved(device)}

        from torch.profiler import ProfilerActivity, profile, record_function

        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
        update_wall = []
        harness._sync(device)
        with profile(activities=activities, record_shapes=True, profile_memory=False, with_stack=False) as prof:
            for index in range(args.profile_steps):
                marked.calls = 0
                begun = time.perf_counter()
                with record_function(f"{MARK_UPDATE}[{index}]"):
                    result = engine.train_step(batches, step=args.warmup_steps + index)
                    harness._sync(device)
                update_wall.append(time.perf_counter() - begun)
        receipt["profiled_update_wall_seconds"] = update_wall
        receipt["profiled_last_loss"] = float(result.loss)
        trace_json = rd / "trace.json"
        prof.export_chrome_trace(str(trace_json))
        trace_path = rd / "trace.json.gz"
        with trace_json.open("rb") as source, gzip.open(trace_path, "wb", compresslevel=6) as target:
            while chunk := source.read(1 << 20):
                target.write(chunk)
        trace_json.unlink()
        receipt["trace_path"] = str(trace_path)
        receipt["trace_sha256"] = _sha256(trace_path)
        if hasattr(engine.optimizer, "synchronize"):
            engine.optimizer.synchronize()
        receipt["status"] = "ok"
    except Exception as exc:
        receipt.update(status="error", error=f"{type(exc).__name__}: {exc}")
        harness.write_json(rd / "receipt.json", receipt)
        raise
    finally:
        torch.set_num_threads(original_threads)
        for logger in dynamo_loggers:
            logger.removeHandler(handler)
        handler.close()
    harness.write_json(rd / "receipt.json", receipt)
    return trace_path


# ----------------------------------------------------------------------- analyze


def _load_trace(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["traceEvents"] if isinstance(data, dict) else data


def _phase_of(path: list[str]) -> str:
    for name in reversed(path):
        if name.startswith(MARK_FORWARD):
            return "forward"
        if name.startswith(MARK_CLIP):
            return "clip"
        if name.startswith(MARK_OPTIMIZER):
            return "optimizer"
    if any(BACKWARD_HINT.search(name) for name in path):
        return "backward"
    return "other"


def _microbatch_of(path: list[str]) -> int | None:
    for name in reversed(path):
        if name.startswith(MARK_FORWARD):
            match = re.search(r"mb=(\d+)", name)
            return int(match.group(1)) if match else None
    return None


def _gemm_info(path_events: list[dict]) -> tuple[str, float] | None:
    """Precision and FLOPs of the innermost ATen GEMM op on the launch path."""
    for event in reversed(path_events):
        name = event["name"]
        if name not in GEMM_OPS:
            continue
        args = event.get("args", {})
        types = [str(t) for t in args.get("Input type", [])]
        dims = args.get("Input Dims", [])
        joined = " ".join(types)
        if "Float4" in joined:
            precision, packed = "fp4", 2
        elif "Float8" in joined:
            precision, packed = "fp8", 1
        elif "BFloat16" in joined or "Half" in joined:
            precision, packed = "bf16", 1
        else:
            precision, packed = "other", 1
        flops = 0.0
        matrices = [d for d in dims if isinstance(d, list) and len(d) >= 2]
        if name in {"aten::_scaled_mm", "aten::_scaled_mm.out", "aten::mm", "aten::mm.out", "aten::matmul"} and len(matrices) >= 2:
            a, b = matrices[0], matrices[1]
            m, k = a[-2], a[-1] * packed
            n = b[-1] if b[-2] * packed == k or b[-2] == a[-1] else b[-2]
            flops = 2.0 * m * n * k
        elif name in {"aten::addmm", "aten::addmm.out"} and len(matrices) >= 3:
            a, b = matrices[1], matrices[2]
            flops = 2.0 * a[-2] * b[-1] * a[-1]
        elif name in {"aten::bmm", "aten::baddbmm"} and len(matrices) >= 2:
            a, b = matrices[-2], matrices[-1]
            flops = 2.0 * a[0] * a[1] * a[2] * b[-1]
        elif name == "aten::linear" and len(matrices) >= 2:
            x, w = matrices[0], matrices[1]
            rows = 1
            for size in x[:-1]:
                rows *= size
            flops = 2.0 * rows * w[-2] * w[-1]
        return precision, flops
    return None


def _family(name: str, cat: str, gemm: tuple[str, float] | None) -> str:
    if gemm is not None:
        return f"gemm {gemm[0]}"
    if cat == "gpu_memcpy":
        return "memcpy"
    if cat == "gpu_memset":
        return "memset"
    for pattern, family in FAMILY_RULES:
        if pattern.search(name):
            return family
    return "other"


def _union(intervals: list[tuple[float, float]]) -> float:
    total, current_start, current_end = 0.0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def analyze_trace(path: Path, *, gap_threshold_ms: float = 0.5, top: int = 40) -> dict:
    events = [e for e in _load_trace(path) if e.get("ph") == "X" and "ts" in e and "dur" in e]
    scope_cats = {"cpu_op", "user_annotation"}
    launch_cats = {"cuda_runtime", "cuda_driver"}
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}

    # Per-thread sweep: ancestors (scope stack) of every launch, keyed by correlation id.
    by_thread: dict[tuple, list[dict]] = defaultdict(list)
    for event in events:
        if event.get("cat") in scope_cats or event.get("cat") in launch_cats:
            by_thread[(event.get("pid"), event.get("tid"))].append(event)
    launch_path: dict[int, list[dict]] = {}
    for thread_events in by_thread.values():
        thread_events.sort(key=lambda e: (e["ts"], -e["dur"]))
        stack: list[dict] = []
        for event in thread_events:
            while stack and stack[-1]["ts"] + stack[-1]["dur"] < event["ts"]:
                stack.pop()
            if event.get("cat") in launch_cats:
                correlation = event.get("args", {}).get("correlation")
                if correlation is not None:
                    launch_path[int(correlation)] = list(stack)
            else:
                stack.append(event)

    # Update windows come from the synchronized main-thread markers.
    updates = sorted((e for e in events if e.get("cat") in scope_cats and str(e["name"]).startswith(MARK_UPDATE)),
                     key=lambda e: e["ts"])
    if not updates:
        raise ValueError("trace has no sprint.update markers; was it produced by this script?")
    update_starts = [u["ts"] for u in updates]

    gpu_events = []
    for event in events:
        if event.get("cat") not in gpu_cats:
            continue
        correlation = event.get("args", {}).get("correlation")
        path_events = launch_path.get(int(correlation), []) if correlation is not None else []
        names = [str(p["name"]) for p in path_events]
        gemm = _gemm_info(path_events)
        index = bisect.bisect_right(update_starts, event["ts"]) - 1
        update = index if 0 <= index and event["ts"] <= updates[index]["ts"] + updates[index]["dur"] else None
        gpu_events.append({
            "name": str(event["name"]), "cat": event["cat"], "ts": event["ts"], "dur": event["dur"],
            "end": event["ts"] + event["dur"], "update": update, "phase": _phase_of(names),
            "microbatch": _microbatch_of(names), "family": _family(str(event["name"]), event["cat"], gemm),
            "gemm": gemm, "compiled": any(COMPILED_REGION.search(n) for n in names),
            "parent": names[-1] if names else None,
        })
    gpu_events.sort(key=lambda e: e["ts"])

    summary = {"schema": "yylab-sprint-step-profile-summary-v1", "trace": str(path), "updates": []}
    per_update = defaultdict(list)
    for event in gpu_events:
        if event["update"] is not None:
            per_update[event["update"]].append(event)

    for index, update in enumerate(updates):
        items = per_update.get(index, [])
        wall_ms = update["dur"] / 1e3
        intervals = [(e["ts"], e["end"]) for e in items]
        busy_ms = _union(intervals) / 1e3
        # Idle gaps on the device timeline within this update.
        gaps = []
        if items:
            current_end = items[0]["end"]
            previous = items[0]
            for event in items[1:]:
                gap = (event["ts"] - current_end) / 1e3
                if gap >= gap_threshold_ms:
                    gaps.append({"gap_ms": round(gap, 3), "after": previous["name"][:80], "after_phase": previous["phase"],
                                 "before": event["name"][:80], "before_phase": event["phase"],
                                 "at_ms": round((current_end - update["ts"]) / 1e3, 3)})
                if event["end"] > current_end:
                    current_end, previous = event["end"], event
        gaps.sort(key=lambda g: -g["gap_ms"])

        phase_ms = defaultdict(float)
        family_ms = defaultdict(float)
        phase_family_ms = defaultdict(lambda: defaultdict(float))
        kernel_ms = defaultdict(lambda: {"ms": 0.0, "count": 0, "family": None, "phase": Counter(), "parent": Counter()})
        gemm_stats = defaultdict(lambda: {"ms": 0.0, "count": 0, "flop": 0.0})
        compiled_ms = defaultdict(float)
        for event in items:
            ms = event["dur"] / 1e3
            phase_ms[event["phase"]] += ms
            family_ms[event["family"]] += ms
            phase_family_ms[event["phase"]][event["family"]] += ms
            entry = kernel_ms[event["name"]]
            entry["ms"] += ms
            entry["count"] += 1
            entry["family"] = event["family"]
            entry["phase"][event["phase"]] += 1
            entry["parent"][event["parent"] or "-"] += 1
            if event["gemm"] is not None:
                stats = gemm_stats[(event["gemm"][0], event["phase"])]
                stats["ms"] += ms
                stats["count"] += 1
                stats["flop"] += event["gemm"][1]
            if event["phase"] in {"forward", "backward"}:
                compiled_ms["inside compiled region" if event["compiled"] else "outside compiled region"] += ms

        # Device-timeline spans per phase and per microbatch (forward/backward).
        def span(selected):
            return (min(e["ts"] for e in selected), max(e["end"] for e in selected)) if selected else None

        phase_span = {}
        for phase in ("forward", "backward", "clip", "optimizer", "other"):
            selected = [e for e in items if e["phase"] == phase]
            s = span(selected)
            phase_span[phase] = {"kernel_ms": round(phase_ms[phase], 3), "count": len(selected),
                                 "span_ms": round((s[1] - s[0]) / 1e3, 3) if s else 0.0,
                                 "busy_ms": round(_union([(e["ts"], e["end"]) for e in selected]) / 1e3, 3)}
        forwards = sorted({e["microbatch"] for e in items if e["phase"] == "forward" and e["microbatch"] is not None})
        microbatches = []
        backward_items = [e for e in items if e["phase"] == "backward"]
        for position, mb in enumerate(forwards):
            f_items = [e for e in items if e["phase"] == "forward" and e["microbatch"] == mb]
            f_span = span(f_items)
            next_f = [e for e in items if e["phase"] == "forward" and e["microbatch"] == forwards[position + 1]] if position + 1 < len(forwards) else []
            limit = span(next_f)[0] if next_f else float("inf")
            b_items = [e for e in backward_items if f_span and f_span[1] <= e["ts"] < limit]
            b_span = span(b_items)
            microbatches.append({
                "microbatch": mb,
                "forward_kernel_ms": round(sum(e["dur"] for e in f_items) / 1e3, 3),
                "forward_span_ms": round((f_span[1] - f_span[0]) / 1e3, 3) if f_span else 0.0,
                "backward_kernel_ms": round(sum(e["dur"] for e in b_items) / 1e3, 3),
                "backward_span_ms": round((b_span[1] - b_span[0]) / 1e3, 3) if b_span else 0.0,
                "forward_kernels": len(f_items), "backward_kernels": len(b_items),
            })
        last_backward = max((e["end"] for e in backward_items), default=None)
        first_forward_next = None
        if index + 1 < len(updates):
            nxt = [e for e in per_update.get(index + 1, []) if e["phase"] == "forward"]
            first_forward_next = min((e["ts"] for e in nxt), default=None)
        tail_ms = ((first_forward_next - last_backward) / 1e3 if first_forward_next and last_backward
                   else (update["ts"] + update["dur"] - last_backward) / 1e3 if last_backward else None)

        top_kernels = sorted(kernel_ms.items(), key=lambda kv: -kv[1]["ms"])[:top]
        gemm_rows = []
        for (precision, phase), stats in sorted(gemm_stats.items()):
            tflops = stats["flop"] / (stats["ms"] / 1e3) / 1e12 if stats["ms"] else 0.0
            peak = {"fp4": PEAK_TFLOPS["fp4"], "fp8": PEAK_TFLOPS["fp8_fp16acc"], "bf16": PEAK_TFLOPS["bf16"]}.get(precision)
            gemm_rows.append({"precision": precision, "phase": phase, "count": stats["count"], "ms": round(stats["ms"], 3),
                              "tflop": round(stats["flop"] / 1e12, 3), "achieved_tflops": round(tflops, 1),
                              "fraction_of_dense_peak": round(tflops / peak, 3) if peak else None,
                              "peak_reference_tflops": peak})
        summary["updates"].append({
            "update": index, "wall_ms": round(wall_ms, 3), "device_busy_ms": round(busy_ms, 3),
            "device_idle_ms": round(wall_ms - busy_ms, 3), "device_busy_fraction": round(busy_ms / wall_ms, 4) if wall_ms else None,
            "gpu_events": len(items), "distinct_kernel_names": len(kernel_ms),
            "mean_kernel_us": round(sum(e["dur"] for e in items) / len(items), 2) if items else None,
            "phase": phase_span, "microbatches": microbatches,
            "tail_after_last_backward_ms": round(tail_ms, 3) if tail_ms is not None else None,
            "family_ms": {k: round(v, 3) for k, v in sorted(family_ms.items(), key=lambda kv: -kv[1])},
            "phase_family_ms": {p: {k: round(v, 3) for k, v in sorted(f.items(), key=lambda kv: -kv[1])} for p, f in phase_family_ms.items()},
            "compiled_region_ms": {k: round(v, 3) for k, v in compiled_ms.items()},
            "gemm": gemm_rows,
            "top_kernels": [{"name": name, "ms": round(v["ms"], 3), "count": v["count"], "family": v["family"],
                             "phases": dict(v["phase"]), "parent": v["parent"].most_common(1)[0][0]} for name, v in top_kernels],
            "idle_gaps": gaps[:25],
        })
    return summary


def _markdown(summary: dict, receipt: dict | None) -> str:
    lines = ["# Sprint step profile", ""]
    if receipt:
        case = receipt.get("case", {})
        lines += [f"Recipe `{receipt.get('config_path')}`; {receipt.get('hardware')}; torch {receipt.get('runtime_versions', {}).get('torch')}, "
                  f"torchao {receipt.get('runtime_versions', {}).get('torchao')}; precision `{case.get('precision')}`, "
                  f"microbatch {case.get('micro_batch')} x {case.get('grad_accum')}, compile `{case.get('compile_mode')}`; "
                  f"{receipt.get('planned_parameters', 0):,} parameters.", "",
                  "Profiler instrumentation and the per-update synchronization change overlap. These are diagnostics, not throughput results.", ""]
        counters = receipt.get("dynamo_counters", {})
        breaks = counters.get("graph_break", {})
        frames = counters.get("frames", {})
        lines += ["## Dynamo (warmup)", "",
                  f"Frames: {frames}. Graph breaks recorded: {sum(int(v) for v in breaks.values()) if breaks else 0}.", ""]
        for reason, count in sorted(breaks.items(), key=lambda kv: -int(kv[1]))[:20]:
            lines.append(f"- {count} x {reason}")
        lines.append("")
    for update in summary["updates"]:
        lines += [f"## Update {update['update']}", "",
                  f"Wall {update['wall_ms']:.1f} ms; device busy {update['device_busy_ms']:.1f} ms "
                  f"({(update['device_busy_fraction'] or 0):.1%}); idle {update['device_idle_ms']:.1f} ms; "
                  f"{update['gpu_events']} device events, {update['distinct_kernel_names']} distinct kernels, "
                  f"mean {update['mean_kernel_us']} us; tail after last backward kernel {update['tail_after_last_backward_ms']} ms.", "",
                  "| Phase | Kernel ms | Busy ms | Span ms | Events |", "| --- | ---: | ---: | ---: | ---: |"]
        for phase, row in update["phase"].items():
            lines.append(f"| {phase} | {row['kernel_ms']:.1f} | {row['busy_ms']:.1f} | {row['span_ms']:.1f} | {row['count']} |")
        lines += ["", "| Microbatch | Fwd kernel ms | Fwd span ms | Bwd kernel ms | Bwd span ms | Fwd / bwd kernels |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for mb in update["microbatches"]:
            lines.append(f"| {mb['microbatch']} | {mb['forward_kernel_ms']:.1f} | {mb['forward_span_ms']:.1f} | {mb['backward_kernel_ms']:.1f} | {mb['backward_span_ms']:.1f} | {mb['forward_kernels']} / {mb['backward_kernels']} |")
        lines += ["", "| Kernel family | ms | Share of busy |", "| --- | ---: | ---: |"]
        busy = update["device_busy_ms"] or 1.0
        for family, ms in update["family_ms"].items():
            lines.append(f"| {family} | {ms:.1f} | {ms / busy:.1%} |")
        if update["compiled_region_ms"]:
            lines += ["", "Forward/backward kernel time by launch site: " + ", ".join(f"{k} {v:.1f} ms" for k, v in update["compiled_region_ms"].items()) + ".", ""]
        lines += ["", "| GEMM | Phase | Count | ms | TFLOP | Achieved TFLOP/s | Fraction of dense peak |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in update["gemm"]:
            fraction = f"{row['fraction_of_dense_peak']:.0%} of {row['peak_reference_tflops']}" if row["fraction_of_dense_peak"] is not None else "-"
            lines.append(f"| {row['precision']} | {row['phase']} | {row['count']} | {row['ms']:.1f} | {row['tflop']:.2f} | {row['achieved_tflops']:.0f} | {fraction} |")
        lines += ["", "| Top kernels | Family | Phase(s) | Count | ms | Launched from |", "| --- | --- | --- | ---: | ---: | --- |"]
        for kernel in update["top_kernels"]:
            phases = ", ".join(f"{p}:{c}" for p, c in kernel["phases"].items())
            lines.append(f"| `{kernel['name'][:70]}` | {kernel['family']} | {phases} | {kernel['count']} | {kernel['ms']:.1f} | `{kernel['parent'][:40]}` |")
        lines += ["", "| Largest device idle gaps | At ms | After (phase) | Before (phase) |", "| --- | ---: | --- | --- |"]
        for gap in update["idle_gaps"][:12]:
            lines.append(f"| {gap['gap_ms']:.2f} ms | {gap['at_ms']:.1f} | `{gap['after'][:40]}` ({gap['after_phase']}) | `{gap['before'][:40]}` ({gap['before_phase']}) |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="official Sprint precision recipe JSON")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sprint-step-profile"))
    parser.add_argument("--precision", help="override the recipe's first precision")
    parser.add_argument("--batch", type=int, help="override the recipe's first microbatch")
    parser.add_argument("--compile-mode", help="override the recipe's first compile mode (eager disables torch.compile)")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--gap-threshold-ms", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--analyze-only", type=Path, help="re-analyze an existing trace.json[.gz] without running anything")
    args = parser.parse_args()
    if args.analyze_only is not None:
        trace_path = args.analyze_only
        rd = trace_path.parent
    else:
        if args.config is None:
            parser.error("--config is required unless --analyze-only is given")
        trace_path = run_profile(args)
        rd = Path(args.output)
    summary = analyze_trace(trace_path, gap_threshold_ms=args.gap_threshold_ms, top=args.top)
    receipt_path = rd / "receipt.json"
    receipt = json.loads(receipt_path.read_text("utf-8")) if receipt_path.is_file() else None
    (rd / "profile_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (rd / "profile_summary.md").write_text(_markdown(summary, receipt), encoding="utf-8")
    for update in summary["updates"]:
        print(f"update {update['update']}: wall {update['wall_ms']:.1f} ms, device busy {update['device_busy_ms']:.1f} ms "
              f"({(update['device_busy_fraction'] or 0):.1%}), {update['gpu_events']} device events")
        for phase, row in update["phase"].items():
            print(f"  {phase:9s} kernel {row['kernel_ms']:8.1f} ms  span {row['span_ms']:8.1f} ms  events {row['count']}")
        print("  families: " + ", ".join(f"{k} {v:.0f}" for k, v in list(update["family_ms"].items())[:8]))
    print(f"wrote {rd / 'profile_summary.md'}")


if __name__ == "__main__":
    main()
