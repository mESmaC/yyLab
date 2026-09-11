"""Disposable, explicit-precision Sprint training comparison.

Every candidate runs real forward/backward/optimizer steps on the same seeded
model shape and synthetic effective batch. Unsupported precision never falls
back to another precision. This experiment does not establish language quality.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from itertools import product
from pathlib import Path
from typing import Any

import torch

from ..artifacts import write_json
from ..models.sprint import SprintConfig, SprintForCausalLM, estimate_parameter_count, plan_sprint_model
from .sprint_precision import DEFAULT_PRECISION_MODES, PRECISION_MODES, apply_sprint_precision

COMPILE_MODES = ("eager", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
ATTENTION_BACKENDS = ("auto", "cudnn", "efficient", "flash", "math")
PROVENANCE_SOURCE_PATHS = (
    "yylab/train/benchmark_sprint_precision.py", "yylab/train/sprint_precision.py",
    "yylab/train/mxfp8.py", "yylab/train/mxfp8_fast.py", "yylab/train/low_precision.py",
    "yylab/train/bitnet_benchmark.py", "yylab/train/bitnet_int8_benchmark.py", "yylab/train/sprint_nvfp4_training.py",
    "yylab/train/sprint_nvfp4_portable.py", "yylab/models/sprint.py",
    "yylab/train/sprint.py", "yylab/train/sprint_cpu_adam.py",
)


def _source_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in PROVENANCE_SOURCE_PATHS if (root / relative).is_file()}


def _runtime_versions() -> dict[str, str | None]:
    versions = {"torch": str(torch.__version__)}
    for name, distributions in (("torchao", ("torchao",)), ("transformers", ("transformers",)),
                                ("triton", ("triton", "triton-windows"))):
        versions[name] = None
        for distribution in distributions:
            try:
                versions[name] = metadata.version(distribution)
                break
            except metadata.PackageNotFoundError:
                continue
    return versions


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
        if isinstance(value, bool) or parsed != value or parsed < 1:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    return parsed


def _requested_cpu_adam_threshold(cfg: Mapping[str, Any]) -> int | None:
    """Validate the benchmark-only override before device/model allocation."""
    key = "experimental_cpu_adam_min_elements"
    if key not in cfg:
        return None
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be an explicit positive integer")
    if cfg.get("cpu_optimizer_kernel") != "compiled":
        raise ValueError("experimental CPU Adam threshold requires cpu_optimizer_kernel='compiled'")
    if cfg.get("optimizer_state_offload", "cpu") != "cpu":
        raise ValueError("experimental CPU Adam threshold requires CPU optimizer offload")
    return value


def _configure_cpu_adam_threshold(engine, cfg: Mapping[str, Any]) -> dict | None:
    """Configure a disposable engine before warmup; leave training defaults alone."""
    threshold = _requested_cpu_adam_threshold(cfg)
    if threshold is None:
        return None
    optimizer = engine.optimizer
    if getattr(optimizer, "_compiled_cpu_adam", None) is None:
        raise RuntimeError("experimental CPU Adam threshold requires an already prepared compiled kernel")
    if threshold > optimizer.bucket_elements:
        raise ValueError("experimental CPU Adam threshold exceeds the actual optimizer bucket capacity")
    default_threshold = optimizer._min_compiled_elements
    compiled = native = compiled_slices = native_slices = 0
    for group in optimizer.param_groups:
        for entries, _ in optimizer._buckets(group["params"]):
            for _, _, count, _ in entries:
                if count >= threshold:
                    compiled += count
                    compiled_slices += 1
                else:
                    native += count
                    native_slices += 1
    optimizer._min_compiled_elements = threshold
    return {"requested_min_elements": threshold, "effective_min_elements": optimizer._min_compiled_elements,
            "default_min_elements": default_threshold, "bucket_elements": optimizer.bucket_elements,
            "eligible_compiled_parameters": compiled, "eligible_native_parameters": native,
            "eligible_compiled_slices": compiled_slices, "eligible_native_slices": native_slices,
            "eligible_compiled_fraction": compiled / (compiled + native) if compiled + native else 0.0,
            "scope": "Eligibility from actual optimizer groups and bucket splits; only parameters with gradients update. Benchmark only; not a production/resume setting."}


def _choices(values: Any, allowed: tuple[str, ...], name: str) -> list[str]:
    if not isinstance(values, (list, tuple)) or not values or any(value not in allowed for value in values):
        raise ValueError(f"{name} must be a nonempty list containing only {', '.join(allowed)}")
    return list(dict.fromkeys(values))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _finite_metric(row: dict, name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        row[name] = None
        row[f"{name}_nonfinite"] = str(result)
        raise RuntimeError(f"{name} is not finite: {result}")
    row[name] = result
    return result


@torch.no_grad()
def _decoder_samples(model) -> dict[str, torch.Tensor]:
    """Bounded decoder-weight samples, avoiding embedding-only update evidence."""
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.ndim == 2 and name.endswith("weight") and (
            "layers" in name.split(".") or "blocks" in name.split(".")
        ):
            flat = parameter.detach().reshape(-1)
            stride = max(1, flat.numel() // 128)
            result[name] = flat[::stride][:128].to(device="cpu", dtype=torch.float32).clone()
    if not result:
        raise RuntimeError("no decoder projection weights are available for update verification")
    return result


@torch.no_grad()
def _final_gradient_norm(model) -> float:
    norms = [torch.linalg.vector_norm(parameter.grad.detach(), dtype=torch.float32)
             for parameter in model.parameters() if parameter.grad is not None]
    if not norms:
        raise RuntimeError("measured training steps produced no parameter gradients")
    return float(torch.linalg.vector_norm(torch.stack(norms), dtype=torch.float32))


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def _storage_inventory(values, *, seen: set | None = None) -> dict:
    """Count unique allocated tensor storage, including aliased offload buffers once."""
    seen = set() if seen is None else seen
    by_device, by_dtype = {}, {}
    total, count = 0, 0
    for tensor in _tensors(values):
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        key = (str(tensor.device), storage.data_ptr(), size)
        if key in seen:
            continue
        seen.add(key)
        count += 1
        total += size
        by_device[str(tensor.device)] = by_device.get(str(tensor.device), 0) + size
        by_dtype[str(tensor.dtype)] = by_dtype.get(str(tensor.dtype), 0) + size
    return {"bytes": total, "storages": count, "by_device": by_device, "by_dtype": by_dtype}


def _training_memory(model, optimizer) -> dict:
    seen = set()
    parameters = _storage_inventory(list(model.parameters()), seen=seen)
    gradients = _storage_inventory([p.grad for p in model.parameters() if p.grad is not None], seen=seen)
    state = _storage_inventory(optimizer.state, seen=seen)
    # Exclude parameter groups and state already counted above. Direct tensor
    # attributes include both staging buffers and the FP32 gradient scratch.
    buffers = _storage_inventory({key: value for key, value in vars(optimizer).items()
                                  if key not in {"state", "param_groups"}}, seen=seen)
    model_buffers = _storage_inventory({name: vars(module) for name, module in model.named_modules()}, seen=seen)
    return {"parameters": parameters, "gradients": gradients, "optimizer_state": state,
            "optimizer_buffers": buffers, "model_buffers": model_buffers,
            "total_tensor_storage_bytes": sum(value["bytes"] for value in (
                parameters, gradients, state, buffers, model_buffers)),
            "scope": "Unique retained parameter, gradient, model-buffer, optimizer-state, and direct optimizer-buffer tensor storage. CUDA allocator peaks also include execution workspaces; Python objects and compiler caches are excluded."}


def _measure_precision_case(config: SprintConfig, cfg: dict, precision: str, batch: int,
                            mode: str, device: torch.device, inputs: torch.Tensor) -> dict:
    from .sprint import SprintStepEngine

    model = engine = None
    row = {"precision": precision, "micro_batch": batch, "grad_accum": len(inputs) // batch,
           "torch_compile": mode != "eager", "compile_mode": "default" if mode == "eager" else mode,
           "parameter_dtype": "bfloat16", "optimizer_state_offload": "cpu",
           "cpu_threads": torch.get_num_threads(), "offload_bucket_mb": cfg["offload_bucket_mb"],
           "cpu_optimizer_kernel_requested": cfg["cpu_optimizer_kernel"],
           "attention_backend": cfg["attention_backend"],
           "gradient_checkpointing": cfg["gradient_checkpointing"],
           "stage": "initialization"}
    try:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(cfg["seed"])
        started = time.perf_counter()
        model = SprintForCausalLM(SprintConfig(**config.to_dict())).to(device=device, dtype=torch.bfloat16)
        row["parameters_before_conversion"] = sum(p.numel() for p in model.parameters())
        row["stage"] = "precision_conversion"
        precision_metadata = apply_sprint_precision(model, precision, device)
        if not isinstance(precision_metadata, Mapping) or precision_metadata.get("precision") != precision:
            raise RuntimeError("precision adapter did not attest the requested precision")
        # This also rejects non-JSON provider metadata before any training.
        row["precision_metadata"] = json.loads(json.dumps(dict(precision_metadata), allow_nan=False))
        row["parameters"] = sum(p.numel() for p in model.parameters())
        row["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        row["stage"] = "optimizer_initialization"
        step_cfg = {**cfg, **row, "max_steps": 1_000_000, "warmup_steps": 0}
        engine = SprintStepEngine(model, step_cfg, device=device)
        threshold_runtime = _configure_cpu_adam_threshold(engine, step_cfg)
        row["runtime"] = {
            "parameter_dtype": engine.parameter_dtype, "compute_dtype": engine.compute_dtype,
            "optimizer": type(engine.optimizer).__name__, "optimizer_state_offload": engine.optimizer_state_offload,
            "cpu_optimizer_kernel": getattr(engine, "cpu_optimizer_kernel", None),
            "cpu_optimizer_kernel_reason": getattr(engine.optimizer, "cpu_kernel_reason", None),
            "cpu_threads": torch.get_num_threads(), "offload_bucket_mb": cfg["offload_bucket_mb"],
            "attention_backend": engine.attention_backend, "torch_compile": row["torch_compile"],
            "compile_mode": row["compile_mode"], "gradient_checkpointing": cfg["gradient_checkpointing"],
        }
        if threshold_runtime is not None:
            row["runtime"]["experimental_cpu_adam_threshold"] = threshold_runtime
        batches = [{"input_ids": part, "labels": part} for part in inputs.split(batch)]
        _sync(device)
        row["initialization_seconds"] = time.perf_counter() - started
        row["stage"] = "warmup"
        started = time.perf_counter()
        for step in range(cfg["warmup_steps"]):
            result = engine.train_step(batches, step=step)
        _sync(device)
        row["warmup_and_compile_seconds"] = time.perf_counter() - started
        _finite_metric(row, "warmup_last_loss", result.loss)
        before_weights = _decoder_samples(model)
        _sync(device)
        row["stage"] = "measured_training"
        started = time.perf_counter()
        tokens = supervised_tokens = 0
        for step in range(cfg["measure_steps"]):
            result = engine.train_step(batches, step=cfg["warmup_steps"] + step)
            tokens += result.tokens
            supervised_tokens += result.supervised_tokens
        _sync(device)
        elapsed = time.perf_counter() - started
        row.update(measured_seconds=elapsed, measured_steps=cfg["measure_steps"],
                   tokens=tokens, supervised_tokens=supervised_tokens,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0)
        row["stage"] = "untimed_validation"
        _finite_metric(row, "last_loss", result.loss)
        _finite_metric(row, "final_grad_norm", _final_gradient_norm(model))
        after_weights = _decoder_samples(model)
        if any(not bool(torch.isfinite(values).all()) for values in after_weights.values()):
            raise RuntimeError("decoder parameter samples are not finite after the measured optimizer steps")
        changed = [name for name, before in before_weights.items()
                   if name in after_weights and not torch.equal(before, after_weights[name])]
        row["parameter_update"] = {"verified": bool(changed), "sampled_decoder_tensors": len(before_weights),
                                   "changed_sample_tensors": len(changed), "example": changed[0] if changed else None,
                                   "scope": "Up to 128 evenly spaced values per decoder matrix, before versus after measured steps; embeddings excluded."}
        if not changed:
            raise RuntimeError("no decoder parameter update was observed during measured optimizer steps")
        if tokens <= 0 or supervised_tokens <= 0 or elapsed <= 0:
            raise RuntimeError("measurement produced no positive token counts or elapsed time")
        memory = _training_memory(model, engine.optimizer)
        row["memory"] = memory
        row["optimizer_state_bytes"] = memory["optimizer_state"]["bytes"]
        row["cpu_optimizer_state_bytes"] = memory["optimizer_state"]["by_device"].get("cpu", 0)
        row["offload_buffer_bytes"] = memory["optimizer_buffers"]["bytes"]
        row["tokens_per_second"] = tokens / elapsed
        row["supervised_tokens_per_second"] = supervised_tokens / elapsed
        row["seconds_per_step"] = elapsed / cfg["measure_steps"]
        row["projected_seconds_per_billion_targets"] = 1e9 / row["supervised_tokens_per_second"]
        row.update(status="ok", stage="complete")
    except torch.OutOfMemoryError as exc:
        row.update(status="oom", error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        if engine is not None and hasattr(engine.optimizer, "synchronize"):
            try:
                engine.optimizer.synchronize()
            except Exception as exc:
                row.update(status="error", cleanup_error=f"{type(exc).__name__}: {exc}")
        del engine, model
        gc.collect()
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                row.update(status="error", cleanup_error=f"{type(exc).__name__}: {exc}")
    return row


def run_sprint_precision_benchmark(cfg: dict, rd: Path) -> dict:
    cfg = dict(cfg)
    original_threads = torch.get_num_threads()
    for key, default in (("effective_batch", 8), ("seq_len", 512), ("warmup_steps", 5),
                         ("measure_steps", 20), ("offload_bucket_mb", 32), ("cpu_threads", original_threads)):
        cfg[key] = _positive_int(cfg.get(key, default), key)
    if cfg["seq_len"] < 2:
        raise ValueError("seq_len must be at least two for shifted causal supervision")
    cfg["seed"] = int(cfg.get("seed", 42))
    cfg["lr"] = float(cfg.get("lr", 3e-4))
    if not math.isfinite(cfg["lr"]) or cfg["lr"] <= 0:
        raise ValueError("lr must be finite and positive")
    raw_batches = cfg.get("batch_sizes", [1, 2, 4, 8])
    if not isinstance(raw_batches, (list, tuple)) or not raw_batches:
        raise ValueError("batch_sizes must be a nonempty list")
    batches = list(dict.fromkeys(_positive_int(value, "batch_sizes") for value in raw_batches))
    if any(cfg["effective_batch"] % batch for batch in batches):
        raise ValueError("every microbatch must divide the fixed effective_batch")
    precisions = _choices(cfg.get("precisions", list(DEFAULT_PRECISION_MODES)), PRECISION_MODES, "precisions")
    compile_modes = _choices(cfg.get("compile_modes", ["eager"]), COMPILE_MODES, "compile_modes")
    cfg.setdefault("cpu_optimizer_kernel", "auto")
    cfg.setdefault("attention_backend", "auto")
    cfg.setdefault("gradient_checkpointing", False)
    if cfg["cpu_optimizer_kernel"] not in {"auto", "native", "compiled"}:
        raise ValueError("invalid cpu_optimizer_kernel")
    _requested_cpu_adam_threshold(cfg)
    if cfg["attention_backend"] not in ATTENTION_BACKENDS:
        raise ValueError("invalid attention_backend")
    if not isinstance(cfg["gradient_checkpointing"], bool):
        raise ValueError("gradient_checkpointing must be Boolean")
    if cfg.get("parameter_dtype", "bfloat16") not in {"bf16", "bfloat16"}:
        raise ValueError("all precision candidates must use BF16 trainable parameter storage")
    if cfg.get("optimizer_state_offload", "cpu") != "cpu":
        raise ValueError("this comparison uses identical FP32 CPU master weights and optimizer states in every mode")
    raw_device = str(cfg.get("device", "auto"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if raw_device == "auto" else torch.device(raw_device)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("precision benchmark device must be CPU or CUDA")
    config = SprintConfig(**cfg["model_config"]) if cfg.get("model_config") else plan_sprint_model(
        _positive_int(cfg.get("target_parameters", 50_000_000), "target_parameters"),
        _positive_int(cfg.get("vocab_size", 8192), "vocab_size"),
        max_position_embeddings=int(cfg.get("max_position_embeddings", max(2048, cfg["seq_len"]))),
        profile=cfg.get("shape_profile", "throughput"),
    )
    config.validate()
    if cfg["seq_len"] > config.max_position_embeddings:
        raise ValueError("benchmark sequence length exceeds model context")
    rd = Path(rd)
    rd.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    inputs = torch.randint(config.vocab_size, (cfg["effective_batch"], cfg["seq_len"]), generator=generator)
    if device.type == "cuda":
        inputs = inputs.pin_memory()
    # Capture reproducibility metadata before model creation, outside every
    # initialization/warmup/measurement timer. Keep this snapshot separate from
    # cfg so adding receipt fields cannot change the numerical execution path.
    recipe = json.loads(json.dumps({
        **cfg, "model_config": config.to_dict(), "device": str(device),
        "precisions": precisions, "batch_sizes": batches, "compile_modes": compile_modes,
        "parameter_dtype": "bfloat16", "optimizer_state_offload": "cpu",
        "compute_dtype": cfg.get("compute_dtype", "auto"),
        "weight_decay": cfg.get("weight_decay", 0.1), "betas": cfg.get("betas", [0.9, 0.95]),
        "eps": cfg.get("eps", 1e-8), "grad_clip": cfg.get("grad_clip", 1.0),
        "min_lr_ratio": cfg.get("min_lr_ratio", 0.1),
    }, allow_nan=False))
    report = {
        "schema": "yylab-sprint-precision-benchmark-v1", "experimental": True, "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_sha256": _source_sha256(), "runtime_versions": _runtime_versions(), "recipe": recipe,
        "optimizer_schedule": {"max_steps": 1_000_000, "warmup_steps": 0},
        "device": str(device), "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__, "host_processor": platform.processor(), "host_logical_cores": os.cpu_count(),
        "model_config": config.to_dict(), "planned_parameters": estimate_parameter_count(config),
        "sequence_length": cfg["seq_len"], "effective_batch": cfg["effective_batch"], "seed": cfg["seed"],
        "warmup_steps": cfg["warmup_steps"], "measure_steps": cfg["measure_steps"],
        "comparison": {"precisions": precisions, "batch_sizes": batches, "compile_modes": compile_modes,
                       "cpu_threads": cfg["cpu_threads"], "offload_bucket_mb": cfg["offload_bucket_mb"],
                       "cpu_optimizer_kernel": cfg["cpu_optimizer_kernel"], "attention_backend": cfg["attention_backend"],
                       "gradient_checkpointing": cfg["gradient_checkpointing"], "lr": cfg["lr"]},
        "initialization": "Every seeded model is rounded to BF16 before its precision adapter and optimizer are created. FP32 CPU masters therefore begin as copies of the same BF16-rounded initialization. BitNet may change the parameter count; each row records the exact result.",
        "measurement": "Synchronized complete forward, backward, gradient clipping, and optimizer steps on repeated synthetic tokens including CPU-to-device transfer. Initialization, warmup/compilation, parameter-update checks, gradient-norm checks, data preparation, evaluation, and checkpoint writes are excluded from steady-state throughput. Final gradient norms describe the accumulated gradients after clipping. No convergence or final-quality claim.",
        "cases": [],
    }
    total = len(precisions) * len(batches) * len(compile_modes)
    write_json(rd / "benchmark.json", report)
    try:
        torch.set_num_threads(cfg["cpu_threads"])
        for precision, mode, batch in product(precisions, compile_modes, batches):
            message = f"Sprint precision benchmark: {precision}, batch {batch}, accumulation {cfg['effective_batch'] // batch}, {mode}"
            print(message, flush=True)
            write_json(rd / "progress.json", {"stage": "Sprint precision benchmark", "step": len(report["cases"]),
                       "max_steps": total, "metric_label": "tokens/s", "value": 0, "msg": message})
            row = _measure_precision_case(config, cfg, precision, batch, mode, device, inputs)
            report["cases"].append(row)
            write_json(rd / "benchmark.json", report)
            print(json.dumps(row, allow_nan=False), flush=True)
        successful = [row for row in report["cases"] if row["status"] == "ok"]
        failed = len(report["cases"]) - len(successful)
        report.update(status="failed" if not successful else "partial" if failed else "ok",
                      successful_cases=len(successful), failed_cases=failed)
        if successful:
            best = max(successful, key=lambda row: row["supervised_tokens_per_second"])
            report["fastest_measured"] = {key: best[key] for key in (
                "precision", "micro_batch", "grad_accum", "torch_compile", "compile_mode",
                "supervised_tokens_per_second", "parameters")}
            report["fastest_measured"]["runtime"] = best["runtime"]
        write_json(rd / "benchmark.json", report)
        write_json(rd / "summary.json", report)
        write_json(rd / "progress.json", {"stage": "Sprint precision benchmark", "step": total, "max_steps": total,
                   "metric_label": "tokens/s", "value": report.get("fastest_measured", {}).get("supervised_tokens_per_second", 0),
                   "msg": f"{report['status']}: {len(successful)} successful cases; {failed} failed cases"})
        if not successful:
            raise RuntimeError("All Sprint precision candidates failed; inspect benchmark.json for explicit per-mode errors")
        return report
    finally:
        torch.set_num_threads(original_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sprint-precision-benchmark"))
    parser.add_argument("--target-params", type=int, default=50_000_000)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--shape-profile", choices=["throughput", "balanced"], default="throughput")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--precisions", default=",".join(DEFAULT_PRECISION_MODES))
    parser.add_argument("--compile-modes", default="eager")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--offload-bucket-mb", type=int, default=32)
    parser.add_argument("--cpu-optimizer-kernel", choices=["auto", "native", "compiled"], default="auto")
    parser.add_argument("--cpu-optimizer-min-elements", type=int,
                        help="Experimental benchmark-only compiled CPU Adam cutoff; requires --cpu-optimizer-kernel compiled")
    parser.add_argument("--attention-backend", choices=ATTENTION_BACKENDS, default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text("utf-8")) if args.config else {
        "target_parameters": args.target_params, "vocab_size": args.vocab_size, "shape_profile": args.shape_profile,
        "seq_len": args.seq_len, "effective_batch": args.effective_batch,
        "batch_sizes": [int(value) for value in args.batch_sizes.split(",")],
        "precisions": [value.strip() for value in args.precisions.split(",")],
        "compile_modes": [value.strip() for value in args.compile_modes.split(",")],
        "warmup_steps": args.warmup_steps, "measure_steps": args.measure_steps, "device": args.device,
        "offload_bucket_mb": args.offload_bucket_mb, "cpu_optimizer_kernel": args.cpu_optimizer_kernel,
        "attention_backend": args.attention_backend, "gradient_checkpointing": args.gradient_checkpointing,
        "lr": args.lr, "seed": args.seed,
        **({"cpu_threads": args.cpu_threads} if args.cpu_threads is not None else {}),
    }
    if args.cpu_optimizer_min_elements is not None:
        cfg["experimental_cpu_adam_min_elements"] = args.cpu_optimizer_min_elements
    report = run_sprint_precision_benchmark(cfg, args.output)
    print(json.dumps({key: report.get(key) for key in ("status", "successful_cases", "failed_cases", "fastest_measured")}, indent=2))
    if report["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
