"""Synthetic complete-step benchmark for yyLab's joint learned per-token MoLE path."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from ..models.precision_capabilities import model_compute_dtype
from ..mole.inject import DEFAULT_TARGET_MODULES
from .models import canonical_json
from .quality import _artifact_lock

ProgressCallback = Callable[[str, int, int, float, str], None]


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def synthetic_tokens_per_step(cfg: Mapping[str, object]) -> int:
    return (
        int(cfg.get("seq_len") or 2048) * int(cfg.get("micro_batch") or 1) * int(cfg.get("grad_accum") or 1)
    )


def summarize_step_metrics(
    measured: Sequence[Mapping[str, object]],
    *,
    useful_tokens_per_step: int,
) -> dict[str, object]:
    seconds = [float(item["step_seconds"]) for item in measured]
    losses = [float(item["loss"]) for item in measured]
    total_seconds = sum(seconds)
    tokens = useful_tokens_per_step * len(measured)
    return {
        "measured_steps": len(measured),
        "step_seconds_mean": statistics.fmean(seconds) if seconds else None,
        "step_seconds_median": percentile(seconds, 0.5),
        "step_seconds_p95": percentile(seconds, 0.95),
        "useful_tokens": tokens,
        "useful_tokens_per_second": tokens / total_seconds if total_seconds > 0 else None,
        "loss_mean": statistics.fmean(losses) if losses else None,
        "loss_finite": bool(losses) and all(math.isfinite(value) for value in losses),
    }


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _candidate_fingerprint(receipt: Mapping[str, object]) -> str:
    checkpoint_lock = receipt.get("checkpoint_lock")
    return hashlib.sha256(
        canonical_json(
            {
                "configuration": {
                    key: value
                    for key, value in receipt.items()
                    if key
                    not in {
                        "schema_version",
                        "created_at",
                        "label",
                        "checkpoint_id",
                        "gpu_before",
                        "checkpoint_lock",
                        "candidate_fingerprint",
                    }
                },
                "checkpoint_identity": (
                    checkpoint_lock.get("sha256") or checkpoint_lock.get("ref")
                    if isinstance(checkpoint_lock, Mapping)
                    else None
                ),
            }
        ).encode("utf-8")
    ).hexdigest()


def _prepare_kbit(model, quant: str) -> None:
    if quant not in {"nf4", "int8"}:
        return
    from peft import prepare_model_for_kbit_training

    prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)


def _enable_grad_ckpt(model) -> None:
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    enable_inputs = getattr(model, "enable_input_require_grads", None)
    if callable(enable_inputs):
        enable_inputs()


def _enable_liger(model, enabled: bool, full_kernels: bool) -> bool:
    if not enabled or getattr(model.config, "model_type", None) != "qwen3":
        return False
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        from liger_kernel.transformers.model.qwen3 import lce_forward

        apply_liger_kernel_to_qwen3(
            rope=full_kernels,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=full_kernels,
            swiglu=full_kernels,
            model=model,
        )
        if getattr(model.forward, "__func__", None) is not lce_forward:
            raise RuntimeError("Liger did not install the fused Qwen3 forward")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Qwen3 fused linear cross-entropy setup failed") from exc
    return True


def _gpu_snapshot() -> dict[str, object]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return {
            "name": str(name),
            "memory_used_gb": memory.used / 2**30,
            "memory_total_gb": memory.total / 2**30,
        }
    except Exception:  # noqa: BLE001
        return {}


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    if summary["status"] != "ok":
        result = f"{str(summary['status']).upper()}: {summary.get('error', 'benchmark did not run')}"
    else:
        result = (
            f"{float(summary['useful_tokens_per_second']):.1f} useful tok/s; "
            f"median {float(summary['step_seconds_median']):.3f}s/step; "
            f"p95 {float(summary['step_seconds_p95']):.3f}s/step"
        )
    path.write_text(
        "\n".join(
            [
                f"# yyLab training performance — {summary.get('label') or 'run'}",
                "",
                f"**{result}**",
                "",
                f"Checkpoint: `{summary.get('checkpoint_ref')}`",
                "",
                (
                    f"MoLE: E{summary.get('num_experts')}/r{summary.get('rank')}/"
                    f"k{summary.get('top_k')} · seq {summary.get('seq_len')} · "
                    f"batch {summary.get('micro_batch')} × accum {summary.get('grad_accum')}"
                ),
                "",
                f"Optimizer: `{summary.get('optimizer_impl')}` ({summary.get('optimizer_state')})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_training_performance(
    cfg: Mapping[str, object],
    output_dir: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Benchmark the actual yyLab per-token MoLE forward/backward/optimizer path.

    Warmup steps are excluded. A candidate is never silently resized after an
    OOM; the requested shape and explicit OOM result are kept as the receipt.
    """

    import torch
    from transformers.optimization import get_cosine_schedule_with_warmup

    from ..models.loading import load_base
    from ..mole.inject import MoLEConfig, inject_mole, set_active_expert, set_stage
    from ..mole.losses import aux_loss, router_usage_metrics
    from ..train.loop import attention_backend_context, make_optimizer

    output_dir.mkdir(parents=True, exist_ok=True)
    warmup_value = cfg.get("warmup_steps")
    warmup_steps = 1 if warmup_value is None else int(warmup_value)
    measured_steps = int(cfg.get("measured_steps") or 5)
    if warmup_steps < 0 or measured_steps < 1:
        raise ValueError("warmup must be non-negative and measured steps must be positive")
    seq_len = int(cfg.get("seq_len") or 2048)
    micro_batch = int(cfg.get("micro_batch") or 1)
    grad_accum = int(cfg.get("grad_accum") or 1)
    if min(seq_len, micro_batch, grad_accum) < 1:
        raise ValueError("sequence length, micro-batch, and accumulation must be positive")
    top_k = int(cfg.get("top_k") or 2)
    num_experts = int(cfg.get("num_experts") or 8)
    rank = int(cfg.get("rank") or 8)
    if not 1 <= top_k <= num_experts:
        raise ValueError("top-k must be within the expert count")
    gpu_before = _gpu_snapshot()
    receipt: dict[str, object] = {
        "schema_version": "yylab.training-performance-receipt.v1",
        "created_at": _now(),
        "label": str(cfg.get("label") or "training performance"),
        "checkpoint_ref": str(cfg["checkpoint_ref"]),
        "checkpoint_id": cfg.get("checkpoint_id"),
        "quant": str(cfg.get("quant") or "bf16"),
        "num_experts": num_experts,
        "rank": rank,
        "top_k": top_k,
        "seq_len": seq_len,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "warmup_steps": warmup_steps,
        "measured_steps_requested": measured_steps,
        "optimizer_state": str(cfg.get("optimizer_state") or "auto"),
        "target_modules": list(cfg.get("target_modules") or DEFAULT_TARGET_MODULES),
        "gpu_before": gpu_before,
        "require_idle_gpu": bool(cfg.get("require_idle_gpu", True)),
        "max_existing_gpu_gb": float(
            1.0 if cfg.get("max_existing_gpu_gb") is None else cfg["max_existing_gpu_gb"]
        ),
    }
    receipt["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    used_gpu = gpu_before.get("memory_used_gb")
    if (
        receipt["require_idle_gpu"]
        and isinstance(used_gpu, (int, float))
        and float(used_gpu) > float(receipt["max_existing_gpu_gb"])
    ):
        receipt["checkpoint_lock"] = {
            "ref": str(cfg["checkpoint_ref"]),
            "content_addressed": False,
            "skipped": "busy_gpu_preflight",
        }
        receipt["candidate_fingerprint"] = _candidate_fingerprint(receipt)
        _atomic_json(output_dir / "benchmark-receipt.json", receipt)
        summary = {
            **receipt,
            "schema_version": "yylab.training-performance-summary.v1",
            "status": "busy_gpu",
            "error": (
                f"GPU already has {float(used_gpu):.2f} GB in use, above the "
                f"{float(receipt['max_existing_gpu_gb']):.2f} GB benchmark limit. "
                "Close GPU applications or disable the idle-GPU requirement explicitly."
            ),
            "measured_steps": 0,
            "useful_tokens_per_second": None,
            "step_seconds_median": None,
            "step_seconds_p95": None,
        }
        _atomic_json(output_dir / "summary.json", summary)
        _write_report(output_dir / "report.md", summary)
        print(summary["error"], flush=True)
        if progress:
            progress("training performance", 0, measured_steps, 0.0, "GPU busy; benchmark skipped")
        return summary
    if progress:
        progress(
            "benchmark provenance",
            0,
            measured_steps,
            0.0,
            "Hashing checkpoint bytes",
        )
    print("benchmark provenance: hashing checkpoint bytes", flush=True)
    receipt["checkpoint_lock"] = _artifact_lock(str(cfg["checkpoint_ref"]))
    receipt["candidate_fingerprint"] = _candidate_fingerprint(receipt)
    _atomic_json(output_dir / "benchmark-receipt.json", receipt)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    measured: list[dict[str, object]] = []
    optimizer_impl = ""
    started_wall = time.perf_counter()
    try:
        model, tokenizer, cuda, resolved_quant, notes = load_base(
            str(cfg["checkpoint_ref"]),
            str(cfg.get("quant") or "bf16"),
            attn_implementation=str(cfg.get("attn_implementation") or "sdpa"),
        )
        device = torch.device("cuda" if cuda else "cpu")
        model.config.use_cache = False
        for note in notes:
            print(f"note: {note}", flush=True)
        _prepare_kbit(model, resolved_quant)
        torch.manual_seed(int(cfg.get("seed") or 42))
        if cuda:
            torch.cuda.manual_seed_all(int(cfg.get("seed") or 42))
        mole_cfg = MoLEConfig(
            num_experts=num_experts,
            rank=rank,
            alpha=float(cfg.get("alpha") or rank * 2),
            top_k=top_k,
            target_modules=tuple(receipt["target_modules"]),
            router_type="per_token",
        )
        replaced = inject_mole(
            model,
            mole_cfg,
            expert_dtype=model_compute_dtype(resolved_quant),
        )
        if not replaced:
            raise RuntimeError("no target linear modules matched the requested MoLE architecture")
        set_active_expert(model, None)
        set_stage(model, "both")
        if bool(cfg.get("grad_ckpt", True)):
            _enable_grad_ckpt(model)
        fused_ce = _enable_liger(
            model,
            bool(cfg.get("liger_fused_ce", True)),
            bool(cfg.get("liger_full_kernels", True)),
        )
        optimizer = make_optimizer(
            model,
            float(cfg.get("lr") or 2e-4),
            cuda=cuda,
            optimizer_state=str(cfg.get("optimizer_state") or "auto"),
        )
        optimizer_impl = type(optimizer).__name__
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=warmup_steps + measured_steps,
        )
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        vocab_size = int(getattr(model.config, "vocab_size", 32000) or 32000)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(cfg.get("seed") or 42))
        cpu_ids = torch.randint(
            low=16,
            high=max(17, vocab_size),
            size=(micro_batch, seq_len),
            dtype=torch.long,
            generator=generator,
        )
        print(
            f"synthetic joint per-token MoLE: E={num_experts} r={rank} k={top_k}; "
            f"seq={seq_len}; batch={micro_batch}x{grad_accum}; trainable={trainable:,}; "
            f"optimizer={optimizer_impl}; fused_ce={fused_ce}",
            flush=True,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_steps = warmup_steps + measured_steps
        prefer_cudnn = bool(cfg.get("cudnn_sdpa", True))
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(total_steps):
            if cuda:
                torch.cuda.synchronize(device)
            step_started = time.perf_counter()
            losses: list[float] = []
            for _micro in range(grad_accum):
                ids = cpu_ids.to(device=device, non_blocking=cuda)
                with attention_backend_context(model, device, enabled=prefer_cudnn):
                    output = model(input_ids=ids, labels=ids)
                    task_loss = output.loss
                    loss = task_loss / grad_accum
                    loss = (
                        loss
                        + aux_loss(
                            model,
                            top_k,
                            float(cfg.get("aux_weight") or 0.01),
                        )
                        / grad_accum
                    )
                    loss.backward()
                losses.append(float(task_loss.detach().float().item()))
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(cfg.get("grad_clip") or 1.0),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if cuda:
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - step_started
            record = {
                "step": step + 1,
                "phase": "warmup" if step < warmup_steps else "measured",
                "step_seconds": seconds,
                "loss": statistics.fmean(losses),
                "useful_tokens": seq_len * micro_batch * grad_accum,
                "useful_tokens_per_second": seq_len * micro_batch * grad_accum / seconds,
                **router_usage_metrics(model),
            }
            if step >= warmup_steps:
                measured.append(record)
                with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
                measured_index = step - warmup_steps + 1
                message = (
                    f"{record['useful_tokens_per_second']:.1f} tok/s · "
                    f"{record['step_seconds']:.3f}s · loss {record['loss']:.4f}"
                )
                print(f"[{measured_index}/{measured_steps}] {message}", flush=True)
                if progress:
                    progress(
                        "training performance",
                        measured_index,
                        measured_steps,
                        float(record["loss"]),
                        message,
                    )
            else:
                print(f"warmup [{step + 1}/{warmup_steps}] {seconds:.3f}s", flush=True)
                if step + 1 == warmup_steps and cuda:
                    torch.cuda.reset_peak_memory_stats(device)
        summarized = summarize_step_metrics(
            measured,
            useful_tokens_per_step=synthetic_tokens_per_step(cfg),
        )
        finite_parameters = all(
            bool(torch.isfinite(parameter.detach()).all().item())
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        summary: dict[str, object] = {
            **receipt,
            **summarized,
            "schema_version": "yylab.training-performance-summary.v1",
            "status": "ok",
            "resolved_quant": resolved_quant,
            "cuda": cuda,
            "optimizer_impl": optimizer_impl,
            "trainable_parameters": trainable,
            "fused_linear_cross_entropy": fused_ce,
            "parameters_finite": finite_parameters,
            "wall_seconds": time.perf_counter() - started_wall,
            "route_usage_final": router_usage_metrics(model),
        }
        try:
            import psutil

            memory = psutil.Process().memory_info()
            # Windows exposes the process lifetime peak as peak_wset; other
            # platforms retain current RSS when no portable peak is available.
            summary["peak_rss_gb"] = getattr(memory, "peak_wset", memory.rss) / 2**30
        except Exception:  # noqa: BLE001
            summary["peak_rss_gb"] = None
        if cuda:
            summary["peak_cuda_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
            summary["peak_cuda_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 2**30
        else:
            summary["peak_cuda_allocated_gb"] = 0.0
            summary["peak_cuda_reserved_gb"] = 0.0
        summary["gpu_after"] = _gpu_snapshot()
    except torch.OutOfMemoryError as exc:
        summary = {
            **receipt,
            "schema_version": "yylab.training-performance-summary.v1",
            "status": "oom",
            "error": f"{type(exc).__name__}: {exc}",
            "optimizer_impl": optimizer_impl or None,
            "measured_steps": len(measured),
            "useful_tokens_per_second": None,
            "step_seconds_median": None,
            "step_seconds_p95": None,
            "wall_seconds": time.perf_counter() - started_wall,
            "gpu_after": _gpu_snapshot(),
        }
        print("benchmark OOM; candidate was not resized", flush=True)
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _atomic_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    if progress:
        message = (
            f"{float(summary['useful_tokens_per_second']):.1f} useful tok/s"
            if summary["status"] == "ok"
            else "OOM (candidate preserved)"
        )
        progress(
            "training performance",
            int(summary.get("measured_steps") or 0),
            measured_steps,
            float(summary.get("loss_mean") or 0.0),
            message,
        )
    return summary


__all__ = [
    "percentile",
    "run_training_performance",
    "summarize_step_metrics",
    "synthetic_tokens_per_step",
]
