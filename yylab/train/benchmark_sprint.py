"""Disposable complete-step throughput tuning; never trains a registry model.

Run: python -m yylab.train.benchmark_sprint --target-params 50000000 --vocab-size 8192
All candidates keep the same model, context, tokens and effective batch. Synthetic
inputs measure compute/transfer throughput, not corpus IO or model quality.
Optional --cpu-thread-counts 4,8,12,20 --offload-bucket-sizes-mb 16,32,64
sweeps CPU optimizer settings alongside microbatch and compilation choices.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
from itertools import product
from pathlib import Path

import torch

from ..artifacts import write_json
from ..models.sprint import SprintConfig, SprintForCausalLM, estimate_parameter_count, plan_sprint_model

ATTENTION_BACKENDS = ("auto", "cudnn", "efficient", "flash", "math")


def _positive_int(value, name):
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, bool) or parsed != value or parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sweep_values(cfg, sweep_key, scalar_key, default):
    values = cfg.get(sweep_key, [cfg.get(scalar_key, default)])
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{sweep_key} must be a nonempty list of positive integers")
    return list(dict.fromkeys(_positive_int(value, sweep_key) for value in values))


def _measure_case(config, cfg, batch, mode, device, inputs):
    from .sprint import SprintStepEngine

    model = engine = None
    row = {"micro_batch": batch, "grad_accum": inputs.shape[0] // batch,
           "torch_compile": mode != "eager", "compile_mode": "default" if mode == "eager" else mode,
           "parameter_dtype": cfg.get("parameter_dtype", "bfloat16" if device.type == "cuda" else "float32"),
           "optimizer_state_offload": cfg.get("optimizer_state_offload", "cpu"),
           "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", cfg.get("activation_checkpointing", False))),
           "cpu_threads": torch.get_num_threads(), "offload_bucket_mb": int(cfg.get("offload_bucket_mb", 32)),
           "attention_backend": cfg.get("attention_backend", "auto")}
    try:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(int(cfg.get("seed", 42)))
        start = time.perf_counter()
        model = SprintForCausalLM(config)
        step_cfg = {**cfg, **row, "lr": float(cfg.get("lr", 3e-4)), "max_steps": 1000000,
                    "warmup_steps": 0}
        engine = SprintStepEngine(model, step_cfg, device=device)
        row["cpu_optimizer_kernel"] = getattr(
            engine, "cpu_optimizer_kernel", "native" if row["optimizer_state_offload"] == "cpu" else None) or "auto"
        row["cpu_optimizer_kernel_reason"] = getattr(engine.optimizer, "cpu_kernel_reason", None)
        row["cpu_threads"] = torch.get_num_threads()
        bucket_elements = getattr(engine.optimizer, "bucket_elements", None)
        row["offload_bucket_bytes"] = (
            int(bucket_elements) * next(model.parameters()).element_size() if bucket_elements is not None else None
        )
        batches = [{"input_ids": part, "labels": part} for part in inputs.split(batch)]
        _sync(device)
        row["initialization_seconds"] = time.perf_counter() - start
        warm_start = time.perf_counter()
        for step in range(cfg["warmup_steps"]):
            result = engine.train_step(batches, step=step)
        _sync(device)
        row["warmup_and_compile_seconds"] = time.perf_counter() - warm_start
        before = time.perf_counter()
        supervised_tokens = tokens = 0
        for step in range(cfg["measure_steps"]):
            result = engine.train_step(batches, step=step + cfg["warmup_steps"])
            tokens += result.tokens
            supervised_tokens += result.supervised_tokens
        _sync(device)
        elapsed = time.perf_counter() - before
        loss = float(result.loss)
        if not math.isfinite(loss):
            raise RuntimeError("benchmark produced a non-finite loss")
        row.update(status="ok", measured_seconds=elapsed, measured_steps=cfg["measure_steps"],
                   tokens=tokens, supervised_tokens=supervised_tokens,
                   tokens_per_second=tokens / elapsed,
                   supervised_tokens_per_second=supervised_tokens / elapsed,
                   seconds_per_step=elapsed / cfg["measure_steps"], last_loss=loss,
                   compute_dtype=str(engine.compute_dtype) if hasattr(engine, "compute_dtype") else "auto",
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0)
        row["cpu_optimizer_state_bytes"] = sum(
            value.numel() * value.element_size()
            for state in engine.optimizer.state.values() for value in state.values()
            if isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.ndim > 0
        )
        row["projected_seconds_per_billion_targets"] = 1e9 / row["supervised_tokens_per_second"]
    except torch.OutOfMemoryError as exc:
        row.update(status="oom", error=str(exc).splitlines()[0])
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        del engine, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def run_sprint_benchmark(cfg: dict, rd: Path) -> dict:
    cfg = dict(cfg)
    rd = Path(rd)
    rd.mkdir(parents=True, exist_ok=True)
    raw_device = str(cfg.get("device", "auto"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if raw_device == "auto" else torch.device(raw_device)
    effective_batch = _positive_int(cfg.get("effective_batch", 8), "effective_batch")
    sequence_length = _positive_int(cfg.get("seq_len", 512), "seq_len")
    if sequence_length < 2:
        raise ValueError("seq_len must be at least two for causal supervision")
    batch_sizes = [_positive_int(b, "batch_sizes") for b in cfg.get("batch_sizes", [1, 2, 4, 8])]
    if not batch_sizes or any(effective_batch % b for b in batch_sizes):
        raise ValueError("every batch size must divide effective_batch; the comparison holds it fixed")
    modes = list(cfg.get("compile_modes", ["eager", "default"]))
    if not modes or any(mode not in {"eager", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"} for mode in modes):
        raise ValueError("invalid compile_modes")
    for name, default in (("warmup_steps", 5), ("measure_steps", 20)):
        cfg[name] = _positive_int(cfg.get(name, default), name)
    original_cpu_threads = torch.get_num_threads()
    cpu_thread_counts = _sweep_values(cfg, "cpu_thread_counts", "cpu_threads", original_cpu_threads)
    bucket_sizes = _sweep_values(cfg, "offload_bucket_sizes_mb", "offload_bucket_mb", 32)
    attention_backends = cfg.get("attention_backends", [cfg.get("attention_backend", "auto")])
    if not isinstance(attention_backends, (list, tuple)) or not attention_backends or any(
        backend not in ATTENTION_BACKENDS for backend in attention_backends
    ):
        raise ValueError("attention_backends must be a nonempty list of auto, cudnn, efficient, flash, or math")
    attention_backends = list(dict.fromkeys(attention_backends))
    if cfg.get("optimizer_state_offload", "cpu") != "cpu" and len(bucket_sizes) > 1:
        raise ValueError("offload bucket sweeps require CPU optimizer state offload")
    if cfg.get("model_config"):
        config = SprintConfig(**cfg["model_config"])
    else:
        config = plan_sprint_model(_positive_int(cfg.get("target_parameters", 50000000), "target_parameters"),
                                   _positive_int(cfg.get("vocab_size", 8192), "vocab_size"),
                                   max_position_embeddings=max(sequence_length, int(cfg.get("max_position_embeddings", 2048))),
                                   profile=cfg.get("shape_profile", "throughput"))
    if sequence_length > config.max_position_embeddings:
        raise ValueError("benchmark sequence exceeds the model context")
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 42)))
    inputs = torch.randint(config.vocab_size, (effective_batch, sequence_length), generator=generator)
    if device.type == "cuda":
        inputs = inputs.pin_memory()
    report = {
        "schema": "yylab-sprint-benchmark-v1", "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__, "device": str(device), "model_config": config.to_dict(),
        "host_processor": platform.processor(), "host_logical_cores": os.cpu_count(),
        "torch_cpu_threads": original_cpu_threads,
        "sweeps": {"cpu_thread_counts": cpu_thread_counts, "offload_bucket_sizes_mb": bucket_sizes,
                   "attention_backends": attention_backends},
        "parameters": estimate_parameter_count(config), "effective_batch": effective_batch,
        "execution": {
            "parameter_dtype": cfg.get("parameter_dtype", "bfloat16" if device.type == "cuda" else "float32"),
            "optimizer_state_offload": cfg.get("optimizer_state_offload", "cpu"),
            "cpu_optimizer_kernel": cfg.get("cpu_optimizer_kernel", "auto"),
            "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", cfg.get("activation_checkpointing", False))),
            **({"attention_backend": attention_backends[0]} if len(attention_backends) == 1 else {}),
        },
        "sequence_length": sequence_length, "warmup_steps": cfg["warmup_steps"], "measure_steps": cfg["measure_steps"],
        "measurement": "Synthetic repeated tokens; synchronized complete forward/backward/clip/optimizer updates including CPU-to-device transfer. Warmup, compilation, validation, data preparation and checkpoints excluded from steady-state throughput. No language-quality assessment.",
        "cases": [],
    }
    total = len(batch_sizes) * len(modes) * len(cpu_thread_counts) * len(bucket_sizes) * len(attention_backends)
    try:
        for threads, bucket_mb, attention, mode, batch in product(
            cpu_thread_counts, bucket_sizes, attention_backends, modes, batch_sizes,
        ):
            torch.set_num_threads(threads)
            case_cfg = {**cfg, "cpu_threads": threads, "offload_bucket_mb": bucket_mb, "attention_backend": attention}
            message = (f"Sprint benchmark: batch {batch}, accumulation {effective_batch // batch}, {mode}, "
                       f"CPU threads {threads}, offload bucket {bucket_mb} MiB, attention {attention}")
            print(message, flush=True)
            write_json(rd / "progress.json", {"stage": "Sprint throughput benchmark", "step": len(report["cases"]),
                       "max_steps": total, "value": 0, "loss": 0, "metric_label": "tokens/s", "msg": message})
            row = _measure_case(config, case_cfg, batch, mode, device, inputs)
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
            write_json(rd / "benchmark.json", report)
        successful = [row for row in report["cases"] if row["status"] == "ok"]
        if successful:
            best = max(successful, key=lambda row: row["supervised_tokens_per_second"])
            report["recommended"] = {key: best[key] for key in (
                "micro_batch", "grad_accum", "torch_compile", "compile_mode", "parameter_dtype",
                "optimizer_state_offload", "gradient_checkpointing", "cpu_threads", "offload_bucket_mb",
                "attention_backend", "cpu_optimizer_kernel")}
            report["best_supervised_tokens_per_second"] = best["supervised_tokens_per_second"]
            report["projected_seconds_per_billion_targets"] = best["projected_seconds_per_billion_targets"]
            report["recommendation_scope"] = "Fastest measured candidate for this exact shape and effective batch, including the recorded CPU thread count, offload bucket size, and attention backend; short-run noise, IO and startup may change the best end-to-end choice. Re-run finalists before a long training run."
        write_json(rd / "benchmark.json", report)
        write_json(rd / "summary.json", report)
        write_json(rd / "progress.json", {"stage": "Sprint throughput benchmark", "step": total, "max_steps": total,
                   "value": report.get("best_supervised_tokens_per_second", 0), "loss": 0, "metric_label": "tokens/s",
                   "msg": "Benchmark complete" if successful else "No candidate completed; inspect benchmark.json"})
        if not successful:
            raise RuntimeError("All Sprint benchmark candidates failed; see benchmark.json for errors")
        return report
    finally:
        torch.set_num_threads(original_cpu_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON benchmark recipe (overrides other shape flags)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sprint-benchmark"))
    parser.add_argument("--target-params", type=int, default=50000000)
    parser.add_argument("--shape-profile", choices=["throughput", "balanced"], default="throughput")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--compile-modes", default="eager,default")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parameter-dtype", choices=["bfloat16", "float32"])
    parser.add_argument("--optimizer-state-offload", choices=["cpu", "none"], default="cpu")
    parser.add_argument("--cpu-optimizer-kernel", choices=["auto", "native", "compiled"], default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--cpu-threads", type=int, help="CPU intra-op threads; omitted preserves the current setting")
    parser.add_argument("--cpu-thread-counts", help="Comma-separated CPU thread sweep, e.g. 4,8,12,20")
    parser.add_argument("--offload-bucket-mb", type=int, default=32)
    parser.add_argument("--offload-bucket-sizes-mb", help="Comma-separated CPU offload bucket sweep, e.g. 16,32,64")
    parser.add_argument("--attention-backend", choices=ATTENTION_BACKENDS, default="auto")
    parser.add_argument("--attention-backends", help="Comma-separated attention sweep, e.g. auto,cudnn,efficient")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text("utf-8")) if args.config else {
        "target_parameters": args.target_params, "vocab_size": args.vocab_size,
        "shape_profile": args.shape_profile,
        "seq_len": args.seq_len, "effective_batch": args.effective_batch,
        "batch_sizes": [int(v) for v in args.batch_sizes.split(",")],
        "compile_modes": args.compile_modes.split(","), "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps, "device": args.device,
        "parameter_dtype": args.parameter_dtype or ("bfloat16" if args.device != "cpu" and torch.cuda.is_available() else "float32"),
        "optimizer_state_offload": args.optimizer_state_offload,
        "cpu_optimizer_kernel": args.cpu_optimizer_kernel,
        "gradient_checkpointing": args.gradient_checkpointing,
        "offload_bucket_mb": args.offload_bucket_mb,
        "attention_backend": args.attention_backend,
    }
    if not args.config:
        if args.cpu_threads is not None:
            cfg["cpu_threads"] = args.cpu_threads
        if args.cpu_thread_counts is not None:
            cfg["cpu_thread_counts"] = [int(value) for value in args.cpu_thread_counts.split(",")]
        if args.offload_bucket_sizes_mb is not None:
            cfg["offload_bucket_sizes_mb"] = [int(value) for value in args.offload_bucket_sizes_mb.split(",")]
        if args.attention_backends is not None:
            cfg["attention_backends"] = [value.strip() for value in args.attention_backends.split(",")]
    report = run_sprint_benchmark(cfg, args.output)
    print(json.dumps({key: report[key] for key in ("recommended", "best_supervised_tokens_per_second")}, indent=2))


if __name__ == "__main__":
    main()
