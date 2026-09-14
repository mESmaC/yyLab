"""Per-step allocator diagnostic for ONE official Sprint precision benchmark case.

Launch in a fresh process with --config <saved recipe> --output <new directory>.
Extra synchronizations and logging change execution: DIAGNOSTIC_NOT_THROUGHPUT.
The memory guard applies to every completed step; the elapsed-time guard starts
after configured compile warmup. A guard stops between steps, never mid-update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path

MARKER = "DIAGNOSTIC_NOT_THROUGHPUT"
NOTICE = (
    "DIAGNOSTIC_NOT_THROUGHPUT: device synchronization and per-step logging disturb "
    "normal overlap. These timings cannot be used as accepted throughput results. "
    "Guards act only after a synchronized completed step; they cannot interrupt a "
    "blocked kernel or driver call. CUDA allocator counters do not measure Windows "
    "shared-GPU-memory residency or all driver/library allocations."
)


class DiagnosticStop(RuntimeError):
    pass


def _utc():
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _mark(value):
    if isinstance(value, list):
        return [_mark(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in {"fastest_measured", "projected_seconds_per_billion_targets"}:
            continue
        if key in {"tokens_per_second", "supervised_tokens_per_second", "seconds_per_step"}:
            key = "diagnostic_" + key
        result[key] = _mark(item)
    if "schema" in result:
        result["original_schema"] = result["schema"]
        result["schema"] = "yylab-sprint-allocator-diagnostic-v1"
    return result


@contextmanager
def _instrument(harness, sprint, torch, report, emit):
    original_step = sprint.SprintStepEngine.train_step
    original_case = harness._measure_precision_case
    original_write = harness.write_json
    helper_names = ("_sync", "_final_gradient_norm", "_decoder_samples", "_training_memory")
    original_helpers = {name: getattr(harness, name) for name in helper_names}
    original_optimizer_sync = sprint.SprintCPUOffloadAdamW.synchronize
    warmup_steps = None

    def sync(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def counters(device):
        if device.type != "cuda":
            return {"allocated_bytes": 0, "reserved_bytes": 0,
                    "max_allocated_bytes": 0, "max_reserved_bytes": 0}
        stats = torch.cuda.memory_stats(device)
        return {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "active_bytes": stats.get("active_bytes.all.current"),
            "max_active_bytes": stats.get("active_bytes.all.peak"),
            "inactive_split_bytes": stats.get("inactive_split_bytes.all.current"),
            "max_inactive_split_bytes": stats.get("inactive_split_bytes.all.peak"),
            "segments": stats.get("segment.all.current"),
            "allocation_retries": stats.get("num_alloc_retries"),
            "allocator_ooms": stats.get("num_ooms"),
            "device_allocations": stats.get("num_device_alloc"),
            "device_frees": stats.get("num_device_free"),
        }

    def phase():
        case = report.get("case", {})
        count = len(report["steps"])
        if not count:
            return "initialization"
        if count < case.get("warmup_steps", 0):
            return "warmup_and_compile"
        if count == case.get("warmup_steps", 0):
            return "warmup_finalize_or_pre_measurement"
        if count < case.get("warmup_steps", 0) + case.get("measure_steps", 0):
            return "measured_training"
        return "measurement_finalize_or_untimed_validation"

    def traced_helper(name, original):
        @wraps(original)
        def traced(*args, **kwargs):
            identity = {"stage": phase(), "operation": name, "completed_steps": len(report["steps"])}
            emit({"event": "operation_begin", **identity})
            started = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                emit({"event": "operation_error", **identity, "elapsed_seconds": time.perf_counter() - started,
                      "error": f"{type(exc).__name__}: {exc}"})
                raise
            emit({"event": "operation_end", **identity, "elapsed_seconds": time.perf_counter() - started})
            return result
        return traced

    @wraps(original_step)
    def train_step(engine, cpu_batches, *, step=0):
        if warmup_steps is None:
            raise RuntimeError("diagnostic step was called outside the official benchmark case")
        stage = "warmup_and_compile" if step < warmup_steps else "measured_training"
        identity = {"step": int(step), "completed_step": int(step) + 1, "stage": stage,
                    "measurement_index": None if step < warmup_steps else int(step) - warmup_steps}
        emit({"event": "step_start", **identity})
        started = None
        try:
            sync(engine.device)
            emit({"event": "step_ready", **identity})
            started = time.perf_counter()
            result = original_step(engine, cpu_batches, step=step)
            emit({"event": "step_returned_awaiting_device", **identity})
            sync(engine.device)
            elapsed = time.perf_counter() - started
            row = {"event": "step_complete", **identity, "elapsed_seconds": elapsed,
                   "device": str(engine.device), **counters(engine.device)}
            reasons = []
            if row["reserved_bytes"] > report["guards"]["max_reserved_bytes"]:
                reasons.append("reserved_memory_limit")
            if stage == "measured_training" and elapsed > report["guards"]["max_step_seconds"]:
                reasons.append("measured_step_time_limit")
            row["guard_reasons"] = reasons
            report["steps"].append(row)
            emit(row)
            if reasons:
                report["guard_stop"] = {**identity, "reasons": reasons, "elapsed_seconds": elapsed,
                                        "reserved_bytes": row["reserved_bytes"]}
                raise DiagnosticStop("completed step crossed " + ", ".join(reasons))
            return result
        except DiagnosticStop:
            raise
        except BaseException as exc:
            # Do not synchronize again after a CUDA failure or retry an update.
            row = {"event": "step_error", **identity, "error": f"{type(exc).__name__}: {exc}",
                   "elapsed_seconds": None if started is None else time.perf_counter() - started}
            report["step_error"] = row
            emit(row)
            raise

    @wraps(original_case)
    def measure_case(config, cfg, precision, batch, mode, device, inputs):
        nonlocal warmup_steps
        warmup_steps = cfg["warmup_steps"]
        report["case"] = {"precision": precision, "micro_batch": batch,
                          "grad_accum": len(inputs) // batch, "compile_mode": mode,
                          "device": str(device), "warmup_steps": warmup_steps,
                          "measure_steps": cfg["measure_steps"]}
        emit({"event": "stage", "stage": "initialization", **report["case"]})
        try:
            row = original_case(config, cfg, precision, batch, mode, device, inputs)
            report["case_result"] = _mark(row)
            emit({"event": "case_result", "stage": row.get("stage"), "status": row.get("status"),
                  "error": row.get("error"), "cleanup_error": row.get("cleanup_error")})
            return row
        finally:
            warmup_steps = None

    def marked_write(path, value):
        marked = _mark(value)
        marked.update(diagnostic_marker=MARKER, comparison_eligible=False, measurement_notice=NOTICE)
        if "recipe" in marked:
            report["normalized_recipe"] = marked["recipe"]
            report["source_sha256"] = marked.get("source_sha256", {})
            report["runtime_versions"] = marked.get("runtime_versions", {})
            report["hardware"] = marked.get("hardware")
        return original_write(path, marked)

    try:
        sprint.SprintStepEngine.train_step = train_step
        harness._measure_precision_case = measure_case
        harness.write_json = marked_write
        for name, original in original_helpers.items():
            setattr(harness, name, traced_helper(name, original))
        sprint.SprintCPUOffloadAdamW.synchronize = traced_helper("optimizer.synchronize", original_optimizer_sync)
        yield
    finally:
        sprint.SprintCPUOffloadAdamW.synchronize = original_optimizer_sync
        for name, original in original_helpers.items():
            setattr(harness, name, original)
        harness.write_json = original_write
        harness._measure_precision_case = original_case
        sprint.SprintStepEngine.train_step = original_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-reserved-gib", type=float, default=11.0)
    parser.add_argument("--max-step-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in (args.max_reserved_gib, args.max_step_seconds)):
        parser.error("guard thresholds must be finite and positive")
    config_bytes = args.config.read_bytes()
    payload = json.loads(config_bytes.decode("utf-8-sig"))
    cfg = payload.get("recipe", payload)
    if not isinstance(cfg, dict):
        parser.error("config must be a recipe object or a receipt containing recipe")
    for key in ("precisions", "batch_sizes", "compile_modes"):
        if not isinstance(cfg.get(key), list) or len(cfg[key]) != 1:
            parser.error(f"single-case diagnostic requires exactly one explicit {key} entry")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output must be new/empty; never mix diagnostics with benchmark receipts")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "yylab-sprint-allocator-diagnostic-v1", "diagnostic_marker": MARKER,
        "comparison_eligible": False, "measurement_notice": NOTICE, "status": "running",
        "created_at_utc": _utc(), "input_recipe": cfg,
        "config_path": str(args.config.resolve()), "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "allocator_environment": {key: os.environ.get(key) for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
        "guards": {"max_reserved_bytes": int(args.max_reserved_gib * 2**30),
                   "max_step_seconds": args.max_step_seconds, "elapsed_guard_excludes_warmup": True},
        "steps": [],
    }
    _write_json(output / "diagnostic.json", report)
    code = 2
    with (output / "steps.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        def emit(value):
            value = {"timestamp_utc": _utc(), "diagnostic_marker": MARKER, **value}
            line = json.dumps(value, allow_nan=False)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        try:
            emit({"event": "stage", "stage": "imports", "notice": NOTICE})
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            import torch
            from yylab.train import benchmark_sprint_precision as harness
            from yylab.train import sprint

            with _instrument(harness, sprint, torch, report, emit):
                benchmark = harness.run_sprint_precision_benchmark(cfg, output / "instrumented-only")
            report["benchmark_status"] = benchmark["status"]
            report["status"] = "complete" if benchmark["status"] == "ok" else "failed"
            code = 0 if report["status"] == "complete" else 2
        except BaseException as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            # The official single-case harness records the guard as an explicit
            # case error, then raises its all-candidates-failed summary exception.
            report["status"] = "guard_stopped" if report.get("guard_stop") else "failed"
            code = 3 if report["status"] == "guard_stopped" else 2
        finally:
            report["finished_at_utc"] = _utc()
            report["completed_steps"] = len(report["steps"])
            _write_json(output / "diagnostic.json", report)
            emit({"event": "final", "stage": "complete", "status": report["status"],
                  "completed_steps": report["completed_steps"], "guard_stop": report.get("guard_stop"),
                  "error": report.get("error")})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
