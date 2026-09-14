"""Fresh-process allocator experiment around the unchanged official harness.

PYTORCH_ALLOC_CONF is set before Torch imports. Optional empty_cache runs once
after warmup and before the timer, never in a training update. FP32 CPU Adam
only in this initial wrapper; optimizer/cache composition is rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

KEY = "experimental_allocator"
VARIANT = "native_allocator_residency_v1"
POLICIES = {"backend:native": 1.0,
            "garbage_collection_threshold:0.8,per_process_memory_fraction:0.85": 0.85}


def validate(cfg):
    if not isinstance(cfg, dict) or "recipe" in cfg:
        raise ValueError("allocator experiment requires a flat recipe")
    policy = cfg.get(KEY)
    if (not isinstance(policy, dict) or set(policy) != {"config", "release_after_warmup"}
            or not isinstance(policy.get("config"), str) or policy["config"] not in POLICIES
            or type(policy.get("release_after_warmup")) is not bool):
        raise ValueError("experimental_allocator requires one allowed exact config string and Boolean release_after_warmup")
    if "experimental_optimizer_candidate" in cfg or "experimental_weight_cache" in cfg:
        raise ValueError("initial allocator wrapper supports official FP32-moment training only")
    if cfg.get("device", "auto") not in {"auto", "cuda", "cuda:0"}:
        raise ValueError("allocator experiment requires CUDA")
    if "torch" in sys.modules:
        raise RuntimeError("allocator policy must be set in a fresh process before importing Torch")
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        if key in os.environ and os.environ[key] != policy["config"]:
            raise ValueError(f"conflicting inherited {key}; use a clean fresh-process environment")
    return dict(policy)


def run(cfg, output):
    policy = validate(cfg)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("allocator benchmark output must be new")
    inherited = {key: os.environ.get(key) for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}
    os.environ["PYTORCH_ALLOC_CONF"] = policy["config"]
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import torch
    from yylab.train import benchmark_sprint_precision as harness

    source_path = Path(__file__).resolve().relative_to(root).as_posix()
    identity = {**policy, "variant": VARIANT, "expected_backend": "native",
                "expected_per_process_memory_fraction": POLICIES[policy["config"]],
                "fraction_attestation_absolute_tolerance": 1e-8,
                "set_before_torch_import": True, "inherited_environment": inherited,
                "effective_environment": {key: os.environ.get(key) for key in inherited},
                "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "training_math_unchanged": True, "optimizer_recipe": "FP32 CPU masters and moments",
                "scope": "Allocation/reservation policy only. Optional release is outside steady-state timing; no update or timer is modified."}
    original_case, original_samples = harness._measure_precision_case, harness._decoder_samples
    original_write, original_sources = harness.write_json, harness.PROVENANCE_SOURCE_PATHS
    active = None

    def memory(device):
        return {"allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device)}

    def samples(model):
        nonlocal active
        if active is None:
            raise RuntimeError("allocator observer called outside an active benchmark case")
        active["decoder_sample_calls"] += 1
        if active["decoder_sample_calls"] == 1:
            # Official source calls this after warmup and before it starts
            # the measurement clock; the second call is after measurement.
            device = next(model.parameters()).device
            active["after_warmup_before_release"] = memory(device)
            if policy["release_after_warmup"]:
                started = time.perf_counter()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                active["release_count"] += 1
                active["release_seconds_outside_measurement"] = time.perf_counter() - started
            active["after_warmup_after_release"] = memory(device)
        return original_samples(model)

    def case(config, step_cfg, precision, batch, mode, device, inputs):
        nonlocal active
        active = {**identity, "decoder_sample_calls": 0, "release_count": 0,
                  "release_seconds_outside_measurement": 0.0}
        try:
            if device.type != "cuda":
                raise RuntimeError("allocator experiment resolved to a non-CUDA device")
            active["actual_backend"] = torch.cuda.memory.get_allocator_backend()
            active["actual_per_process_memory_fraction"] = torch.cuda.get_per_process_memory_fraction(device)
            if active["actual_backend"] != "native" or not math.isclose(
                    active["actual_per_process_memory_fraction"], POLICIES[policy["config"]], abs_tol=1e-8, rel_tol=0):
                raise RuntimeError("active allocator backend/fraction does not match requested policy")
            row = original_case(config, step_cfg, precision, batch, mode, device, inputs)
            if row.get("status") == "ok" and (
                    active["decoder_sample_calls"] != 2
                    or active["release_count"] != int(policy["release_after_warmup"])):
                row.update(status="error", error="allocator warmup-release observer did not attest the expected call count")
        except Exception as error:
            row = {"precision": precision, "micro_batch": batch,
                   "grad_accum": step_cfg["effective_batch"] // batch,
                   "status": "error", "stage": "allocator_validation",
                   "error": f"{type(error).__name__}: {error}"}
        finally:
            receipt, active = active, None
        row["allocator_experiment"] = receipt
        row.setdefault("runtime", {})[KEY] = receipt
        return row

    def write(path, value):
        if value.get("schema") == "yylab-sprint-precision-benchmark-v1":
            value["allocator_experiment"] = identity
            value["allocator_measurement"] = (
                "Normal complete training-step timer is unchanged. Optional synchronize/empty_cache runs once "
                "after warmup before parameter sampling and the timer, and has a separate duration. "
                "Allocator peaks still include initialization/warmup; no peak counters are reset by this wrapper.")
        return original_write(path, value)

    harness._measure_precision_case, harness._decoder_samples = case, samples
    harness.write_json = write
    harness.PROVENANCE_SOURCE_PATHS = (*original_sources, source_path)
    try:
        return harness.run_sprint_precision_benchmark(cfg, output)
    finally:
        harness._measure_precision_case, harness._decoder_samples = original_case, original_samples
        harness.write_json, harness.PROVENANCE_SOURCE_PATHS = original_write, original_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true", help="standard-library admission check; no Torch or CUDA")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.validate_only:
        policy = validate(cfg)
        if args.output.exists():
            parser.error("output must be new")
        print(json.dumps({"status": "valid_not_executed", KEY: policy}), flush=True)
        return 0
    report = run(cfg, args.output)
    print(json.dumps({key: report.get(key) for key in ("status", "successful_cases", "failed_cases", "fastest_measured")}), flush=True)
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
