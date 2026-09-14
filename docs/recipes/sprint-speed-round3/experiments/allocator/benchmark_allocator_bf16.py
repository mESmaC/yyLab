"""Fresh-process allocator experiment composed with explicit BF16-moment Adam.

PYTORCH_ALLOC_CONF is set before Torch imports. Optional empty_cache runs once
after the first warmup update or all warmup, never in measured updates. This
separate wrapper preserves the explicit BF16-moment experiment and its warnings.
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
    if (not isinstance(policy, dict)
            or not {"config", "release_after_warmup"} <= set(policy)
            or set(policy) - {"config", "release_after_warmup", "release_after_first_warmup_step"}
            or not isinstance(policy.get("config"), str) or policy["config"] not in POLICIES
            or type(policy.get("release_after_warmup")) is not bool
            or type(policy.get("release_after_first_warmup_step", False)) is not bool):
        raise ValueError("experimental_allocator requires an allowed exact config and Boolean release policies")
    release_first = policy.get("release_after_first_warmup_step", False)
    if release_first and policy["release_after_warmup"]:
        raise ValueError("first-warmup and after-warmup release policies are mutually exclusive")
    if release_first and (type(cfg.get("warmup_steps", 5)) is not int or cfg.get("warmup_steps", 5) < 1):
        raise ValueError("first-warmup release requires at least one warmup step")
    if cfg.get("experimental_optimizer_candidate") != "bf16_moments_large_slices_v1":
        raise ValueError("composed wrapper requires exactly bf16_moments_large_slices_v1")
    if (cfg.get("cpu_optimizer_kernel") != "compiled"
            or cfg.get("optimizer_state_offload", "cpu") != "cpu"
            or cfg.get("parameter_dtype", "bfloat16") != "bfloat16"):
        raise ValueError("composed wrapper requires BF16 weights and explicitly compiled CPU offload")
    if "experimental_cpu_adam_min_elements" in cfg or "experimental_weight_cache" in cfg:
        raise ValueError("composed wrapper does not support threshold overrides or weight caching")
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
    from yylab.train import sprint
    import bf16_moments_candidate as candidate

    candidate_path = Path(candidate.__file__).resolve()
    candidate_source = {candidate_path.relative_to(root).as_posix():
                        hashlib.sha256(candidate_path.read_bytes()).hexdigest()}
    cfg = {**cfg, "experimental_optimizer_candidate_source_sha256": candidate_source}
    print(candidate.NOTICE, flush=True)
    # The candidate must surround the allocator hook installation, so restoration
    # unwinds allocator hooks first and then the optimizer/metadata overrides.
    with candidate._candidate_runner(harness, sprint, candidate_source):
        return _run_allocator(cfg, output, policy, inherited, root, torch, harness, sprint)


def _run_allocator(cfg, output, policy, inherited, root, torch, harness, sprint):
    source_path = Path(__file__).resolve().relative_to(root).as_posix()
    identity = {**policy, "variant": VARIANT, "expected_backend": "native",
                "expected_per_process_memory_fraction": POLICIES[policy["config"]],
                "fraction_attestation_absolute_tolerance": 1e-8,
                "set_before_torch_import": True, "inherited_environment": inherited,
                "effective_environment": {key: os.environ.get(key) for key in inherited},
                "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "training_math_unchanged": False, "allocator_training_math_unchanged": True,
                "changed_optimizer_numerics": True,
                "optimizer_recipe": "FP32 CPU masters; BF16 moments for compiled large slices, FP32 moments for native tails",
                "first_warmup_release_timing_scope": "inside_warmup_after_step_0_before_measured_training",
                "scope": "Allocation/reservation policy only. Optional release is outside steady-state timing; allocator release changes no update or timer; optimizer numerics follow the separately attested BF16-moment candidate."}
    original_case, original_samples = harness._measure_precision_case, harness._decoder_samples
    original_write, original_sources = harness.write_json, harness.PROVENANCE_SOURCE_PATHS
    original_train_step = sprint.SprintStepEngine.train_step
    active = None

    def memory(device):
        return {"allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device)}

    def train_step(engine, batches, *, step=0):
        if active is None:
            raise RuntimeError("allocator observer called outside an active benchmark case")
        if (active["train_step_calls"] == 0
                and policy.get("release_after_first_warmup_step", False)
                and (step != 0 or active["warmup_steps"] < 1)):
            raise RuntimeError("first-warmup release requires the first engine update to be warmup step zero")
        result = original_train_step(engine, batches, step=step)
        active["train_step_calls"] += 1
        if active["train_step_calls"] == 1 and policy.get("release_after_first_warmup_step", False):
            # The optimizer's update and stream ordering are already complete.
            # This synchronization also completes asynchronous writeback before
            # freeing unused reservations. It remains inside total warmup time.
            device = engine.device
            active["first_warmup_before_release"] = memory(device)
            started = time.perf_counter()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            active["first_warmup_release_seconds_outside_measurement"] = time.perf_counter() - started
            active["first_warmup_after_release"] = memory(device)
            active["first_warmup_release_count"] += 1
            active["first_warmup_step"] = step
        return result

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
                  "release_seconds_outside_measurement": 0.0,
                  "train_step_calls": 0, "warmup_steps": step_cfg["warmup_steps"],
                  "first_warmup_release_count": 0, "first_warmup_step": None,
                  "first_warmup_release_seconds_outside_measurement": 0.0}
        try:
            if device.type != "cuda":
                raise RuntimeError("allocator experiment resolved to a non-CUDA device")
            active["actual_backend"] = torch.cuda.memory.get_allocator_backend()
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            active["actual_per_process_memory_fraction"] = torch.cuda.get_per_process_memory_fraction(device_index)
            if active["actual_backend"] != "native" or not math.isclose(
                    active["actual_per_process_memory_fraction"], POLICIES[policy["config"]], abs_tol=1e-8, rel_tol=0):
                raise RuntimeError("active allocator backend/fraction does not match requested policy")
            row = original_case(config, step_cfg, precision, batch, mode, device, inputs)
            if row.get("status") == "ok" and (
                    active["decoder_sample_calls"] != 2
                    or active["release_count"] != int(policy["release_after_warmup"])
                    or active["first_warmup_release_count"] != int(policy.get("release_after_first_warmup_step", False))
                    or active["train_step_calls"] != step_cfg["warmup_steps"] + step_cfg["measure_steps"]):
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
                "after the first warmup step or after all warmup before the measured timer, with a separate duration. "
                "First-step release is included in total warmup_seconds and precedes every remaining warmup step. "
                "Allocator peaks still include initialization/warmup; no peak counters are reset by this wrapper.")
        return original_write(path, value)

    harness._measure_precision_case, harness._decoder_samples = case, samples
    sprint.SprintStepEngine.train_step = train_step
    harness.write_json = write
    harness.PROVENANCE_SOURCE_PATHS = (*original_sources, source_path)
    try:
        return harness.run_sprint_precision_benchmark(cfg, output)
    finally:
        harness._measure_precision_case, harness._decoder_samples = original_case, original_samples
        sprint.SprintStepEngine.train_step = original_train_step
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
