"""Disposable BF16-moment optimizer experiment; not equivalent to standard AdamW.

Run in a fresh process with --config <official precision recipe> --output <newdir>.
Large compiled slices retain BF16 moments between steps; tiny native slices keep
FP32 moments. All masters and update arithmetic remain FP32. The current update
uses newly calculated FP32 moments, rounded once when stored for the NEXT step.
No convergence/quality claim and no production checkpoint/resume support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
from types import FunctionType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from yylab.train import sprint_cpu_adam as baseline_kernel
from yylab.train.sprint import SprintCPUOffloadAdamW

NAME = "bf16_moments_large_slices_v1"
NOTICE = (
    "Changed optimizer numerics: compiled large slices store first/second moments in BF16, "
    "rounding them each step; native small slices retain FP32 moments. Masters and update "
    "arithmetic are FP32, and accelerator weights/gradients are BF16. Current updates use "
    "unrounded newly computed FP32 moments before BF16 storage for the next step. This is "
    "not equivalent to standard FP32-state AdamW. Experimental speed/memory result only; "
    "no convergence or final-quality claim. No production checkpoint/resume support."
)


def _fused_update(master, moment, variance, staging_bf16, scalars):
    gradient = staging_bf16.float()
    # Explicit casts are essential: BF16 is storage, not update arithmetic.
    next_moment = moment.float() * scalars[0] + gradient * scalars[1]
    next_variance = variance.float() * scalars[2] + gradient.square() * scalars[3]
    denominator = next_variance.sqrt() * scalars[4] + scalars[5]
    next_master = master * scalars[6] - (next_moment / denominator) * scalars[7]
    moment.copy_(next_moment)
    variance.copy_(next_variance)
    master.copy_(next_master)
    staging_bf16.copy_(next_master)


@torch.no_grad()
def _validate_kernel(kernel):
    master = torch.linspace(-0.8, 0.8, 257)
    moment = torch.linspace(-0.03, 0.03, 257).bfloat16()
    variance = torch.linspace(0.02, 0.2, 257).bfloat16()
    expected = [value.clone() for value in (master, moment, variance)]
    staging = torch.empty(257, dtype=torch.bfloat16)
    for step in range(1, 5):
        lr, beta1, beta2, eps, decay = 0.001 / step, 0.8, 0.95, 1e-7, 0.2
        staging.copy_(torch.linspace(-1.2, 1.7, 257) * step)
        gradient = staging.float()
        # Independent recurrence with persistent BF16 state, not stock AdamW.
        next_m = expected[1].float() * beta1 + gradient * (1 - beta1)
        next_v = expected[2].float() * beta2 + gradient.square() * (1 - beta2)
        denominator = (next_v / (1 - beta2**step)).sqrt() + eps
        expected[0].mul_(1 - lr * decay).add_(-(lr / (1 - beta1**step)) * next_m / denominator)
        expected[1].copy_(next_m)
        expected[2].copy_(next_v)
        kernel(master, moment, variance, staging,
               baseline_kernel._coefficients(step, lr, beta1, beta2, eps, decay))
        torch.testing.assert_close(master, expected[0], rtol=1e-5, atol=2e-6)
        # Compiler contraction may cross a BF16 rounding boundary; allow one
        # BF16 ULP here. Small eager oracle tests check the recurrence directly.
        torch.testing.assert_close(moment.float(), expected[1].float(), rtol=0.008, atol=1e-6)
        torch.testing.assert_close(variance.float(), expected[2].float(), rtol=0.008, atol=1e-6)
        if not torch.equal(staging, master.bfloat16()):
            raise RuntimeError("candidate kernel broke shared BF16 gradient/writeback alias")


@lru_cache(maxsize=8)
def _prepare_kernel(cpu_threads, warmup_elements):
    baseline_kernel._activate_local_compiler()
    from torch._dynamo.backends.registry import lookup_backend
    from torch._inductor import config
    from torch._inductor.cpp_builder import get_cpp_compiler

    compiler = get_cpp_compiler()
    inductor = lookup_backend("inductor")

    def backend(graph, inputs):
        with config.patch({"cpp.threads": cpu_threads, "compile_threads": 1, "fx_graph_cache": False}):
            return inductor(graph, inputs)

    name = f"_candidate_bf16_moments_{cpu_threads}_{warmup_elements}"
    code = _fused_update.__code__.replace(co_name=name, co_qualname=name)
    function = FunctionType(code, _fused_update.__globals__, name=name)
    kernel = torch.compile(function, backend=backend, fullgraph=True, dynamic=True)
    with torch.no_grad():
        master = torch.zeros(warmup_elements, dtype=torch.float32)
        moment = torch.zeros(warmup_elements, dtype=torch.bfloat16)
        variance = torch.zeros_like(moment)
        staging = torch.ones_like(moment)
        kernel(master, moment, variance, staging,
               baseline_kernel._coefficients(1, 0.0003, 0.9, 0.95, 1e-8, 0.1))
        del master, moment, variance, staging
        _validate_kernel(kernel)
    return kernel, str(compiler)


class CompiledBF16Moments(baseline_kernel.CompiledCPUAdam):
    def __init__(self, *, cpu_threads, warmup_elements):
        if cpu_threads < 1 or warmup_elements < 1:
            raise ValueError("candidate kernel needs positive thread and representative slice sizes")
        self.cpu_threads, self.warmup_elements = cpu_threads, warmup_elements
        self.kernel, self.compiler = _prepare_kernel(cpu_threads, warmup_elements)
        self.scalars = torch.empty(8, dtype=torch.float32)
        self._last_settings = None
        # Inherited update only supplies FP32 scalars then calls the kernel once.
        # It never retries a partially mutated live state after failure.


class BF16MomentsAdamW(SprintCPUOffloadAdamW):
    """Keep the stock step/transfer pipeline; partition moment state by fixed slice."""

    def __init__(self, params, *, bucket_mb=32, cpu_kernel="compiled",
                 _min_elements=None, _bucket_elements=None, _kernel_factory=None, **kwargs):
        if cpu_kernel != "compiled":
            raise ValueError("BF16-moment candidate requires explicitly compiled CPU optimizer")
        # Skip baseline compilation: this experiment prepares its own kernel.
        super().__init__(params, bucket_mb=bucket_mb, cpu_kernel="native", **kwargs)
        if self.parameter_dtype != torch.bfloat16:
            raise ValueError("BF16-moment candidate requires BF16 weights and gradients")
        self._min_compiled_elements = int(baseline_kernel.MIN_COMPILED_ELEMENTS if _min_elements is None else _min_elements)
        if _bucket_elements is not None:
            self.bucket_elements = int(_bucket_elements)
        if self._min_compiled_elements < 1 or self.bucket_elements < self._min_compiled_elements:
            raise ValueError("candidate bucket must fit a positive compiled-slice threshold")
        self._fixed_buckets = [list(super(BF16MomentsAdamW, self)._buckets(group["params"]))
                               for group in self.param_groups]
        self._slice_layout = {}
        self._slice_index = {}
        for buckets in self._fixed_buckets:
            for entries, _ in buckets:
                for parameter, offset, count, _ in entries:
                    layout = self._slice_layout.setdefault(id(parameter), [])
                    self._slice_index[(id(parameter), offset, count)] = len(layout)
                    layout.append((offset, count, count >= self._min_compiled_elements))
        large = [count for layout in self._slice_layout.values() for _, count, compiled in layout if compiled]
        if not large:
            raise ValueError("candidate model has no eligible large compiled slices")
        factory = _kernel_factory or CompiledBF16Moments
        # Preparation and disposable-buffer validation happen before model steps.
        self._compiled_cpu_adam = factory(cpu_threads=torch.get_num_threads(), warmup_elements=max(large))
        self.resolved_cpu_kernel = "compiled_bf16_moments"
        self.cpu_kernel_reason = NOTICE
        self._moments_ready = False

    def initialize_master_weights(self, initial_weights):
        super().initialize_master_weights(initial_weights)
        self._moments_ready = False
        self._initialize_moments()

    def _initialize_moments(self):
        if self._moments_ready:
            return
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                if "master_param" not in state:
                    state.update(step=0, master_param=parameter.detach().to(device="cpu", dtype=torch.float32, copy=True))
                if int(state.get("step", 0)):
                    raise RuntimeError("candidate cannot convert nonempty live optimizer state")
                state["exp_avg"] = [torch.zeros(count, dtype=torch.bfloat16 if compiled else torch.float32, device="cpu")
                                    for _, count, compiled in self._slice_layout[id(parameter)]]
                state["exp_avg_sq"] = [torch.zeros_like(value) for value in state["exp_avg"]]
        self._moments_ready = True

    def _buckets(self, parameters):
        # Missing gradients leave holes in the original layout. This keeps every
        # parameter slice's storage dtype fixed instead of changing it mid-run.
        selected = {id(parameter) for parameter in parameters}
        for buckets in self._fixed_buckets:
            for entries, _ in buckets:
                active = [entry for entry in entries if id(entry[0]) in selected]
                if active:
                    yield active, max(source + count for _, _, count, source in active)

    @torch.no_grad()
    def step(self, closure=None):
        self._initialize_moments()
        return super().step(closure)

    def _update_bucket(self, group, entries, used, slot):
        from torch.optim import _functional

        staging = self._staging_buffers[slot]
        masters, gradients, moments, variances, steps, destinations = [], [], [], [], [], []
        for parameter, offset, count, source in entries:
            state = self.state[parameter]
            index = self._slice_index[(id(parameter), offset, count)]
            master = state["master_param"].view(-1)[offset:offset + count]
            moment, variance = state["exp_avg"][index], state["exp_avg_sq"][index]
            if moment.dtype == torch.bfloat16:
                self._compiled_cpu_adam.update(
                    master, moment, variance, staging[source:source + count], step=state["step"],
                    lr=group["lr"], beta1=group["betas"][0], beta2=group["betas"][1],
                    eps=group["eps"], weight_decay=group["weight_decay"],
                )
            else:
                gradient = self._gradient[source:source + count]
                gradient.copy_(staging[source:source + count])
                masters.append(master)
                gradients.append(gradient)
                moments.append(moment)
                variances.append(variance)
                steps.append(torch.tensor(float(state["step"] - 1), dtype=torch.float32))
                destinations.append((source, count))
        if masters:
            _functional.adamw(
                masters, gradients, moments, variances, [], steps,
                fused=self.cpu_fused, foreach=not self.cpu_fused, capturable=False, differentiable=False,
                amsgrad=False, beta1=group["betas"][0], beta2=group["betas"][1], lr=group["lr"],
                weight_decay=group["weight_decay"], eps=group["eps"], maximize=False,
            )
            for master, (source, count) in zip(masters, destinations, strict=True):
                staging[source:source + count].copy_(master)

    def candidate_metadata(self):
        bf16 = sum(count for layout in self._slice_layout.values() for _, count, compiled in layout if compiled)
        fp32 = sum(count for layout in self._slice_layout.values() for _, count, compiled in layout if not compiled)
        return {"name": NAME, "changed_optimizer_numerics": True, "notice": NOTICE,
                "master_dtype": "float32", "moment_storage": "mixed_bfloat16_float32",
                "update_arithmetic": "float32", "bf16_moment_parameter_elements": bf16,
                "fp32_moment_parameter_elements": fp32, "expected_state_bytes": 8 * bf16 + 12 * fp32,
                "minimum_compiled_slice_elements": self._min_compiled_elements,
                "kernel_prepared_before_steps": True,
                "compiler": getattr(self._compiled_cpu_adam, "compiler", None),
                "kernel_cpu_threads": getattr(self._compiled_cpu_adam, "cpu_threads", None),
                "kernel_warmup_elements": getattr(self._compiled_cpu_adam, "warmup_elements", None),
                "theoretical_large_slice_bytes_per_update": {"fp32_moments": 28, "bf16_moments": 20},
                "traffic_scope": "Idealized CPU tensor reads+writes: FP32 master, two moments, shared BF16 gradient/weight staging; excludes caches, scratch, transfers and native tails."}

    def load_state_dict(self, state_dict):
        raise RuntimeError("disposable BF16-moment candidate has no production checkpoint/resume contract")

    def state_dict(self):
        raise RuntimeError("disposable BF16-moment state must not be saved as a production AdamW checkpoint")


@contextmanager
def _candidate_runner(harness, sprint, candidate_source):
    original_optimizer = sprint.SprintCPUOffloadAdamW
    original_case, original_write = harness._measure_precision_case, harness.write_json
    original_memory = harness._training_memory
    active_metadata = None

    def memory(model, optimizer):
        nonlocal active_metadata
        result = original_memory(model, optimizer)
        active_metadata = optimizer.candidate_metadata()
        expected = active_metadata["expected_state_bytes"]
        if result["optimizer_state"]["bytes"] != expected:
            raise RuntimeError(f"candidate optimizer state inventory mismatch: expected {expected} bytes")
        return result

    @wraps(original_case)
    def case(*args, **kwargs):
        nonlocal active_metadata
        active_metadata = None
        row = original_case(*args, **kwargs)
        row["optimizer_candidate"] = active_metadata or {
            "name": NAME, "changed_optimizer_numerics": True, "notice": NOTICE, "attestation_complete": False}
        row.setdefault("runtime", {})["optimizer_candidate"] = row["optimizer_candidate"]
        return row

    def write(path, value):
        value = dict(value)
        value["optimizer_candidate"] = {"name": NAME, "changed_optimizer_numerics": True, "notice": NOTICE}
        if "schema" in value:
            value["comparison"] = {**value.get("comparison", {}), "optimizer_numerics": NAME}
            value["source_sha256"] = {**value.get("source_sha256", {}), **candidate_source}
            value["measurement"] = value.get("measurement", "") + " " + NOTICE
            # The base harness description assumes all-FP32 moments; replace it.
            value["initialization"] = (
                "Every seeded model is rounded to BF16 before its precision adapter and optimizer. "
                "FP32 CPU masters start from those BF16-rounded weights. Moment storage is mixed "
                "BF16/FP32 as attested per case; BitNet may add parameters. " + NOTICE)
        return original_write(path, value)

    try:
        sprint.SprintCPUOffloadAdamW = BF16MomentsAdamW
        harness._measure_precision_case = case
        harness._training_memory = memory
        harness.write_json = write
        yield
    finally:
        harness.write_json = original_write
        harness._training_memory = original_memory
        harness._measure_precision_case = original_case
        sprint.SprintCPUOffloadAdamW = original_optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8-sig"))
    cfg = dict(payload.get("recipe", payload))
    if cfg.get("cpu_optimizer_kernel") != "compiled" or cfg.get("optimizer_state_offload", "cpu") != "cpu":
        parser.error("candidate requires compiled CPU optimizer and CPU offload")
    if "experimental_cpu_adam_min_elements" in cfg:
        parser.error("candidate uses the baseline large-slice cutoff; threshold overrides are unsupported")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new/empty; never mix optimizer experiments")
    from yylab.train import benchmark_sprint_precision as harness
    from yylab.train import sprint

    source_path = Path(__file__).resolve()
    source = {source_path.relative_to(ROOT).as_posix(): hashlib.sha256(source_path.read_bytes()).hexdigest()}
    cfg["experimental_optimizer_candidate"] = NAME
    cfg["experimental_optimizer_candidate_source_sha256"] = source
    print(NOTICE, flush=True)
    with _candidate_runner(harness, sprint, source):
        report = harness.run_sprint_precision_benchmark(cfg, args.output)
    print(json.dumps({"status": report["status"], "optimizer_candidate": NAME,
                      "changed_optimizer_numerics": True}), flush=True)
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
