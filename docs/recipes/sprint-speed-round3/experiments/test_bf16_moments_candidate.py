"""Small CPU-only oracles; no GPU or actual compiler execution.

Run after the GPU benchmark queue is idle:
    python -m pytest artifacts/sprint-1b-speed-20260911/test_bf16_moments_candidate.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location("bf16_moments_candidate", Path(__file__).with_name("bf16_moments_candidate.py"))
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class EagerKernel:
    calls = []

    def __init__(self, **kwargs):
        self.preparation = kwargs

    @torch.no_grad()
    def update(self, master, moment, variance, staging, **settings):
        self.calls.append(staging.numel())
        candidate._fused_update(master, moment, variance, staging,
                                candidate.baseline_kernel._coefficients(**settings))


@torch.no_grad()
def oracle(master, moment, variance, gradient, *, step, lr, betas, eps, decay):
    beta1, beta2 = betas
    next_m = beta1 * moment.float() + (1 - beta1) * gradient.float()
    next_v = beta2 * variance.float() + (1 - beta2) * gradient.float().square()
    correction = (next_v / (1 - beta2**step)).sqrt() + eps
    result = master * (1 - lr * decay) - lr * (next_m / (1 - beta1**step)) / correction
    return result, next_m.to(moment.dtype), next_v.to(variance.dtype)


def test_float_arithmetic_with_bf16_state_rounding_each_step():
    master = torch.linspace(-0.8, 0.7, 37)
    moment = torch.linspace(-0.04, 0.09, 37).bfloat16()
    variance = torch.linspace(0.001, 0.12, 37).bfloat16()
    expected = [tensor.clone() for tensor in (master, moment, variance)]
    pointers = [tensor.data_ptr() for tensor in (master, moment, variance)]
    retained_fp32_m = moment.float()
    observed_rounding = False
    for step in range(1, 9):
        gradient = (torch.linspace(-1.23, 1.71, 37) / step).bfloat16()
        lr, betas, eps, decay = 0.003 / step, (0.83, 0.97), 1e-6, 0.2
        expected = oracle(*expected, gradient, step=step, lr=lr, betas=betas, eps=eps, decay=decay)
        retained_fp32_m = betas[0] * retained_fp32_m + (1 - betas[0]) * gradient.float()
        staging = gradient.clone()
        EagerKernel().update(master, moment, variance, staging, step=step, lr=lr,
                             beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=decay)
        torch.testing.assert_close(master, expected[0], rtol=1e-5, atol=2e-6)
        assert torch.equal(moment, expected[1])
        assert torch.equal(variance, expected[2])
        assert torch.equal(staging, master.bfloat16())
        observed_rounding |= not torch.equal(moment, retained_fp32_m.bfloat16())
    assert observed_rounding, "must test persistent rounding, not just final-state compression"
    assert pointers == [tensor.data_ptr() for tensor in (master, moment, variance)]


def test_mixed_large_native_tails_missing_gradients_decay_and_state_inventory():
    initial = [torch.linspace(-0.7, 0.8, 45), torch.linspace(-0.2, 0.3, 7)]
    parameters = [torch.nn.Parameter(value.bfloat16()) for value in initial]
    optimizer = candidate.BF16MomentsAdamW(
        parameters, lr=0.002, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.15,
        _min_elements=16, _bucket_elements=32, _kernel_factory=EagerKernel,
    )
    optimizer.initialize_master_weights([value.clone() for value in initial])
    states = [optimizer.state[p] for p in parameters]
    assert [v.dtype for v in states[0]["exp_avg"]] == [torch.bfloat16, torch.float32]
    assert [v.dtype for v in states[1]["exp_avg"]] == [torch.float32]
    assert [v.numel() for v in states[0]["exp_avg"]] == [32, 13]
    expected = [[state["master_param"].clone(), [v.clone() for v in state["exp_avg"]],
                 [v.clone() for v in state["exp_avg_sq"]], 0] for state in states]
    pointers = [[v.data_ptr() for key in ("exp_avg", "exp_avg_sq") for v in state[key]] for state in states]
    EagerKernel.calls = []
    for iteration in range(4):
        lr = 0.002 / (iteration + 1)
        optimizer.param_groups[0]["lr"] = lr
        for index, parameter in enumerate(parameters):
            parameter.grad = (torch.linspace(-1.3, 1.2, parameter.numel()) * (iteration + 1)).bfloat16()
            if index == 1 and iteration % 2:
                parameter.grad = None
                continue
            reference = expected[index]
            reference[3] += 1
            for slice_index, (offset, count, _) in enumerate(optimizer._slice_layout[id(parameter)]):
                end = offset + count
                updated = oracle(reference[0][offset:end], reference[1][slice_index], reference[2][slice_index],
                                 parameter.grad[offset:end], step=reference[3], lr=lr,
                                 betas=(0.8, 0.95), eps=1e-7, decay=0.15)
                reference[0][offset:end] = updated[0]
                reference[1][slice_index], reference[2][slice_index] = updated[1:]
        optimizer.step()
        for parameter, state, reference in zip(parameters, states, expected, strict=True):
            assert state["step"] == reference[3]
            torch.testing.assert_close(state["master_param"], reference[0], rtol=1e-5, atol=2e-6)
            for key, index in (("exp_avg", 1), ("exp_avg_sq", 2)):
                for actual, wanted in zip(state[key], reference[index], strict=True):
                    torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=2e-6)
            assert torch.equal(parameter, state["master_param"].bfloat16())
    assert EagerKernel.calls == [32] * 4
    assert pointers == [[v.data_ptr() for key in ("exp_avg", "exp_avg_sq") for v in state[key]] for state in states]
    metadata = optimizer.candidate_metadata()
    assert metadata["bf16_moment_parameter_elements"] == 32
    assert metadata["fp32_moment_parameter_elements"] == 20
    assert metadata["expected_state_bytes"] == 32 * 8 + 20 * 12
    from yylab.train.benchmark_sprint_precision import _storage_inventory
    assert _storage_inventory(optimizer.state)["bytes"] == metadata["expected_state_bytes"]


def test_live_kernel_failure_never_retries_or_changes_native_tail():
    class Failure(EagerKernel):
        calls = 0

        @torch.no_grad()
        def update(self, master, *args, **kwargs):
            self.calls += 1
            master.add_(1)
            raise RuntimeError("intentional failure after mutation")

    parameter = torch.nn.Parameter(torch.zeros(37, dtype=torch.bfloat16))
    optimizer = candidate.BF16MomentsAdamW([parameter], _min_elements=16, _bucket_elements=32,
                                         _kernel_factory=Failure)
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="after mutation"):
        optimizer.step()
    assert optimizer._compiled_cpu_adam.calls == 1
    assert torch.equal(optimizer.state[parameter]["master_param"][:32], torch.ones(32))
    assert torch.equal(optimizer.state[parameter]["master_param"][32:], torch.zeros(5))
    assert torch.equal(parameter, torch.zeros_like(parameter))
    assert all(torch.count_nonzero(v) == 0 for v in optimizer.state[parameter]["exp_avg"])


def test_parameter_group_decay_and_learning_rates_are_preserved():
    initial = [torch.linspace(-0.4, 0.7, 20), torch.linspace(-0.2, 0.3, 7)]
    parameters = [torch.nn.Parameter(value.bfloat16()) for value in initial]
    optimizer = candidate.BF16MomentsAdamW(
        [{"params": [parameters[0]], "weight_decay": 0.3, "lr": 0.002},
         {"params": [parameters[1]], "weight_decay": 0.0, "lr": 0.001}],
        betas=(0.85, 0.97), eps=1e-6, _min_elements=16, _bucket_elements=32, _kernel_factory=EagerKernel,
    )
    optimizer.initialize_master_weights([value.clone() for value in initial])
    expected = []
    for parameter, group in zip(parameters, optimizer.param_groups, strict=True):
        parameter.grad = torch.linspace(-0.8, 1.3, parameter.numel()).bfloat16()
        state = optimizer.state[parameter]
        expected.append(oracle(state["master_param"].clone(), state["exp_avg"][0].clone(),
                               state["exp_avg_sq"][0].clone(), parameter.grad,
                               step=1, lr=group["lr"], betas=group["betas"], eps=group["eps"],
                               decay=group["weight_decay"]))
    optimizer.step()
    for parameter, reference in zip(parameters, expected, strict=True):
        state = optimizer.state[parameter]
        torch.testing.assert_close(state["master_param"], reference[0], rtol=1e-5, atol=2e-6)
        assert torch.equal(parameter, state["master_param"].bfloat16())


def test_preparation_failure_is_fatal_before_live_state_is_created():
    def fail(**kwargs):
        raise RuntimeError("preparation failed")

    parameter = torch.nn.Parameter(torch.zeros(32, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="preparation failed"):
        candidate.BF16MomentsAdamW([parameter], _min_elements=16, _bucket_elements=32, _kernel_factory=fail)
    assert torch.equal(parameter, torch.zeros_like(parameter))


def test_disposable_recurrence_validation_and_preparation_are_cpu_small(monkeypatch):
    from torch._inductor import cpp_builder
    compiled = []

    def eager_compile(function, **kwargs):
        compiled.append((function, kwargs))
        return function

    monkeypatch.setattr(candidate.baseline_kernel, "_activate_local_compiler", lambda: None)
    monkeypatch.setattr(cpp_builder, "get_cpp_compiler", lambda: "mock-compiler")
    monkeypatch.setattr(torch, "compile", eager_compile)
    candidate._prepare_kernel.cache_clear()
    try:
        first = candidate._prepare_kernel(1, 37)
        assert candidate._prepare_kernel(1, 37) is first
        second = candidate._prepare_kernel(1, 45)
        assert len(compiled) == 2
        assert compiled[0][0].__code__ is not compiled[1][0].__code__
        assert first[0] is not second[0]
        assert all(kwargs["fullgraph"] and kwargs["dynamic"] for _, kwargs in compiled)
    finally:
        candidate._prepare_kernel.cache_clear()
