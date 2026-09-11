"""Compiled CPU Adam admission, shared staging, and exact checkpoint continuation."""

from __future__ import annotations

import copy

import pytest
import torch

from yylab.train import sprint_cpu_adam as kernels
from yylab.train.sprint import SprintCPUOffloadAdamW


@pytest.fixture(autouse=True)
def two_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_native_and_tiny_auto_do_not_activate_a_compiler(monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("native optimizer attempted compilation")

    monkeypatch.setattr(kernels, "CompiledCPUAdam", unexpected)
    for mode, dtype, count in [("native", torch.bfloat16, kernels.MIN_COMPILED_ELEMENTS),
                               ("auto", torch.bfloat16, 17), ("auto", torch.float32, 17)]:
        parameter = torch.nn.Parameter(torch.ones(count, dtype=dtype))
        optimizer = SprintCPUOffloadAdamW([parameter], cpu_kernel=mode)
        assert optimizer.resolved_cpu_kernel == "native"
        assert optimizer.cpu_kernel_reason


@pytest.mark.parametrize("mode,dtype,count", [("unknown", torch.bfloat16, 17),
                                              ("compiled", torch.bfloat16, 17),
                                              ("compiled", torch.float32, 17)])
def test_invalid_compiled_admission_fails_before_state_allocation(mode, dtype, count):
    parameter = torch.nn.Parameter(torch.ones(count, dtype=dtype))
    with pytest.raises(ValueError, match="kernel|BF16"):
        SprintCPUOffloadAdamW([parameter], cpu_kernel=mode)
    assert torch.equal(parameter, torch.ones_like(parameter))


def test_preparation_failure_falls_back_only_for_auto(monkeypatch, capsys):
    def unavailable(**kwargs):
        raise RuntimeError("test compiler unavailable")

    monkeypatch.setattr(kernels, "MIN_COMPILED_ELEMENTS", 16)
    monkeypatch.setattr(kernels, "CompiledCPUAdam", unavailable)
    parameter = torch.nn.Parameter(torch.ones(32, dtype=torch.bfloat16))
    optimizer = SprintCPUOffloadAdamW([parameter], cpu_kernel="auto")
    assert optimizer.resolved_cpu_kernel == "native"
    assert "test compiler unavailable" in optimizer.cpu_kernel_reason
    assert "using native CPU Adam" in capsys.readouterr().out
    assert len(optimizer.state) == 0
    with pytest.raises(RuntimeError, match="test compiler unavailable"):
        SprintCPUOffloadAdamW([parameter], cpu_kernel="compiled")


class _EagerKernel:
    calls = []

    def __init__(self, **kwargs):
        pass

    def update(self, master, moment, variance, staging, **settings):
        self.calls.append(staging.numel())
        kernels._fused_update(master, moment, variance, staging,
                              kernels._coefficients(**settings))


def test_mixed_large_and_small_slices_keep_gradients_and_per_parameter_steps(monkeypatch):
    monkeypatch.setattr(kernels, "MIN_COMPILED_ELEMENTS", 16)
    monkeypatch.setattr(kernels, "CompiledCPUAdam", _EagerKernel)
    _EagerKernel.calls = []
    initial = [torch.linspace(-0.7, 0.8, 45), torch.linspace(-0.2, 0.3, 7)]
    parameters = [torch.nn.Parameter(value.bfloat16()) for value in initial]
    references = [torch.nn.Parameter(value.clone()) for value in initial]
    optimizer = SprintCPUOffloadAdamW(parameters, lr=0.002, betas=(0.8, 0.95), cpu_kernel="compiled")
    optimizer.bucket_elements = 32
    optimizer.initialize_master_weights([value.clone() for value in initial])
    native = torch.optim.AdamW(references, lr=0.002, betas=(0.8, 0.95), fused=True, weight_decay=0)
    for step in range(4):
        for index, (parameter, reference) in enumerate(zip(parameters, references, strict=True)):
            gradient = torch.linspace(-1.3, 1.2, parameter.numel()).bfloat16() * (step + 1)
            parameter.grad = gradient if index == 0 or step % 2 == 0 else None
            reference.grad = parameter.grad.float() if parameter.grad is not None else None
        optimizer.step()
        native.step()
    assert _EagerKernel.calls == [32] * 4
    for parameter, reference in zip(parameters, references, strict=True):
        actual = optimizer.state[parameter]
        expected = native.state[reference]
        assert actual["step"] == int(expected["step"])
        torch.testing.assert_close(actual["master_param"], reference, rtol=1e-5, atol=2e-6)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=2e-6)
        assert torch.equal(parameter, actual["master_param"].bfloat16())


def test_live_compiled_failure_is_never_retried_as_native(monkeypatch):
    class MutatingFailure(_EagerKernel):
        def update(self, master, *args, **kwargs):
            master.add_(1)
            raise RuntimeError("failed after live mutation")

    monkeypatch.setattr(kernels, "MIN_COMPILED_ELEMENTS", 16)
    monkeypatch.setattr(kernels, "CompiledCPUAdam", MutatingFailure)
    parameter = torch.nn.Parameter(torch.zeros(32, dtype=torch.bfloat16))
    optimizer = SprintCPUOffloadAdamW([parameter], cpu_kernel="auto")
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="failed after live mutation"):
        optimizer.step()
    assert optimizer.resolved_cpu_kernel == "compiled"
    assert torch.equal(optimizer.state[parameter]["master_param"], torch.ones(32))
    assert torch.equal(optimizer.state[parameter]["exp_avg"], torch.zeros(32))


def test_preparation_keys_isolate_dynamo_code_and_reuse_cached_kernels(monkeypatch):
    from torch._inductor import cpp_builder

    compiled_functions = []

    def fake_compile(function, **kwargs):
        compiled_functions.append(function)
        return function

    monkeypatch.setattr(kernels, "_activate_local_compiler", lambda: None)
    monkeypatch.setattr(cpp_builder, "get_cpp_compiler", lambda: "mock-compiler")
    monkeypatch.setattr(torch, "compile", fake_compile)
    kernels._prepare_kernel.cache_clear()
    try:
        first = kernels._prepare_kernel(2, 17)
        assert kernels._prepare_kernel(2, 17) is first
        second = kernels._prepare_kernel(2, 19)
        third = kernels._prepare_kernel(4, 17)
        assert len(compiled_functions) == 3
        assert len({id(function.__code__) for function in compiled_functions}) == 3
        assert len({function.__code__.co_name for function in compiled_functions}) == 3
        assert first[0] is not second[0] and first[0] is not third[0]
    finally:
        kernels._prepare_kernel.cache_clear()


@pytest.fixture(scope="module")
def compiled_kernel():
    from torch._inductor import config

    try:
        kernels._activate_local_compiler()
    except RuntimeError as error:
        pytest.skip(f"installed CPU compiler unavailable: {error}")
    settings = (config.cpp.threads, config.compile_threads, config.fx_graph_cache)
    kernel = kernels.CompiledCPUAdam(cpu_threads=2, warmup_elements=kernels.MIN_COMPILED_ELEMENTS + 257)
    assert settings == (config.cpp.threads, config.compile_threads, config.fx_graph_cache)
    return kernel


def test_real_compiled_kernel_shared_alias_and_dynamic_scalars(compiled_kernel):
    master = torch.linspace(-0.8, 0.8, 257)
    moment, variance = torch.zeros_like(master), torch.zeros_like(master)
    staging = torch.empty(257, dtype=torch.bfloat16)
    reference = torch.nn.Parameter(master.clone())
    native = torch.optim.AdamW([reference], lr=0.001, betas=(0.8, 0.95), weight_decay=0.2, fused=True)
    for step in range(1, 7):
        lr = 0.001 / step
        staging.copy_(torch.linspace(-0.9, 1.4, 257) * step)
        reference.grad = staging.float()
        native.param_groups[0]["lr"] = lr
        native.step()
        compiled_kernel.update(master, moment, variance, staging, step=step, lr=lr,
                               beta1=0.8, beta2=0.95, eps=1e-8, weight_decay=0.2)
        torch.testing.assert_close(master, reference, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(moment, native.state[reference]["exp_avg"], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(variance, native.state[reference]["exp_avg_sq"], rtol=1e-5, atol=2e-6)
        assert torch.equal(staging, master.bfloat16())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_real_compiled_large_updates_and_checkpoint_resume(compiled_kernel, device, tmp_path):
    initial = torch.linspace(-0.7, 0.8, kernels.MIN_COMPILED_ELEMENTS + 257)
    parameter = torch.nn.Parameter(initial.to(device=device, dtype=torch.bfloat16))
    reference = torch.nn.Parameter(initial.clone())
    kwargs = {"bucket_mb": 4, "cpu_kernel": "compiled", "lr": 0.001,
              "betas": (0.8, 0.95), "weight_decay": 0.1}
    optimizer = SprintCPUOffloadAdamW([parameter], **kwargs)
    optimizer.initialize_master_weights([initial.clone()])
    assert optimizer.resolved_cpu_kernel == "compiled"
    native = torch.optim.AdamW([reference], lr=0.001, betas=(0.8, 0.95), weight_decay=0.1, fused=True)
    resumed = resumed_parameter = None
    for step in range(3):
        gradient = torch.linspace(-1.2, 1.5, initial.numel()).bfloat16() * (step + 1)
        parameter.grad = gradient.to(device)
        reference.grad = gradient.float()
        optimizer.step()
        native.step()
        state = optimizer.state_dict()
        actual = optimizer.state[parameter]
        torch.testing.assert_close(actual["master_param"], reference, rtol=1e-5, atol=2e-6)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual[key], native.state[reference][key], rtol=1e-5, atol=2e-6)
        assert torch.equal(parameter.cpu(), actual["master_param"].bfloat16())
        if resumed is not None:
            resumed_parameter.grad = gradient.to(device)
            resumed.step()
            resumed.synchronize()
            assert torch.equal(resumed_parameter, parameter)
            restored = resumed.state[resumed_parameter]
            assert restored["step"] == actual["step"]
            for key in ("master_param", "exp_avg", "exp_avg_sq"):
                assert torch.equal(restored[key], actual[key])
        if step == 0:
            checkpoint = tmp_path / "optimizer.pt"
            torch.save(state, checkpoint)
            resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
            resumed = SprintCPUOffloadAdamW([resumed_parameter], **kwargs)
            resumed.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            assert resumed.resolved_cpu_kernel == "compiled"
    assert all(value.numel() <= 2 * 2**20 for value in optimizer._staging_buffers)
    # state_dict owns CPU FP32 tensors; no dtype/device conversion on restoration.
    payload = copy.deepcopy(optimizer.state_dict())
    assert all(value.dtype == torch.float32 and value.device.type == "cpu"
               for value in payload["state"][0].values() if isinstance(value, torch.Tensor))
