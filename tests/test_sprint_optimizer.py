"""Small numerical checks for Sprint's bounded CPU-offload update pipeline."""

from __future__ import annotations

import copy

import pytest
import torch

from yylab.train.sprint import SprintCPUOffloadAdamW


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _exercise_updates(device: str, dtype: torch.dtype, fused: bool):
    generator = torch.Generator().manual_seed(123)
    initial = [torch.randn(11, 9, generator=generator), torch.randn(23, generator=generator),
               torch.randn(7, generator=generator)]
    parameters = [torch.nn.Parameter(value.to(device=device, dtype=dtype)) for value in initial]
    reference = [torch.nn.Parameter(value.clone()) for value in initial]

    def groups(values):
        return [{"params": values[:2], "lr": 0.002, "weight_decay": 0.15},
                {"params": values[2:], "lr": 0.003, "weight_decay": 0.0}]

    offload = SprintCPUOffloadAdamW(groups(parameters), betas=(0.8, 0.95))
    offload.cpu_fused = fused
    offload.bucket_elements = 13
    offload.initialize_master_weights([value.clone() for value in initial])
    expected = torch.optim.AdamW(groups(reference), betas=(0.8, 0.95), fused=fused, foreach=not fused)
    resumed = resumed_parameters = None
    for step in range(5):
        grads = [torch.randn(value.shape, generator=generator).to(dtype) if (index + step) % 4 != 1 else None
                 for index, value in enumerate(initial)]
        for parameter, master, grad in zip(parameters, reference, grads, strict=True):
            parameter.grad = grad.to(device) if grad is not None else None
            master.grad = grad.float() if grad is not None else None
        offload.step()
        expected.step()
        if resumed is not None:
            for parameter, grad in zip(resumed_parameters, grads, strict=True):
                parameter.grad = grad.to(device) if grad is not None else None
            resumed.step()
        # state_dict is a public synchronization boundary, including writeback.
        offload.state_dict()
        for index, (parameter, master) in enumerate(zip(parameters, reference, strict=True)):
            state = offload.state[parameter]
            if "exp_avg" in state:
                # SIMD tails at bucket boundaries can differ by a few FP32
                # rounding bits from a single whole-tensor CPU Adam call.
                torch.testing.assert_close(state["master_param"], master, rtol=2e-6, atol=1e-7)
                for key in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(state[key], expected.state[master][key], rtol=2e-6, atol=1e-7)
                assert state["step"] == int(expected.state[master]["step"])
            torch.testing.assert_close(parameter.cpu(), master.to(dtype), rtol=2e-6, atol=1e-7)
            if resumed is not None:
                resumed.synchronize()
                torch.testing.assert_close(resumed_parameters[index], parameter, rtol=0, atol=0)
                torch.testing.assert_close(resumed.state[resumed_parameters[index]]["master_param"],
                                           state["master_param"], rtol=0, atol=0)
        if step == 1:
            payload = copy.deepcopy(offload.state_dict())
            resumed_parameters = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
            resumed = SprintCPUOffloadAdamW(groups(resumed_parameters), betas=(0.8, 0.95))
            resumed.cpu_fused = fused
            resumed.bucket_elements = 13
            resumed.load_state_dict(payload)
    assert len(offload._staging_buffers) == (2 if device.startswith("cuda") else 1)
    assert all(value.numel() <= 13 for value in offload._staging_buffers)
    assert offload._gradient.numel() <= 13
    if device.startswith("cuda"):
        assert all(value.is_pinned() for value in offload._staging_buffers)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fused", [True, False])
def test_cpu_offload_bucket_updates_and_resume_match_cpu_adam(dtype, fused):
    _exercise_updates("cpu", dtype, fused)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_offload_pipeline_matches_cpu_master_updates(dtype):
    _exercise_updates("cuda", dtype, True)
