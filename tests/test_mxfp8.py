from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yylab.train.mxfp8 import (
    EMULATED_MXFP8_BACKEND,
    NATIVE_MXFP8_BACKEND,
    mxfp8_linear_backward,
    mxfp8_mm_nt,
    native_mxfp8_status,
    resolve_mxfp8_backend,
)


def test_emulated_mxfp8_backward_handles_unaligned_leading_dimensions() -> None:
    pytest.importorskip("torchao")
    generator = torch.Generator().manual_seed(808)
    input = torch.randn((2, 5, 79), generator=generator)
    weight = torch.randn((113, 79), generator=generator)
    grad_output = torch.randn((2, 5, 113), generator=generator)

    grad_input, grad_weight = mxfp8_linear_backward(
        input,
        weight,
        grad_output,
        backend="emulated",
    )

    reference_input = grad_output @ weight
    reference_weight = grad_output.reshape(-1, 113).t() @ input.reshape(-1, 79)
    assert grad_input.shape == input.shape
    assert grad_weight.shape == weight.shape
    assert grad_input.dtype == input.dtype
    assert grad_weight.dtype == weight.dtype
    assert (grad_input - reference_input).norm() / reference_input.norm() < 0.12
    assert (grad_weight - reference_weight).norm() / reference_weight.norm() < 0.12


def test_mxfp8_mm_rejects_mismatched_contraction_dimensions() -> None:
    with pytest.raises(ValueError, match="contraction dimension"):
        mxfp8_mm_nt(torch.randn(3, 31), torch.randn(7, 32), backend="emulated")


def test_mxfp8_backend_resolution_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    from yylab.train import mxfp8

    monkeypatch.setattr(mxfp8, "_native_prerequisites", lambda: (False, "test gap"))
    backend, note = resolve_mxfp8_backend("auto")
    assert backend == EMULATED_MXFP8_BACKEND
    assert "test gap" in note
    with pytest.raises(RuntimeError, match="test gap"):
        resolve_mxfp8_backend("native")


def test_emulated_nvfp4_qat_really_uses_explicit_mxfp8_backward() -> None:
    from yylab.train.low_precision import QATLinear, apply_qat, qat_metadata

    metadata = qat_metadata(
        "nvfp4",
        backend="emulated",
        backward_precision="mxfp8",
        backward_backend="emulated",
    )
    assert metadata is not None
    model = torch.nn.Sequential(torch.nn.Linear(79, 35, bias=True))
    model.config = SimpleNamespace()
    apply_qat(model, metadata)
    assert isinstance(model[0], QATLinear)
    assert model[0].qat_backward_backend == "emulated"

    value = torch.randn(2, 5, 79, requires_grad=True)
    output = model(value)
    output.square().mean().backward()

    assert output.shape == (2, 5, 35)
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert model[0].weight.grad is not None
    assert torch.isfinite(model[0].weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_mxfp8_executes_both_backward_gemms() -> None:
    available, note = native_mxfp8_status(force_refresh=True)
    if not available:
        pytest.skip(note)
    backend, _ = resolve_mxfp8_backend("native")
    assert backend == NATIVE_MXFP8_BACKEND

    generator = torch.Generator(device="cpu").manual_seed(120)
    input = torch.randn((19, 47), generator=generator).to("cuda", torch.bfloat16)
    weight = torch.randn((35, 47), generator=generator).to("cuda", torch.bfloat16)
    grad_output = torch.randn((19, 35), generator=generator).to(
        "cuda", torch.bfloat16
    )
    grad_input, grad_weight = mxfp8_linear_backward(
        input,
        weight,
        grad_output,
        backend="native",
    )
    torch.cuda.synchronize()
    assert grad_input.shape == input.shape
    assert grad_weight.shape == weight.shape
    assert grad_input.dtype == torch.bfloat16
    assert grad_weight.dtype == torch.bfloat16
    assert torch.isfinite(grad_input).all()
    assert torch.isfinite(grad_weight).all()
