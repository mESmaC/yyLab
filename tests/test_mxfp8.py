from __future__ import annotations

import pytest
import torch

from yylab.train.mxfp8 import (
    NATIVE_MXFP8_BACKEND,
    MXFP8Error,
    mxfp8_linear_backward,
    mxfp8_mm_nt,
    native_mxfp8_status,
    resolve_mxfp8_backend,
)


def test_mxfp8_rejects_emulated_backend() -> None:
    with pytest.raises(ValueError, match="unknown MXFP8 backend"):
        resolve_mxfp8_backend("emulated")


def test_mxfp8_mm_rejects_mismatched_contraction_dimensions() -> None:
    with pytest.raises(ValueError, match="contraction dimension"):
        mxfp8_mm_nt(torch.randn(3, 31), torch.randn(7, 32), backend="native")


def test_mxfp8_backend_resolution_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    from yylab.train import mxfp8

    monkeypatch.setattr(
        mxfp8,
        "_native_prerequisites",
        lambda _device=None: (False, "test gap"),
    )
    with pytest.raises(MXFP8Error, match="test gap"):
        resolve_mxfp8_backend("auto")
    with pytest.raises(MXFP8Error, match="test gap"):
        resolve_mxfp8_backend("native")


def test_mxfp8_backend_accepts_its_exact_resolved_provider_token(monkeypatch) -> None:
    from yylab.train import mxfp8

    monkeypatch.setattr(
        mxfp8,
        "_native_prerequisites",
        lambda _device=None: (True, "exact provider available"),
    )

    resolved, note = resolve_mxfp8_backend(NATIVE_MXFP8_BACKEND)

    assert resolved == NATIVE_MXFP8_BACKEND
    assert note == "exact provider available"


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
