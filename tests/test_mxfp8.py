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


def test_mxfp8_custom_cast_fake_contract_has_correct_shapes_and_no_mutation() -> None:
    from yylab.train.mxfp8 import _mx_quantize_rows_raw

    values = torch.empty((129, 160), dtype=torch.bfloat16, device="meta")
    packed, scales = _mx_quantize_rows_raw(values)
    assert packed.shape == values.shape and packed.dtype == torch.float8_e4m3fn
    assert scales.shape == (129, 5) and scales.dtype == torch.float8_e8m0fnu
    assert packed.device.type == scales.device.type == "meta"
    schema = torch.ops.yylab.mxfp8_quantize_rows.default._schema
    assert all(argument.alias_info is None for argument in schema.arguments)
    assert all(result.alias_info is None for result in schema.returns)


@pytest.mark.parametrize("left,right,error,message", [
    (torch.zeros(2, 3, 4), torch.zeros(3, 4), ValueError, "2-D"),
    (torch.zeros(2, 4), torch.zeros(3, 4, device="meta"), ValueError, "same device"),
    (torch.zeros(2, 4), torch.zeros(3, 4, dtype=torch.bfloat16), TypeError, "same high-precision dtype"),
])
def test_mxfp8_operand_contract_fails_before_kernel_dispatch(left, right, error, message, monkeypatch):
    from yylab.train import mxfp8

    def unexpected(*args, **kwargs):
        raise AssertionError("input validation must run before native dispatch")

    monkeypatch.setattr(mxfp8, "resolve_mxfp8_backend", unexpected)
    with pytest.raises(error, match=message):
        mxfp8_mm_nt(left, right, backend="native")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_mxfp8_custom_cast_matches_original_torchao_bytes_without_mutation() -> None:
    from yylab.train.mxfp8 import _mx_quantize_rows_raw

    available, note = native_mxfp8_status()
    if not available:
        pytest.skip(note)
    from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0

    generator = torch.Generator(device="cpu").manual_seed(902)
    values = torch.randn((129, 160), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    original = values.clone()
    expected_data, expected_scales = triton_to_mxfp8_dim0(values, inner_block_size=32, scaling_mode="rceil")
    actual_data, actual_scales = _mx_quantize_rows_raw(values)
    torch.cuda.synchronize()
    assert actual_data.shape == values.shape and actual_data.dtype == torch.float8_e4m3fn
    assert actual_scales.shape == (129, 5) and actual_scales.dtype == torch.float8_e8m0fnu
    assert torch.equal(values, original)
    assert torch.equal(actual_data.view(torch.uint8), expected_data.view(torch.uint8))
    assert torch.equal(actual_scales.view(torch.uint8), expected_scales.view(torch.uint8))
    assert actual_data.untyped_storage().data_ptr() != values.untyped_storage().data_ptr()
    assert actual_scales.untyped_storage().data_ptr() != values.untyped_storage().data_ptr()
    assert actual_data.untyped_storage().data_ptr() != actual_scales.untyped_storage().data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_native_mxfp8_fullgraph_forward_backward_matches_eager() -> None:
    import copy

    from yylab.train.sprint_precision import MXFP8Linear

    available, note = native_mxfp8_status()
    if not available:
        pytest.skip(note)
    generator = torch.Generator(device="cpu").manual_seed(903)
    eager = MXFP8Linear(47, 35, bias=True, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        eager.weight.copy_(torch.randn((35, 47), generator=generator) * 0.2)
        eager.bias.copy_(torch.randn((35,), generator=generator) * 0.01)
    traced = copy.deepcopy(eager)
    inputs = torch.randn((2, 19, 47), generator=generator).to(device="cuda", dtype=torch.bfloat16).requires_grad_()
    traced_inputs = inputs.detach().clone().requires_grad_()
    upstream = torch.randn((2, 19, 35), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    expected = eager(inputs)
    expected.backward(upstream)
    compiled = torch.compile(traced, fullgraph=True, dynamic=False)
    actual = compiled(traced_inputs)
    actual.backward(upstream)
    torch.cuda.synchronize()
    for result, reference in ((actual, expected), (traced_inputs.grad, inputs.grad),
                              (traced.weight.grad, eager.weight.grad), (traced.bias.grad, eager.bias.grad)):
        assert result is not None and reference is not None
        assert result.dtype == torch.bfloat16
        assert torch.isfinite(result).all() and torch.count_nonzero(result)
        torch.testing.assert_close(result, reference, rtol=0.02, atol=0.005)
    # The native low-bit results should also approximate the dense FP32 GEMMs;
    # equality to eager alone would not catch both paths sharing a bad result.
    dense_output = torch.nn.functional.linear(inputs.float(), eager.weight.float(), eager.bias.float())
    dense_dgrad = upstream.float().reshape(-1, 35).mm(eager.weight.float()).reshape_as(inputs)
    dense_wgrad = upstream.float().reshape(-1, 35).t().mm(inputs.float().reshape(-1, 47))
    for result, reference in ((actual, dense_output), (traced_inputs.grad, dense_dgrad),
                              (traced.weight.grad, dense_wgrad)):
        relative_error = (result.float() - reference).norm() / reference.norm().clamp_min(1e-12)
        assert float(relative_error) < 0.15
