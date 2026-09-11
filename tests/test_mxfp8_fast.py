"""Validation and optional native GPU checks for MXFP8 transpose-cast candidate."""

from __future__ import annotations

import pytest
import torch

from yylab.train import mxfp8, mxfp8_fast


def test_fast_cast_fake_outputs_preserve_row_and_transposed_scale_orientation():
    value = torch.empty((128, 256), dtype=torch.bfloat16, device="meta")
    row_data, row_scales = mxfp8_fast._quantize_rows_raw(value)
    transposed_data, transposed_scales = mxfp8_fast._quantize_transposed_rows_raw(value)
    assert row_data.shape == (128, 256) and row_data.is_contiguous()
    assert row_scales.shape == (128, 8) and row_scales.is_contiguous()
    assert transposed_data.shape == (256, 128) and transposed_data.is_contiguous()
    assert transposed_scales.shape == (256, 4) and transposed_scales.is_contiguous()
    assert row_data.dtype == transposed_data.dtype == torch.float8_e4m3fn
    assert row_scales.dtype == transposed_scales.dtype == torch.float8_e8m0fnu
    for operator in (torch.ops.yylab.mxfp8_fast_rows.default, torch.ops.yylab.mxfp8_fast_transposed_rows.default):
        assert all(argument.alias_info is None for argument in operator._schema.arguments)


def test_irregular_fast_gemm_delegates_to_the_existing_native_path(monkeypatch):
    left, right = torch.zeros(19, 47), torch.zeros(35, 47)
    result = torch.empty(19, 35)
    received = []

    def native_call(actual_left, actual_right, *, backend):
        received.append((actual_left, actual_right, backend))
        return result

    monkeypatch.setattr(mxfp8, "mxfp8_mm_nt", native_call)
    assert mxfp8_fast.mxfp8_mm_nt_fast(left, right, backend="native") is result
    assert len(received) == 1 and received[0][0] is left and received[0][1] is right
    assert received[0][2] == "native"


def test_empty_fast_backward_delegates_to_existing_native_shape_semantics(monkeypatch):
    inputs, weight, grad = torch.zeros(0, 47), torch.zeros(35, 47), torch.zeros(0, 35)
    result = (torch.empty_like(inputs), torch.empty_like(weight))
    received = []

    def native_call(actual_input, actual_weight, actual_grad, *, backend):
        received.append((actual_input, actual_weight, actual_grad, backend))
        return result

    monkeypatch.setattr(mxfp8, "mxfp8_linear_backward", native_call)
    assert mxfp8_fast.mxfp8_linear_backward_fast(inputs, weight, grad, backend="native") is result
    assert len(received) == 1
    assert all(actual is expected for actual, expected in zip(received[0][:3], (inputs, weight, grad), strict=True))
    assert received[0][3] == "native"


def test_padded_backward_prepares_each_original_once_and_reuses_padded_gradient(monkeypatch):
    inputs = torch.ones(37, 79, dtype=torch.bfloat16)
    weight = torch.full((113, 79), 2, dtype=torch.bfloat16)
    grad = torch.full((37, 113), 3, dtype=torch.bfloat16)
    casts = []
    monkeypatch.setattr(mxfp8_fast, "_preflight", lambda *args: None)

    def fake_cast(value, *, transposed=False):
        casts.append((value, transposed))
        return value.t() if transposed else value

    def fake_mm(left, right):
        return left.new_zeros((left.shape[0], right.shape[1]))

    monkeypatch.setattr(mxfp8_fast, "_as_mx", fake_cast)
    monkeypatch.setattr(torch, "mm", fake_mm)
    dgrad, wgrad = mxfp8_fast.mxfp8_linear_backward_fast(inputs, weight, grad, backend="native")
    assert dgrad.shape == inputs.shape and wgrad.shape == weight.shape
    assert dgrad.is_contiguous() and wgrad.is_contiguous()
    assert len(casts) == 4 and [transposed for _, transposed in casts] == [False, True, True, True]
    assert casts[0][0] is casts[2][0]
    for padded, original in ((casts[0][0], grad), (casts[1][0], weight), (casts[3][0], inputs)):
        assert padded.shape == (128, 128) and padded.is_contiguous()
        assert torch.equal(padded[:original.shape[0], :original.shape[1]], original)
        assert torch.count_nonzero(padded[original.shape[0]:]) == 0
        assert torch.count_nonzero(padded[:, original.shape[1]:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_dim1_cast_matches_row_quantization_of_bf16_transpose_exactly():
    available, note = mxfp8.native_mxfp8_status()
    if not available:
        pytest.skip(note)
    generator = torch.Generator(device="cpu").manual_seed(981)
    value = torch.randn((128, 256), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    # Give rows/columns different scales so using the wrong contraction axis
    # cannot accidentally pass on a uniform-magnitude test tensor.
    value.mul_(torch.linspace(0.1, 8.0, 128, device="cuda", dtype=torch.bfloat16).unsqueeze(1))
    value[:, 32:64].mul_(16)
    original = value.clone()
    expected_data, expected_scales = mxfp8_fast._quantize_rows_raw(value.t().contiguous())
    actual_data, actual_scales = mxfp8_fast._quantize_transposed_rows_raw(value)
    torch.cuda.synchronize()
    assert torch.equal(value, original)
    assert actual_data.shape == (256, 128) and actual_data.is_contiguous()
    assert actual_scales.shape == (256, 4) and actual_scales.is_contiguous()
    assert torch.equal(actual_data.view(torch.uint8), expected_data.view(torch.uint8))
    assert torch.equal(actual_scales.view(torch.uint8), expected_scales.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_aligned_fast_forward_and_gradients_match_native_and_dense_reference():
    available, note = mxfp8.native_mxfp8_status()
    if not available:
        pytest.skip(note)
    generator = torch.Generator(device="cpu").manual_seed(982)
    inputs = torch.randn((2, 64, 256), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((384, 256), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    grad = torch.randn((2, 64, 384), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    expected_output = mxfp8.mxfp8_mm_nt(inputs.reshape(-1, 256), weight, backend="native")
    expected_dgrad, expected_wgrad = mxfp8.mxfp8_linear_backward(inputs, weight, grad, backend="native")
    output = mxfp8_fast.mxfp8_mm_nt_fast(inputs.reshape(-1, 256), weight, backend="native")
    dgrad, wgrad = mxfp8_fast.mxfp8_linear_backward_fast(inputs, weight, grad, backend="native")
    torch.cuda.synchronize()
    for actual, expected in ((output, expected_output), (dgrad, expected_dgrad), (wgrad, expected_wgrad)):
        assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    dense_output = inputs.float().reshape(-1, 256).mm(weight.float().t())
    dense_dgrad = grad.float().reshape(-1, 384).mm(weight.float()).reshape_as(inputs)
    dense_wgrad = grad.float().reshape(-1, 384).t().mm(inputs.float().reshape(-1, 256))
    for actual, expected in ((output, dense_output), (dgrad, dense_dgrad), (wgrad, dense_wgrad)):
        assert float((actual.float() - expected).norm() / expected.norm().clamp_min(1e-12)) < 0.15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows,input_features,output_features", [(37, 79, 113), (128, 128, 3008), (128, 6976, 128)])
def test_gpu_padded_fast_gradients_match_existing_native_semantics(rows, input_features, output_features):
    available, note = mxfp8.native_mxfp8_status()
    if not available:
        pytest.skip(note)
    generator = torch.Generator(device="cpu").manual_seed(984)
    inputs = torch.randn((rows, input_features), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((output_features, input_features), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    grad = torch.randn((rows, output_features), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    expected_dgrad, expected_wgrad = mxfp8.mxfp8_linear_backward(inputs, weight, grad, backend="native")
    dgrad, wgrad = mxfp8_fast.mxfp8_linear_backward_fast(inputs, weight, grad, backend="native")
    torch.cuda.synchronize()
    dense_dgrad = grad.float().mm(weight.float())
    dense_wgrad = grad.float().t().mm(inputs.float())
    for actual, native_reference, dense_reference in ((dgrad, expected_dgrad, dense_dgrad),
                                                      (wgrad, expected_wgrad, dense_wgrad)):
        assert actual.shape == native_reference.shape and actual.dtype == torch.bfloat16
        assert actual.is_contiguous() and torch.isfinite(actual).all() and torch.count_nonzero(actual)
        torch.testing.assert_close(actual, native_reference, rtol=0, atol=0)
        assert float((actual.float() - dense_reference).norm() / dense_reference.norm().clamp_min(1e-12)) < 0.15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_fast_forward_backward_are_fullgraph_compile_compatible():
    available, note = mxfp8.native_mxfp8_status()
    if not available:
        pytest.skip(note)
    generator = torch.Generator(device="cpu").manual_seed(983)
    inputs = torch.randn((128, 256), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((128, 256), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    grad = torch.randn((128, 128), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    forward = torch.compile(mxfp8_fast.mxfp8_mm_nt_fast, fullgraph=True, dynamic=False)
    backward = torch.compile(mxfp8_fast.mxfp8_linear_backward_fast, fullgraph=True, dynamic=False)
    expected = mxfp8_fast.mxfp8_mm_nt_fast(inputs, weight, backend="native")
    expected_dgrad, expected_wgrad = mxfp8_fast.mxfp8_linear_backward_fast(inputs, weight, grad, backend="native")
    actual = forward(inputs, weight, backend="native")
    dgrad, wgrad = backward(inputs, weight, grad, backend="native")
    torch.cuda.synchronize()
    for result, reference in ((actual, expected), (dgrad, expected_dgrad), (wgrad, expected_wgrad)):
        assert torch.isfinite(result).all() and torch.count_nonzero(result)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
