"""Contract checks and optional native-GPU parity for INT8-forward BitNet QAT."""

from __future__ import annotations

import copy
import json

import pytest
import torch
from torch import nn

from yylab.models.sprint import SprintConfig, SprintForCausalLM
from yylab.train import bitnet_int8_benchmark as int8
from yylab.train.bitnet_benchmark import (
    BitNetQATLinear,
    convert_sprint_to_bitnet_qat,
    quantize_int8_per_token,
    quantize_ternary_absmean,
)


def _tiny_model():
    return SprintForCausalLM(SprintConfig(
        vocab_size=19, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=8, loss_backend="torch")).bfloat16()


def test_integer_codes_dequantize_to_the_exact_reference_bf16_operands():
    weight = torch.linspace(-1.7, 2.1, 7 * 11).reshape(7, 11).bfloat16()
    activation = torch.linspace(-2.2, 1.9, 3 * 11).reshape(3, 11).bfloat16()
    weight_codes, weight_inverse = int8._weight_codes(weight)
    activation_codes, activation_inverse = int8._activation_codes(activation)
    assert weight_codes.dtype == activation_codes.dtype == torch.int8
    assert set(weight_codes.flatten().tolist()) <= {-1, 0, 1}
    assert activation_codes.min() >= -128 and activation_codes.max() <= 127
    assert torch.equal((weight_codes.float() / weight_inverse).bfloat16(), quantize_ternary_absmean(weight))
    assert torch.equal((activation_codes.float() / activation_inverse).bfloat16(), quantize_int8_per_token(activation))


def test_cpu_conversion_and_execution_fail_closed_without_mutation():
    model = _tiny_model()
    modules = {name: id(module) for name, module in model.named_modules()}
    available, note = int8.native_bitnet_int8_prerequisites("cpu")
    assert not available and "CUDA" in note
    with pytest.raises(RuntimeError, match="requires CUDA"):
        int8.convert_bitnet_to_int8_forward(model)
    assert modules == {name: id(module) for name, module in model.named_modules()}
    with pytest.raises(RuntimeError, match="same CUDA device"):
        int8.bitnet_int8_linear(torch.zeros(2, 16, dtype=torch.bfloat16),
                               torch.zeros(32, 16, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="no dense fallback"):
        int8._native_int8_mm(torch.zeros(32, 32, dtype=torch.int8), torch.zeros(32, 32, dtype=torch.int8))


def test_conversion_preserves_bitnet_parameter_objects_with_mocked_admission(monkeypatch):
    model = _tiny_model()
    base_metadata = convert_sprint_to_bitnet_qat(model)
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    module_ids = {name: id(module) for name, module in model.named_modules()}
    monkeypatch.setattr(int8, "native_bitnet_int8_prerequisites", lambda device: (True, "mocked admission only"))
    metadata = int8.convert_bitnet_to_int8_forward(model)
    assert parameter_ids == {id(parameter) for parameter in model.parameters()}
    assert module_ids == {name: id(module) for name, module in model.named_modules()}
    assert metadata["total_parameters"] == base_metadata["total_parameters"]
    assert metadata["added_norm_parameters"] == base_metadata["added_norm_parameters"]
    assert metadata["converted_linear_count"] == 4
    assert metadata["native_low_bit_gemm"] and not metadata["native_low_bit_backward"]
    assert not metadata["native_packed_ternary"]
    assert metadata["forward_gemm"] == "int8" and metadata["backward_gemm"] == "bf16"
    assert "BF16-dequantized" in metadata["numerical_difference"]
    assert model.lm_head.weight is model.embed_tokens.weight
    assert int8.convert_bitnet_to_int8_forward(model) == metadata
    json.dumps(metadata)


def test_int8_custom_op_fake_contract_is_int32_and_nonmutating():
    left = torch.empty(32, 96, device="meta", dtype=torch.int8)
    right = torch.empty(128, 96, device="meta", dtype=torch.int8)
    output = int8._native_int8_mm(left, right)
    assert output.shape == (32, 128) and output.dtype == torch.int32
    schema = torch.ops.yylab.bitnet_native_int8_mm.default._schema
    assert all(argument.alias_info is None for argument in schema.arguments)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows,features,outputs", [(32, 64, 96), (37, 79, 113)])
def test_gpu_int8_forward_matches_explicit_integer_reference_and_bf16_ste(rows, features, outputs):
    available, note = int8.native_bitnet_int8_prerequisites("cuda")
    if not available:
        pytest.skip(note)
    generator = torch.Generator(device="cpu").manual_seed(771)
    inputs = torch.randn((rows, features), generator=generator).to(device="cuda", dtype=torch.bfloat16).requires_grad_()
    weight = torch.randn((outputs, features), generator=generator).to(device="cuda", dtype=torch.bfloat16).requires_grad_()
    upstream = torch.randn((rows, outputs), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    input_codes, input_inverse = int8._activation_codes(inputs.detach())
    weight_codes, weight_inverse = int8._weight_codes(weight.detach())
    integer_reference = input_codes.cpu().to(torch.int64).mm(weight_codes.cpu().to(torch.int64).t())
    reference = (integer_reference.to(device="cuda", dtype=torch.float32)
                 * (input_inverse.reciprocal() * weight_inverse.reciprocal())).bfloat16()
    quant_input = quantize_int8_per_token(inputs.detach())
    quant_weight = quantize_ternary_absmean(weight.detach())
    expected_dgrad, expected_wgrad = upstream.mm(quant_weight), upstream.t().mm(quant_input)
    actual = int8.bitnet_int8_linear(inputs, weight)
    actual.backward(upstream)
    torch.cuda.synchronize()
    assert actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for gradient, expected in ((inputs.grad, expected_dgrad), (weight.grad, expected_wgrad)):
        assert gradient is not None and gradient.isfinite().all() and torch.count_nonzero(gradient)
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_int8_wrapper_ste_reaches_internal_norm_like_dense_qat():
    available, note = int8.native_bitnet_int8_prerequisites("cuda")
    if not available:
        pytest.skip(note)
    torch.manual_seed(772)
    reference = BitNetQATLinear(nn.Linear(64, 96, bias=False, device="cuda", dtype=torch.bfloat16))
    candidate = copy.deepcopy(reference)
    candidate.__class__ = int8.BitNetInt8ForwardLinear
    reference_input = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    candidate_input = reference_input.detach().clone().requires_grad_()
    upstream = torch.randn(2, 16, 96, device="cuda", dtype=torch.bfloat16)
    reference(reference_input).backward(upstream)
    candidate(candidate_input).backward(upstream)
    torch.cuda.synchronize()
    for actual, expected in ((candidate_input.grad, reference_input.grad),
                              (candidate.weight.grad, reference.weight.grad),
                              (candidate.norm.weight.grad, reference.norm.weight.grad)):
        assert actual is not None and actual.isfinite().all() and torch.count_nonzero(actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_int8_wrapper_fullgraph_forward_and_backward_match_eager():
    available, note = int8.native_bitnet_int8_prerequisites("cuda")
    if not available:
        pytest.skip(note)
    torch.manual_seed(773)
    eager = BitNetQATLinear(nn.Linear(64, 96, bias=False, device="cuda", dtype=torch.bfloat16))
    eager.__class__ = int8.BitNetInt8ForwardLinear
    traced = copy.deepcopy(eager)
    first = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    second = first.detach().clone().requires_grad_()
    upstream = torch.randn(2, 16, 96, device="cuda", dtype=torch.bfloat16)
    expected = eager(first)
    expected.backward(upstream)
    compiled = torch.compile(traced, fullgraph=True, dynamic=False)
    actual = compiled(second)
    actual.backward(upstream)
    torch.cuda.synchronize()
    # Default Inductor fusion can remove an intermediate BF16 rounding before
    # absmax, moving activations across one-code quantization boundaries. This
    # fixture measured 0.52% output / 0.58% weight-gradient relative L2 error;
    # emulate_precision_casts=True makes every output/gradient bitwise equal.
    # Keep the throughput recipe and bound aggregate plus worst absolute error.
    # Separate tests above require exact integer-reference and STE gradients.
    pairs = {"output": (actual, expected), "input_gradient": (second.grad, first.grad),
             "weight_gradient": (traced.weight.grad, eager.weight.grad),
             "norm_gradient": (traced.norm.weight.grad, eager.norm.weight.grad)}
    for name, (result, reference) in pairs.items():
        assert result is not None and result.isfinite().all()
        result, reference = result.detach().float(), reference.detach().float()
        difference = (result - reference).abs()
        relative_l2 = float(difference.norm() / reference.norm().clamp_min(1e-12))
        max_absolute = float(difference.max())
        max_reference = float(reference.abs().max())
        assert relative_l2 < 0.01, f"{name}: relative L2 error {relative_l2:.6f}"
        assert max_absolute < 0.01 * max_reference + 1e-6, (
            f"{name}: max absolute error {max_absolute:.6f}, reference maximum {max_reference:.6f}")
