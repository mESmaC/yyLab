"""CPU checks for the isolated Sprint-shape BitNet training comparison."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from yylab.models.sprint import SprintConfig, SprintForCausalLM
from yylab.train.bitnet_benchmark import (
    BitNetQATLinear,
    convert_sprint_to_bitnet_qat,
    int8_activation_ste,
    quantize_int8_per_token,
    quantize_ternary_absmean,
    ternary_weight_ste,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(*, mlp_type="gelu", tied=True):
    return SprintForCausalLM(SprintConfig(
        vocab_size=19, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, max_position_embeddings=8, loss_backend="torch",
        mlp_type=mlp_type, tie_word_embeddings=tied))


def test_weight_quantization_has_one_ternary_absmean_scale_per_matrix():
    weights = torch.tensor([[-2.0, -0.1, 0.0], [0.2, 4.0, 1.0]])
    scale = weights.abs().mean()
    actual = quantize_ternary_absmean(weights)
    expected = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]) * scale
    torch.testing.assert_close(actual, expected)
    assert quantize_ternary_absmean(weights.bfloat16()).dtype == torch.bfloat16
    zeros = quantize_ternary_absmean(torch.zeros(3, 7, dtype=torch.bfloat16))
    assert torch.isfinite(zeros).all() and not torch.count_nonzero(zeros)


def test_activation_quantization_is_per_token_with_signed_int8_levels():
    activation = torch.tensor([[[-2.0, 0.0, 1.0, 2.0], [-20.0, 0.0, 10.0, 20.0]]])
    actual = quantize_int8_per_token(activation)
    expected_first = torch.tensor([-127.0, 0.0, 64.0, 127.0]) * (2.0 / 127)
    torch.testing.assert_close(actual[0, 0], expected_first)
    torch.testing.assert_close(actual[0, 1], expected_first * 10)
    assert actual.shape == activation.shape
    zeros = quantize_int8_per_token(torch.zeros(2, 3, 4, dtype=torch.bfloat16))
    assert torch.isfinite(zeros).all() and not torch.count_nonzero(zeros)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("operation,quantizer", [(ternary_weight_ste, quantize_ternary_absmean),
                                                 (int8_activation_ste, quantize_int8_per_token)])
def test_ste_preserves_exact_quantized_forward_and_identity_gradient(dtype, operation, quantizer):
    values = torch.linspace(-1.7, 2.1, 24).reshape(4, 6).to(dtype).requires_grad_()
    upstream = torch.linspace(-0.4, 0.8, 24).reshape_as(values).to(dtype)
    actual = operation(values)
    assert torch.equal(actual, quantizer(values.detach()))
    actual.backward(upstream)
    assert torch.equal(values.grad, upstream)
    assert torch.count_nonzero(values.grad)


@pytest.mark.parametrize("mlp_type,tied", [("gelu", True), ("swiglu", False)])
def test_conversion_preserves_weights_ties_shapes_and_accounts_for_norms(mlp_type, tied):
    model = tiny_model(mlp_type=mlp_type, tied=tied)
    originals = {id(parameter): parameter.detach().clone() for parameter in model.parameters()}
    embedding, head = model.embed_tokens, model.lm_head
    old_input_norms = [block.input_layernorm for block in model.layers]
    old_mlp_norms = [block.post_attention_layernorm for block in model.layers]
    random_state = torch.random.get_rng_state().clone()
    metadata = convert_sprint_to_bitnet_qat(model)
    assert torch.equal(random_state, torch.random.get_rng_state())
    assert model.embed_tokens is embedding and model.lm_head is head
    assert (model.embed_tokens.weight is model.lm_head.weight) == tied
    assert metadata["converted_linear_count"] == 8
    assert metadata["added_norm_parameters"] == 2 * (16 + 32)
    assert metadata["total_parameters"] == metadata["original_parameters"] + 96
    assert metadata["native_low_bit_gemm"] is False
    json.dumps(metadata)
    for parameter in model.parameters():
        if id(parameter) in originals:
            assert torch.equal(parameter, originals.pop(id(parameter)))
        else:
            assert parameter.ndim == 1 and torch.equal(parameter, torch.ones_like(parameter))
    assert not originals
    for index, block in enumerate(model.layers):
        assert isinstance(block.input_layernorm, nn.Identity)
        assert isinstance(block.post_attention_layernorm, nn.Identity)
        assert block.self_attn.qkv_proj.norm is old_input_norms[index]
        assert block.up_proj.norm is old_mlp_norms[index]
        assert isinstance(block.self_attn.o_proj, BitNetQATLinear)
        assert isinstance(block.down_proj, BitNetQATLinear)
    identities = [id(parameter) for parameter in model.parameters()]
    assert convert_sprint_to_bitnet_qat(model) == metadata
    assert identities == [id(parameter) for parameter in model.parameters()]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_converted_causal_loss_backpropagates_into_every_projection(dtype):
    torch.manual_seed(7)
    model = tiny_model().to(dtype)
    convert_sprint_to_bitnet_qat(model)
    inputs = torch.randint(0, model.config.vocab_size, (2, 7))
    loss = model(inputs, labels=inputs, use_cache=False).loss
    assert loss.isfinite()
    loss.backward()
    for name, module in model.named_modules():
        if isinstance(module, BitNetQATLinear):
            assert module.weight.grad is not None, name
            assert module.weight.grad.dtype == dtype
            assert module.weight.grad.isfinite().all(), name
            assert torch.count_nonzero(module.weight.grad), name
            assert module.norm.weight.grad is not None, name
            assert module.norm.weight.grad.isfinite().all(), name


def test_qat_linear_and_backward_are_fullgraph_compile_compatible():
    torch.manual_seed(3)
    layer = BitNetQATLinear(nn.Linear(8, 12, bias=False)).bfloat16()
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    values = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    expected = layer(values)
    actual = compiled(values)
    assert torch.equal(actual, expected)
    actual.float().square().mean().backward()
    assert values.grad is not None and values.grad.isfinite().all() and torch.count_nonzero(values.grad)
    assert layer.weight.grad is not None and torch.count_nonzero(layer.weight.grad)


def test_invalid_topology_is_rejected_before_partial_conversion():
    with pytest.raises(TypeError, match="SprintForCausalLM"):
        convert_sprint_to_bitnet_qat(nn.Linear(8, 12))
    model = tiny_model()
    first = model.layers[0].self_attn.qkv_proj
    model.layers[1].down_proj = nn.Identity()
    with pytest.raises(ValueError, match="unconverted"):
        convert_sprint_to_bitnet_qat(model)
    assert model.layers[0].self_attn.qkv_proj is first


def test_wrapper_preserves_optional_bias_parameter():
    linear = nn.Linear(4, 3, bias=True)
    converted = BitNetQATLinear(linear)
    assert converted.weight is linear.weight and converted.bias is linear.bias
    converted(torch.ones(2, 4)).sum().backward()
    assert torch.equal(linear.bias.grad, torch.full((3,), 2.0))
