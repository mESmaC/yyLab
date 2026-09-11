"""Experimental Sprint-shape BitNet b1.58 QAT for training benchmarks only.

This uses dense floating-point GEMMs, including in backward. BF16 latent
parameters stay BF16; ternary weights and INT8 activations are dequantized to
that dtype for GEMM. It is neither packed ternary training nor bitnet.cpp.

Methodology follows section 3 of the authors' training implementation:
https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
and FP32 quantization reductions/identity STE in Transformers' online BitLinear:
https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/bitnet.py

Sprint's depth, widths, packed projections, activation function, attention,
embeddings and output head are preserved. This is not a reproduction of the
later BitNet 2B4T architecture or its pretraining/quality results.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

BITNET_TRAINING_SOURCE = (
    "https://github.com/microsoft/unilm/blob/master/bitnet/"
    "The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf"
)
BITNET_PAPER = "https://arxiv.org/abs/2402.17764"
BITNET_TRANSFORMERS_SOURCE = (
    "https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/bitnet.py"
)


def quantize_ternary_absmean(weight: torch.Tensor) -> torch.Tensor:
    """Dequantized {-mean_abs, 0, mean_abs}, with one FP32 scale per matrix."""
    values = weight.float()
    inverse_scale = values.abs().mean().clamp_min(1e-5).reciprocal()
    return ((values * inverse_scale).round().clamp(-1, 1) / inverse_scale).to(weight.dtype)


def quantize_int8_per_token(activation: torch.Tensor) -> torch.Tensor:
    """Dequantized signed INT8 values, with one FP32 absmax scale per token."""
    values = activation.float()
    inverse_scale = 127.0 / values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    return ((values * inverse_scale).round().clamp(-128, 127) / inverse_scale).to(activation.dtype)


class _TernaryWeightSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        return quantize_ternary_absmean(weight)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class _Int8ActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation):
        return quantize_int8_per_token(activation)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def ternary_weight_ste(weight: torch.Tensor) -> torch.Tensor:
    """Quantize forward; pass the dense GEMM's weight gradient through unchanged."""
    return _TernaryWeightSTE.apply(weight)


def int8_activation_ste(activation: torch.Tensor) -> torch.Tensor:
    """Quantize forward; retain gradients into the internal RMS normalization."""
    return _Int8ActivationSTE.apply(activation)


class BitNetQATLinear(nn.Module):
    """Internal RMSNorm + W1.58A8 STE + dense GEMM, preserving latent weights.

    The constructor adopts the original Parameter objects without rounding,
    cloning or changing their dtype. Convert the benchmark model to BF16 using
    its normal training engine before measuring BF16 dense QAT throughput.
    """

    def __init__(self, linear: nn.Linear, *, norm: nn.Module | None = None, eps: float = 1e-6):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("BitNetQATLinear requires an ordinary nn.Linear")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight, self.bias = linear.weight, linear.bias
        self.norm = norm if norm is not None else nn.RMSNorm(
            self.in_features, eps=eps, device=self.weight.device, dtype=self.weight.dtype)
        self.train(linear.training)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(inputs)
        return F.linear(int8_activation_ste(normalized), ternary_weight_ste(self.weight), self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, quantization=W1.58A8_QAT, gemm=dense")


def convert_sprint_to_bitnet_qat(model: nn.Module) -> dict:
    """Convert a scratch Sprint in place and return JSON-safe methodology/counts.

    Only decoder projections are quantized. The two existing outer block norms
    move inside QKV/up projections with their original affine parameters. The
    output/down projections receive new internal RMSNorms initialized to one.
    This follows the reference's removal of redundant outer norms while
    retaining every original Parameter object. Tied embeddings/head stay tied.

    Packed QKV shares one ternary scale across Q/K/V. Packed SwiGLU, if selected,
    similarly shares one scale for gate/up. These choices deliberately retain
    Sprint GEMM shapes for a controlled comparison. Apply before constructing
    the optimizer or compiling the model. No production checkpoint integration
    is provided for this experimental topology.
    """
    from yylab.models.sprint import SprintForCausalLM

    if not isinstance(model, SprintForCausalLM):
        raise TypeError("BitNet benchmark conversion requires a SprintForCausalLM")
    if hasattr(model, "_bitnet_qat_benchmark_metadata"):
        return copy.deepcopy(model._bitnet_qat_benchmark_metadata)
    # Validate the entire topology before mutating any decoder module.
    for block in model.layers:
        if not all(isinstance(layer, nn.Linear) for layer in (
                block.self_attn.qkv_proj, block.self_attn.o_proj, block.up_proj, block.down_proj)):
            raise ValueError("BitNet benchmark requires unconverted Sprint decoder linears")
        if not all(isinstance(norm, nn.RMSNorm) for norm in (
                block.input_layernorm, block.post_attention_layernorm)):
            raise ValueError("BitNet benchmark requires Sprint's original outer RMSNorms")

    original_parameters = sum(parameter.numel() for parameter in model.parameters())
    original_ids = {id(parameter) for parameter in model.parameters()}
    added_norm_parameters = 0
    converted = []
    for index, block in enumerate(model.layers):
        block.self_attn.qkv_proj = BitNetQATLinear(
            block.self_attn.qkv_proj, norm=block.input_layernorm, eps=model.config.rms_norm_eps)
        block.self_attn.o_proj = BitNetQATLinear(block.self_attn.o_proj, eps=model.config.rms_norm_eps)
        block.up_proj = BitNetQATLinear(
            block.up_proj, norm=block.post_attention_layernorm, eps=model.config.rms_norm_eps)
        block.down_proj = BitNetQATLinear(block.down_proj, eps=model.config.rms_norm_eps)
        block.input_layernorm = nn.Identity()
        block.post_attention_layernorm = nn.Identity()
        added_norm_parameters += block.self_attn.o_proj.in_features + block.down_proj.in_features
        converted.extend(f"layers.{index}.{name}" for name in (
            "self_attn.qkv_proj", "self_attn.o_proj", "up_proj", "down_proj"))

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if (not original_ids.issubset({id(parameter) for parameter in model.parameters()})
            or total_parameters != original_parameters + added_norm_parameters):
        raise RuntimeError("BitNet conversion failed to preserve the original parameters")
    metadata = {
        "method": "sprint_shape_bitnet_b158_qat",
        "training_kernel": "dense_floating_point_gemm",
        "native_low_bit_gemm": False,
        "benchmark_dtype": "bfloat16",
        "latent_dtype_policy": "preserve; benchmark engine selects BF16",
        "weight_quantization": "per_tensor_absmean_ternary_ste",
        "activation_quantization": "per_token_absmax_int8_ste",
        "quantization_reduction_dtype": "float32",
        "normalization": "internal RMSNorm in every converted projection; outer block norms removed",
        "original_parameters": original_parameters,
        "total_parameters": total_parameters,
        "added_norm_parameters": added_norm_parameters,
        "norm_layers_moved": 2 * len(model.layers),
        "norm_layers_added": 2 * len(model.layers),
        "converted_linear_count": len(converted),
        "converted_modules": converted,
        "packed_qkv_scale": "shared per matrix",
        "packed_gate_up_scale": "shared per matrix" if model.config.mlp_type == "swiglu" else None,
        "preserved": ["original parameter objects", "embedding/head tie", "decoder dimensions", "activation function",
                      "attention implementation", "loss implementation"],
        "sources": [BITNET_TRAINING_SOURCE, BITNET_PAPER, BITNET_TRANSFORMERS_SOURCE],
    }
    model._bitnet_qat_benchmark_metadata = copy.deepcopy(metadata)
    return metadata
