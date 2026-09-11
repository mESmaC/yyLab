"""Benchmark-only BitNet QAT with native CUDA INT8 forward and BF16 backward.

Ternary codes occupy INT8 tensors; this is not packed 1.58-bit/ternary hardware.
Forward accumulates integer products in INT32, rescales in FP32, then rounds
once to BF16. It therefore differs slightly from the dense QAT reference,
which rounds each dequantized operand to BF16 before GEMM. Backward deliberately
uses those reference BF16 dequantized operands with identity STE gradients.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from .bitnet_benchmark import BitNetQATLinear, convert_sprint_to_bitnet_qat


def native_bitnet_int8_prerequisites(device: str | torch.device) -> tuple[bool, str]:
    """Cheap admission only; the benchmark must validate real kernel execution."""
    device = torch.device(device)
    if device.type != "cuda":
        return False, "native BitNet INT8 forward requires CUDA"
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    if not hasattr(torch, "_int_mm") or not torch._C._dispatch_has_kernel_for_dispatch_key("aten::_int_mm", "CUDA"):
        return False, "this PyTorch build has no registered CUDA INT8 matmul"
    return True, "CUDA torch._int_mm is registered; real execution still requires benchmark validation"


def _weight_codes(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = weight.float()
    inverse_scale = values.abs().mean().clamp_min(1e-5).reciprocal()
    return (values * inverse_scale).round().clamp(-1, 1).to(torch.int8), inverse_scale


def _activation_codes(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.float()
    inverse_scale = 127.0 / values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    return (values * inverse_scale).round().clamp(-128, 127).to(torch.int8), inverse_scale


@torch.library.custom_op("yylab::bitnet_native_int8_mm", mutates_args=())
def _native_int8_mm(left: torch.Tensor, right_rows: torch.Tensor) -> torch.Tensor:
    if not left.is_cuda or not right_rows.is_cuda:
        raise RuntimeError("BitNet INT8 forward requires native CUDA operands; no dense fallback exists")
    # Keep this exact INT8 operation opaque to graph rewrites. The right-hand
    # operand is a column-major view of contiguous INT8 weight rows.
    return torch._int_mm(left, right_rows.t())


@_native_int8_mm.register_fake
def _native_int8_mm_fake(left, right_rows):
    return left.new_empty((left.shape[0], right_rows.shape[0]), dtype=torch.int32)


def _pad_codes(values: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    if values.shape == (rows, columns):
        return values.contiguous()
    return F.pad(values, (0, columns - values.shape[1], 0, rows - values.shape[0]))


class _BitNetInt8Forward(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, inputs, weight, bias):
        rows = inputs.numel() // inputs.shape[-1]
        input_flat = inputs.reshape(rows, inputs.shape[-1])
        input_codes, input_inverse = _activation_codes(input_flat)
        weight_codes, weight_inverse = _weight_codes(weight)
        # These shadows match the reference BitNet STE operands exactly. They
        # are independent of the higher-precision integer-forward epilogue.
        quantized_input = (input_codes.float() / input_inverse).to(torch.bfloat16)
        quantized_weight = (weight_codes.float() / weight_inverse).to(torch.bfloat16)
        ctx.save_for_backward(quantized_input, quantized_weight)
        ctx.input_shape, ctx.has_bias = inputs.shape, bias is not None
        aligned_rows = max(32, (rows + 31) // 32 * 32)
        aligned_features = (weight.shape[1] + 31) // 32 * 32
        aligned_outputs = (weight.shape[0] + 31) // 32 * 32
        left = _pad_codes(input_codes, aligned_rows, aligned_features)
        right = _pad_codes(weight_codes, aligned_outputs, aligned_features)
        accumulated = _native_int8_mm(left, right)[:rows, :weight.shape[0]]
        output_scale = input_inverse.reciprocal() * weight_inverse.reciprocal()
        output = accumulated.float() * output_scale
        if bias is not None:
            output = output + bias.float()
        return output.to(torch.bfloat16).reshape(*inputs.shape[:-1], weight.shape[0])

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        quantized_input, quantized_weight = ctx.saved_tensors
        gradient = grad_output.reshape(-1, grad_output.shape[-1]).to(torch.bfloat16)
        grad_input = gradient.mm(quantized_weight).reshape(ctx.input_shape)
        grad_weight = gradient.t().mm(quantized_input)
        grad_bias = gradient.sum(0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias


def bitnet_int8_linear(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """W1.58A8 integer forward with reference dense-BF16 STE backward."""
    if inputs.ndim < 1 or weight.ndim != 2 or inputs.shape[-1] != weight.shape[1]:
        raise ValueError("BitNet INT8 expects inputs[..., K] and weight[N, K]")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0:
        raise ValueError("BitNet INT8 requires positive input and output feature counts")
    if weight.shape[1] > (2**31 - 1) // 128:
        raise ValueError("BitNet INT8 contraction is too large for safe INT32 ternary accumulation")
    if inputs.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("BitNet INT8 benchmark requires BF16 activations and latent weights")
    if inputs.device != weight.device or not inputs.is_cuda:
        raise RuntimeError("BitNet INT8 forward requires activations and weights on the same CUDA device")
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.device != weight.device
                             or bias.dtype != torch.bfloat16):
        raise ValueError("BitNet INT8 bias must match the BF16 output features and CUDA device")
    return _BitNetInt8Forward.apply(inputs, weight, bias)


class BitNetInt8ForwardLinear(BitNetQATLinear):
    def forward(self, inputs):
        return bitnet_int8_linear(self.norm(inputs), self.weight, self.bias)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, forward=native_int8, backward=bf16, packed_ternary=False")


def convert_bitnet_to_int8_forward(model: nn.Module) -> dict:
    """Convert an already placed BF16 Sprint/BitNet benchmark model in place."""
    if hasattr(model, "_bitnet_int8_benchmark_metadata"):
        return copy.deepcopy(model._bitnet_int8_benchmark_metadata)
    parameters = list(model.parameters())
    if not parameters or any(parameter.dtype != torch.bfloat16 for parameter in parameters):
        raise ValueError("BitNet INT8 conversion requires BF16 latent parameters")
    device = parameters[0].device
    if any(parameter.device != device for parameter in parameters):
        raise ValueError("place the BitNet INT8 benchmark model on one CUDA device")
    available, note = native_bitnet_int8_prerequisites(device)
    if not available:
        raise RuntimeError(note)
    metadata = convert_sprint_to_bitnet_qat(model)
    original_ids = {id(parameter) for parameter in model.parameters()}
    converted = 0
    for module in model.modules():
        if isinstance(module, BitNetQATLinear):
            module.__class__ = BitNetInt8ForwardLinear
            converted += 1
    if not converted:
        raise ValueError("BitNet INT8 requested but no converted decoder projections were found")
    if original_ids != {id(parameter) for parameter in model.parameters()}:
        raise RuntimeError("BitNet INT8 conversion changed the existing Parameter objects")
    metadata.update(
        method="sprint_shape_bitnet_b158_native_int8_forward",
        training_kernel="native_int8_forward_bf16_backward", native_low_bit_gemm=True,
        native_low_bit_backward=False, native_packed_ternary=False,
        forward_gemm="int8", backward_gemm="bf16", accumulation_dtype="int32",
        ternary_code_storage="int8 (one byte per code); no 1.58-bit packing",
        forward_rescaling="FP32 per-token activation scale and per-tensor weight scale, then BF16 output",
        numerical_difference="FP32 post-GEMM rescaling instead of separately BF16-dequantized forward operands",
        backward_recipe="dense BF16 GEMMs on saved reference-dequantized quantized activations/weights; identity STE",
        integer_code_alignment=32, scale_computation="original unpadded values",
        kernel_preflight=note, converted_linear_count=converted,
    )
    model._bitnet_int8_benchmark_metadata = copy.deepcopy(metadata)
    return metadata


__all__ = ["BitNetInt8ForwardLinear", "bitnet_int8_linear", "convert_bitnet_to_int8_forward",
           "native_bitnet_int8_prerequisites"]
