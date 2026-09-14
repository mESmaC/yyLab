"""Explicit precision adapters for disposable Sprint training experiments.

These adapters are deliberately separate from saved Sprint training recipes.
All modes retain floating-point trainable parameters and optimizer state.
NVFP4/MXFP8 require native kernels; BitNet is labeled dense-GEMM QAT.
"""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_PRECISION_MODES = ("bf16", "nvfp4_bf16", "nvfp4_mxfp8", "mxfp8", "bitnet_b158")
PRECISION_MODES = (*DEFAULT_PRECISION_MODES, "nvfp4_all", "nvfp4_all_portable", "nvfp4_all_portable_sr4",
                   "mxfp8_fast", "nvfp4_mxfp8_fast", "bitnet_b158_int8")


class _MXFP8LinearFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, inputs, weight, bias):
        from .mxfp8 import mxfp8_mm_nt

        ctx.save_for_backward(inputs, weight)
        ctx.has_bias = bias is not None
        output = mxfp8_mm_nt(inputs.reshape(-1, inputs.shape[-1]), weight, backend="native")
        if bias is not None:
            output = output + bias
        return output.reshape(*inputs.shape[:-1], weight.shape[0])

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        from .mxfp8 import mxfp8_linear_backward

        inputs, weight = ctx.saved_tensors
        grad_input, grad_weight = mxfp8_linear_backward(inputs, weight, grad_output, backend="native")
        grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias


class MXFP8Linear(nn.Linear):
    """Three native MXFP8 GEMMs, with fresh block scaling for each GEMM."""

    def forward(self, inputs):
        return _MXFP8LinearFunction.apply(inputs, self.weight, self.bias)


class _FastLowPrecisionFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, inputs, weight, bias, forward_format):
        from .mxfp8_fast import mxfp8_mm_nt_fast

        ctx.save_for_backward(inputs, weight)
        ctx.has_bias = bias is not None
        if forward_format == "nvfp4":
            from torchao.prototype.mx_formats.nvfp4_tensor import (
                NVFP4Tensor,
                _addmm_nvfp4_dispatch,
                per_tensor_amax_to_scale,
            )

            flat = inputs.reshape(-1, inputs.shape[-1])
            input_scale = per_tensor_amax_to_scale(flat.detach().abs().amax())
            weight_scale = per_tensor_amax_to_scale(weight.detach().abs().amax())
            packed_input = NVFP4Tensor.to_nvfp4(flat, per_tensor_scale=input_scale)
            packed_weight = NVFP4Tensor.to_nvfp4(weight, per_tensor_scale=weight_scale)
            output = _addmm_nvfp4_dispatch(packed_input, packed_weight.t(), None, bias)
        else:
            output = mxfp8_mm_nt_fast(inputs.reshape(-1, inputs.shape[-1]), weight, backend="native")
            if bias is not None:
                output = output + bias
        return output.reshape(*inputs.shape[:-1], weight.shape[0])

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        from .mxfp8_fast import mxfp8_linear_backward_fast

        inputs, weight = ctx.saved_tensors
        grad_input, grad_weight = mxfp8_linear_backward_fast(inputs, weight, grad_output, backend="native")
        grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None


class FastLowPrecisionLinear(nn.Linear):
    forward_format = "mxfp8"

    def forward(self, inputs):
        return _FastLowPrecisionFunction.apply(inputs, self.weight, self.bias, self.forward_format)


def apply_sprint_precision(model: nn.Module, precision: str, device) -> dict:
    """Mutate a disposable, already placed BF16 model before optimizer creation."""
    if precision not in PRECISION_MODES:
        raise ValueError(f"unknown Sprint benchmark precision: {precision}")
    device = torch.device(device)
    parameters = list(model.parameters())
    if not parameters or any(p.dtype != torch.bfloat16 for p in parameters):
        raise ValueError("precision comparison requires BF16 parameters in every mode")
    if any(p.device.type != device.type or device.index is not None and p.device.index != device.index
           for p in parameters):
        raise ValueError("place all model parameters on the benchmark device before conversion")
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and name != "lm_head" and not name.endswith(".lm_head")]
    before = sum(p.numel() for p in parameters)
    metadata = {
        "precision": precision, "experimental": True,
        "scope": "decoder projections; embedding, tied output head, attention and loss retain BF16/FP32 math",
        "parameter_dtype": "bfloat16", "training_state_is_low_bit": False,
        "forward_gemm": "bf16", "backward_gemm": "bf16",
        "native_low_bit_gemm": False, "native_low_bit_backward": False,
        "parameters_before": before, "converted_linear_count": 0,
    }
    if precision in {"mxfp8_fast", "nvfp4_mxfp8_fast"}:
        base = precision.removesuffix("_fast")
        metadata = apply_sprint_precision(model, base, device)
        # Native prerequisites above are identical to the corresponding base
        # recipe. Only cast/padding preparation changes; no dense fallback.
        for _, module in targets:
            module.__class__ = FastLowPrecisionLinear
            module.forward_format = "nvfp4" if base.startswith("nvfp4") else "mxfp8"
        metadata.update(precision=precision, cast_implementation="fused BF16 transpose and MXFP8 column cast")
    elif precision in {"nvfp4_all", "nvfp4_all_portable", "nvfp4_all_portable_sr4"}:
        if device.type != "cuda":
            raise RuntimeError("native all-GEMM NVFP4 execution requires CUDA")
        from .sprint_nvfp4_training import apply_torchao_nvfp4_training

        rounding = {"nvfp4_all": "hardware", "nvfp4_all_portable": "portable",
                    "nvfp4_all_portable_sr4": "portable_sr4"}[precision]
        metadata.update(apply_torchao_nvfp4_training(
            model, stochastic_rounding=rounding, pad_to_alignment=rounding != "hardware"))
        metadata["precision"] = precision
    elif precision.startswith("nvfp4_"):
        from .low_precision import apply_qat, qat_metadata

        recipe = qat_metadata("nvfp4", device=device, backend="native",
                              backward_precision=precision.removeprefix("nvfp4_"))
        metadata["converted_linear_count"] = apply_qat(model, recipe)
        metadata.update(forward_gemm="nvfp4", backward_gemm=recipe["backward_precision"],
                        native_low_bit_gemm=True, native_low_bit_backward=recipe["native_low_bit_backward"],
                        provider=recipe["backend"], provider_recipe=recipe,
                        gradient_recipe="BF16 shadows requantized to selected backward format; not FP4-reconstructed backward")
    elif precision == "mxfp8":
        from .mxfp8 import NATIVE_MXFP8_BACKEND, native_mxfp8_status

        available, note = native_mxfp8_status(device)
        if not available:
            raise RuntimeError(f"native MXFP8 training is unavailable: {note}")
        if not targets:
            raise ValueError("MXFP8 requested but no decoder linear modules were found")
        for _, module in targets:
            module.__class__ = MXFP8Linear
        metadata.update(converted_linear_count=len(targets), forward_gemm="mxfp8", backward_gemm="mxfp8",
                        native_low_bit_gemm=True, native_low_bit_backward=True,
                        provider=NATIVE_MXFP8_BACKEND, provider_note=note,
                        gradient_recipe="BF16 shadows independently block-quantized for each of three native GEMMs")
    elif precision == "bitnet_b158_int8":
        from .bitnet_int8_benchmark import convert_bitnet_to_int8_forward

        metadata["bitnet_recipe"] = convert_bitnet_to_int8_forward(model)
        metadata.update(converted_linear_count=len(targets), forward_gemm="int8",
                        native_low_bit_gemm=True, provider="bitnet_b158_native_int8_forward_bf16_backward",
                        native_packed_ternary=False,
                        gradient_recipe="BF16 dequantized operands and identity STE; INT32 forward sums rescaled in FP32")
    elif precision == "bitnet_b158":
        from .bitnet_benchmark import convert_sprint_to_bitnet_qat

        metadata["bitnet_recipe"] = convert_sprint_to_bitnet_qat(model)
        metadata["converted_linear_count"] = len(targets)
        metadata.update(provider="bitnet_b158_ste_dense_bf16",
                        gradient_recipe="straight-through estimators through ternary weights and per-token INT8 activations",
                        quantized_forward_values="W1.58A8 represented as BF16 for dense GEMMs")
    metadata["parameters_after"] = sum(p.numel() for p in model.parameters())
    return metadata
