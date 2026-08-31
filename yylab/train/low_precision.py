from __future__ import annotations

import importlib.util
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

LOW_PRECISION_METADATA_KEY = "yylab_low_precision"
QAT_FORMATS = {"int4", "nvfp4", "ternary"}
SPARSITY_PATTERNS = {"none", "paired_4:8"}
NATIVE_NVFP4_BACKEND = "torchao_native_nvfp4_qat"
EMULATED_QAT_BACKEND = "yylab_ste_emulation"
COMPRESSED_SPARSE_NVFP4_BACKEND = "cusparselt_compressed_nvfp4"
DENSE_MASKED_SPARSITY_BACKEND = "yylab_paired_magnitude_ste"
SPARSITY_BACKEND_REQUESTS = {"auto", "compressed", "dense_masked"}
BACKWARD_PRECISIONS = {"bf16", "mxfp8"}


def _ste(original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    return original + (quantized.to(original.dtype) - original).detach()


def _grouped_view(value: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    group_size = max(1, int(group_size))
    padding = (-value.shape[-1]) % group_size
    padded = F.pad(value, (0, padding)) if padding else value
    return padded.reshape(*padded.shape[:-1], -1, group_size), padding


def _restore_grouped(value: torch.Tensor, original: torch.Tensor, padding: int) -> torch.Tensor:
    flat = value.reshape(*original.shape[:-1], -1)
    if padding:
        flat = flat[..., : original.shape[-1]]
    return flat.reshape_as(original)


def _canonical_sparsity(pattern: str | None) -> str:
    normalized = str(pattern or "none").strip().lower().replace(" ", "_")
    aliases = {
        "": "none",
        "dense": "none",
        "4:8": "paired_4:8",
        "paired-4:8": "paired_4:8",
        "paired_4_of_8": "paired_4:8",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SPARSITY_PATTERNS:
        raise ValueError(f"unknown structured sparsity pattern: {pattern}")
    return normalized


def _canonical_sparsity_backend(backend: str | None) -> str:
    normalized = str(backend or "auto").strip().lower().replace("-", "_")
    aliases = {
        "cusparselt": "compressed",
        "native": "compressed",
        "dense": "dense_masked",
        "masked": "dense_masked",
        "emulated": "dense_masked",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SPARSITY_BACKEND_REQUESTS:
        raise ValueError(f"unknown structured-sparsity backend: {backend}")
    return normalized


def structured_sparsity_mask(value: torch.Tensor, pattern: str = "paired_4:8") -> torch.Tensor:
    """Return a deterministic magnitude mask compatible with sparse FP4 MMA.

    NVFP4 sparse Tensor Cores use paired 4:8 sparsity: each consecutive
    eight-value K group is split into four pairs, and the two strongest pairs
    are retained in full. This is the FP4 hardware counterpart to 2:4 at 50%
    density; an arbitrary scalar 2:4 mask is not compatible with the kernel.
    """

    pattern = _canonical_sparsity(pattern)
    if pattern == "none":
        return torch.ones_like(value, dtype=torch.bool)
    if value.ndim < 1 or value.shape[-1] % 8:
        raise ValueError(
            "paired 4:8 sparsity requires the weight K dimension to be divisible by 8"
        )
    paired = value.detach().float().reshape(*value.shape[:-1], -1, 4, 2)
    scores = paired.square().sum(dim=-1)
    keep_indices = scores.topk(2, dim=-1, largest=True, sorted=False).indices
    keep_pairs = torch.zeros_like(scores, dtype=torch.bool)
    keep_pairs.scatter_(-1, keep_indices, True)
    return keep_pairs.unsqueeze(-1).expand_as(paired).reshape_as(value)


def fake_structured_sparsity(
    value: torch.Tensor,
    pattern: str = "paired_4:8",
) -> torch.Tensor:
    """Apply a hardware-compatible sparse forward with dense-shadow STE gradients."""

    mask = structured_sparsity_mask(value, pattern)
    return _ste(value, value * mask.to(value.dtype))


def fake_quant_int4(value: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    grouped, padding = _grouped_view(value.float(), group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
    quantized = (grouped / scale).round().clamp(-8, 7) * scale
    return _ste(value, _restore_grouped(quantized, value, padding))


def fake_quant_int8_activation(value: torch.Tensor) -> torch.Tensor:
    work = value.float()
    scale = work.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    quantized = (work / scale).round().clamp(-128, 127) * scale
    return _ste(value, quantized)


def fake_quant_nvfp4(value: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Simulate NVIDIA's two-level NVFP4 quantization exactly.

    Native NVFP4 first derives one FP32 scale for the complete tensor, then
    quantizes each 16-value block scale to E4M3 before rounding the values to
    E2M1. Keeping both scale levels here makes CPU and non-native execution
    match the packed TorchAO forward numerics; it still does not use FP4 GEMMs
    or reduce training-state RAM.
    """

    if int(group_size) != 16:
        raise ValueError("NVFP4 requires a group size of 16")
    if value.numel() == 0:
        return value

    # TorchAO's converter uses round-to-nearest-even bit conversion for E2M1.
    # Reuse those conversion primitives so emulation cannot drift at midpoint
    # ties while retaining yyLab's identity-gradient STE.
    from torchao.prototype.mx_formats.kernels import (
        f4_unpacked_to_f32,
        f32_to_f4_unpacked,
    )

    grouped, padding = _grouped_view(value.detach().float(), group_size)
    tensor_amax = grouped.abs().amax()
    tensor_scale = tensor_amax / (448.0 * 6.0)
    # TorchAO currently produces NaNs for an all-zero tensor because its
    # optional outer scale is zero. A positive neutral floor preserves the
    # exact zero representation and matches all nonzero TorchAO numerics.
    e4m3_tiny = torch.finfo(torch.float8_e4m3fn).tiny
    safe_min_scale = torch.finfo(torch.float32).tiny / e4m3_tiny
    tensor_scale = torch.where(
        tensor_amax > 0,
        tensor_scale.clamp_min(safe_min_scale),
        torch.ones_like(tensor_scale),
    )

    block_scale = grouped.abs().amax(dim=-1, keepdim=True) / 6.0
    block_scale_e4m3 = (
        (block_scale / tensor_scale)
        .clamp(min=e4m3_tiny, max=448.0)
        .to(torch.float8_e4m3fn)
        .float()
    )
    normalized = (
        grouped * ((1.0 / tensor_scale) / block_scale_e4m3)
    ).clamp(-6.0, 6.0)
    e2m1 = f4_unpacked_to_f32(f32_to_f4_unpacked(normalized))
    quantized = e2m1 * (tensor_scale * block_scale_e4m3)
    return _ste(value, _restore_grouped(quantized, value, padding))


def fake_quant_ternary(value: torch.Tensor) -> torch.Tensor:
    work = value.float()
    scale = work.abs().mean().clamp_min(1e-8)
    quantized = (work / scale).round().clamp(-1, 1) * scale
    return _ste(value, quantized)


class QATLinear(nn.Linear):
    """State-dict-compatible Linear with differentiable fake-quant numerics."""

    qat_format: str
    qat_group_size: int
    qat_activation: str
    qat_backend: str
    qat_recipe_backend: str
    qat_runtime_note: str
    qat_sparsity: str
    qat_sparsity_backend: str
    qat_backward_precision: str
    qat_backward_backend: str

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if (
            self.qat_format == "nvfp4"
            and self.qat_sparsity != "none"
            and self.qat_sparsity_backend == COMPRESSED_SPARSE_NVFP4_BACKEND
        ):
            from .sparse_nvfp4 import sparse_nvfp4_linear

            return sparse_nvfp4_linear(
                input,
                weight,
                self.bias,
                backward_precision=self.qat_backward_precision,
                backward_backend=self.qat_backward_backend,
            )
        if self.qat_sparsity != "none":
            weight = fake_structured_sparsity(weight, self.qat_sparsity)

        if (
            self.qat_format == "nvfp4"
            and self.qat_backend == NATIVE_NVFP4_BACKEND
            and input.is_cuda
            and weight.is_cuda
        ):
            return _native_nvfp4_linear(
                input,
                weight,
                self.bias,
                backward_precision=self.qat_backward_precision,
                backward_backend=self.qat_backward_backend,
            )
        if self.qat_format == "nvfp4" and self.qat_backward_precision == "mxfp8":
            return _emulated_nvfp4_linear(
                input,
                weight,
                self.bias,
                backward_backend=self.qat_backward_backend,
            )
        if self.qat_format == "int4":
            weight = fake_quant_int4(weight, self.qat_group_size)
        elif self.qat_format == "nvfp4":
            weight = fake_quant_nvfp4(weight, self.qat_group_size)
        elif self.qat_format == "ternary":
            weight = fake_quant_ternary(weight)
        else:
            raise RuntimeError(f"unknown QAT format: {self.qat_format}")

        activation = input
        if self.qat_activation == "int8":
            activation = fake_quant_int8_activation(input)
        elif self.qat_activation == "nvfp4":
            activation = fake_quant_nvfp4(input, 16)
        return F.linear(activation, weight, self.bias)


class _EmulatedNVFP4Linear(torch.autograd.Function):
    """Format-faithful NVFP4 forward with explicit MXFP8-emulated autograd."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        backward_backend: str,
    ) -> torch.Tensor:
        quantized_input = fake_quant_nvfp4(input, 16)
        quantized_weight = fake_quant_nvfp4(weight, 16)
        ctx.save_for_backward(quantized_input, quantized_weight)
        ctx.input_shape = input.shape
        ctx.has_bias = bias is not None
        ctx.backward_backend = backward_backend
        return F.linear(quantized_input, quantized_weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        quantized_input, quantized_weight = ctx.saved_tensors
        from .mxfp8 import mxfp8_linear_backward

        grad_input, grad_weight = mxfp8_linear_backward(
            quantized_input,
            quantized_weight,
            grad_output,
            backend=ctx.backward_backend,
        )
        grad_bias = (
            grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)
            if ctx.has_bias
            else None
        )
        return grad_input.reshape(ctx.input_shape), grad_weight, grad_bias, None


def _emulated_nvfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    backward_backend: str,
) -> torch.Tensor:
    return _EmulatedNVFP4Linear.apply(input, weight, bias, backward_backend)


class _NativeNVFP4Linear(torch.autograd.Function):
    """Native FP4 forward GEMM with selectable BF16 or MXFP8 backward.

    TorchAO owns the packed E2M1/E4M3 representation and dispatches PyTorch's
    SM120 block-scaled GEMM. yyLab keeps BF16 shadow parameters for backward
    and computes bias gradients (the current prototype helper omits them).
    Structured masks are applied before this function so STE can carry the
    returned dense weight gradient back to the shadow parameter. The optimizer
    may independently keep an FP32 or potentiated-FP8 CPU master.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        backward_precision: str,
        backward_backend: str,
    ) -> torch.Tensor:
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            NVFP4Tensor,
            _addmm_nvfp4_dispatch,
            per_tensor_amax_to_scale,
        )

        original_shape = input.shape
        flat_input = input.reshape(-1, input.shape[-1])
        input_scale = per_tensor_amax_to_scale(flat_input.detach().abs().amax())
        weight_scale = per_tensor_amax_to_scale(weight.detach().abs().amax())
        packed_input = NVFP4Tensor.to_nvfp4(flat_input, per_tensor_scale=input_scale)
        packed_weight = NVFP4Tensor.to_nvfp4(weight, per_tensor_scale=weight_scale)
        ctx.save_for_backward(packed_input, packed_weight)
        ctx.input_shape = original_shape
        ctx.has_bias = bias is not None
        ctx.backward_precision = backward_precision
        ctx.backward_backend = backward_backend
        output = _addmm_nvfp4_dispatch(packed_input, packed_weight.t(), None, bias)
        return output.reshape(*original_shape[:-1], output.shape[-1])

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        packed_input, packed_weight = ctx.saved_tensors
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        input_hp = packed_input.dequantize(packed_input.orig_dtype)
        weight_hp = packed_weight.dequantize(packed_weight.orig_dtype)
        if ctx.backward_precision == "mxfp8":
            from .mxfp8 import mxfp8_linear_backward

            grad_input, grad_weight = mxfp8_linear_backward(
                input_hp.reshape(ctx.input_shape),
                weight_hp,
                grad_output,
                backend=ctx.backward_backend,
            )
        else:
            grad_input = grad_flat.mm(weight_hp).reshape(ctx.input_shape)
            grad_weight = grad_flat.t().mm(input_hp)
        grad_bias = grad_flat.sum(dim=0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None, None


def _native_nvfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    backward_precision: str = "bf16",
    backward_backend: str = "emulated",
) -> torch.Tensor:
    if not input.is_cuda or not weight.is_cuda:
        raise RuntimeError("native NVFP4 QAT requires CUDA tensors")
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise RuntimeError("native NVFP4 QAT currently requires BF16 activations and master weights")
    if backward_precision not in BACKWARD_PRECISIONS:
        raise ValueError(f"unknown low-bit backward precision: {backward_precision}")
    return _NativeNVFP4Linear.apply(
        input,
        weight,
        bias,
        backward_precision,
        backward_backend,
    )


def qat_metadata(
    qat_format: str,
    *,
    group_size: int = 32,
    activation: str = "none",
    parameter_dtype: str = "bf16",
    sparsity: str = "none",
    backend: str = "auto",
    sparsity_backend: str = "auto",
    backward_precision: str = "bf16",
    backward_backend: str = "auto",
) -> dict[str, Any] | None:
    sparsity = _canonical_sparsity(sparsity)
    requested_sparsity_backend = _canonical_sparsity_backend(sparsity_backend)
    backward_precision = str(backward_precision or "bf16").strip().lower()
    requested_backward_backend = str(backward_backend or "auto").strip().lower()
    if backward_precision not in BACKWARD_PRECISIONS:
        raise ValueError(f"unknown low-bit backward precision: {backward_precision}")
    if requested_backward_backend not in {"auto", "native", "emulated"}:
        raise ValueError(f"unknown MXFP8 backward backend: {backward_backend}")
    if qat_format == "none":
        if sparsity != "none":
            raise ValueError("structured sparsity requires a QAT format")
        if str(backend or "auto").strip().lower() == "native":
            raise ValueError("native low-bit backend requires a QAT format")
        if backward_precision != "bf16":
            raise ValueError("MXFP8 backward requires NVFP4 QAT")
        return None
    if qat_format not in QAT_FORMATS:
        raise ValueError(f"unknown QAT format: {qat_format}")
    if sparsity != "none" and qat_format != "nvfp4":
        raise ValueError("paired 4:8 sparsity is currently supported only with NVFP4 QAT")
    requested_backend = str(backend or "auto").strip().lower()
    if requested_backend not in {"auto", "native", "emulated"}:
        raise ValueError(f"unknown QAT backend: {backend}")
    native = False
    native_note = ""
    if qat_format == "nvfp4":
        group_size = 16
        activation = "nvfp4"
        native_available, native_note = native_nvfp4_status()
        if requested_backend == "native" and parameter_dtype != "bf16":
            raise ValueError("native NVFP4 QAT requires BF16 parameter dtype")
        if requested_backend == "native" and not native_available:
            raise RuntimeError(f"native NVFP4 QAT requested but unavailable: {native_note}")
        native = parameter_dtype == "bf16" and (
            requested_backend == "native" or (requested_backend == "auto" and native_available)
        )
    elif qat_format == "ternary":
        activation = "int8"
    elif requested_backend == "native":
        raise ValueError("the native low-bit QAT backend currently supports NVFP4 only")
    if activation not in {"none", "int8", "nvfp4"}:
        raise ValueError(f"unknown activation QAT format: {activation}")
    if backward_precision == "mxfp8" and qat_format != "nvfp4":
        raise ValueError("MXFP8 backward requires NVFP4 QAT")
    if (
        backward_precision == "mxfp8"
        and requested_backward_backend == "native"
        and parameter_dtype != "bf16"
    ):
        raise ValueError("native MXFP8 backward requires BF16 shadow parameters")

    resolved_sparsity_backend = "none"
    native_sparse = False
    sparse_note = "dense weights"
    if sparsity != "none":
        if requested_sparsity_backend == "compressed" and not native:
            raise ValueError(
                "compressed sparse NVFP4 requires the native BF16 NVFP4 backend"
            )
        sparse_available = False
        if native and requested_sparsity_backend != "dense_masked":
            from .sparse_nvfp4 import sparse_nvfp4_status

            sparse_available, sparse_note = sparse_nvfp4_status()
        if requested_sparsity_backend == "compressed" and not sparse_available:
            raise RuntimeError(
                f"compressed sparse NVFP4 requested but unavailable: {sparse_note}"
            )
        native_sparse = native and sparse_available and requested_sparsity_backend in {
            "auto",
            "compressed",
        }
        resolved_sparsity_backend = (
            COMPRESSED_SPARSE_NVFP4_BACKEND
            if native_sparse
            else DENSE_MASKED_SPARSITY_BACKEND
        )

    resolved_backward_backend = "none"
    native_backward = False
    backward_note = "BF16 dense backward"
    if backward_precision == "mxfp8":
        from .mxfp8 import (
            EMULATED_MXFP8_BACKEND,
            NATIVE_MXFP8_BACKEND,
            native_mxfp8_status,
        )

        if requested_backward_backend == "emulated" or parameter_dtype != "bf16":
            resolved_backward_backend = EMULATED_MXFP8_BACKEND
            backward_note = (
                "portable MXFP8 emulation"
                if parameter_dtype == "bf16"
                else f"MXFP8 emulation for {parameter_dtype} shadow parameters"
            )
        else:
            native_backward, backward_note = native_mxfp8_status()
            if requested_backward_backend == "native" and not native_backward:
                raise RuntimeError(
                    f"native MXFP8 backward requested but unavailable: {backward_note}"
                )
            resolved_backward_backend = (
                NATIVE_MXFP8_BACKEND
                if native_backward
                else EMULATED_MXFP8_BACKEND
            )
    return {
        "version": 3,
        "mode": "qat",
        "format": qat_format,
        "backend": NATIVE_NVFP4_BACKEND if native else EMULATED_QAT_BACKEND,
        "backend_request": requested_backend,
        "group_size": int(group_size),
        "activation": activation,
        "parameter_dtype": parameter_dtype,
        "native_low_bit_gemm": native,
        "native_low_bit_note": native_note,
        "backward_precision": backward_precision,
        "backward_backend": resolved_backward_backend,
        "backward_backend_request": requested_backward_backend,
        "native_low_bit_backward": native_backward,
        "backward_note": backward_note,
        "training_state_is_low_bit": False,
        "sparsity": sparsity,
        "sparsity_density": 0.5 if sparsity != "none" else 1.0,
        "sparsity_backend": resolved_sparsity_backend,
        "sparsity_backend_request": requested_sparsity_backend,
        "sparsity_note": sparse_note,
        "native_sparse_gemm": native_sparse,
        "compressed_sparse_checkpoint": False,
    }


def _qat_linear_targets(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    targets: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, QATLinear):
            continue
        # Embeddings and the output projection remain high precision, matching
        # the common low-bit pretraining recipes and Liger's fused CE contract.
        if name == "lm_head" or name.endswith(".lm_head"):
            continue
        targets.append((name, module))
    return targets


def _resolve_qat_runtime_backend(
    targets: list[tuple[str, nn.Linear]],
    *,
    qat_format: str,
    recipe_backend: str,
    allow_runtime_fallback: bool,
) -> tuple[str, str]:
    if recipe_backend == EMULATED_QAT_BACKEND:
        return recipe_backend, "saved recipe uses emulated QAT"
    if recipe_backend != NATIVE_NVFP4_BACKEND:
        raise ValueError(f"unknown saved QAT backend: {recipe_backend}")
    if qat_format != "nvfp4":
        raise ValueError("the native low-bit QAT backend supports NVFP4 only")

    available, note = native_nvfp4_status()
    incompatibility = None
    if available:
        non_cuda = [name or "<root>" for name, module in targets if not module.weight.is_cuda]
        non_bf16 = [
            name or "<root>"
            for name, module in targets
            if module.weight.dtype != torch.bfloat16
        ]
        if non_cuda:
            incompatibility = f"QAT weights are not on CUDA ({non_cuda[0]})"
        elif non_bf16:
            incompatibility = f"QAT weights are not BF16 ({non_bf16[0]})"
    else:
        incompatibility = note

    if incompatibility is None:
        return NATIVE_NVFP4_BACKEND, note
    if allow_runtime_fallback:
        return EMULATED_QAT_BACKEND, f"native recipe emulated at runtime: {incompatibility}"
    raise RuntimeError(f"native NVFP4 QAT runtime is unavailable: {incompatibility}")


def _resolve_sparsity_runtime_backend(
    targets: list[tuple[str, nn.Linear]],
    *,
    recipe_backend: str,
    allow_runtime_fallback: bool,
) -> tuple[str, str]:
    if recipe_backend in {"none", DENSE_MASKED_SPARSITY_BACKEND}:
        note = "dense weights" if recipe_backend == "none" else "dense masked execution"
        return recipe_backend, note
    if recipe_backend != COMPRESSED_SPARSE_NVFP4_BACKEND:
        raise ValueError(f"unknown saved sparsity backend: {recipe_backend}")

    non_cuda = [name or "<root>" for name, module in targets if not module.weight.is_cuda]
    non_bf16 = [
        name or "<root>"
        for name, module in targets
        if module.weight.dtype != torch.bfloat16
    ]
    if non_cuda:
        incompatibility = f"QAT weights are not on CUDA ({non_cuda[0]})"
    elif non_bf16:
        incompatibility = f"QAT weights are not BF16 ({non_bf16[0]})"
    else:
        from .sparse_nvfp4 import sparse_nvfp4_status

        available, note = sparse_nvfp4_status()
        incompatibility = None if available else note
        if available:
            return COMPRESSED_SPARSE_NVFP4_BACKEND, note
    if allow_runtime_fallback:
        return (
            DENSE_MASKED_SPARSITY_BACKEND,
            f"compressed sparse recipe uses dense masked runtime: {incompatibility}",
        )
    raise RuntimeError(f"compressed sparse NVFP4 runtime is unavailable: {incompatibility}")


def _resolve_backward_runtime_backend(
    targets: list[tuple[str, nn.Linear]],
    *,
    precision: str,
    recipe_backend: str,
    allow_runtime_fallback: bool,
) -> tuple[str, str]:
    if precision == "bf16":
        return "none", "BF16 dense backward"
    if precision != "mxfp8":
        raise ValueError(f"unknown saved backward precision: {precision}")

    from .mxfp8 import EMULATED_MXFP8_BACKEND, NATIVE_MXFP8_BACKEND

    if recipe_backend == EMULATED_MXFP8_BACKEND:
        return "emulated", "portable MXFP8 emulation"
    if recipe_backend != NATIVE_MXFP8_BACKEND:
        raise ValueError(f"unknown saved MXFP8 backend: {recipe_backend}")

    non_cuda = [name or "<root>" for name, module in targets if not module.weight.is_cuda]
    non_bf16 = [
        name or "<root>"
        for name, module in targets
        if module.weight.dtype != torch.bfloat16
    ]
    if non_cuda:
        incompatibility = f"QAT weights are not on CUDA ({non_cuda[0]})"
    elif non_bf16:
        incompatibility = f"QAT weights are not BF16 ({non_bf16[0]})"
    else:
        from .mxfp8 import native_mxfp8_status

        available, note = native_mxfp8_status()
        incompatibility = None if available else note
        if available:
            return "native", note
    if allow_runtime_fallback:
        return "emulated", f"native MXFP8 recipe emulated at runtime: {incompatibility}"
    raise RuntimeError(f"native MXFP8 backward runtime is unavailable: {incompatibility}")


def apply_qat(
    model: nn.Module,
    metadata: dict[str, Any],
    *,
    allow_runtime_fallback: bool = False,
) -> int:
    qat_format = str(metadata.get("format") or "")
    if qat_format not in QAT_FORMATS:
        raise ValueError(f"unknown saved QAT format: {qat_format}")
    group_size = int(metadata.get("group_size") or (16 if qat_format == "nvfp4" else 32))
    activation = str(metadata.get("activation") or "none")
    recipe_backend = str(metadata.get("backend") or EMULATED_QAT_BACKEND)
    sparsity = _canonical_sparsity(str(metadata.get("sparsity") or "none"))
    recipe_sparsity_backend = str(
        metadata.get("sparsity_backend")
        or (DENSE_MASKED_SPARSITY_BACKEND if sparsity != "none" else "none")
    )
    backward_precision = str(metadata.get("backward_precision") or "bf16")
    recipe_backward_backend = str(metadata.get("backward_backend") or "none")
    targets = _qat_linear_targets(model)
    if not targets:
        raise RuntimeError("QAT requested but no transformer Linear modules were found")
    runtime_fallback = (
        allow_runtime_fallback or str(metadata.get("backend_request") or "") == "auto"
    )
    runtime_backend, runtime_note = _resolve_qat_runtime_backend(
        targets,
        qat_format=qat_format,
        recipe_backend=recipe_backend,
        allow_runtime_fallback=runtime_fallback,
    )
    runtime_sparsity_backend, runtime_sparsity_note = _resolve_sparsity_runtime_backend(
        targets,
        recipe_backend=recipe_sparsity_backend,
        allow_runtime_fallback=(
            allow_runtime_fallback
            or str(metadata.get("sparsity_backend_request") or "") == "auto"
        ),
    )
    runtime_backward_backend, runtime_backward_note = _resolve_backward_runtime_backend(
        targets,
        precision=backward_precision,
        recipe_backend=recipe_backward_backend,
        allow_runtime_fallback=(
            allow_runtime_fallback
            or str(metadata.get("backward_backend_request") or "") == "auto"
        ),
    )
    for _, module in targets:
        module.__class__ = QATLinear
        module.qat_format = qat_format
        module.qat_group_size = group_size
        module.qat_activation = activation
        module.qat_recipe_backend = recipe_backend
        module.qat_backend = runtime_backend
        module.qat_runtime_note = runtime_note
        module.qat_sparsity = sparsity
        module.qat_sparsity_backend = runtime_sparsity_backend
        module.qat_sparsity_runtime_note = runtime_sparsity_note
        module.qat_backward_precision = backward_precision
        module.qat_backward_backend = runtime_backward_backend
        module.qat_backward_runtime_note = runtime_backward_note
    config = getattr(model, "config", None)
    if config is not None:
        # Persist the recipe, not this process's device/dtype resolution. Dense
        # shadow checkpoints can therefore move between native and emulated
        # runtimes without rewriting their training provenance.
        setattr(config, LOW_PRECISION_METADATA_KEY, dict(metadata))
    return len(targets)


def apply_saved_qat(model: nn.Module) -> tuple[int, dict[str, Any] | None]:
    metadata = getattr(getattr(model, "config", None), LOW_PRECISION_METADATA_KEY, None)
    if not metadata:
        return 0, None
    recipe_metadata = dict(metadata)
    count = apply_qat(model, recipe_metadata, allow_runtime_fallback=True)
    runtime_backends = {
        module.qat_backend for module in model.modules() if isinstance(module, QATLinear)
    }
    runtime_sparsity_backends = {
        module.qat_sparsity_backend
        for module in model.modules()
        if isinstance(module, QATLinear)
    }
    runtime_backward_backends = {
        module.qat_backward_backend
        for module in model.modules()
        if isinstance(module, QATLinear)
    }
    runtime_backend = runtime_backends.pop() if len(runtime_backends) == 1 else "mixed"
    runtime_sparsity_backend = (
        runtime_sparsity_backends.pop() if len(runtime_sparsity_backends) == 1 else "mixed"
    )
    runtime_backward_backend = (
        runtime_backward_backends.pop() if len(runtime_backward_backends) == 1 else "mixed"
    )
    runtime_metadata = dict(recipe_metadata)
    runtime_metadata["runtime_backend"] = runtime_backend
    runtime_metadata["runtime_sparsity_backend"] = runtime_sparsity_backend
    runtime_metadata["runtime_backward_backend"] = runtime_backward_backend
    runtime_metadata["runtime_native_low_bit_gemm"] = runtime_backend == NATIVE_NVFP4_BACKEND
    runtime_metadata["runtime_native_sparse_gemm"] = (
        runtime_sparsity_backend == COMPRESSED_SPARSE_NVFP4_BACKEND
    )
    runtime_metadata["runtime_native_low_bit_backward"] = runtime_backward_backend == "native"
    # Existing callers use this field to describe current execution. The saved
    # config retains the original value because runtime_metadata is ephemeral.
    runtime_metadata["native_low_bit_gemm"] = runtime_backend == NATIVE_NVFP4_BACKEND
    runtime_metadata["native_sparse_gemm"] = (
        runtime_sparsity_backend == COMPRESSED_SPARSE_NVFP4_BACKEND
    )
    runtime_metadata["native_low_bit_backward"] = runtime_backward_backend == "native"
    return count, runtime_metadata


def native_nvfp4_status() -> tuple[bool, str]:
    """Report yyLab's native dense NVFP4 QAT lane, independently of hardware."""

    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    capability = torch.cuda.get_device_capability()
    if capability[0] < 10:
        return False, f"native NVFP4 requires Blackwell or newer, found SM{capability[0]}{capability[1]}"
    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch.ops.aten, "_scaled_mm"):
        return False, "this PyTorch build does not expose packed FP4 scaled GEMM"
    if importlib.util.find_spec("torchao") is None:
        return False, "TorchAO is not installed"
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor  # noqa: F401
    except (ImportError, OSError) as exc:
        return False, f"TorchAO NVFP4 backend failed to import: {exc}"
    return (
        True,
        f"SM{capability[0]}{capability[1]} native block-scaled FP4 via TorchAO; "
        "BF16 shadow parameters with selectable BF16 or MXFP8 backward",
    )


def native_sparse_nvfp4_status(*, smoke: bool = True) -> tuple[bool, str]:
    """Report sparse FP4 via cheap capability or real worker smoke status."""

    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    from .sparse_nvfp4 import sparse_nvfp4_capability_status, sparse_nvfp4_status

    return sparse_nvfp4_status() if smoke else sparse_nvfp4_capability_status()


def qat_state_multiplier(qat_format: str) -> float:
    """Training-state factor: fake quantization keeps the dense shadow weights."""

    if qat_format not in {"none", *QAT_FORMATS}:
        raise ValueError(f"unknown QAT format: {qat_format}")
    return 1.0
