"""Experimental native MXFP8 preparation with fused column casts in backward.

The candidate skips no-op padding and fuses BF16 transpose with columnwise
block-32 quantization through TorchAO's Triton dim1 cast. Quantization remains
E4M3/E8M0 with RCEIL scaling, independently recomputed along the contraction
dimension of each GEMM. Backward pads each original BF16 matrix once when
necessary, then reuses it across casts. There is no dense or emulated fallback.
Irregular forward shapes use yyLab's existing native implementation explicitly.

Provider implementation/reference:
https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/kernels.py
The dim1 provider returns a column-major payload in the input shape and scales
for its transposed rows; interpreting those scales as ordinary row scales of
the original matrix would silently change the quantization recipe.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from . import mxfp8 as native


@torch.library.custom_op("yylab::mxfp8_fast_rows", mutates_args=())
def _quantize_rows_raw(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0

    return triton_to_mxfp8_dim0(value, inner_block_size=32, scaling_mode="rceil")


@_quantize_rows_raw.register_fake
def _quantize_rows_fake(value):
    return (torch.empty_like(value, dtype=torch.float8_e4m3fn),
            value.new_empty((value.shape[0], value.shape[1] // 32), dtype=torch.float8_e8m0fnu))


@torch.library.custom_op("yylab::mxfp8_fast_transposed_rows", mutates_args=())
def _quantize_transposed_rows_raw(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim1

    column_major, scales = triton_to_mxfp8_dim1(value, inner_block_size=32, scaling_mode="rceil")
    # dim1 has already stored the FP8 payload in transposed, contiguous memory.
    # This t() is a view of the fresh FP8 output, not a BF16 transpose copy.
    # Scales already have [original columns, original rows / 32] orientation.
    return column_major.t(), scales


@_quantize_transposed_rows_raw.register_fake
def _quantize_transposed_rows_fake(value):
    return (value.new_empty((value.shape[1], value.shape[0]), dtype=torch.float8_e4m3fn),
            value.new_empty((value.shape[1], value.shape[0] // 32), dtype=torch.float8_e8m0fnu))


def _as_mx(value: torch.Tensor, *, transposed: bool = False):
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

    contiguous = value.contiguous()
    qdata, scales = (_quantize_transposed_rows_raw(contiguous) if transposed
                     else _quantize_rows_raw(contiguous))
    return MXTensor.from_qdata_and_scales(
        qdata, scales, value.dtype, block_size=32, kernel_preference=KernelPreference.AUTO)


def _preflight(value: torch.Tensor, backend: str) -> None:
    native._canonical_backend(backend)
    if not torch.compiler.is_compiling():
        native.resolve_mxfp8_backend(backend, value.device)
    if not value.is_cuda:
        raise native.MXFP8Error("native MXFP8 requires CUDA operands")
    if value.dtype != torch.bfloat16:
        raise native.MXFP8Error("native MXFP8 requires BF16 operands")


def _aligned(*dimensions) -> bool:
    return all(dimension > 0 and dimension % 128 == 0 for dimension in dimensions)


def _pad_original_matrix(value: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """Pad once before any row/column casts; aligned inputs avoid a zero pad."""
    if value.shape == (rows, columns):
        return value.contiguous()
    return F.pad(value, (0, columns - value.shape[1], 0, rows - value.shape[0]))


def mxfp8_mm_nt_fast(left: torch.Tensor, right_rows: torch.Tensor, *, backend: str = "auto") -> torch.Tensor:
    """Native ``left @ right_rows.T``, avoiding padding for aligned M/N/K."""
    if left.ndim != 2 or right_rows.ndim != 2:
        raise ValueError("MXFP8 matrix operands must both be 2-D")
    if left.shape[1] != right_rows.shape[1]:
        raise ValueError("MXFP8 matrix operands must share their contraction dimension")
    if left.device != right_rows.device:
        raise ValueError("MXFP8 matrix operands must be on the same device")
    if left.dtype != right_rows.dtype:
        raise TypeError("MXFP8 matrix operands must use the same high-precision dtype")
    if not _aligned(left.shape[0], left.shape[1], right_rows.shape[0]):
        return native.mxfp8_mm_nt(left, right_rows, backend=backend)
    _preflight(left, backend)
    left_mx, right_mx = _as_mx(left), _as_mx(right_rows)
    return torch.mm(left_mx, right_mx.t()).to(left.dtype).contiguous()


def mxfp8_linear_backward_fast(
    inputs: torch.Tensor, weight: torch.Tensor, grad_output: torch.Tensor, *, backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native dgrad/wgrad with direct dim1 casts instead of BF16 transpose copies."""
    if inputs.ndim < 1 or grad_output.ndim != inputs.ndim:
        raise ValueError("input and grad_output must have matching, non-empty ranks")
    if weight.ndim != 2:
        raise ValueError("weight must be 2-D")
    if inputs.shape[:-1] != grad_output.shape[:-1]:
        raise ValueError("input and grad_output leading dimensions must match")
    if inputs.shape[-1] != weight.shape[1]:
        raise ValueError("input feature dimension does not match weight")
    if grad_output.shape[-1] != weight.shape[0]:
        raise ValueError("grad_output feature dimension does not match weight")
    if inputs.device != weight.device or inputs.device != grad_output.device:
        raise ValueError("MXFP8 backward operands must be on the same device")
    if inputs.dtype != weight.dtype or inputs.dtype != grad_output.dtype:
        raise TypeError("MXFP8 backward operands must use the same high-precision dtype")
    rows = inputs.numel() // inputs.shape[-1] if inputs.shape[-1] else 0
    if not rows or not weight.numel():
        return native.mxfp8_linear_backward(inputs, weight, grad_output, backend=backend)
    _preflight(inputs, backend)
    input_features, output_features = weight.shape[1], weight.shape[0]
    padded_rows = native._ceil_multiple(rows, 128)
    padded_input_features = native._ceil_multiple(input_features, 128)
    padded_output_features = native._ceil_multiple(output_features, 128)
    input_flat = _pad_original_matrix(
        inputs.reshape(-1, input_features), padded_rows, padded_input_features)
    grad_flat = _pad_original_matrix(
        grad_output.reshape(-1, output_features), padded_rows, padded_output_features)
    weight_padded = _pad_original_matrix(weight, padded_output_features, padded_input_features)

    # dX = dY @ W: dY blocks run across output channels; W is independently
    # quantized down its output-channel axis, then represented as rows of W.T.
    grad_rows = _as_mx(grad_flat)
    weight_columns = _as_mx(weight_padded, transposed=True)
    grad_input = torch.mm(grad_rows, weight_columns.t())[:rows, :input_features]
    del grad_rows, weight_columns, weight_padded

    # dW = dY.T @ X: both fresh column casts now block across the token axis.
    # Reusing either dgrad quantization here would change the contraction scale.
    grad_columns = _as_mx(grad_flat, transposed=True)
    input_columns = _as_mx(input_flat, transposed=True)
    grad_weight = torch.mm(grad_columns, input_columns.t())[:output_features, :input_features]
    return (grad_input.to(inputs.dtype).contiguous().reshape_as(inputs),
            grad_weight.to(weight.dtype).contiguous())


__all__ = ["mxfp8_mm_nt_fast", "mxfp8_linear_backward_fast"]
