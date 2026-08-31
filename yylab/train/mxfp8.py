from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

NATIVE_MXFP8_BACKEND = "torchao_native_mxfp8"
EMULATED_MXFP8_BACKEND = "torchao_emulated_mxfp8"
MXFP8_BACKENDS = {"auto", "native", "emulated"}


class MXFP8Error(RuntimeError):
    """Raised when an explicitly requested MXFP8 execution path is unusable."""


def _ceil_multiple(value: int, multiple: int) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


def _canonical_backend(backend: str) -> str:
    normalized = str(backend or "auto").strip().lower()
    if normalized not in MXFP8_BACKENDS:
        raise ValueError(f"unknown MXFP8 backend: {backend}")
    return normalized


def _native_prerequisites() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    capability = torch.cuda.get_device_capability()
    if capability[0] < 10:
        return False, (
            "native MXFP8 block-scaled GEMM requires Blackwell or newer, "
            f"found SM{capability[0]}{capability[1]}"
        )
    if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch.ops.aten, "_scaled_mm"):
        return False, "this PyTorch build does not expose FP8 scaled GEMM"
    try:
        from torchao.prototype.mx_formats.mx_tensor import MXTensor  # noqa: F401
    except (ImportError, OSError) as exc:
        return False, f"TorchAO MXFP8 backend failed to import: {exc}"
    return True, f"SM{capability[0]}{capability[1]} TorchAO block-scaled MXFP8"


def resolve_mxfp8_backend(backend: str = "auto") -> tuple[str, str]:
    """Resolve a recipe request without claiming that a smoke test was run.

    Strict/native callers should additionally call :func:`native_mxfp8_status`
    in worker preflight.  The split keeps UI capability inspection from
    allocating a CUDA context while still making execution failures explicit.
    """

    requested = _canonical_backend(backend)
    if requested == "emulated":
        return EMULATED_MXFP8_BACKEND, "portable TorchAO MXFP8 emulation"
    available, note = _native_prerequisites()
    if available:
        return NATIVE_MXFP8_BACKEND, note
    if requested == "native":
        raise MXFP8Error(f"native MXFP8 requested but unavailable: {note}")
    return EMULATED_MXFP8_BACKEND, f"native MXFP8 unavailable; using emulation: {note}"


def _mx_quantize_rows(value: torch.Tensor, *, native: bool):
    """Quantize contiguous matrix rows in 32-value MXFP8 blocks.

    Explicitly transposing the second operand before this function avoids
    TorchAO's optional dim-1 CUDA quantizer.  Some Windows wheels omit that
    operator even though the native block-scaled GEMM and Triton dim-0 cast are
    present.  Both operands still reach the real MXFP8 Tensor Core GEMM.
    """

    from torchao.prototype.mx_formats.config import (
        MXFP8Dim0CastKernelChoice,
        ScaleCalculationMode,
    )
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    from torchao.quantization.quantize_.common.kernel_preference import (
        KernelPreference,
    )

    preference = KernelPreference.AUTO if native else KernelPreference.EMULATED
    cast_kernel = (
        MXFP8Dim0CastKernelChoice.TRITON
        if native
        else MXFP8Dim0CastKernelChoice.TORCH
    )
    return MXTensor.to_mx(
        value.contiguous(),
        torch.float8_e4m3fn,
        32,
        ScaleCalculationMode.RCEIL,
        preference,
        mxfp8_dim0_cast_kernel_choice=cast_kernel,
    )


def mxfp8_mm_nt(
    left: torch.Tensor,
    right_rows: torch.Tensor,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Compute ``left @ right_rows.T`` with block-32 E4M3/E8M0 MXFP8.

    ``right_rows`` is accepted before transposition so each operand can be
    quantized along the shared contraction dimension.  Native execution pads
    all GEMM axes to 128 for the SM120 scaled-matmul contract and slices the
    BF16 result back to its requested shape.
    """

    if left.ndim != 2 or right_rows.ndim != 2:
        raise ValueError("MXFP8 matrix operands must both be 2-D")
    if left.shape[1] != right_rows.shape[1]:
        raise ValueError(
            "MXFP8 matrix operands must share their contraction dimension: "
            f"{left.shape[1]} != {right_rows.shape[1]}"
        )
    if left.device != right_rows.device:
        raise ValueError("MXFP8 matrix operands must be on the same device")
    if left.dtype != right_rows.dtype:
        raise TypeError("MXFP8 matrix operands must use the same high-precision dtype")
    if left.numel() == 0 or right_rows.numel() == 0:
        return left.new_zeros((left.shape[0], right_rows.shape[0]))

    resolved, _ = resolve_mxfp8_backend(backend)
    native = resolved == NATIVE_MXFP8_BACKEND
    if native:
        if not left.is_cuda:
            raise MXFP8Error("native MXFP8 requires CUDA operands")
        if left.dtype != torch.bfloat16:
            raise MXFP8Error("native MXFP8 backward currently requires BF16 operands")

    m, contraction = left.shape
    n = right_rows.shape[0]
    alignment = 128 if native else 32
    padded_m = _ceil_multiple(m, alignment)
    padded_n = _ceil_multiple(n, alignment)
    padded_k = _ceil_multiple(contraction, alignment)
    left_padded = F.pad(left, (0, padded_k - contraction, 0, padded_m - m))
    right_padded = F.pad(
        right_rows,
        (0, padded_k - contraction, 0, padded_n - n),
    )

    left_mx = _mx_quantize_rows(left_padded, native=native)
    right_mx = _mx_quantize_rows(right_padded, native=native)
    output = torch.mm(left_mx, right_mx.t())
    return output[:m, :n].to(dtype=left.dtype).contiguous()


def mxfp8_linear_backward(
    input: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Linear dgrad and wgrad using native or emulated MXFP8 GEMMs."""

    if input.ndim < 1 or grad_output.ndim != input.ndim:
        raise ValueError("input and grad_output must have matching, non-empty ranks")
    if weight.ndim != 2:
        raise ValueError("weight must be 2-D")
    if input.shape[:-1] != grad_output.shape[:-1]:
        raise ValueError("input and grad_output leading dimensions must match")
    if input.shape[-1] != weight.shape[1]:
        raise ValueError("input feature dimension does not match weight")
    if grad_output.shape[-1] != weight.shape[0]:
        raise ValueError("grad_output feature dimension does not match weight")

    input_flat = input.reshape(-1, input.shape[-1]).contiguous()
    grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
    grad_input = mxfp8_mm_nt(
        grad_flat,
        weight.t().contiguous(),
        backend=backend,
    ).reshape_as(input)
    grad_weight = mxfp8_mm_nt(
        grad_flat.t().contiguous(),
        input_flat.t().contiguous(),
        backend=backend,
    )
    return grad_input, grad_weight


@lru_cache(maxsize=1)
def _smoke_native_mxfp8() -> tuple[bool, str]:
    available, note = _native_prerequisites()
    if not available:
        return False, note
    try:
        generator = torch.Generator(device="cpu").manual_seed(1208)
        input_cpu = torch.randn((37, 79), generator=generator)
        weight_cpu = torch.randn((113, 79), generator=generator)
        grad_cpu = torch.randn((37, 113), generator=generator)
        input = input_cpu.to(device="cuda", dtype=torch.bfloat16)
        weight = weight_cpu.to(device="cuda", dtype=torch.bfloat16)
        grad_output = grad_cpu.to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            grad_input, grad_weight = mxfp8_linear_backward(
                input,
                weight,
                grad_output,
                backend="native",
            )
        torch.cuda.synchronize(input.device)
        reference_input = grad_output.float().mm(weight.float())
        reference_weight = grad_output.float().t().mm(input.float())
        dgrad_error = (
            (grad_input.float() - reference_input).norm()
            / reference_input.norm().clamp_min(1e-12)
        )
        wgrad_error = (
            (grad_weight.float() - reference_weight).norm()
            / reference_weight.norm().clamp_min(1e-12)
        )
        if not bool(torch.isfinite(dgrad_error)) or not bool(torch.isfinite(wgrad_error)):
            raise MXFP8Error("MXFP8 smoke test returned non-finite error")
        if float(dgrad_error) > 0.15 or float(wgrad_error) > 0.15:
            raise MXFP8Error(
                "MXFP8 smoke error exceeded tolerance: "
                f"dgrad={float(dgrad_error):.4f}, wgrad={float(wgrad_error):.4f}"
            )
        return (
            True,
            f"{note}; native dgrad/wgrad smoke passed "
            f"(relative error {float(dgrad_error):.4f}/{float(wgrad_error):.4f})",
        )
    except Exception as exc:
        return False, f"native MXFP8 dgrad/wgrad smoke failed: {exc}"


def native_mxfp8_status(*, force_refresh: bool = False) -> tuple[bool, str]:
    """Report native MXFP8 only after both gradient GEMMs execute successfully."""

    if force_refresh:
        _smoke_native_mxfp8.cache_clear()
    return _smoke_native_mxfp8()


def mxfp8_capability_status() -> tuple[bool, str]:
    """Cheap UI preflight that leaves real dgrad/wgrad validation to a worker."""

    available, note = _native_prerequisites()
    if not available:
        return available, note
    return True, f"{note} is configured; the training worker requires both GEMM smokes"


__all__ = [
    "EMULATED_MXFP8_BACKEND",
    "MXFP8Error",
    "NATIVE_MXFP8_BACKEND",
    "mxfp8_linear_backward",
    "mxfp8_mm_nt",
    "mxfp8_capability_status",
    "native_mxfp8_status",
    "resolve_mxfp8_backend",
]
