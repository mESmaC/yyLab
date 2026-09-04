from __future__ import annotations

from functools import lru_cache

import torch

BITSANDBYTES_QUANTIZATIONS = ("nf4", "int8")
_BITSANDBYTES_COMPUTE_DTYPES = {
    "nf4": torch.bfloat16,
    # Linear8bitLt quantizes activations through FP16. Probing it with BF16
    # inputs only observes the layer's cast-back dtype, not its real compute.
    "int8": torch.float16,
}


def bitsandbytes_compute_precision(quantization: str) -> str:
    """Return the explicit arithmetic precision of yyLab's bnb runtime path."""

    normalized = str(quantization).strip().casefold()
    try:
        dtype = _BITSANDBYTES_COMPUTE_DTYPES[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown bitsandbytes quantization: {quantization}") from exc
    return "bf16" if dtype == torch.bfloat16 else "fp16"


def _require_native_compute_dtype(device: torch.device, dtype: torch.dtype) -> None:
    """Reject emulated or unavailable arithmetic before invoking bnb kernels."""

    capability = torch.cuda.get_device_capability(device)
    if dtype == torch.bfloat16:
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError(
                f"CUDA device {device} has no native BF16 arithmetic for NF4 compute"
            )
        return
    if dtype == torch.float16 and capability < (5, 3):
        raise RuntimeError(
            f"CUDA device {device} compute capability {capability[0]}.{capability[1]} "
            "has no native FP16 arithmetic for INT8 compute"
        )


def _probe_one(quantization: str, device: torch.device) -> tuple[bool, str]:
    """Execute the same quantized CUDA layer family used by HF loading.

    Import/version checks are deliberately insufficient here. A format is
    advertised only after its quantized storage, CUDA forward operator, and
    input-gradient autograd path all execute successfully.
    """

    try:
        import bitsandbytes as bnb

        compute_dtype = _BITSANDBYTES_COMPUTE_DTYPES[quantization]
        compute_precision = bitsandbytes_compute_precision(quantization)
        with (
            torch.cuda.device(int(device.index)),
            torch.inference_mode(False),
            torch.enable_grad(),
        ):
            _require_native_compute_dtype(device, compute_dtype)
            generator = torch.Generator(device="cpu").manual_seed(4815)
            initial_weight = torch.randn((32, 64), generator=generator) * 0.02
            if quantization == "nf4":
                layer = bnb.nn.Linear4bit(
                    64,
                    32,
                    bias=False,
                    compute_dtype=compute_dtype,
                    compress_statistics=True,
                    quant_type="nf4",
                )
                expected_storage = torch.uint8
            elif quantization == "int8":
                layer = bnb.nn.Linear8bitLt(
                    64,
                    32,
                    bias=False,
                    has_fp16_weights=False,
                    threshold=6.0,
                )
                expected_storage = torch.int8
            else:  # pragma: no cover - callers are constrained by the public constant
                raise ValueError(f"unknown bitsandbytes quantization: {quantization}")

            with torch.no_grad():
                layer.weight.copy_(initial_weight)
            layer = layer.to(device)
            if layer.weight.dtype != expected_storage:
                raise RuntimeError(
                    f"weight storage stayed {layer.weight.dtype}, expected {expected_storage}"
                )
            if layer.weight.device != device:
                raise RuntimeError(
                    f"weight storage landed on {layer.weight.device}, expected {device}"
                )
            if quantization == "nf4":
                if getattr(layer.weight, "quant_state", None) is None:
                    raise RuntimeError("NF4 weight has no quantization state")
                if getattr(layer, "compute_dtype", None) != compute_dtype:
                    raise RuntimeError(
                        "NF4 layer did not retain its requested BF16 compute dtype"
                    )

            inputs = torch.randn(
                (2, 64),
                generator=generator,
                device="cpu",
            ).to(device=device, dtype=compute_dtype).requires_grad_(True)
            output = layer(inputs)
            (grad_input,) = torch.autograd.grad(output.float().square().mean(), (inputs,))
            torch.cuda.synchronize(device)
        if output.device != device or grad_input.device != device:
            raise RuntimeError("operator returned a tensor on the wrong device")
        if output.dtype != compute_dtype or grad_input.dtype != compute_dtype:
            raise RuntimeError(
                "operator substituted its advertised compute dtype: "
                f"expected {compute_dtype}, got output={output.dtype}, dgrad={grad_input.dtype}"
            )
        if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(grad_input).all()):
            raise RuntimeError("forward/backward returned non-finite values")
        if not bool(output.abs().max() > 0) or not bool(grad_input.abs().max() > 0):
            raise RuntimeError("forward/backward returned an all-zero tensor")
        return (
            True,
            "quantized CUDA storage plus native "
            f"{compute_precision.upper()} forward/backward smoke passed",
        )
    except Exception as exc:  # capability reports must fail closed
        return False, f"CUDA operator/autograd smoke failed: {exc}"


@lru_cache(maxsize=8)
def _probe_bitsandbytes_cuda(device_index: int) -> tuple[tuple[str, bool, str], ...]:
    device = torch.device("cuda", int(device_index))
    return tuple(
        (quantization, *_probe_one(quantization, device))
        for quantization in BITSANDBYTES_QUANTIZATIONS
    )


def _resolved_cuda_index(device: str | torch.device | None = None) -> tuple[int | None, str]:
    if not torch.cuda.is_available():
        return None, "CUDA is unavailable"
    resolved = torch.device(device or "cuda")
    if resolved.type != "cuda":
        return None, f"bitsandbytes quantization requires CUDA, got {resolved}"
    index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
    if index < 0 or index >= torch.cuda.device_count():
        return None, f"CUDA device index {index} is unavailable"
    return index, ""


def bitsandbytes_quantization_status(
    device: str | torch.device | None = None,
    *,
    force_refresh: bool = False,
) -> dict[str, tuple[bool, str]]:
    """Return execution-verified NF4/INT8 status for one CUDA device."""

    index, unavailable = _resolved_cuda_index(device)
    if index is None:
        return {
            quantization: (False, unavailable)
            for quantization in BITSANDBYTES_QUANTIZATIONS
        }
    if force_refresh:
        _probe_bitsandbytes_cuda.cache_clear()
    return {
        quantization: (available, detail)
        for quantization, available, detail in _probe_bitsandbytes_cuda(index)
    }


def available_bitsandbytes_quantizations(
    device: str | torch.device | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[str, ...]:
    """Return only bitsandbytes formats proven executable on this GPU."""

    status = bitsandbytes_quantization_status(device, force_refresh=force_refresh)
    return tuple(
        quantization
        for quantization in BITSANDBYTES_QUANTIZATIONS
        if status[quantization][0]
    )


def require_bitsandbytes_quantization(
    quantization: str,
    device: str | torch.device | None = None,
) -> None:
    """Fail closed when a requested HF quantized-load path was not verified."""

    normalized = str(quantization).strip().casefold()
    if normalized not in BITSANDBYTES_QUANTIZATIONS:
        raise ValueError(f"unknown bitsandbytes quantization: {quantization}")
    available, detail = bitsandbytes_quantization_status(device)[normalized]
    if not available:
        raise RuntimeError(
            f"{normalized.upper()} bitsandbytes loading is unavailable: {detail}"
        )


__all__ = [
    "BITSANDBYTES_QUANTIZATIONS",
    "available_bitsandbytes_quantizations",
    "bitsandbytes_compute_precision",
    "bitsandbytes_quantization_status",
    "require_bitsandbytes_quantization",
]
