from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache

import torch

DENSE_MODEL_PRECISIONS = ("bf16", "fp16", "fp32")

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def model_compute_dtype(precision: str) -> torch.dtype:
    """Map a resolved base precision to its actual activation compute dtype.

    ``load_base`` verifies the selected precision on the exact execution device
    before returning it.  Downstream train/inference components use this strict
    mapping so quantized weight storage is never mistaken for adapter compute:
    NF4 executes in BF16, while bitsandbytes INT8 executes in FP16.
    """

    from .bitsandbytes_capabilities import (
        BITSANDBYTES_QUANTIZATIONS,
        bitsandbytes_compute_precision,
    )

    normalized = str(precision).strip().casefold()
    if normalized in DENSE_MODEL_PRECISIONS:
        return _DTYPES[normalized]
    if normalized in BITSANDBYTES_QUANTIZATIONS:
        return _DTYPES[bitsandbytes_compute_precision(normalized)]
    raise ValueError(
        f"unsupported base precision for adapter compute: {normalized or '<empty>'}"
    )


def model_execution_device() -> torch.device:
    """Return the exact device used by yyLab's single-model loading paths."""

    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def _resolve_device(device: str | torch.device | None = None) -> tuple[torch.device | None, str]:
    if device is None:
        device = model_execution_device()
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return torch.device("cpu"), ""
    if resolved.type != "cuda":
        return None, f"dense model precision requires a CPU or CUDA device, got {resolved}"
    if not torch.cuda.is_available():
        return None, "CUDA is unavailable"
    index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
    if index < 0 or index >= torch.cuda.device_count():
        return None, f"CUDA device index {index} is unavailable"
    return torch.device("cuda", index), ""


def _probe_one(precision: str, device: torch.device) -> tuple[bool, str]:
    """Exercise a dense forward plus input/weight backward in the requested dtype."""

    try:
        if device.type == "cpu" and precision != "fp32":
            raise RuntimeError("CPU execution is conservatively limited to FP32")
        device_context = (
            torch.cuda.device(int(device.index))
            if device.type == "cuda"
            else nullcontext()
        )
        with device_context:
            if device.type == "cuda":
                capability = torch.cuda.get_device_capability(device)
                if precision == "bf16" and not torch.cuda.is_bf16_supported(
                    including_emulation=False
                ):
                    raise RuntimeError("the selected CUDA device has no native BF16 support")
                if precision == "fp16" and capability < (5, 3):
                    raise RuntimeError(
                        f"compute capability {capability[0]}.{capability[1]} "
                        "has no native FP16 arithmetic"
                    )

            dtype = _DTYPES[precision]
            generator = torch.Generator(device="cpu").manual_seed(4816)
            with torch.inference_mode(False), torch.enable_grad():
                inputs = torch.randn((4, 64), generator=generator).to(
                    device=device, dtype=dtype
                ).requires_grad_(True)
                weight = torch.randn((32, 64), generator=generator).to(
                    device=device, dtype=dtype
                ).requires_grad_(True)
                output = torch.nn.functional.linear(inputs, weight)
                grad_input, grad_weight = torch.autograd.grad(
                    output.float().square().mean(),
                    (inputs, weight),
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        for name, tensor in (
            ("output", output),
            ("input gradient", grad_input),
            ("weight gradient", grad_weight),
        ):
            if tensor.device != device:
                raise RuntimeError(f"{name} returned on {tensor.device}, expected {device}")
            if tensor.dtype != dtype:
                raise RuntimeError(f"{name} widened to {tensor.dtype}, expected {dtype}")
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"{name} contains non-finite values")
            if not bool(tensor.abs().max() > 0):
                raise RuntimeError(f"{name} is all zero")
        return True, f"native {device.type.upper()} dense forward/backward smoke passed"
    except Exception as exc:  # capability reports must fail closed
        return False, f"native dense forward/backward smoke failed: {exc}"


@lru_cache(maxsize=16)
def _probe_dense_device(device_type: str, device_index: int | None) -> tuple[tuple[str, bool, str], ...]:
    device = torch.device(device_type, device_index)
    candidates = DENSE_MODEL_PRECISIONS if device_type == "cuda" else ("fp32",)
    probed = {
        precision: _probe_one(precision, device)
        for precision in candidates
    }
    return tuple(
        (
            precision,
            *probed.get(
                precision,
                (False, "CPU execution is conservatively limited to FP32"),
            ),
        )
        for precision in DENSE_MODEL_PRECISIONS
    )


def dense_precision_status(
    device: str | torch.device | None = None,
    *,
    force_refresh: bool = False,
) -> dict[str, tuple[bool, str]]:
    """Return exact-device dense model precision availability."""

    resolved, unavailable = _resolve_device(device)
    if resolved is None:
        return {
            precision: (False, unavailable)
            for precision in DENSE_MODEL_PRECISIONS
        }
    if force_refresh:
        _probe_dense_device.cache_clear()
    return {
        precision: (available, detail)
        for precision, available, detail in _probe_dense_device(
            resolved.type,
            resolved.index,
        )
    }


def available_dense_precisions(
    device: str | torch.device | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[str, ...]:
    status = dense_precision_status(device, force_refresh=force_refresh)
    return tuple(
        precision
        for precision in DENSE_MODEL_PRECISIONS
        if status[precision][0]
    )


def preferred_dense_precision(device: str | torch.device | None = None) -> str:
    available = available_dense_precisions(device)
    if not available:
        raise RuntimeError("no dense model precision passed the exact-device capability probe")
    return available[0]


def require_dense_precision(
    precision: str,
    device: str | torch.device | None = None,
) -> None:
    normalized = str(precision).strip().casefold()
    if normalized not in DENSE_MODEL_PRECISIONS:
        raise ValueError(f"unknown dense model precision: {precision}")
    available, detail = dense_precision_status(device)[normalized]
    if not available:
        raise RuntimeError(f"{normalized.upper()} dense model execution is unavailable: {detail}")


def available_model_precisions(
    device: str | torch.device | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[str, ...]:
    """Return every base-model precision proven executable on one device."""

    from .bitsandbytes_capabilities import available_bitsandbytes_quantizations

    return (
        *available_dense_precisions(device, force_refresh=force_refresh),
        *available_bitsandbytes_quantizations(device, force_refresh=force_refresh),
    )


def require_model_precision(
    precision: str,
    device: str | torch.device | None = None,
) -> None:
    """Enforce the shared dense/bitsandbytes model-load capability contract."""

    from .bitsandbytes_capabilities import (
        BITSANDBYTES_QUANTIZATIONS,
        require_bitsandbytes_quantization,
    )

    normalized = str(precision).strip().casefold()
    if normalized in DENSE_MODEL_PRECISIONS:
        require_dense_precision(normalized, device)
        return
    if normalized in BITSANDBYTES_QUANTIZATIONS:
        require_bitsandbytes_quantization(normalized, device)
        return
    if normalized == "fp8":
        raise ValueError("FP8 base loading is not implemented; choose a verified runtime precision")
    raise ValueError(f"unsupported base precision: {normalized or '<empty>'}")


__all__ = [
    "DENSE_MODEL_PRECISIONS",
    "available_dense_precisions",
    "available_model_precisions",
    "dense_precision_status",
    "model_compute_dtype",
    "model_execution_device",
    "preferred_dense_precision",
    "require_dense_precision",
    "require_model_precision",
]
