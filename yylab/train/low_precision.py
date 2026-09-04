from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import Any

import torch
from torch import nn

LOW_PRECISION_METADATA_KEY = "yylab_low_precision"
QAT_FORMATS = {"nvfp4"}
SPARSITY_PATTERNS = {"none", "paired_4:8"}
NATIVE_NVFP4_BACKEND = "torchao_native_nvfp4_qat"
COMPRESSED_SPARSE_NVFP4_BACKEND = "cusparselt_compressed_nvfp4"
SPARSITY_BACKEND_REQUESTS = {"auto", "compressed"}
BACKWARD_PRECISIONS = {"bf16", "mxfp8"}


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
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SPARSITY_BACKEND_REQUESTS:
        raise ValueError(f"unknown structured-sparsity backend: {backend}")
    return normalized


class QATLinear(nn.Linear):
    """State-dict-compatible Linear backed only by native low-bit kernels."""

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
        if self.qat_format != "nvfp4" or self.qat_backend != NATIVE_NVFP4_BACKEND:
            raise RuntimeError(
                "saved low-precision recipe does not name a supported native NVFP4 backend"
            )
        if self.qat_group_size != 16 or self.qat_activation != "nvfp4":
            raise RuntimeError(
                "native NVFP4 execution requires exact W4A4 metadata "
                "(group_size=16, activation='nvfp4')"
            )
        if not input.is_cuda or not self.weight.is_cuda:
            raise RuntimeError("native NVFP4 training requires CUDA activations and weights")
        if self.qat_sparsity != "none":
            if self.qat_sparsity_backend != COMPRESSED_SPARSE_NVFP4_BACKEND:
                raise RuntimeError(
                    "paired 4:8 NVFP4 requires the compressed native sparse backend"
                )
            from .sparse_nvfp4 import sparse_nvfp4_linear

            return sparse_nvfp4_linear(
                input,
                self.weight,
                self.bias,
                backward_precision=self.qat_backward_precision,
                backward_backend=self.qat_backward_backend,
            )
        if self.qat_sparsity_backend != "none":
            raise RuntimeError(f"unknown native sparsity backend: {self.qat_sparsity_backend}")
        return _native_nvfp4_linear(
            input,
            self.weight,
            self.bias,
            backward_precision=self.qat_backward_precision,
            backward_backend=self.qat_backward_backend,
        )


class _NativeNVFP4Linear(torch.autograd.Function):
    """Native FP4 forward GEMM with selectable BF16 or MXFP8 backward.

    TorchAO owns the packed E2M1/E4M3 representation and dispatches PyTorch's
    SM120 block-scaled GEMM. yyLab keeps BF16 shadow parameters for backward
    and computes bias gradients (the current prototype helper omits them).
    The dense path has no structured mask. The separate compressed sparse
    wrapper masks its returned weight gradient to the exact 4:8 forward pattern.
    Full optimizer offload keeps the master and Adam moments in FP32 CPU memory.
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
        # The native forward consumes packed FP4 operands, but backward uses
        # the original BF16 shadow tensors. Saving dequantized packed values
        # here would silently turn the advertised BF16-shadow backward into an
        # FP4-reconstructed backward.
        ctx.save_for_backward(flat_input, weight)
        ctx.input_shape = original_shape
        ctx.has_bias = bias is not None
        ctx.backward_precision = backward_precision
        ctx.backward_backend = backward_backend
        output = _addmm_nvfp4_dispatch(packed_input, packed_weight.t(), None, bias)
        return output.reshape(*original_shape[:-1], output.shape[-1])

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        input_hp, weight_hp = ctx.saved_tensors
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
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
    backward_backend: str = "none",
) -> torch.Tensor:
    if not input.is_cuda or not weight.is_cuda:
        raise RuntimeError("native NVFP4 QAT requires CUDA tensors")
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise RuntimeError("native NVFP4 QAT currently requires BF16 activations and master weights")
    if backward_precision not in BACKWARD_PRECISIONS:
        raise ValueError(f"unknown low-bit backward precision: {backward_precision}")
    if backward_precision == "bf16" and backward_backend != "none":
        raise RuntimeError("BF16 backward does not use a low-precision backend")
    if backward_precision == "mxfp8":
        from .mxfp8 import NATIVE_MXFP8_BACKEND

        if backward_backend != NATIVE_MXFP8_BACKEND:
            raise RuntimeError(
                f"MXFP8 backward requires {NATIVE_MXFP8_BACKEND!r}, "
                f"got {backward_backend!r}"
            )
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
    device: str | torch.device | None = None,
    group_size: int = 16,
    activation: str | None = None,
    parameter_dtype: str = "bf16",
    sparsity: str = "none",
    backend: str = "auto",
    sparsity_backend: str = "auto",
    backward_precision: str = "bf16",
    backward_backend: str = "auto",
) -> dict[str, Any] | None:
    qat_format = str(qat_format or "none").strip().lower()
    requested_activation = str(
        activation or ("none" if qat_format == "none" else "nvfp4")
    ).strip().lower()
    sparsity = _canonical_sparsity(sparsity)
    requested_sparsity_backend = _canonical_sparsity_backend(sparsity_backend)
    backward_precision = str(backward_precision or "bf16").strip().lower()
    requested_backward_backend = str(backward_backend or "auto").strip().lower()
    requested_backend = str(backend or "auto").strip().lower()
    if backward_precision not in BACKWARD_PRECISIONS:
        raise ValueError(f"unknown low-bit backward precision: {backward_precision}")
    if requested_backward_backend not in {"auto", "native"}:
        raise ValueError(f"unknown MXFP8 backward backend: {backward_backend}")
    if requested_backend not in {"auto", "native"}:
        raise ValueError(
            f"unsupported low-bit backend {backend!r}; yyLab permits native execution only"
        )
    if qat_format == "none":
        if requested_activation != "none":
            raise ValueError(
                "activation quantization requires a native low-precision forward format"
            )
        if sparsity != "none":
            raise ValueError("structured sparsity requires native NVFP4")
        if requested_sparsity_backend != "auto":
            raise ValueError("a compressed sparsity backend requires structured sparsity")
        if requested_backend == "native":
            raise ValueError("native low-bit backend requires a QAT format")
        if backward_precision != "bf16":
            raise ValueError("MXFP8 backward requires NVFP4 QAT")
        if requested_backward_backend != "auto":
            raise ValueError("a native backward backend requires MXFP8 backward precision")
        return None
    if qat_format not in QAT_FORMATS:
        raise ValueError(
            f"unsupported low-precision format {qat_format!r}; no native training backend is integrated"
        )
    if parameter_dtype != "bf16":
        raise ValueError("native NVFP4 training requires BF16 parameter storage")
    if requested_activation != "nvfp4":
        raise ValueError(
            f"native NVFP4 training requires activation='nvfp4'; got {activation!r}"
        )
    group_size = int(group_size)
    if group_size != 16:
        raise ValueError(
            f"native NVFP4 uses an exact block size of 16; got group_size={group_size}"
        )
    if sparsity == "none" and requested_sparsity_backend != "auto":
        raise ValueError("a compressed sparsity backend requires paired 4:8 sparsity")
    if backward_precision == "bf16" and requested_backward_backend != "auto":
        raise ValueError("a native backward backend requires MXFP8 backward precision")
    activation = "nvfp4"
    native_available, native_note = native_nvfp4_status(device)
    if not native_available:
        raise RuntimeError(f"native NVFP4 training is unavailable: {native_note}")

    resolved_sparsity_backend = "none"
    native_sparse = False
    sparse_note = "dense weights"
    if sparsity != "none":
        from .sparse_nvfp4 import sparse_nvfp4_status

        native_sparse, sparse_note = sparse_nvfp4_status(device)
        if not native_sparse:
            raise RuntimeError(f"native compressed sparse NVFP4 is unavailable: {sparse_note}")
        resolved_sparsity_backend = COMPRESSED_SPARSE_NVFP4_BACKEND

    resolved_backward_backend = "none"
    native_backward = False
    backward_note = "BF16 dense backward"
    if backward_precision == "mxfp8":
        from .mxfp8 import (
            NATIVE_MXFP8_BACKEND,
            native_mxfp8_status,
        )

        native_backward, backward_note = native_mxfp8_status(device)
        if not native_backward:
            raise RuntimeError(f"native MXFP8 backward is unavailable: {backward_note}")
        resolved_backward_backend = NATIVE_MXFP8_BACKEND
    combined_autograd_note = None
    if native_sparse and native_backward:
        combined_available, combined_autograd_note = native_sparse_nvfp4_mxfp8_status(
            device
        )
        if not combined_available:
            raise RuntimeError(
                "the requested sparse NVFP4 + MXFP8 combination is unavailable: "
                f"{combined_autograd_note}"
            )
    return {
        "version": 4,
        "mode": "qat",
        "format": qat_format,
        "backend": NATIVE_NVFP4_BACKEND,
        "backend_request": requested_backend,
        "group_size": int(group_size),
        "activation": activation,
        "parameter_dtype": parameter_dtype,
        "native_low_bit_gemm": True,
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
        "combined_autograd_note": combined_autograd_note,
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


def _qat_target_device(targets: list[tuple[str, nn.Linear]]) -> torch.device:
    devices = {module.weight.device for _, module in targets}
    if len(devices) != 1:
        raise RuntimeError(
            "native low-precision QAT requires all target weights on one exact device; "
            f"found {sorted(map(str, devices))}"
        )
    return next(iter(devices))


def preflight_saved_qat(
    metadata: dict[str, Any],
    device: str | torch.device | None = None,
) -> None:
    """Reject a saved recipe unless every requested low-bit lane is native."""

    qat_format = str(metadata.get("format") or "")
    if qat_format not in QAT_FORMATS:
        raise RuntimeError(
            f"saved QAT format {qat_format!r} has no native yyLab training backend"
        )
    recipe_backend = str(metadata.get("backend") or "legacy_unspecified")
    if recipe_backend != NATIVE_NVFP4_BACKEND:
        raise RuntimeError(
            f"saved QAT backend {recipe_backend!r} is not native; emulated checkpoints "
            "must be loaded with restore_saved_qat=False"
        )
    available, note = native_nvfp4_status(device)
    if not available:
        raise RuntimeError(f"native NVFP4 QAT runtime is unavailable: {note}")
    group_size = int(metadata.get("group_size") or 0)
    activation = str(metadata.get("activation") or "legacy_unspecified")
    parameter_dtype = str(metadata.get("parameter_dtype") or "legacy_unspecified")
    if group_size != 16 or activation != "nvfp4" or parameter_dtype != "bf16":
        raise RuntimeError(
            "saved NVFP4 recipe does not describe the exact native W4A4 contract: "
            f"group_size={group_size}, activation={activation!r}, "
            f"parameter_dtype={parameter_dtype!r}"
        )
    backend_request = str(metadata.get("backend_request") or "native")
    if backend_request not in {"auto", "native"}:
        raise RuntimeError(
            f"saved NVFP4 backend request {backend_request!r} is not native-capable"
        )

    sparsity = _canonical_sparsity(str(metadata.get("sparsity") or "none"))
    recipe_sparsity_backend = str(
        metadata.get("sparsity_backend")
        or ("legacy_unspecified" if sparsity != "none" else "none")
    )
    sparsity_backend_request = str(metadata.get("sparsity_backend_request") or "auto")
    if sparsity_backend_request not in SPARSITY_BACKEND_REQUESTS:
        raise RuntimeError(
            f"saved sparsity backend request {sparsity_backend_request!r} is unsupported"
        )
    if sparsity == "none":
        if recipe_sparsity_backend != "none":
            raise RuntimeError(
                f"saved dense QAT recipe names unexpected sparsity backend "
                f"{recipe_sparsity_backend!r}"
            )
        if sparsity_backend_request != "auto":
            raise RuntimeError(
                "saved dense NVFP4 recipe requests a compressed sparsity backend"
            )
    else:
        if recipe_sparsity_backend != COMPRESSED_SPARSE_NVFP4_BACKEND:
            raise RuntimeError(
                f"saved sparsity backend {recipe_sparsity_backend!r} is not native "
                "compressed execution"
            )
        sparse_available, sparse_note = native_sparse_nvfp4_status(device)
        if not sparse_available:
            raise RuntimeError(
                f"compressed sparse NVFP4 runtime is unavailable: {sparse_note}"
            )

    backward_precision = str(metadata.get("backward_precision") or "bf16")
    recipe_backward_backend = str(metadata.get("backward_backend") or "none")
    backward_backend_request = str(metadata.get("backward_backend_request") or "auto")
    if backward_backend_request not in {"auto", "native"}:
        raise RuntimeError(
            f"saved backward backend request {backward_backend_request!r} is unsupported"
        )
    if backward_precision == "bf16":
        if recipe_backward_backend != "none":
            raise RuntimeError(
                f"saved BF16 backward recipe names unexpected backend "
                f"{recipe_backward_backend!r}"
            )
        if backward_backend_request != "auto":
            raise RuntimeError("saved BF16 recipe requests a native low-precision backward backend")
    elif backward_precision == "mxfp8":
        from .mxfp8 import NATIVE_MXFP8_BACKEND, native_mxfp8_status

        if recipe_backward_backend != NATIVE_MXFP8_BACKEND:
            raise RuntimeError(
                f"saved MXFP8 backend {recipe_backward_backend!r} is not native; "
                "emulation is disabled"
            )
        backward_available, backward_note = native_mxfp8_status(device)
        if not backward_available:
            raise RuntimeError(
                f"native MXFP8 backward runtime is unavailable: {backward_note}"
            )
    else:
        raise RuntimeError(f"unknown saved backward precision: {backward_precision}")
    if sparsity != "none" and backward_precision == "mxfp8":
        combined_available, combined_note = native_sparse_nvfp4_mxfp8_status(device)
        if not combined_available:
            raise RuntimeError(
                "saved sparse NVFP4 + MXFP8 recipe is unavailable: "
                f"{combined_note}"
            )


def _resolve_qat_runtime_backend(
    targets: list[tuple[str, nn.Linear]],
    *,
    qat_format: str,
    recipe_backend: str,
) -> tuple[str, str]:
    if recipe_backend != NATIVE_NVFP4_BACKEND:
        raise RuntimeError(
            f"saved QAT backend {recipe_backend!r} is not native; emulated checkpoints "
            "must be loaded with restore_saved_qat=False"
        )
    if qat_format != "nvfp4":
        raise ValueError("the native low-bit QAT backend supports NVFP4 only")

    target_device = _qat_target_device(targets)
    available, note = native_nvfp4_status(target_device)
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
    raise RuntimeError(f"native NVFP4 QAT runtime is unavailable: {incompatibility}")


def _resolve_sparsity_runtime_backend(
    targets: list[tuple[str, nn.Linear]],
    *,
    recipe_backend: str,
) -> tuple[str, str]:
    if recipe_backend == "none":
        return recipe_backend, "dense weights"
    if recipe_backend != COMPRESSED_SPARSE_NVFP4_BACKEND:
        raise RuntimeError(
            f"saved sparsity backend {recipe_backend!r} is not native compressed execution"
        )

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

        available, note = sparse_nvfp4_status(_qat_target_device(targets))
        incompatibility = None if available else note
        if available:
            return COMPRESSED_SPARSE_NVFP4_BACKEND, note
    raise RuntimeError(f"compressed sparse NVFP4 runtime is unavailable: {incompatibility}")


def _resolve_backward_runtime_backend(
    targets: list[tuple[str, nn.Linear]],
    *,
    precision: str,
    recipe_backend: str,
) -> tuple[str, str]:
    if precision == "bf16":
        return "none", "BF16 dense backward"
    if precision != "mxfp8":
        raise ValueError(f"unknown saved backward precision: {precision}")

    from .mxfp8 import NATIVE_MXFP8_BACKEND

    if recipe_backend != NATIVE_MXFP8_BACKEND:
        raise RuntimeError(
            f"saved MXFP8 backend {recipe_backend!r} is not native; emulation is disabled"
        )

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

        available, note = native_mxfp8_status(_qat_target_device(targets))
        incompatibility = None if available else note
        if available:
            return NATIVE_MXFP8_BACKEND, note
    raise RuntimeError(f"native MXFP8 backward runtime is unavailable: {incompatibility}")


def apply_qat(
    model: nn.Module,
    metadata: dict[str, Any],
    *,
    allow_runtime_fallback: bool = False,
) -> int:
    if allow_runtime_fallback:
        raise ValueError("runtime low-precision fallback has been removed; native execution is required")
    targets = _qat_linear_targets(model)
    if not targets:
        raise RuntimeError("QAT requested but no transformer Linear modules were found")
    target_device = _qat_target_device(targets)
    preflight_saved_qat(metadata, target_device)
    qat_format = str(metadata.get("format") or "")
    if qat_format not in QAT_FORMATS:
        raise ValueError(f"unknown saved QAT format: {qat_format}")
    group_size = int(metadata.get("group_size") or (16 if qat_format == "nvfp4" else 32))
    activation = str(metadata.get("activation") or "none")
    recipe_backend = str(metadata.get("backend") or "legacy_unspecified")
    sparsity = _canonical_sparsity(str(metadata.get("sparsity") or "none"))
    recipe_sparsity_backend = str(
        metadata.get("sparsity_backend") or ("legacy_unspecified" if sparsity != "none" else "none")
    )
    backward_precision = str(metadata.get("backward_precision") or "bf16")
    recipe_backward_backend = str(metadata.get("backward_backend") or "none")
    runtime_backend, runtime_note = _resolve_qat_runtime_backend(
        targets,
        qat_format=qat_format,
        recipe_backend=recipe_backend,
    )
    runtime_sparsity_backend, runtime_sparsity_note = _resolve_sparsity_runtime_backend(
        targets,
        recipe_backend=recipe_sparsity_backend,
    )
    runtime_backward_backend, runtime_backward_note = _resolve_backward_runtime_backend(
        targets,
        precision=backward_precision,
        recipe_backend=recipe_backward_backend,
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
        # Persist the native recipe, not transient device-specific notes.
        setattr(config, LOW_PRECISION_METADATA_KEY, dict(metadata))
    return len(targets)


def apply_saved_qat(model: nn.Module) -> tuple[int, dict[str, Any] | None]:
    metadata = getattr(getattr(model, "config", None), LOW_PRECISION_METADATA_KEY, None)
    if not metadata:
        return 0, None
    recipe_metadata = dict(metadata)
    count = apply_qat(model, recipe_metadata)
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
    from .mxfp8 import NATIVE_MXFP8_BACKEND

    runtime_metadata = dict(recipe_metadata)
    runtime_metadata["runtime_backend"] = runtime_backend
    runtime_metadata["runtime_sparsity_backend"] = runtime_sparsity_backend
    runtime_metadata["runtime_backward_backend"] = runtime_backward_backend
    runtime_metadata["runtime_native_low_bit_gemm"] = runtime_backend == NATIVE_NVFP4_BACKEND
    runtime_metadata["runtime_native_sparse_gemm"] = (
        runtime_sparsity_backend == COMPRESSED_SPARSE_NVFP4_BACKEND
    )
    runtime_metadata["runtime_native_low_bit_backward"] = (
        runtime_backward_backend == NATIVE_MXFP8_BACKEND
    )
    # Existing callers use this field to describe current execution. The saved
    # config retains the original value because runtime_metadata is ephemeral.
    runtime_metadata["native_low_bit_gemm"] = runtime_backend == NATIVE_NVFP4_BACKEND
    runtime_metadata["native_sparse_gemm"] = (
        runtime_sparsity_backend == COMPRESSED_SPARSE_NVFP4_BACKEND
    )
    runtime_metadata["native_low_bit_backward"] = (
        runtime_backward_backend == NATIVE_MXFP8_BACKEND
    )
    return count, runtime_metadata


def _resolved_cuda_index(
    device: str | torch.device | None = None,
) -> tuple[int | None, str]:
    if not torch.cuda.is_available():
        return None, "CUDA is unavailable"
    resolved = torch.device(device if device is not None else "cuda")
    if resolved.type != "cuda":
        return None, f"native NVFP4 requires CUDA, got {resolved}"
    index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
    if index < 0 or index >= torch.cuda.device_count():
        return None, f"CUDA device index {index} is unavailable"
    return index, ""


def _native_nvfp4_prerequisites(
    device: str | torch.device | None = None,
) -> tuple[bool, str]:
    index, unavailable = _resolved_cuda_index(device)
    if index is None:
        return False, unavailable
    capability = torch.cuda.get_device_capability(index)
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
    return True, f"SM{capability[0]}{capability[1]} native block-scaled FP4 via TorchAO"


@lru_cache(maxsize=8)
def _smoke_native_nvfp4(device_index: int) -> tuple[bool, str]:
    device = torch.device("cuda", int(device_index))
    available, note = _native_nvfp4_prerequisites(device)
    if not available:
        return False, note
    try:
        generator = torch.Generator(device="cpu").manual_seed(4104)
        input_cpu = torch.randn((128, 128), generator=generator) * 0.125
        weight_cpu = torch.randn((128, 128), generator=generator) * 0.125
        bias_cpu = torch.randn((128,), generator=generator) * 0.01
        grad_cpu = torch.randn((128, 128), generator=generator) * 0.125
        with torch.cuda.device(device), torch.inference_mode(False), torch.enable_grad():
            input = input_cpu.to(device=device, dtype=torch.bfloat16).requires_grad_(True)
            weight = weight_cpu.to(device=device, dtype=torch.bfloat16).requires_grad_(True)
            bias = bias_cpu.to(device=device, dtype=torch.bfloat16).requires_grad_(True)
            grad_output = grad_cpu.to(device=device, dtype=torch.bfloat16)
            output = _native_nvfp4_linear(
                input,
                weight,
                bias,
                backward_precision="bf16",
                backward_backend="none",
            )
            grad_input, grad_weight, grad_bias = torch.autograd.grad(
                output,
                (input, weight, bias),
                grad_outputs=grad_output,
            )
        torch.cuda.synchronize(input.device)
        reference = input.detach().float().mm(weight.detach().float().t())
        reference.add_(bias.detach().float())
        error = (
            (output.detach().float() - reference).norm()
            / reference.norm().clamp_min(1e-12)
        )
        reference_dgrad = grad_output.float().mm(weight.detach().float())
        reference_wgrad = grad_output.float().t().mm(input.detach().float())
        dgrad_error = (
            (grad_input.float() - reference_dgrad).norm()
            / reference_dgrad.norm().clamp_min(1e-12)
        )
        wgrad_error = (
            (grad_weight.float() - reference_wgrad).norm()
            / reference_wgrad.norm().clamp_min(1e-12)
        )
        tensors = (output, grad_input, grad_weight, grad_bias)
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise RuntimeError("forward/backward returned non-finite values")
        if any(not bool(value.abs().max() > 0) for value in tensors):
            raise RuntimeError("forward/backward returned an all-zero tensor")
        if not bool(torch.isfinite(error)) or float(error) > 0.55:
            raise RuntimeError(f"forward relative error is too large ({float(error):.4f})")
        if (
            not bool(torch.isfinite(dgrad_error))
            or not bool(torch.isfinite(wgrad_error))
            or float(dgrad_error) > 0.02
            or float(wgrad_error) > 0.02
        ):
            raise RuntimeError(
                "BF16-shadow backward relative error is too large: "
                f"dgrad={float(dgrad_error):.4f}, wgrad={float(wgrad_error):.4f}"
            )
        return (
            True,
            f"{note}; native BF16-shadow forward/backward smoke passed "
            f"(forward/dgrad/wgrad relative error "
            f"{float(error):.4f}/{float(dgrad_error):.4f}/{float(wgrad_error):.4f})",
        )
    except Exception as exc:
        return False, f"native NVFP4 forward/backward smoke failed: {exc}"


def native_nvfp4_status(
    device: str | torch.device | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[bool, str]:
    """Report native dense NVFP4 only after a real forward/backward smoke."""

    if force_refresh:
        _smoke_native_nvfp4.cache_clear()
    index, unavailable = _resolved_cuda_index(device)
    if index is None:
        return False, unavailable
    return _smoke_native_nvfp4(index)


def native_sparse_nvfp4_status(
    device: str | torch.device | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[bool, str]:
    """Report sparse FP4 only after its compressed forward/backward smoke."""

    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    from .sparse_nvfp4 import sparse_nvfp4_status

    return sparse_nvfp4_status(device, force_refresh=force_refresh)


def native_sparse_nvfp4_mxfp8_status(
    device: str | torch.device | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[bool, str]:
    """Report the combined lane only after its exact autograd path passes."""

    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    from .sparse_nvfp4 import sparse_nvfp4_mxfp8_status

    return sparse_nvfp4_mxfp8_status(device, force_refresh=force_refresh)


def qat_state_multiplier(qat_format: str) -> float:
    """Training-state factor for BF16 shadow parameters used by native NVFP4."""

    if qat_format not in {"none", *QAT_FORMATS}:
        raise ValueError(f"unknown QAT format: {qat_format}")
    return 1.0
