"""Isolated TorchAO all-three-GEMM NVFP4 experiment for Sprint benchmarks.

This is a training recipe, not packed optimizer state: parameters remain BF16
and the caller selects the optimizer. TorchAO applies 2D weight scaling,
Hadamard transforms and stochastic gradient rounding, saving packed activations
for backward. This differs from yyLab's NVFP4-forward/BF16-shadow-backward QAT.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


def _load_provider():
    try:
        from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
            prepare_for_cuda_graph,
        )
        from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
            NVFP4Linear,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "The all-GEMM NVFP4 benchmark requires TorchAO's NVFP4TrainingConfig "
            "prototype and a compatible PyTorch/Triton installation."
        ) from exc
    if not hasattr(torch.nn.functional, "scaled_mm"):
        raise RuntimeError("The all-GEMM NVFP4 benchmark requires torch.nn.functional.scaled_mm.")
    return NVFP4Linear, prepare_for_cuda_graph


def _validate_hardware_rounding(capability: tuple[int, int]) -> None:
    # PTX's cvt.rs E2M1 conversion exists on SM100a/SM103a, not SM120a.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
    if capability not in {(10, 0), (10, 3)}:
        raise RuntimeError(
            f"Stock TorchAO all-GEMM NVFP4 stochastic rounding is unsupported on SM{capability[0]}"
            f"{capability[1]}: cvt.rs requires SM100/SM103. The RTX 50-series can run native FP4 GEMMs, "
            "but this training recipe needs a separately validated portable stochastic-rounding kernel."
        )


def apply_torchao_nvfp4_training(
    model: nn.Module, *, stochastic_rounding: str = "hardware", pad_to_alignment: bool = False,
) -> dict:
    """Convert a disposable BF16 Sprint before compilation/optimizer creation.

    CPU construction is supported for validation; execution requires CUDA
    Blackwell and a compatible native scaled-MM kernel. Stock hardware
    stochastic rounding additionally requires SM100/SM103; the separately
    labeled portable variant uses distance-proportional random rounding.
    For CUDA graphs, convert after placing the model on its final CUDA device
    so persistent provider workspaces are prepared outside graph capture.

    Projection dimensions must be divisible by 128 unless portable SR and
    pad_to_alignment=True are explicitly selected. That variant pads only GEMM
    operands, then crops outputs/gradients while preserving every parameter and
    model dimension. TorchAO additionally requires the sequence dimension of
    inputs to be divisible by 128 in either mode.
    """
    from yylab.models.sprint import SprintForCausalLM

    if stochastic_rounding not in {"hardware", "portable", "portable_sr4"}:
        raise ValueError("stochastic_rounding must be 'hardware', 'portable' or 'portable_sr4'.")
    portable = stochastic_rounding in {"portable", "portable_sr4"}
    if not isinstance(pad_to_alignment, bool):
        raise ValueError("pad_to_alignment must be a boolean.")
    if pad_to_alignment and not portable:
        raise ValueError("Operand padding is supported only by the portable stochastic-rounding variant.")
    if not isinstance(model, SprintForCausalLM):
        raise TypeError("The all-GEMM NVFP4 benchmark requires a SprintForCausalLM.")
    if hasattr(model, "_torchao_nvfp4_training_benchmark_metadata"):
        metadata = model._torchao_nvfp4_training_benchmark_metadata
        if metadata["stochastic_rounding"] != stochastic_rounding:
            raise ValueError("Cannot change stochastic rounding on an already converted benchmark model.")
        if metadata["pad_to_alignment"] != pad_to_alignment:
            raise ValueError("Cannot change operand padding on an already converted benchmark model.")
        return copy.deepcopy(metadata)

    parameters = list(model.parameters())
    if any(parameter.dtype != torch.bfloat16 for parameter in parameters):
        raise ValueError("The all-GEMM NVFP4 benchmark requires BF16 parameters before conversion.")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1 or next(iter(devices)).type not in {"cpu", "cuda"}:
        raise ValueError("Place the complete BF16 model on one CPU or CUDA device before conversion.")
    device = next(iter(devices))
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 10:
        raise RuntimeError("The all-GEMM NVFP4 benchmark requires CUDA Blackwell (SM100+).")
    if device.type == "cuda" and stochastic_rounding == "hardware":
        _validate_hardware_rounding(torch.cuda.get_device_capability(device))

    targets = []
    padded_projections = []
    for index, block in enumerate(model.layers):
        for suffix, module in (
            ("self_attn.qkv_proj", block.self_attn.qkv_proj),
            ("self_attn.o_proj", block.self_attn.o_proj),
            ("up_proj", block.up_proj),
            ("down_proj", block.down_proj),
        ):
            name = f"layers.{index}.{suffix}"
            if type(module) is not nn.Linear:
                raise ValueError(f"All-GEMM NVFP4 requires an unconverted Sprint projection: {name}.")
            if module.in_features % 128 or module.out_features % 128:
                if not pad_to_alignment:
                    raise ValueError(
                        f"All-GEMM NVFP4 requires projection dimensions divisible by 128; "
                        f"{name} has in_features={module.in_features}, out_features={module.out_features}. "
                        "Choose one 128-aligned shape or explicitly enable portable operand padding."
                    )
                padded_projections.append({
                    "name": name,
                    "original_in_features": module.in_features,
                    "original_out_features": module.out_features,
                    "gemm_in_features": (module.in_features + 127) // 128 * 128,
                    "gemm_out_features": (module.out_features + 127) // 128 * 128,
                })
            targets.append((name, module))
    if not targets:
        raise ValueError("The all-GEMM NVFP4 benchmark found no Sprint decoder projections.")

    linear_cls, prepare = _load_provider()
    if stochastic_rounding == "portable":
        from .sprint_nvfp4_portable import portable_nvfp4_linear_class

        linear_cls = portable_nvfp4_linear_class()
    elif stochastic_rounding == "portable_sr4":
        from .sprint_nvfp4_portable import portable_nvfp4_sr4_linear_class

        linear_cls = portable_nvfp4_sr4_linear_class()
    # Validate and construct everything before replacing the original topology.
    replacements = [
        (name, linear_cls.from_linear(module).train(module.training))
        for name, module in targets
    ]
    for _, module in replacements:
        module._sprint_nvfp4_pad_to_alignment = pad_to_alignment
    if device.type == "cuda":
        prepare(device, sign_vectors=tuple(module.rht_sign_vector for _, module in replacements))

    parameter_ids = {id(parameter) for parameter in parameters}
    for name, module in replacements:
        model.set_submodule(name, module)
    if {id(parameter) for parameter in model.parameters()} != parameter_ids:
        raise RuntimeError("TorchAO NVFP4 conversion failed to preserve the original parameters.")

    metadata = {
        "precision": {"hardware": "nvfp4_all_gemm", "portable": "nvfp4_all_gemm_portable_sr",
                      "portable_sr4": "nvfp4_all_gemm_portable_sr4"}[stochastic_rounding],
        "experimental": True,
        "provider": {"hardware": "torchao_nvfp4_training_triton", "portable": "torchao_nvfp4_training_portable_sr",
                     "portable_sr4": "torchao_nvfp4_training_portable_sr4"}[stochastic_rounding],
        "stochastic_rounding": stochastic_rounding,
        "portable_cast_search": ({"tile": [128, 128], "num_stages": 2, "num_warps": [4, 8]}
                                 if portable else None),
        "stochastic_rounding_details": (
            "hardware cvt.rs instruction" if stochastic_rounding == "hardware" else (
                "software adjacent-level sampling with 24-bit per-element uniforms; four outputs per Philox4x32-10 call; different RNG layout from scalar portable SR and cvt.rs"
                if stochastic_rounding == "portable_sr4" else
                "software adjacent-level sampling with 24-bit per-element uniforms; different RNG layout from cvt.rs")
        ),
        "forward_gemm": "nvfp4",
        "backward_gemm": "nvfp4",
        "native_low_bit_gemm": True,
        "native_low_bit_backward": True,
        "parameter_dtype": "bfloat16",
        "training_state_is_low_bit": False,
        "gradient_recipe": "2D weight scaling; Hadamard transforms; stochastic NVFP4 gradient rounding",
        "saved_activations": "packed NVFP4 codes and scales",
        "scope": "decoder projections; embedding, output head, attention and loss retain BF16/FP32 math",
        "required_projection_alignment": 128,
        "required_sequence_alignment": 128,
        "pad_to_alignment": pad_to_alignment,
        "padded_projections": padded_projections,
        "converted_linear_count": len(replacements),
        "converted_modules": [name for name, _ in replacements],
        "parameters_before": sum(parameter.numel() for parameter in parameters),
        "parameters_after": sum(parameter.numel() for parameter in model.parameters()),
        "cuda_graph_workspace_prepared": device.type == "cuda",
    }
    if portable:
        metadata["source_provenance"] = copy.deepcopy(linear_cls._portable_provider_provenance)
    if stochastic_rounding == "portable_sr4":
        from .sprint_nvfp4_portable import portable_sr4_recipe

        metadata["stochastic_rounding_recipe"] = portable_sr4_recipe()
    model._torchao_nvfp4_training_benchmark_metadata = copy.deepcopy(metadata)
    return metadata
