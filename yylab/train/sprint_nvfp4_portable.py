"""Experimental portable stochastic rounding for TorchAO's NVFP4 recipe.

SM120 supports FP4 GEMMs and round-to-nearest conversion but lacks the cvt.rs
instruction used by the stock training kernel. This variant selects adjacent
E2M1 levels with distance-proportional probability using 24 random bits per
element. It retains stochastic rounding, but is not bitwise equivalent to the
hardware instruction's random-bit layout. Scaling, RHT and all GEMMs remain
the installed TorchAO implementation. The explicit SR4 variant consumes all
four Philox outputs per call; the original per-element random layout remains
available independently.

Private provider functions are cloned with isolated globals so neither the
installed package nor another model's stock provider is modified. This is
deliberately confined to disposable benchmarks and must pass a CUDA smoke on
the actual runtime before any throughput result is accepted.
"""

from __future__ import annotations

import functools
import hashlib
import types
from importlib.metadata import version
from pathlib import Path

import torch
import triton
import triton.language as tl
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    convert_8xfp32_to_4xfp4_packed,
)

_REVIEWED_PROVIDER_SHA256 = {
    "nvfp4_linear.py": "4e4c5f0fea8e68e72a99d0711183d2f83ab6a18d5a9399a16bf31cbe73f207ce",
    "hadamard_quantize_row_col_triton.py": "ec82549e5dd731fba6ab153325f90e898ba61f5d2c2a25096462d743f96feb8f",
    "hadamard_utils.py": "e51edf14e44ffdf08522e52e21a540fd04658502d9d7628afb0ae6b95f9f2573",
    "hadamard_amax_triton.py": "a7be8a00b853aa7eac88f0e3f0fc9c77e5a445df507dbdc41a881a0811a9275d",
    "quantize_2d_triton.py": "370e13a5658df1e475e5b4e1402fb31c4df5e9c61cd6e294daa55a5d3d392514",
    "nvfp4_training.py": "43e052f781c922ffc2f7c5d56fc1f68f06d2599565e0838d895e67ac48a53999",
}

_REVIEWED_SR4_CANDIDATE_SHA256 = "4d091890b0047859a56e2a60585325f338eb38a9fff76806c6b4f8466642354f"


def portable_sr4_recipe() -> dict:
    """Describe the explicit SR4 random layout, independent of model state."""
    return {
        "variant": "portable_sr4",
        "algorithm": "philox4x32_10_all_outputs",
        "philox_rounds": 10,
        "random_values_per_call": 4,
        "uniform_bits_per_element": 24,
        "random_layout": "counter=(call_offset_low32, unique_four_element_group_high32); output lanes 0,1,2,3 map to adjacent columns",
        "stream_separation": "unchanged caller seed and seed XOR 1, independent row/column offsets",
        "baseline_bitwise_equivalent": False,
        "hardware_bitwise_equivalent": False,
        "rounding": "distance-proportional sampling between adjacent E2M1 values",
        "reviewed_candidate_source_sha256": _REVIEWED_SR4_CANDIDATE_SHA256,
    }


def _provider_source_provenance() -> dict:
    """Refuse to clone prototype code whose semantics have not been reviewed."""
    from torchao.prototype.moe_training import nvfp4_training

    directory = Path(nvfp4_training.__file__).parent
    actual = {}
    for name, expected in _REVIEWED_PROVIDER_SHA256.items():
        try:
            actual[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(f"Cannot verify portable NVFP4 source provenance: {name}") from exc
        if actual[name] != expected:
            raise RuntimeError(
                f"Portable NVFP4 source guard rejected changed TorchAO file {name}: "
                f"expected SHA256 {expected}, found {actual[name]}. "
                "Review and validate the new prototype before updating the source manifest."
            )
    return {
        "reviewed_provider": "torchao_nvfp4_training_0.18.0",
        "torchao_version": version("torchao"),
        "torch_version": str(torch.__version__),
        "triton_version": triton.__version__,
        "upstream_source_sha256": actual,
        "portable_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def stochastic_e2m1_reference(values: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """CPU/GPU oracle: unpacked E2M1 codes with supplied uniform values in [0,1)."""
    if values.shape != uniform.shape:
        raise ValueError("values and uniform must have identical shapes")
    if not bool(torch.isfinite(uniform).all()) or not bool(((uniform >= 0) & (uniform < 1)).all()):
        raise ValueError("uniform values must be finite and in [0, 1)")
    values = values.float()
    magnitude = values.abs().nan_to_num(nan=6.0, posinf=6.0).clamp_max(6.0)
    lower = torch.where(magnitude < 2, (magnitude * 2).floor() * 0.5,
                        torch.where(magnitude < 4, magnitude.floor(), 4.0))
    spacing = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    lower_code = torch.where(magnitude < 2, (magnitude * 2).floor(),
                             torch.where(magnitude < 4, magnitude.floor() + 2, 6.0)).to(torch.uint8)
    code = lower_code + (uniform < (magnitude - lower) / spacing).to(torch.uint8)
    sign = (torch.signbit(values) & ~values.isnan()).to(torch.uint8) * 8
    return code | sign


@triton.jit
def _portable_e2m1_codes(values, uniform):
    magnitude = tl.minimum(tl.abs(values), 6.0)
    magnitude = tl.where(values != values, 6.0, magnitude)
    lower = tl.where(magnitude < 2.0, tl.floor(magnitude * 2.0) * 0.5,
                     tl.where(magnitude < 4.0, tl.floor(magnitude), 4.0))
    spacing = tl.where(magnitude < 2.0, 0.5, tl.where(magnitude < 4.0, 1.0, 2.0))
    lower_code = tl.where(magnitude < 2.0, tl.floor(magnitude * 2.0),
                          tl.where(magnitude < 4.0, tl.floor(magnitude) + 2.0, 6.0)).to(tl.uint8)
    code = lower_code + (uniform < (magnitude - lower) / spacing).to(tl.uint8)
    sign = ((values.to(tl.uint32, bitcast=True) >> 28) & 8).to(tl.uint8)
    sign = tl.where(values != values, 0, sign).to(tl.uint8)
    return code | sign


@triton.jit
def _portable_pack_fp4(
    scaled,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    STOCHASTIC_ROUNDING: tl.constexpr,
    seed_ptr,
    offset_base_ptr,
    tile_id,
):
    if STOCHASTIC_ROUNDING:
        seed = tl.load(seed_ptr)
        offset_base = tl.load(offset_base_ptr).to(tl.uint64) & 0xFFFFFFFF
        local_n = tl.arange(0, BLOCK_N)[:, None]
        local_m = tl.arange(0, BLOCK_M)[None, :]
        linear_idx = (tile_id.to(tl.int64) * (BLOCK_N * BLOCK_M)
                      + (local_n * BLOCK_M + local_m).to(tl.int64))
        offset = (linear_idx.to(tl.uint64) << 32) | offset_base
        random_bits = tl.randint(seed, offset).to(tl.uint32)
        uniform = (random_bits >> 8).to(tl.float32) * (1.0 / 16777216.0)
        codes = _portable_e2m1_codes(scaled, uniform)
        low, high = codes.reshape(BLOCK_N, BLOCK_M // 2, 2).split()
        return low | (high << 4)
    else:
        return convert_8xfp32_to_4xfp4_packed(scaled.reshape(BLOCK_N, BLOCK_M // 2, 2).split())


@triton.jit
def _sr4_random_bits(BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                     seed, offset_base, tile_id):
    tl.static_assert(BLOCK_M % 4 == 0)
    local_n = tl.arange(0, BLOCK_N)[:, None]
    local_g = tl.arange(0, BLOCK_M // 4)[None, :]
    group_idx = (tile_id.to(tl.int64) * (BLOCK_N * (BLOCK_M // 4))
                 + (local_n * (BLOCK_M // 4) + local_g).to(tl.int64))
    offset = (group_idx.to(tl.uint64) << 32) | (offset_base.to(tl.uint64) & 0xFFFFFFFF)
    r0, r1, r2, r3 = tl.randint4x(seed, offset, n_rounds=10)
    # Nested joins flatten to r0,r1,r2,r3 for four adjacent columns.
    bits = tl.join(tl.join(r0, r2), tl.join(r1, r3))
    return bits.reshape(BLOCK_N, BLOCK_M).to(tl.uint32)


@triton.jit
def _portable_pack_fp4_sr4(scaled, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                          STOCHASTIC_ROUNDING: tl.constexpr,
                          seed_ptr, offset_base_ptr, tile_id):
    if STOCHASTIC_ROUNDING:
        seed = tl.load(seed_ptr)
        offset_base = tl.load(offset_base_ptr)
        random_bits = _sr4_random_bits(BLOCK_N, BLOCK_M, seed, offset_base, tile_id)
        uniform = (random_bits >> 8).to(tl.float32) * (1.0 / 16777216.0)
        codes = _portable_e2m1_codes(scaled, uniform)
        low, high = codes.reshape(BLOCK_N, BLOCK_M // 2, 2).split()
        return low | (high << 4)
    else:
        return convert_8xfp32_to_4xfp4_packed(scaled.reshape(BLOCK_N, BLOCK_M // 2, 2).split())


def _clone_function(function, **global_replacements):
    namespace = dict(function.__globals__)
    namespace.update(global_replacements)
    result = types.FunctionType(function.__code__, namespace, function.__name__,
                                function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = dict(function.__annotations__)
    return result


@functools.lru_cache(maxsize=1)
def portable_nvfp4_linear_class():
    """Create an isolated TorchAO module variant; imports/registration only."""
    return _build_portable_nvfp4_linear_class(sr4=False)


@functools.lru_cache(maxsize=1)
def portable_nvfp4_sr4_linear_class():
    """Create an independent SR4 variant without changing scalar-SR providers."""
    return _build_portable_nvfp4_linear_class(sr4=True)


def _build_portable_nvfp4_linear_class(*, sr4: bool):
    provenance = _provider_source_provenance()
    if sr4:
        provenance["stochastic_rounding_recipe"] = portable_sr4_recipe()
    from torchao.prototype.moe_training.nvfp4_training import (
        hadamard_quantize_row_col_triton as provider_quant,
    )
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import nvfp4_mm_triton
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import NVFP4Linear

    try:
        original_kernel = provider_quant._hadamard_quantize_row_col_kernel
        pack = _portable_pack_fp4_sr4 if sr4 else _portable_pack_fp4
        kernel_fn = _clone_function(original_kernel.fn.fn, _pack_fp4=pack)
        # Software per-element Philox substantially increases register pressure
        # versus cvt.rs. Keep the portable experiment's startup search bounded;
        # larger/high-stage provider tiles compile very slowly on SM120.
        configurations = [config for config in original_kernel.configs
                          if config.kwargs.get("BLOCK_M") == 128 and config.kwargs.get("BLOCK_N") == 128
                          and config.num_stages == 2 and config.num_warps in {4, 8}]
        if len(configurations) != 2:
            raise RuntimeError("Installed TorchAO lacks the reviewed portable NVFP4 tile configurations.")
        kernel = triton.autotune(configs=configurations,
                                key=["M", "N", "STOCHASTIC_ROUNDING"])(triton.jit(kernel_fn))
        original_op = provider_quant.triton_rht_quantize_row_col
        operation_fn = _clone_function(original_op._init_fn, _hadamard_quantize_row_col_kernel=kernel)
    except AttributeError as exc:
        raise RuntimeError("Installed TorchAO's NVFP4 kernel API is incompatible with portable SR.") from exc

    operation_name = "yylab::nvfp4_portable_sr4_quantize" if sr4 else "yylab::nvfp4_portable_sr_quantize"
    operation = torch.library.custom_op(operation_name, mutates_args=())(operation_fn)

    @operation.register_fake
    def _fake(A, col_global_amax, row_global_amax, sign_vector, stochastic_rounding=False,
              hadamard_dimension=16, scaling_type=0, col_seed_base=None, col_offset_base=None,
              row_offset_base=None, row_seed_base=None):
        m, n = A.shape
        return (
            A.new_empty((n, m // 2), dtype=torch.uint8),
            A.new_empty((n // 128, m // 64, 32, 16), dtype=torch.float8_e4m3fn),
            A.new_empty((m, n // 2), dtype=torch.uint8),
            A.new_empty((m // 128, n // 64, 32, 16), dtype=torch.float8_e4m3fn),
        )

    replacements = {"triton_rht_quantize_row_col": operation}
    function_name = "PortableSR4NVFP4Function" if sr4 else "PortableSRNVFP4Function"
    function_cls = type(function_name, (torch.autograd.Function,), {
        "forward": staticmethod(_clone_function(nvfp4_mm_triton.forward, **replacements)),
        "backward": staticmethod(_clone_function(nvfp4_mm_triton.backward, **replacements)),
    })
    torch._dynamo.allow_in_graph(function_cls)

    class PortableSRNVFP4Linear(NVFP4Linear):
        _portable_provider_provenance = provenance

        def forward(self, inputs):
            if self.process_group is not None:
                raise RuntimeError("Portable NVFP4 SR is only validated for single-device benchmarks.")
            weight, bias = self.weight, self.bias
            if getattr(self, "_sprint_nvfp4_pad_to_alignment", False):
                pad_k = (-self.in_features) % 128
                pad_n = (-self.out_features) % 128
                if pad_k:
                    inputs = torch.nn.functional.pad(inputs, (0, pad_k))
                if pad_k or pad_n:
                    weight = torch.nn.functional.pad(weight, (0, pad_k, 0, pad_n))
                if bias is not None and pad_n:
                    bias = torch.nn.functional.pad(bias, (0, pad_n))
            output = function_cls.apply(inputs, weight, bias, self._sr_seed, self.rht_sign_vector)
            return output[..., :self.out_features].contiguous()

    if sr4:
        PortableSRNVFP4Linear.__name__ = "PortableSR4NVFP4Linear"
    return PortableSRNVFP4Linear
