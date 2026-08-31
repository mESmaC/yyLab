"""CPU-backed quantized KV cache for Transformers generation.

Transformers exposes quantized and offloaded caches as separate implementations.
This module combines the useful properties of both: inactive layers are retained
on CPU as int8 or packed int4, and only the active layer is transferred and
dequantized on the model device.  A small, recent residual can optionally remain
in the model dtype to avoid repeatedly quantizing the newest tokens.

The implementation deliberately has no third-party quantizer dependency.  It is
an inference cache and is not intended for training with autograd enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import Cache, CacheLayerMixin, get_layer_types_and_kwargs

_SUPPORTED_NBITS = (4, 8)


@dataclass(frozen=True, slots=True)
class QuantizedKVBlock:
    """A group-wise symmetric quantized tensor stored on CPU.

    For int4, two signed values are encoded in each uint8 byte.  ``scales`` has
    one entry per group on the tensor's final dimension.  The sequence dimension
    remains ``-2``, matching the Transformers KV layout ``[B, H, S, D]``.
    """

    data: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype
    nbits: int
    group_size: int

    @property
    def sequence_length(self) -> int:
        return self.original_shape[-2]

    @property
    def storage_bytes(self) -> int:
        return self.data.numel() * self.data.element_size() + self.scales.numel() * self.scales.element_size()


def quantized_offload_cache_availability() -> tuple[bool, str]:
    """Return whether the local Transformers cache contract is compatible.

    The quantizer itself is pure PyTorch and works on CPU.  Pinned transfers are
    opportunistic and therefore are not an availability requirement.
    """

    required_layer_methods = ("update", "get_mask_sizes", "get_seq_length", "get_max_length")
    missing = [name for name in required_layer_methods if not hasattr(CacheLayerMixin, name)]
    if not hasattr(Cache, "update"):
        missing.append("Cache.update")
    if missing:
        return False, "Transformers Cache API is missing: " + ", ".join(missing)
    return True, "available (pure PyTorch int8/packed-int4 CPU storage)"


def _validate_quantization_args(nbits: int, group_size: int, scale_dtype: torch.dtype) -> None:
    if nbits not in _SUPPORTED_NBITS:
        raise ValueError(f"nbits must be one of {_SUPPORTED_NBITS}, got {nbits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if scale_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("scale_dtype must be float16, bfloat16, or float32")


def _copy_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    # A synchronous copy makes ownership/lifetime explicit.  The much larger
    # CPU->accelerator read is allowed to be asynchronous when pinned storage is
    # available (see ``dequantize_kv_tensor``).
    return tensor.detach().to(device="cpu", non_blocking=False).contiguous()


def quantize_kv_tensor(
    tensor: torch.Tensor,
    *,
    nbits: int = 4,
    group_size: int = 64,
    scale_dtype: torch.dtype = torch.float16,
) -> QuantizedKVBlock:
    """Quantize a KV tensor group-wise along its head dimension.

    Quantization arithmetic runs on ``tensor.device`` so a CUDA caller sends
    quantized bytes, rather than full precision values, across PCIe.  Returned
    data and scales are always ordinary CPU tensors.
    """

    _validate_quantization_args(nbits, group_size, scale_dtype)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"KV tensors must be floating point, got {tensor.dtype}")
    if tensor.ndim < 2:
        raise ValueError("KV tensors need at least sequence and head dimensions")

    original_shape = tuple(tensor.shape)
    head_dim = original_shape[-1]
    if head_dim <= 0:
        raise ValueError("KV head dimension must be non-zero")

    groups = (head_dim + group_size - 1) // group_size
    padded_head_dim = groups * group_size
    work = tensor.detach().to(torch.float32)
    if padded_head_dim != head_dim:
        work = torch.nn.functional.pad(work, (0, padded_head_dim - head_dim))
    grouped = work.reshape(*original_shape[:-1], groups, group_size)

    qmax = 127 if nbits == 8 else 7
    absmax = grouped.abs().amax(dim=-1)
    scales = absmax / qmax
    scales = torch.where(absmax > 0, scales, torch.ones_like(scales))
    quantized = torch.round(grouped / scales.unsqueeze(-1)).clamp(-qmax, qmax).to(torch.int8)
    quantized = quantized.reshape(*original_shape[:-1], padded_head_dim)

    if nbits == 8:
        data = quantized
    else:
        # Offset signed [-7, 7] values to [1, 15].  Code 0 is intentionally
        # unused, which keeps zero exactly representable as nibble value 8.
        encoded = (quantized.to(torch.int16) + 8).to(torch.uint8)
        if encoded.shape[-1] % 2:
            encoded = torch.nn.functional.pad(encoded, (0, 1), value=8)
        data = encoded[..., 0::2] | (encoded[..., 1::2] << 4)

    return QuantizedKVBlock(
        data=_copy_to_cpu(data),
        scales=_copy_to_cpu(scales.to(scale_dtype)),
        original_shape=original_shape,
        original_dtype=tensor.dtype,
        nbits=nbits,
        group_size=group_size,
    )


def dequantize_kv_tensor(
    block: QuantizedKVBlock,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    non_blocking: bool = True,
) -> torch.Tensor:
    """Transfer a quantized CPU block and dequantize it on ``device``.

    No full-precision CPU staging tensor is created.  With pinned cache storage,
    the byte/scales transfer can overlap preceding accelerator work on the
    current stream.
    """

    if not isinstance(block, QuantizedKVBlock):
        raise TypeError("block must be a QuantizedKVBlock")
    _validate_quantization_args(block.nbits, block.group_size, block.scales.dtype)
    if block.data.device.type != "cpu" or block.scales.device.type != "cpu":
        raise ValueError("quantized KV blocks must be CPU-backed")
    if len(block.original_shape) < 2 or block.original_shape[-1] <= 0:
        raise ValueError("quantized KV block has an invalid original shape")

    target_device = torch.device("cpu" if device is None else device)
    target_dtype = block.original_dtype if dtype is None else dtype
    if not target_dtype.is_floating_point:
        raise TypeError(f"dequantization dtype must be floating point, got {target_dtype}")

    data = block.data.to(target_device, non_blocking=non_blocking)
    scales = block.scales.to(target_device, dtype=torch.float32, non_blocking=non_blocking)
    groups = scales.shape[-1]
    padded_head_dim = groups * block.group_size

    if block.nbits == 8:
        unpacked = data.to(torch.int16)
    else:
        low = (data & 0x0F).to(torch.int16) - 8
        high = (data >> 4).to(torch.int16) - 8
        unpacked = torch.stack((low, high), dim=-1).flatten(-2)

    unpacked = unpacked[..., :padded_head_dim]
    grouped = unpacked.reshape(*block.original_shape[:-1], groups, block.group_size)
    restored = grouped.to(torch.float32) * scales.unsqueeze(-1)
    restored = restored.flatten(-2)[..., : block.original_shape[-1]]
    return restored.to(target_dtype).reshape(block.original_shape)


class _QuantizedSequence:
    """Amortized CPU storage for quantized blocks along the sequence axis."""

    def __init__(self, *, pin_memory: bool, growth_chunk_size: int) -> None:
        self.pin_memory = pin_memory
        self.growth_chunk_size = growth_chunk_size
        self.data: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.length = 0
        self.capacity = 0
        self.original_shape: tuple[int, ...] | None = None
        self.original_dtype: torch.dtype | None = None
        self.nbits: int | None = None
        self.group_size: int | None = None

    @staticmethod
    def _with_seq_length(shape: tuple[int, ...], length: int) -> tuple[int, ...]:
        mutable = list(shape)
        mutable[-2] = length
        return tuple(mutable)

    def _allocate(self, shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=self.pin_memory)

    def _reserve(self, required: int, block: QuantizedKVBlock) -> None:
        if required <= self.capacity:
            return
        rounded = ((required + self.growth_chunk_size - 1) // self.growth_chunk_size) * self.growth_chunk_size
        new_capacity = max(rounded, max(self.capacity * 2, self.growth_chunk_size))
        data_shape = self._with_seq_length(tuple(block.data.shape), new_capacity)
        scale_shape = self._with_seq_length(tuple(block.scales.shape), new_capacity)
        new_data = self._allocate(data_shape, dtype=block.data.dtype)
        new_scales = self._allocate(scale_shape, dtype=block.scales.dtype)
        if self.length:
            new_data[..., : self.length, :].copy_(self.data[..., : self.length, :])
            new_scales[..., : self.length, :].copy_(self.scales[..., : self.length, :])
        self.data, self.scales, self.capacity = new_data, new_scales, new_capacity

    def append(self, block: QuantizedKVBlock) -> None:
        if block.data.device.type != "cpu" or block.scales.device.type != "cpu":
            raise ValueError("quantized sequence accepts CPU-backed blocks only")
        if self.original_shape is None:
            self.original_shape = block.original_shape
            self.original_dtype = block.original_dtype
            self.nbits = block.nbits
            self.group_size = block.group_size
        else:
            expected = list(self.original_shape)
            incoming = list(block.original_shape)
            expected[-2] = incoming[-2]
            if tuple(expected) != tuple(incoming):
                raise ValueError(
                    f"incompatible KV shape: expected {tuple(expected)}, got {block.original_shape}"
                )
            if (self.nbits, self.group_size, self.original_dtype) != (
                block.nbits,
                block.group_size,
                block.original_dtype,
            ):
                raise ValueError("quantized KV format changed within one cache layer")

        incoming_length = block.sequence_length
        required = self.length + incoming_length
        self._reserve(required, block)
        if incoming_length:
            target = slice(self.length, required)
            self.data[..., target, :].copy_(block.data)
            self.scales[..., target, :].copy_(block.scales)
        self.length = required

    def as_block(self) -> QuantizedKVBlock | None:
        if self.length == 0:
            return None
        shape = self._with_seq_length(self.original_shape, self.length)
        return QuantizedKVBlock(
            data=self.data[..., : self.length, :],
            scales=self.scales[..., : self.length, :],
            original_shape=shape,
            original_dtype=self.original_dtype,
            nbits=self.nbits,
            group_size=self.group_size,
        )

    def dequantize(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        block = self.as_block()
        if block is None:
            return None
        return dequantize_kv_tensor(block, device=device, dtype=dtype, non_blocking=self.pin_memory)

    def crop(self, length: int) -> None:
        if not 0 <= length <= self.length:
            raise ValueError(f"crop length must be in [0, {self.length}], got {length}")
        self.length = length

    def reset(self) -> None:
        self.length = 0

    def _replace_batch(self, data: torch.Tensor, scales: torch.Tensor) -> None:
        data_shape = self._with_seq_length(tuple(data.shape), self.capacity)
        scale_shape = self._with_seq_length(tuple(scales.shape), self.capacity)
        new_data = self._allocate(data_shape, dtype=data.dtype)
        new_scales = self._allocate(scale_shape, dtype=scales.dtype)
        if self.length:
            new_data[..., : self.length, :].copy_(data)
            new_scales[..., : self.length, :].copy_(scales)
        self.data, self.scales = new_data, new_scales
        shape = list(self.original_shape)
        shape[0] = data.shape[0]
        self.original_shape = tuple(shape)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if not self.length:
            return
        cpu_idx = beam_idx.to(device="cpu", dtype=torch.long)
        data = self.data[..., : self.length, :].index_select(0, cpu_idx)
        scales = self.scales[..., : self.length, :].index_select(0, cpu_idx)
        self._replace_batch(data, scales)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if not self.length:
            return
        data = self.data[..., : self.length, :].repeat_interleave(repeats, dim=0)
        scales = self.scales[..., : self.length, :].repeat_interleave(repeats, dim=0)
        self._replace_batch(data, scales)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self.reorder_cache(indices)

    @property
    def logical_storage_bytes(self) -> int:
        if not self.length:
            return 0
        return (
            self.data[..., : self.length, :].numel() * self.data.element_size()
            + self.scales[..., : self.length, :].numel() * self.scales.element_size()
        )


class QuantizedOffloadedLayer(CacheLayerMixin):
    """One dynamic KV layer with quantized history on CPU."""

    is_compileable = False
    is_sliding = False

    def __init__(
        self,
        *,
        nbits: int = 4,
        q_group_size: int = 64,
        residual_length: int = 128,
        scale_dtype: torch.dtype = torch.float16,
        pin_memory: bool = True,
        growth_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        _validate_quantization_args(nbits, q_group_size, scale_dtype)
        if residual_length < 0:
            raise ValueError(f"residual_length must be non-negative, got {residual_length}")
        if growth_chunk_size <= 0:
            raise ValueError(f"growth_chunk_size must be positive, got {growth_chunk_size}")

        self.nbits = nbits
        self.q_group_size = q_group_size
        self.residual_capacity = residual_length
        self.scale_dtype = scale_dtype
        self.pin_memory_requested = pin_memory
        self.pin_memory = bool(pin_memory and torch.cuda.is_available())
        self.growth_chunk_size = growth_chunk_size
        self.cumulative_length = 0
        self._quantized_keys = _QuantizedSequence(
            pin_memory=self.pin_memory, growth_chunk_size=growth_chunk_size
        )
        self._quantized_values = _QuantizedSequence(
            pin_memory=self.pin_memory, growth_chunk_size=growth_chunk_size
        )

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.batch_size = key_states.shape[0]
        empty_shape = list(key_states.shape)
        empty_shape[-2] = 0
        self.keys = torch.empty(empty_shape, dtype=key_states.dtype, device=key_states.device)
        value_empty_shape = list(value_states.shape)
        value_empty_shape[-2] = 0
        self.values = torch.empty(value_empty_shape, dtype=value_states.dtype, device=value_states.device)
        self.is_initialized = True

    @staticmethod
    def _validate_update(key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if key_states.ndim < 2 or value_states.ndim < 2:
            raise ValueError("KV states need at least sequence and head dimensions")
        if key_states.shape != value_states.shape:
            raise ValueError(
                f"key and value cache shapes must match, got {key_states.shape} and {value_states.shape}"
            )
        if key_states.device != value_states.device:
            raise ValueError("key and value cache tensors must be on the same device")
        if key_states.dtype != value_states.dtype or not key_states.is_floating_point():
            raise TypeError("key and value cache tensors must use the same floating-point dtype")

    def _spill(self) -> None:
        resident_length = self.keys.shape[-2]
        overflow = max(0, resident_length - self.residual_capacity)
        if overflow == 0:
            return
        key_prefix = self.keys[..., :overflow, :].contiguous()
        value_prefix = self.values[..., :overflow, :].contiguous()
        self._quantized_keys.append(
            quantize_kv_tensor(
                key_prefix,
                nbits=self.nbits,
                group_size=self.q_group_size,
                scale_dtype=self.scale_dtype,
            )
        )
        self._quantized_values.append(
            quantize_kv_tensor(
                value_prefix,
                nbits=self.nbits,
                group_size=self.q_group_size,
                scale_dtype=self.scale_dtype,
            )
        )
        self.keys = self.keys[..., overflow:, :]
        self.values = self.values[..., overflow:, :]

    def materialize(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize this layer on a target device without changing storage."""

        if not self.is_initialized:
            raise RuntimeError("cannot materialize an uninitialized KV cache layer")
        target_device = self.device if device is None else torch.device(device)
        target_dtype = self.dtype if dtype is None else dtype
        quantized_keys = self._quantized_keys.dequantize(device=target_device, dtype=target_dtype)
        quantized_values = self._quantized_values.dequantize(device=target_device, dtype=target_dtype)
        residual_keys = self.keys.to(target_device, dtype=target_dtype, non_blocking=self.pin_memory)
        residual_values = self.values.to(target_device, dtype=target_dtype, non_blocking=self.pin_memory)
        if quantized_keys is None:
            return residual_keys, residual_values
        if residual_keys.shape[-2] == 0:
            return quantized_keys, quantized_values
        return (
            torch.cat((quantized_keys, residual_keys), dim=-2),
            torch.cat((quantized_values, residual_values), dim=-2),
        )

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        self._validate_update(key_states, value_states)
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        elif key_states.shape[:-2] + key_states.shape[-1:] != self.keys.shape[:-2] + self.keys.shape[-1:]:
            raise ValueError("KV batch/head/head-dimension shape changed within one cache layer")

        # Explicitly restore an offloaded residual to the active layer device.
        self.device = key_states.device
        self.dtype = key_states.dtype
        self.keys = self.keys.to(key_states.device, dtype=key_states.dtype, non_blocking=self.pin_memory)
        self.values = self.values.to(
            value_states.device, dtype=value_states.dtype, non_blocking=self.pin_memory
        )

        # Materialize the *previous* cache, then append the current states at
        # their original precision for this attention call.  Quantizing before
        # returning would unnecessarily perturb tokens that are still part of
        # the current prefill.  They become quantized storage only for later
        # calls, matching Transformers' QuantizedLayer semantics.
        previous_keys, previous_values = self.materialize(device=key_states.device, dtype=key_states.dtype)
        keys_to_return = torch.cat((previous_keys, key_states), dim=-2)
        values_to_return = torch.cat((previous_values, value_states), dim=-2)
        self.keys = torch.cat((self.keys, key_states.detach()), dim=-2)
        self.values = torch.cat((self.values, value_states.detach()), dim=-2)
        self.cumulative_length += key_states.shape[-2]
        self._spill()
        return keys_to_return, values_to_return

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_length(self) -> int:
        return -1

    @property
    def quantized_length(self) -> int:
        return self._quantized_keys.length

    @property
    def resident_length(self) -> int:
        return 0 if not self.is_initialized else self.keys.shape[-2]

    @property
    def quantized_storage_bytes(self) -> int:
        return self._quantized_keys.logical_storage_bytes + self._quantized_values.logical_storage_bytes

    @property
    def quantized_keys(self) -> QuantizedKVBlock | None:
        return self._quantized_keys.as_block()

    @property
    def quantized_values(self) -> QuantizedKVBlock | None:
        return self._quantized_values.as_block()

    def offload(self) -> None:
        """Move the optional full-precision residual to CPU; history is already there."""

        if self.is_initialized:
            self.keys = self.keys.to("cpu", non_blocking=False)
            self.values = self.values.to("cpu", non_blocking=False)

    def prefetch(self) -> None:
        """Restore only the optional residual; history stays packed on CPU."""

        if self.is_initialized and self.keys.device != self.device:
            self.keys = self.keys.to(self.device, non_blocking=self.pin_memory)
            self.values = self.values.to(self.device, non_blocking=self.pin_memory)

    def reset(self) -> None:
        self._quantized_keys.reset()
        self._quantized_values.reset()
        self.cumulative_length = 0
        if self.is_initialized:
            self.keys = self.keys[..., :0, :]
            self.values = self.values[..., :0, :]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if not self.cumulative_length:
            return
        self._quantized_keys.reorder_cache(beam_idx)
        self._quantized_values.reorder_cache(beam_idx)
        self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
        self.values = self.values.index_select(0, beam_idx.to(self.values.device))
        self.batch_size = beam_idx.numel()

    def crop(self, tokens_to_remove: int) -> None:
        current = self.get_seq_length()
        if tokens_to_remove > 0:
            if tokens_to_remove >= current:
                return
            remove = current - tokens_to_remove
        else:
            remove = abs(tokens_to_remove)
        if remove == 0:
            return
        target = max(0, current - remove)
        quantized_target = min(target, self.quantized_length)
        residual_target = max(0, target - quantized_target)
        self._quantized_keys.crop(quantized_target)
        self._quantized_values.crop(quantized_target)
        self.keys = self.keys[..., :residual_target, :]
        self.values = self.values[..., :residual_target, :]
        self.cumulative_length = target

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError(f"repeats must be positive, got {repeats}")
        if not self.cumulative_length:
            return
        self._quantized_keys.batch_repeat_interleave(repeats)
        self._quantized_values.batch_repeat_interleave(repeats)
        self.keys = self.keys.repeat_interleave(repeats, dim=0)
        self.values = self.values.repeat_interleave(repeats, dim=0)
        self.batch_size *= repeats

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self.reorder_cache(indices)


class QuantizedOffloadedCache(Cache):
    """Transformers ``Cache`` with packed, CPU-backed full-attention layers.

    Pass an instance as ``past_key_values`` to ``model.forward`` or
    ``model.generate``.  This cache currently supports decoder-only models whose
    cache consists entirely of full-attention layers.
    """

    def __init__(
        self,
        config: Any | None = None,
        *,
        num_hidden_layers: int | None = None,
        nbits: int = 4,
        q_group_size: int = 64,
        residual_length: int = 128,
        scale_dtype: torch.dtype = torch.float16,
        pin_memory: bool = True,
        growth_chunk_size: int = 256,
    ) -> None:
        available, reason = quantized_offload_cache_availability()
        if not available:
            raise RuntimeError(reason)
        if config is None and num_hidden_layers is None:
            raise ValueError("provide config or num_hidden_layers")

        if config is not None:
            text_config = (
                config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
            )
            config_layers = getattr(text_config, "num_hidden_layers", None)
            if config_layers is None:
                raise ValueError("config does not define num_hidden_layers")
            if num_hidden_layers is not None and num_hidden_layers != config_layers:
                raise ValueError(
                    f"num_hidden_layers={num_hidden_layers} does not match config value {config_layers}"
                )
            num_hidden_layers = config_layers
            declared_layer_types = getattr(text_config, "layer_types", None)
            declared_invalid = (
                sorted(set(declared_layer_types) - {"full_attention"})
                if declared_layer_types is not None
                else []
            )
            if declared_invalid:
                raise ValueError(
                    "QuantizedOffloadedCache currently supports only full-attention layers; "
                    f"unsupported layer types: {declared_invalid}"
                )
            layer_types, _ = get_layer_types_and_kwargs(text_config)
            invalid = sorted(set(layer_types) - {"full_attention"})
            if invalid:
                raise ValueError(
                    "QuantizedOffloadedCache currently supports only full-attention layers; "
                    f"unsupported layer types: {invalid}"
                )

        if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be a positive integer, got {num_hidden_layers!r}")

        layers = [
            QuantizedOffloadedLayer(
                nbits=nbits,
                q_group_size=q_group_size,
                residual_length=residual_length,
                scale_dtype=scale_dtype,
                pin_memory=pin_memory,
                growth_chunk_size=growth_chunk_size,
            )
            for _ in range(num_hidden_layers)
        ]
        # Do not enable Cache's ordinary offloading machinery: these layers own
        # quantized CPU residency and materialize only inside their update path.
        super().__init__(layers=layers, offloading=False)

        self.nbits = nbits
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.scale_dtype = scale_dtype
        self.pin_memory = layers[0].pin_memory

    @property
    def quantized_storage_bytes(self) -> int:
        return sum(layer.quantized_storage_bytes for layer in self.layers)


__all__ = [
    "QuantizedKVBlock",
    "QuantizedOffloadedCache",
    "QuantizedOffloadedLayer",
    "dequantize_kv_tensor",
    "quantize_kv_tensor",
    "quantized_offload_cache_availability",
]
