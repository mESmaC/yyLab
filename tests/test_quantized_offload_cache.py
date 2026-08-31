from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from yylab.infer.quantized_offload_cache import (
    QuantizedOffloadedCache,
    QuantizedOffloadedLayer,
    dequantize_kv_tensor,
    quantize_kv_tensor,
    quantized_offload_cache_availability,
)


@pytest.mark.parametrize(
    ("nbits", "data_dtype", "max_error"),
    [(8, torch.int8, 0.01), (4, torch.uint8, 0.08)],
)
def test_quantize_roundtrip_is_cpu_backed_and_packs_int4(
    nbits: int, data_dtype: torch.dtype, max_error: float
) -> None:
    tensor = torch.linspace(-1.0, 1.0, 2 * 3 * 5 * 17, dtype=torch.float32).reshape(2, 3, 5, 17)

    block = quantize_kv_tensor(
        tensor,
        nbits=nbits,
        group_size=8,
        scale_dtype=torch.float32,
    )
    restored = dequantize_kv_tensor(block, device="cpu", dtype=torch.float32)

    assert block.data.device.type == "cpu"
    assert block.scales.device.type == "cpu"
    assert block.data.dtype == data_dtype
    assert block.scales.shape == (2, 3, 5, 3)
    assert block.data.shape == (2, 3, 5, 12 if nbits == 4 else 24)
    assert restored.shape == tensor.shape
    torch.testing.assert_close(restored, tensor, atol=max_error, rtol=0)


@pytest.mark.parametrize("nbits", [4, 8])
def test_zero_tensor_roundtrips_exactly(nbits: int) -> None:
    tensor = torch.zeros((1, 2, 3, 9), dtype=torch.bfloat16)
    block = quantize_kv_tensor(tensor, nbits=nbits, group_size=5)

    restored = dequantize_kv_tensor(block)

    assert restored.dtype == torch.bfloat16
    assert torch.count_nonzero(restored) == 0


@pytest.mark.parametrize("nbits", [4, 8])
def test_layer_keeps_recent_residual_and_returns_current_states_unquantized(nbits: int) -> None:
    torch.manual_seed(7)
    keys = torch.randn((2, 3, 5, 17), dtype=torch.float32)
    values = torch.randn_like(keys)
    layer = QuantizedOffloadedLayer(
        nbits=nbits,
        q_group_size=8,
        residual_length=2,
        scale_dtype=torch.float32,
        pin_memory=False,
        growth_chunk_size=4,
    )

    returned_keys, returned_values = layer.update(keys, values)

    # The current prefill must not be perturbed merely because it is cached.
    assert torch.equal(returned_keys, keys)
    assert torch.equal(returned_values, values)
    assert layer.get_seq_length() == 5
    assert layer.get_mask_sizes(3) == (8, 0)
    assert layer.get_max_length() == -1
    assert layer.quantized_length == 3
    assert layer.resident_length == 2
    assert layer.quantized_keys.data.device.type == "cpu"
    assert layer.quantized_values.data.device.type == "cpu"

    next_keys = torch.randn((2, 3, 1, 17), dtype=torch.float32)
    next_values = torch.randn_like(next_keys)
    all_keys, all_values = layer.update(next_keys, next_values)

    assert all_keys.shape[-2] == 6
    assert all_values.shape[-2] == 6
    assert layer.quantized_length == 4
    assert layer.resident_length == 2
    # The full-precision residual and current token remain bit-identical.
    assert torch.equal(all_keys[..., -3:, :], torch.cat((keys[..., -2:, :], next_keys), dim=-2))
    assert torch.equal(all_values[..., -3:, :], torch.cat((values[..., -2:, :], next_values), dim=-2))


def test_layer_materialize_crop_reset_and_batch_operations() -> None:
    torch.manual_seed(11)
    keys = torch.randn((2, 2, 6, 8), dtype=torch.float32)
    layer = QuantizedOffloadedLayer(
        nbits=8,
        q_group_size=4,
        residual_length=2,
        scale_dtype=torch.float32,
        pin_memory=False,
        growth_chunk_size=2,
    )
    layer.update(keys, keys + 10)
    before_keys, before_values = layer.materialize()

    layer.reorder_cache(torch.tensor([1, 0]))
    reordered_keys, reordered_values = layer.materialize()
    torch.testing.assert_close(reordered_keys, before_keys[[1, 0]], atol=0, rtol=0)
    torch.testing.assert_close(reordered_values, before_values[[1, 0]], atol=0, rtol=0)

    layer.batch_select_indices(torch.tensor([0]))
    assert layer.batch_size == 1
    layer.batch_repeat_interleave(3)
    assert layer.batch_size == 3
    repeated_keys, _ = layer.materialize()
    torch.testing.assert_close(repeated_keys[0], repeated_keys[1], atol=0, rtol=0)

    layer.crop(-2)
    assert layer.get_seq_length() == 4
    assert layer.quantized_length == 4
    assert layer.resident_length == 0
    layer.crop(3)  # Transformers' legacy positive form means final absolute length.
    assert layer.get_seq_length() == 3
    layer.crop(99)
    assert layer.get_seq_length() == 3

    layer.reset()
    assert layer.get_seq_length() == 0
    assert layer.quantized_length == 0
    assert layer.resident_length == 0


def test_cache_matches_transformers_contract_and_updates_independent_layers() -> None:
    config = SimpleNamespace(num_hidden_layers=2, layer_types=["full_attention", "full_attention"])
    cache = QuantizedOffloadedCache(
        config,
        nbits=4,
        q_group_size=4,
        residual_length=0,
        scale_dtype=torch.float32,
        pin_memory=False,
        growth_chunk_size=2,
    )

    assert isinstance(cache, Cache)
    assert all(isinstance(layer, CacheLayerMixin) for layer in cache.layers)
    states = torch.randn((1, 2, 3, 8), dtype=torch.float32)
    returned_keys, returned_values = cache.update(states, states + 1, layer_idx=1)

    assert torch.equal(returned_keys, states)
    assert torch.equal(returned_values, states + 1)
    assert cache.get_seq_length(0) == 0
    assert cache.get_seq_length(1) == 3
    assert cache.get_mask_sizes(1, 1) == (4, 0)
    assert cache.quantized_storage_bytes > 0
    assert cache.layers[1].resident_length == 0


def test_tiny_qwen_forward_accepts_cache_instance() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    model = Qwen3ForCausalLM(config).eval()
    cache = QuantizedOffloadedCache(
        config,
        nbits=4,
        q_group_size=4,
        residual_length=1,
        scale_dtype=torch.float32,
        pin_memory=False,
        growth_chunk_size=2,
    )

    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[1, 2, 3]]),
            past_key_values=cache,
            use_cache=True,
        )

    assert output.logits.shape == (1, 3, 64)
    assert cache.get_seq_length() == 3
    assert all(layer.quantized_length == 2 for layer in cache.layers)


def test_clear_validation_and_availability_messages() -> None:
    available, reason = quantized_offload_cache_availability()
    assert available
    assert "int8" in reason

    with pytest.raises(ValueError, match="nbits"):
        quantize_kv_tensor(torch.ones(1, 1, 1, 4), nbits=2)
    with pytest.raises(ValueError, match="group_size"):
        quantize_kv_tensor(torch.ones(1, 1, 1, 4), group_size=0)
    with pytest.raises(TypeError, match="floating point"):
        quantize_kv_tensor(torch.ones(1, 1, 1, 4, dtype=torch.int64))
    with pytest.raises(ValueError, match="provide config"):
        QuantizedOffloadedCache(pin_memory=False)
    hybrid = SimpleNamespace(num_hidden_layers=2, layer_types=["full_attention", "sliding_attention"])
    with pytest.raises(ValueError, match="only full-attention"):
        QuantizedOffloadedCache(hybrid, pin_memory=False)
