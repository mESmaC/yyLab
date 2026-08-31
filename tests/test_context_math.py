from __future__ import annotations

from types import SimpleNamespace

from yylab.infer.context_math import (
    capacity_rows,
    kv_bytes_per_token,
    kv_gb,
    kv_shape,
    max_context_for_budget,
    model_weight_gb,
    quantized_kv_bytes_per_token,
)

QWEN_SMALL = SimpleNamespace(
    num_hidden_layers=24,
    num_attention_heads=14,
    num_key_value_heads=2,
    hidden_size=896,
    head_dim=128,
)


def test_kv_shape():
    layers, kv_heads, head_dim = kv_shape(QWEN_SMALL)
    assert (layers, kv_heads, head_dim) == (24, 2, 128)


def test_kv_bytes_per_token():
    assert kv_bytes_per_token(QWEN_SMALL, "bf16") == 2 * 24 * 2 * 128 * 2
    assert kv_bytes_per_token(QWEN_SMALL, "int8") == 2 * 24 * 2 * 128 * 1
    assert kv_bytes_per_token(QWEN_SMALL, "int4") == 2 * 24 * 2 * 128 * 0.5


def test_kv_gb():
    gb_at_1m = kv_gb(QWEN_SMALL, 1_000_000, "bf16")
    assert 22.0 < gb_at_1m < 24.0


def test_quantized_kv_bytes_include_group_scales():
    raw_int4 = kv_bytes_per_token(QWEN_SMALL, "int4")
    packed_int4 = quantized_kv_bytes_per_token(QWEN_SMALL, "int4", group_size=64)
    packed_int8 = quantized_kv_bytes_per_token(QWEN_SMALL, "int8", group_size=64)

    assert packed_int4 > raw_int4
    assert packed_int8 > packed_int4


def test_capacity_rows():
    rows = capacity_rows(QWEN_SMALL, 100_000, vram_free_gb=3.5, ram_gb=32.0, model_gb=0.5)
    by_label = {r["label"]: r for r in rows}
    assert by_label["KV bf16"]["kv_gb"] < by_label["KV offloaded"]["kv_gb"] + 0.01
    assert by_label["KV int8 (quant cache)"]["kv_gb"] < by_label["KV bf16"]["kv_gb"]
    assert by_label["KV int4 (quant cache)"]["fits"] in (True, False)
    offloaded = by_label["KV offloaded"]
    assert offloaded["location"] == "RAM"
    assert by_label["KV int8 offloaded"]["location"] == "RAM"
    assert by_label["KV int4 offloaded"]["kv_gb"] < by_label["KV int8 offloaded"]["kv_gb"]


def test_max_context_for_budget():
    n = max_context_for_budget(QWEN_SMALL, 16.0, "bf16")
    per_tok = kv_bytes_per_token(QWEN_SMALL, "bf16")
    assert n * per_tok <= 16 * 2**30
    assert (n + 1) * per_tok > 16 * 2**30


def test_model_weight_gb():
    assert 0.2 < model_weight_gb(0.5, "nf4") < 0.35
    assert 0.9 < model_weight_gb(0.5, "bf16") < 1.1
