from __future__ import annotations

import math

KV_BYTES = {"bf16": 2.0, "fp16": 2.0, "int8": 1.0, "fp8": 1.0, "int4": 0.5}


def kv_shape(config) -> tuple[int, int, int]:
    layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", 0) or 0
    heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head", 0) or 1
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    hidden = getattr(config, "hidden_size", None) or getattr(config, "n_embd", 0) or 0
    head_dim = getattr(config, "head_dim", None) or (hidden // heads if heads else 0)
    return int(layers), int(kv_heads), int(head_dim)


def kv_bytes_per_token(config, kv_quant: str = "bf16") -> float:
    layers, kv_heads, head_dim = kv_shape(config)
    return 2.0 * layers * kv_heads * head_dim * KV_BYTES.get(kv_quant, 2.0)


def kv_gb(config, ctx_len: int, kv_quant: str = "bf16") -> float:
    return kv_bytes_per_token(config, kv_quant) * max(int(ctx_len), 0) / 2**30


def quantized_kv_bytes_per_token(config, kv_quant: str, group_size: int = 64) -> float:
    """Packed KV bytes including one fp16 scale per head-dimension group."""

    if kv_quant not in {"int4", "int8"}:
        raise ValueError("quantized KV storage must be int4 or int8")
    layers, kv_heads, head_dim = kv_shape(config)
    packed = kv_bytes_per_token(config, kv_quant)
    scale_groups = math.ceil(head_dim / max(1, int(group_size)))
    scale_bytes = 2.0 * layers * kv_heads * scale_groups * 2.0
    return packed + scale_bytes


def quantized_kv_gb(config, ctx_len: int, kv_quant: str, group_size: int = 64) -> float:
    return quantized_kv_bytes_per_token(config, kv_quant, group_size) * max(int(ctx_len), 0) / 2**30


def model_weight_gb(params_b: float, quant: str = "bf16") -> float:
    per_param = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.05, "nf4": 0.55}.get(quant, 2.0)
    return params_b * 1e9 * per_param / 2**30


def capacity_rows(
    config,
    ctx_len: int,
    vram_free_gb: float,
    ram_gb: float,
    model_gb: float,
) -> list[dict]:
    vram_budget = max(vram_free_gb, 0.0)
    ram_budget = max(ram_gb - 4.0, 0.0)

    rows = []
    for label, quant, location, quantized in (
        ("KV bf16", "bf16", "vram", False),
        ("KV int8 (quant cache)", "int8", "vram", True),
        ("KV int4 (quant cache)", "int4", "vram", True),
        ("KV offloaded", "bf16", "ram", False),
        ("KV int8 offloaded", "int8", "ram", True),
        ("KV int4 offloaded", "int4", "ram", True),
    ):
        size = quantized_kv_gb(config, ctx_len, quant) if quantized else kv_gb(config, ctx_len, quant)
        if location == "vram":
            headroom = vram_budget - model_gb
            fits = size <= headroom
            budget = headroom
        else:
            fits = size <= ram_budget
            budget = ram_budget
        rows.append(
            {
                "label": label,
                "location": location.upper(),
                "kv_gb": round(size, 2),
                "budget_gb": round(budget, 2),
                "fits": bool(fits),
            }
        )
    return rows


def max_context_for_budget(config, budget_gb: float, kv_quant: str = "bf16") -> int:
    per_tok = kv_bytes_per_token(config, kv_quant)
    if per_tok <= 0:
        return 0
    return int(budget_gb * 2**30 / per_tok)
