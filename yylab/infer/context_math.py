from __future__ import annotations

import math

KV_BYTES = {"bf16": 2.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}


def kv_shape(config) -> tuple[int, int, int]:
    layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", 0) or 0
    heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head", 0) or 1
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    hidden = getattr(config, "hidden_size", None) or getattr(config, "n_embd", 0) or 0
    head_dim = getattr(config, "head_dim", None) or (hidden // heads if heads else 0)
    return int(layers), int(kv_heads), int(head_dim)


def kv_bytes_per_token(config, kv_quant: str = "bf16") -> float:
    if kv_quant not in KV_BYTES:
        if kv_quant == "fp8":
            raise ValueError("FP8 KV storage is not implemented and is not an alias for INT8")
        raise ValueError(f"unknown KV storage precision: {kv_quant!r}")
    layers, kv_heads, head_dim = kv_shape(config)
    return 2.0 * layers * kv_heads * head_dim * KV_BYTES[kv_quant]


def kv_gb(config, ctx_len: int, kv_quant: str = "bf16") -> float:
    return kv_bytes_per_token(config, kv_quant) * max(int(ctx_len), 0) / 2**30


def quantized_kv_bytes_per_token(config, kv_quant: str, group_size: int = 64) -> float:
    """Packed KV bytes including one fp16 scale per head-dimension group."""

    if kv_quant not in {"int4", "int8"}:
        raise ValueError("quantized KV storage must be int4 or int8")
    if group_size <= 0:
        raise ValueError("quantized KV group_size must be positive")
    layers, kv_heads, head_dim = kv_shape(config)
    scale_groups = math.ceil(head_dim / int(group_size))
    padded_head_dim = scale_groups * int(group_size)
    bytes_per_vector = padded_head_dim if kv_quant == "int8" else math.ceil(padded_head_dim / 2)
    packed = 2.0 * layers * kv_heads * bytes_per_vector
    scale_bytes = 2.0 * layers * kv_heads * scale_groups * 2.0
    return packed + scale_bytes


def quantized_kv_gb(config, ctx_len: int, kv_quant: str, group_size: int = 64) -> float:
    return quantized_kv_bytes_per_token(config, kv_quant, group_size) * max(int(ctx_len), 0) / 2**30


def model_weight_gb(params_b: float, quant: str = "bf16") -> float:
    normalized = str(quant or "").strip().casefold()
    bytes_per_parameter = {
        "bf16": 2.0,
        "fp16": 2.0,
        "fp32": 4.0,
        "int8": 1.05,
        "nf4": 0.55,
    }
    if normalized not in bytes_per_parameter:
        raise ValueError(
            f"unsupported model precision {normalized or '<empty>'!r}; "
            "capacity estimates never assume an unimplemented format"
        )
    per_param = bytes_per_parameter[normalized]
    return params_b * 1e9 * per_param / 2**30


def capacity_rows(
    config,
    ctx_len: int,
    vram_free_gb: float,
    ram_gb: float,
    model_gb: float,
    *,
    quant_group_size: int = 64,
    offload_residual_length: int = 32,
    resident_int4_available: bool = True,
    offload_available: bool = True,
    quantized_offload_available: bool = True,
) -> list[dict]:
    vram_budget = max(vram_free_gb, 0.0)
    ram_budget = max(ram_gb - 4.0, 0.0)

    if offload_residual_length < 0:
        raise ValueError("offload_residual_length must be non-negative")

    def quantized_size(quant: str, residual_length: int, group_size: int) -> float:
        residual_tokens = min(max(int(ctx_len), 0), residual_length)
        quantized_tokens = max(0, int(ctx_len) - residual_tokens)
        return kv_gb(config, residual_tokens, "bf16") + quantized_kv_gb(
            config,
            quantized_tokens,
            quant,
            group_size,
        )

    candidates = [("KV BF16", "vram", kv_gb(config, ctx_len, "bf16"))]
    if resident_int4_available:
        candidates.append(
            (
                "KV INT4 resident (Quanto; 128-token BF16 residual)",
                "vram",
                quantized_size("int4", 128, 64),
            )
        )
    if offload_available:
        candidates.append(("KV BF16 offloaded", "ram", kv_gb(config, ctx_len, "bf16")))
    if quantized_offload_available:
        candidates.extend(
            (
                (
                    f"KV INT8 offloaded ({offload_residual_length}-token BF16 residual)",
                    "ram",
                    quantized_size("int8", offload_residual_length, quant_group_size),
                ),
                (
                    f"KV INT4 offloaded ({offload_residual_length}-token BF16 residual)",
                    "ram",
                    quantized_size("int4", offload_residual_length, quant_group_size),
                ),
            )
        )

    rows = []
    for label, location, size in candidates:
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
