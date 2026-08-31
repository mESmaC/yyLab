from __future__ import annotations


def trained_context_length(base_cfg) -> int:
    """Return the checkpoint's trained RoPE length rather than a padded runtime limit."""

    native_context = int(getattr(base_cfg, "max_position_embeddings", 0) or 0)
    if getattr(base_cfg, "model_type", "") == "qwen3" and native_context >= 32_768:
        # Qwen3 reserves an additional 8,192 runtime positions around its
        # 32,768-token trained sequence. YaRN's original length must remain the
        # trained length or the scale factor is applied to the wrong base.
        return 32_768
    return native_context


def _merged_rope_parameters(base_cfg, rope_scaling: dict) -> dict:
    merged = dict(getattr(base_cfg, "rope_scaling", None) or {})
    merged.update(rope_scaling)
    if merged.get("rope_type") != "yarn" or merged.get("original_max_position_embeddings"):
        return merged
    original_context = trained_context_length(base_cfg)
    if original_context:
        merged["original_max_position_embeddings"] = original_context
    return merged


def scan_checkpoint(path) -> dict:
    import json
    import math
    from pathlib import Path

    from safetensors import safe_open

    path = Path(path)
    cfg_file = path / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError("config.json not found — not a HuggingFace checkpoint folder")
    cfg = json.loads(cfg_file.read_text("utf-8"))
    arch = (cfg.get("architectures") or ["unknown"])[0]
    params = 0
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), framework="numpy") as sf:
            for key in sf.keys():
                shape = sf.get_slice(key).get_shape()
                params += math.prod(shape) if shape else 0
    return {
        "arch": arch,
        "params_b": round(params / 1e9, 3),
        "hidden_size": cfg.get("hidden_size", 0) or cfg.get("n_embd", 0),
        "num_layers": cfg.get("num_hidden_layers", 0) or cfg.get("n_layer", 0),
        "base_cfg": cfg,
    }


def load_base(
    ref: str,
    quant: str,
    attn_implementation: str = "sdpa",
    rope_scaling: dict | None = None,
    max_position_embeddings: int | None = None,
):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from ..train.low_precision import apply_saved_qat

    notes: list[str] = []
    cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(ref)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base_cfg = AutoConfig.from_pretrained(ref)
    saved_qat = getattr(base_cfg, "yylab_low_precision", None)
    if saved_qat and quant in {"nf4", "int8"}:
        raise ValueError(
            "A QAT-trained base cannot also be loaded through bitsandbytes NF4/INT8. "
            "Load its saved BF16/FP16/FP32 shadow checkpoint instead."
        )

    kwargs: dict = {
        "attn_implementation": attn_implementation,
        # Stream checkpoint shards directly into their destination tensors.
        # This avoids a second full-sized CPU state dict during worker startup.
        "low_cpu_mem_usage": True,
    }
    if rope_scaling:
        # Qwen3's default 40,960 runtime allocation is 8,192 prompt
        # positions plus its native 32,768-token trained sequence.
        merged_rope = _merged_rope_parameters(base_cfg, rope_scaling)
        kwargs["rope_scaling"] = merged_rope
        if max_position_embeddings is not None:
            target_context = int(max_position_embeddings)
            if target_context < 1:
                raise ValueError("max_position_embeddings must be positive")
            kwargs["max_position_embeddings"] = target_context
            notes.append(
                f"{str(merged_rope.get('rope_type') or 'RoPE').upper()} context extension: "
                f"{target_context:,} positions."
            )
    if quant == "fp8":
        raise ValueError("FP8 base loading is not implemented; choose NF4, INT8, or BF16")
    if cuda and quant == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["dtype"] = torch.bfloat16
    elif cuda and quant == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["dtype"] = torch.bfloat16
    else:
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "": torch.bfloat16,
        }
        requested_dtype = dtype_map.get(quant, torch.bfloat16)
        kwargs["dtype"] = requested_dtype if cuda else torch.float32
        if not cuda and quant not in {"bf16", "fp16", "fp32", ""}:
            notes.append("No CUDA GPU: quantization disabled, using fp32 on CPU (slow).")
    if cuda:
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(ref, **kwargs)
    qat_count, qat_meta = apply_saved_qat(model)
    if qat_meta is not None:
        execution = (
            "native low-bit forward"
            if qat_meta.get("native_low_bit_gemm")
            else "emulated low-bit forward"
        )
        sparsity = str(qat_meta.get("sparsity") or "none")
        if qat_meta.get("native_sparse_gemm"):
            sparsity_note = f"; {sparsity} compressed sparse forward"
        elif sparsity != "none":
            sparsity_note = f"; {sparsity} dense-masked forward"
        else:
            sparsity_note = ""
        backward = str(qat_meta.get("backward_precision") or "bf16")
        backward_note = (
            "; native MXFP8 backward"
            if qat_meta.get("native_low_bit_backward")
            else f"; {backward} backward"
        )
        notes.append(
            f"restored {qat_meta.get('format')} QAT numerics on {qat_count} Linear modules "
            f"({execution}{sparsity_note}{backward_note}; high-precision shadow checkpoint)"
        )
    model.config.use_cache = True
    return model, tokenizer, cuda, quant, notes
