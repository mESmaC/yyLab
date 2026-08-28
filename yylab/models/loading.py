from __future__ import annotations


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


def load_base(ref: str, quant: str, attn_implementation: str = "sdpa", rope_scaling: dict | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    notes: list[str] = []
    cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(ref)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"attn_implementation": attn_implementation}
    if rope_scaling:
        from transformers import AutoConfig

        base_cfg = AutoConfig.from_pretrained(ref)
        merged_rope = dict(getattr(base_cfg, "rope_scaling", None) or {})
        merged_rope.update(rope_scaling)
        kwargs["rope_scaling"] = merged_rope
    if quant == "fp8":
        quant = "nf4"
        notes.append("FP8 base weights unavailable in the loader; fell back to NF4.")
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
        kwargs["dtype"] = torch.bfloat16 if cuda else torch.float32
        if not cuda and quant not in {"bf16", "fp16", ""}:
            notes.append("No CUDA GPU: quantization disabled, using fp32 on CPU (slow).")
    if cuda:
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(ref, **kwargs)
    model.config.use_cache = True
    return model, tokenizer, cuda, quant, notes
