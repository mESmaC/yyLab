from __future__ import annotations

from pathlib import Path

from ..mole.inject import MoLEConfig, _parent_and_attr, inject_mole, mole_layers
from ..mole.store import apply_mole_weights, load_mole_config, load_mole_weights


def load_base_full(ref: str, dtype=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype or torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(ref)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ref, dtype=dtype, device_map="cpu", attn_implementation="sdpa"
    )
    model.config.use_cache = True
    return model, tokenizer


def _save(model, tokenizer, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def merge_lora(ref: str, adapter_path: str, out_dir: Path, dtype=None) -> Path:
    from peft import PeftModel

    model, tokenizer = load_base_full(ref, dtype)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    return _save(model, tokenizer, out_dir)


def expert_usage_weights(model, calib_ids: list[list[int]]) -> dict[str, object]:
    import torch

    weights: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    model.eval()
    with torch.no_grad():
        for ids in calib_ids:
            inp = torch.tensor([ids], dtype=torch.long)
            model(input_ids=inp)
            for name, layer in mole_layers(model):
                logits = layer.last_router_logits
                if logits is None:
                    continue
                probs = torch.softmax(logits.float(), dim=-1).mean(dim=0)
                if name not in weights:
                    weights[name] = torch.zeros_like(probs)
                    counts[name] = 0
                weights[name] += probs
                counts[name] += 1
            for _, layer in mole_layers(model):
                layer.last_router_logits = None
    return {name: weights[name] / counts[name] for name in weights}


def materialize_mole(model, weights: dict[str, object] | None = None) -> int:
    import torch
    from torch import nn

    merged = 0
    for name, layer in mole_layers(model):
        base = layer.base
        w = weights.get(name) if weights else None
        if w is None:
            w = torch.full((layer.num_experts,), 1.0 / layer.num_experts)
        w = w.to(torch.float32)
        w = w / w.sum()
        delta = None
        for e, expert in enumerate(layer.experts):
            d = (expert.lora_B.float() @ expert.lora_A.float()) * expert.scaling * float(w[e])
            delta = d if delta is None else delta + d
        merged_w = base.weight.detach().float() + delta
        new = nn.Linear(layer.in_features, layer.out_features, bias=base.bias is not None)
        new.weight = nn.Parameter(merged_w.to(base.weight.dtype))
        if base.bias is not None:
            new.bias = nn.Parameter(base.bias.detach().clone())
        parent, attr = _parent_and_attr(model, name)
        setattr(parent, attr, new)
        merged += 1
    return merged


def merge_mole(
    ref: str,
    adapter_path: str,
    out_dir: Path,
    router_path: str | None = None,
    calib_ids: list[list[int]] | None = None,
    dtype=None,
) -> Path:
    import torch

    model, tokenizer = load_base_full(ref, dtype)
    cfg = load_mole_config(adapter_path)
    mole_cfg = MoLEConfig.from_dict(cfg["config"])
    inject_mole(model, mole_cfg, expert_dtype=torch.bfloat16 if dtype is None else dtype)
    apply_mole_weights(model, load_mole_weights(router_path or adapter_path))
    weights = expert_usage_weights(model, calib_ids) if calib_ids else None
    materialize_mole(model, weights)
    return _save(model, tokenizer, out_dir)
