from __future__ import annotations

from pathlib import Path

from ..mole.inject import (
    SHARED_ROUTER_MODULE,
    MoLEConfig,
    _parent_and_attr,
    inject_mole,
    mole_layers,
    shared_router,
    token_routers,
)
from ..mole.store import apply_mole_weights, load_mole_config, load_mole_weights


def load_base_full(ref: str, dtype=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..train.low_precision import apply_saved_qat

    dtype = dtype or torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(ref)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ref, dtype=dtype, device_map="cpu", attn_implementation="sdpa"
    )
    apply_saved_qat(model)
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
    """Measure the sparse mixtures that were actually applied during calibration."""

    import torch

    weights: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def accumulate(name: str, indices: torch.Tensor, selected: torch.Tensor) -> None:
        if indices.ndim != 2 or selected.shape != indices.shape:
            raise ValueError("router calibration expected matching [items, top_k] routes")
        num_experts = (
            next(layer.num_experts for layer_name, layer in mole_layers(model) if layer_name == name)
            if name != "__shared_prompt__"
            else selected.shape[-1]
        )
        if name == "__shared_prompt__":
            prompt = shared_router(model)
            if prompt is None:
                raise RuntimeError("shared prompt route disappeared during calibration")
            num_experts = prompt.num_experts
        totals = torch.zeros(num_experts, dtype=torch.float32, device=indices.device)
        totals.scatter_add_(0, indices.reshape(-1), selected.float().reshape(-1))
        weights[name] = weights.get(name, torch.zeros_like(totals)) + totals
        counts[name] = counts.get(name, 0) + indices.shape[0]

    model.eval()
    with torch.no_grad():
        for ids in calib_ids:
            if not ids:
                continue
            inp = torch.tensor([ids], dtype=torch.long)
            model(input_ids=inp)
            prompt_router = shared_router(model)
            if (
                prompt_router is not None
                and prompt_router.last_router_indices is not None
                and prompt_router.last_router_weights is not None
            ):
                indices = prompt_router.last_router_indices
                selected = prompt_router.last_router_weights.gather(-1, indices)
                accumulate("__shared_prompt__", indices, selected)
                continue
            for name, layer in mole_layers(model):
                block_router = layer.shared_token_router()
                if (
                    block_router is not None
                    and block_router.last_router_indices is not None
                    and block_router.last_router_weights is not None
                ):
                    accumulate(
                        name,
                        block_router.last_router_indices,
                        block_router.last_router_weights,
                    )
                    continue
                logits = layer.last_router_logits
                if logits is None:
                    continue
                top_values, indices = torch.topk(logits.float(), layer.top_k, dim=-1)
                selected = (
                    torch.ones_like(top_values) if layer.top_k == 1 else torch.softmax(top_values, dim=-1)
                )
                accumulate(name, indices, selected)
            for _, layer in mole_layers(model):
                layer.last_router_logits = None
    return {name: weights[name] / counts[name] for name in weights}


def _remove_routing_modules(model) -> None:
    """Remove calibration-only gates and hooks from a statically merged model."""

    for attribute in ("_mole_token_router_hook_handles", "_mole_router_hook_handles"):
        for handle in getattr(model, attribute, ()):
            handle.remove()
        if hasattr(model, attribute):
            delattr(model, attribute)
    for name, _ in token_routers(model):
        parent, attr = _parent_and_attr(model, name)
        delattr(parent, attr)
    if hasattr(model, SHARED_ROUTER_MODULE):
        delattr(model, SHARED_ROUTER_MODULE)


def materialize_mole(model, weights: dict[str, object] | None = None) -> int:
    import torch
    from torch import nn

    merged = 0
    for name, layer in mole_layers(model):
        base = layer.base
        w = weights.get("__shared_prompt__", weights.get(name)) if weights else None
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
    _remove_routing_modules(model)
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

    adapter_cfg = load_mole_config(adapter_path, expected_base_ref=ref)
    cfg = load_mole_config(router_path, expected_base_ref=ref) if router_path else adapter_cfg
    if cfg.get("layers") != adapter_cfg.get("layers"):
        raise ValueError("router snapshot targets a different MoLE layer bank")
    mole_cfg = MoLEConfig.from_dict(cfg["config"])
    if mole_cfg.adaptive.enabled:
        raise ValueError(
            "adaptive MoLE v4 cannot be losslessly merged into one dense checkpoint; "
            "use the native sharded artifact or implement an explicitly calibrated static policy"
        )
    if mole_cfg.router_type == "per_token" and not calib_ids:
        raise ValueError(
            "learned per-token MoLE export requires calibration data; "
            "a static merge cannot preserve dynamic token routing"
        )
    model, tokenizer = load_base_full(ref, dtype)
    inject_mole(model, mole_cfg, expert_dtype=torch.bfloat16 if dtype is None else dtype)
    apply_mole_weights(model, load_mole_weights(router_path or adapter_path))
    weights = expert_usage_weights(model, calib_ids) if calib_ids else None
    if mole_cfg.router_type == "per_token":
        missing = sorted(name for name, _ in mole_layers(model) if name not in (weights or {}))
        if missing:
            raise RuntimeError(
                "calibration produced no learned token-route usage for: " + ", ".join(missing[:3])
            )
    materialize_mole(model, weights)
    return _save(model, tokenizer, out_dir)
