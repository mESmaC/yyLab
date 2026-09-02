from __future__ import annotations

import torch
from torch import nn

from yylab.mole.inject import (
    MoLEConfig,
    count_trainable,
    inject_mole,
    mole_layers,
    set_stage,
    set_trainable_expert,
)
from yylab.mole.losses import aux_loss, collect_router_logits, load_balance_loss
from yylab.mole.store import apply_mole_weights, load_mole_config, load_mole_weights, save_mole


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(32, 32)
        self.v_proj = nn.Linear(32, 64)
        self.other = nn.Linear(64, 16)

    def forward(self, x):
        return self.other(self.v_proj(self.q_proj(x)))


def _inject(top_k: int = 2, num_experts: int = 4) -> nn.Module:
    model = TinyModel()
    cfg = MoLEConfig(num_experts=num_experts, rank=8, alpha=16.0, top_k=top_k, target_modules=("q_proj", "v_proj"))
    replaced = inject_mole(model, cfg)
    assert len(replaced) == 2
    return model


def test_forward_matches_base_at_init():
    torch.manual_seed(0)
    model = TinyModel()
    x = torch.randn(3, 5, 32)
    base_out = model(x).detach().clone()
    _inject_into = MoLEConfig(num_experts=4, rank=8, alpha=16.0, top_k=2, target_modules=("q_proj", "v_proj"))
    inject_mole(model, _inject_into)
    mole_out = model(x)
    assert torch.allclose(base_out, mole_out, atol=1e-6)


def _scatter_router(model: nn.Module) -> None:
    for _, layer in mole_layers(model):
        layer.router.gate.weight.data.normal_(0, 0.5)


def test_backward_trains_only_mole():
    torch.manual_seed(0)
    model = _inject()
    set_stage(model, "both")
    _scatter_router(model)
    x = torch.randn(4, 8, 32)
    model(x).sum().backward()
    for name, p in model.named_parameters():
        is_mole = "experts" in name or "router" in name
        if is_mole:
            assert p.requires_grad and p.grad is not None, name
        else:
            assert not p.requires_grad, name


def test_isolated_stage_forces_one_coherent_expert_without_router_graph():
    torch.manual_seed(0)
    model = _inject()
    set_trainable_expert(model, 2)
    model(torch.randn(3, 4, 32)).sum().backward()
    for _, layer in mole_layers(model):
        assert layer.last_router_logits is None
        assert layer.router.gate.weight.grad is None
        for index, expert in enumerate(layer.experts):
            if index == 2:
                assert expert.lora_A.requires_grad
                assert expert.lora_A.grad is not None
                assert expert.lora_B.grad is not None
            else:
                assert not expert.lora_A.requires_grad
                assert expert.lora_A.grad is None
                assert expert.lora_B.grad is None


def test_topk_routing_and_aux_loss():
    torch.manual_seed(0)
    model = _inject(top_k=2, num_experts=4)
    model.eval()
    x = torch.randn(2, 3, 32)
    out = model(x)
    assert out.shape == (2, 3, 16)
    logits = collect_router_logits(model)
    assert len(logits) == 2
    for lg in logits:
        assert lg.shape[-1] == 4
        loss = load_balance_loss(lg, 2)
        assert loss.item() > 0
    assert aux_loss(model, 2).item() >= 0


def test_top1():
    torch.manual_seed(0)
    model = _inject(top_k=1, num_experts=3)
    out = model(torch.randn(2, 2, 32))
    assert out.shape == (2, 2, 16)


def test_soft_lora_config_roundtrip_and_validation():
    config = MoLEConfig(num_experts=2, rank=4, top_k=3, identity_expert=True)
    restored = MoLEConfig.from_dict(config.to_dict())

    assert restored.identity_expert is True
    assert restored.top_k == 3
    try:
        MoLEConfig(num_experts=2, rank=4, top_k=3)
        raise AssertionError("top-k should require the explicitly configured identity route")
    except ValueError as exc:
        assert "top-k" in str(exc)


def test_soft_lora_artifact_uses_identity_aware_v5_contract(tmp_path):
    config = MoLEConfig(
        num_experts=2,
        rank=4,
        top_k=3,
        identity_expert=True,
        target_modules=("q_proj", "v_proj"),
    )
    model = TinyModel()
    inject_mole(model, config)
    save_mole(tmp_path, model, config, base_ref="tiny")

    payload = load_mole_config(tmp_path, expected_base_ref="tiny")
    fresh = TinyModel()
    inject_mole(fresh, MoLEConfig.from_dict(payload["config"]))
    applied = apply_mole_weights(fresh, load_mole_weights(tmp_path))

    assert payload["format_version"] == 5
    assert applied == 4
    assert all(layer.router.gate.out_features == 3 for _, layer in mole_layers(fresh))


def test_stage_freezing():
    model = _inject()
    set_stage(model, "router")
    for name, p in model.named_parameters():
        if "router" in name:
            assert p.requires_grad
        else:
            assert not p.requires_grad
    set_stage(model, "experts")
    for name, p in model.named_parameters():
        if "experts" in name:
            assert p.requires_grad
        else:
            assert not p.requires_grad
    trainable, total = count_trainable(model)
    assert 0 < trainable < total


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _inject()
    _scatter_router(model)
    x = torch.randn(2, 2, 32)
    set_stage(model, "both")
    for _ in range(3):
        model.zero_grad()
        model(x).sum().backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-0.1)
    ref_out = model(x).detach().clone()

    cfg = MoLEConfig(num_experts=4, rank=8, alpha=16.0, top_k=2, target_modules=("q_proj", "v_proj"))
    save_mole(tmp_path, model, cfg, base_ref="tiny")

    fresh = _inject()
    base_sd = {
        k: v for k, v in model.state_dict().items() if "experts" not in k and "router" not in k
    }
    merged = dict(fresh.state_dict())
    merged.update(base_sd)
    fresh.load_state_dict(merged)
    state = load_mole_weights(tmp_path)
    n = apply_mole_weights(fresh, state)
    assert n == 8
    assert torch.allclose(fresh(x), ref_out, atol=1e-6)


def test_strict_load_rejects_partial_or_wrong_shape_bank(tmp_path):
    cfg = MoLEConfig(num_experts=4, rank=8, alpha=16.0, top_k=2, target_modules=("q_proj", "v_proj"))
    model = _inject()
    save_mole(tmp_path, model, cfg, base_ref="tiny")
    state = load_mole_weights(tmp_path)

    missing = dict(state)
    missing.pop(next(iter(missing)))
    with torch.no_grad():
        try:
            apply_mole_weights(_inject(), missing)
            raise AssertionError("partial MoLE bank should be rejected")
        except ValueError as exc:
            assert "tensor set mismatch" in str(exc)

    wrong_shape = dict(state)
    key = next(key for key in wrong_shape if key.endswith("lora_A"))
    wrong_shape[key] = wrong_shape[key][:-1]
    try:
        apply_mole_weights(_inject(), wrong_shape)
        raise AssertionError("shape-incompatible MoLE bank should be rejected")
    except ValueError as exc:
        assert "shape mismatch" in str(exc)

    assert load_mole_config(tmp_path, expected_base_ref="tiny")["format_version"] == 2
    try:
        load_mole_config(tmp_path, expected_base_ref="another-base")
        raise AssertionError("wrong-base MoLE config should be rejected")
    except ValueError as exc:
        assert "different frozen base" in str(exc)
