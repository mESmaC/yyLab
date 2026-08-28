from __future__ import annotations

import torch
from torch import nn

from yylab.mole.inject import MoLEConfig, count_trainable, inject_mole, mole_layers, set_stage
from yylab.mole.losses import aux_loss, collect_router_logits, load_balance_loss
from yylab.mole.store import apply_mole_weights, load_mole_weights, save_mole


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
