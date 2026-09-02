from __future__ import annotations

import copy

import torch
from torch import nn

from yylab.mole.inject import MoLEConfig, inject_mole, set_active_expert, token_routers
from yylab.mole.layer import LoRAExpert, MoLELinear, SharedTokenRouter


def _legacy_token_dispatch(layer: MoLELinear, x: torch.Tensor) -> torch.Tensor:
    """Numerical oracle for the former synchronized expert loop."""

    y = layer.base(x)
    flat_x = x.reshape(-1, layer.in_features)
    indices, weights, logits = layer.router(flat_x)
    layer.last_router_logits = logits
    add = torch.zeros(
        flat_x.shape[0],
        layer.out_features,
        dtype=y.dtype,
        device=y.device,
    )
    for expert_index, expert in enumerate(layer.experts):
        selected = indices == expert_index
        token_weights = torch.where(selected, weights, torch.zeros_like(weights)).sum(dim=-1)
        mask = selected.any(dim=-1)
        contribution = expert(flat_x[mask])
        contribution = contribution * token_weights[mask].to(contribution.dtype).unsqueeze(-1)
        add[mask] = add[mask] + contribution.to(add.dtype)
    return y + add.reshape(*x.shape[:-1], layer.out_features)


def _initialized_layer(
    *,
    in_features: int = 6,
    out_features: int = 7,
    num_experts: int = 4,
    rank: int = 3,
    top_k: int = 2,
    identity_expert: bool = False,
) -> MoLELinear:
    layer = MoLELinear(
        nn.Linear(in_features, out_features, bias=False),
        num_experts=num_experts,
        rank=rank,
        alpha=rank * 2.0,
        top_k=top_k,
        identity_expert=identity_expert,
    )
    with torch.no_grad():
        layer.router.gate.weight.normal_(0.0, 0.35)
        for expert in layer.experts:
            expert.lora_A.normal_(0.0, 0.25)
            expert.lora_B.normal_(0.0, 0.25)
    return layer


def test_vectorized_token_dispatch_matches_legacy_outputs_and_gradients():
    torch.manual_seed(41)
    reference_layer = _initialized_layer()
    vectorized_layer = copy.deepcopy(reference_layer)

    reference_x = torch.randn(3, 5, 6, requires_grad=True)
    vectorized_x = reference_x.detach().clone().requires_grad_(True)
    probe = torch.randn(3, 5, 7)

    reference = _legacy_token_dispatch(reference_layer, reference_x)
    (reference * probe).sum().backward()
    vectorized = vectorized_layer(vectorized_x)
    (vectorized * probe).sum().backward()

    assert torch.allclose(vectorized, reference, atol=2e-6, rtol=2e-5)
    assert torch.allclose(vectorized_x.grad, reference_x.grad, atol=2e-6, rtol=2e-5)
    for (reference_name, reference_parameter), (name, parameter) in zip(
        reference_layer.named_parameters(),
        vectorized_layer.named_parameters(),
        strict=True,
    ):
        assert name == reference_name
        if reference_parameter.grad is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.allclose(
                parameter.grad,
                reference_parameter.grad,
                atol=3e-6,
                rtol=3e-5,
            ), name


def test_token_dispatch_handles_32_experts_without_expert_loop_or_scalar_branch(monkeypatch):
    torch.manual_seed(43)
    layer = _initialized_layer(
        in_features=11,
        out_features=13,
        num_experts=32,
        rank=2,
        top_k=2,
    )

    def fail_expert_forward(*_args, **_kwargs):
        raise AssertionError("per-token routing called an individual expert")

    def fail_tensor_branch(*_args, **_kwargs):
        raise AssertionError("per-token routing performed a tensor-valued Python branch")

    monkeypatch.setattr(LoRAExpert, "forward", fail_expert_forward)
    monkeypatch.setattr(torch, "any", fail_tensor_branch)
    x = torch.randn(2, 9, 11, requires_grad=True)
    output = layer(x)
    output.square().mean().backward()

    assert output.shape == (2, 9, 13)
    assert layer.router.gate.weight.grad is not None
    assert torch.isfinite(layer.router.gate.weight.grad).all()


def test_top1_router_receives_task_gradient_without_auxiliary_loss():
    torch.manual_seed(47)
    layer = _initialized_layer(num_experts=3, rank=2, top_k=1)
    x = torch.randn(4, 3, 6)
    probe = torch.randn(4, 3, 7)

    _, weights, _ = layer.router(x.reshape(-1, 6))
    assert torch.equal(weights.detach(), torch.ones_like(weights))

    (layer(x) * probe).sum().backward()
    gradient = layer.router.gate.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0


def test_soft_lora_identity_route_can_preserve_exact_base_output():
    torch.manual_seed(49)
    layer = _initialized_layer(num_experts=2, rank=2, top_k=1, identity_expert=True)
    x = torch.ones(2, 3, 6)
    with torch.no_grad():
        layer.router.gate.weight[:2].fill_(-1.0)
        layer.router.gate.weight[2].fill_(1.0)

    expected = layer.base(x)
    output = layer(x)
    indices, weights, logits = layer.router(x.reshape(-1, 6))

    assert logits.shape[-1] == 3
    assert torch.equal(indices, torch.full_like(indices, 2))
    assert torch.equal(weights.detach(), torch.ones_like(weights))
    assert torch.equal(output, expected)


def test_soft_lora_identity_route_keeps_router_and_adapter_gradients():
    torch.manual_seed(51)
    layer = _initialized_layer(num_experts=2, rank=2, top_k=3, identity_expert=True)
    x = torch.randn(3, 4, 6, requires_grad=True)

    layer(x).square().mean().backward()

    assert layer.router.num_routes == 3
    assert layer.router.gate.weight.grad is not None
    assert layer.router.gate.weight.grad.abs().sum().item() > 0
    assert any(expert.lora_B.grad is not None for expert in layer.experts)


def test_block_token_router_route_is_reused_across_different_projection_widths():
    torch.manual_seed(53)
    router = SharedTokenRouter(hidden_size=6, num_experts=4, top_k=2)
    attention_projection = MoLELinear(
        nn.Linear(6, 5, bias=False),
        num_experts=4,
        rank=3,
        alpha=6.0,
        top_k=2,
        shared_token_router=router,
    )
    down_projection = MoLELinear(
        nn.Linear(9, 6, bias=False),
        num_experts=4,
        rank=3,
        alpha=6.0,
        top_k=2,
        shared_token_router=router,
    )
    with torch.no_grad():
        router.gate.weight.normal_(0.0, 0.3)
        for layer in (attention_projection, down_projection):
            for expert in layer.experts:
                expert.lora_A.normal_(0.0, 0.2)
                expert.lora_B.normal_(0.0, 0.2)

    block_states = torch.randn(2, 4, 6)
    indices, weights = router.prepare_route(block_states, force=True)
    attention_output = attention_projection(block_states)
    down_output = down_projection(torch.randn(2, 4, 9))

    assert attention_output.shape == (2, 4, 5)
    assert down_output.shape == (2, 4, 6)
    assert router.last_router_indices is indices
    assert router.last_router_weights is weights
    assert attention_projection.last_router_logits is None
    assert down_projection.last_router_logits is None
    (attention_output.sum() + down_output.sum()).backward()
    assert router.gate.weight.grad is not None
    assert router.gate.weight.grad.abs().sum().item() > 0

    # Weak references keep the block router out of every projection artifact.
    assert not any("shared_token" in key for key in attention_projection.state_dict())


def test_block_token_router_rejects_stale_or_wrong_sized_routes():
    router = SharedTokenRouter(hidden_size=6, num_experts=3, top_k=2)
    layer = MoLELinear(
        nn.Linear(6, 5),
        num_experts=3,
        rank=2,
        top_k=2,
        shared_token_router=router,
    )

    try:
        layer(torch.randn(2, 3, 6))
        raise AssertionError("an unprepared block route should not be consumed")
    except RuntimeError as exc:
        assert "not prepared" in str(exc)

    router.prepare_route(torch.randn(2, 3, 6), force=True)
    try:
        layer(torch.randn(2, 4, 6))
        raise AssertionError("a stale route should not be reused for another token shape")
    except RuntimeError as exc:
        assert "token count" in str(exc)


def test_block_token_router_initialization_breaks_zero_gate_expert_collapse():
    torch.manual_seed(59)
    router = SharedTokenRouter(hidden_size=16, num_experts=8, top_k=2)
    varied_tokens = torch.randn(256, 16)
    indices, _ = router.prepare_route(varied_tokens, force=True)

    assert torch.count_nonzero(router.gate.weight).item() == router.gate.weight.numel()
    assert torch.unique(indices).numel() > router.top_k


def test_forced_expert_can_bypass_block_router_pre_hook():
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(6, 6, bias=False)

        def forward(self, hidden_states):
            return self.q_proj(hidden_states)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 6})()
            self.layers = nn.ModuleList([Block()])

        def forward(self, hidden_states):
            return self.layers[0](hidden_states)

    model = Model()
    inject_mole(
        model,
        MoLEConfig(
            num_experts=3,
            rank=2,
            top_k=2,
            target_modules=("q_proj",),
            router_type="per_token",
        ),
    )
    set_active_expert(model, 1)
    router = token_routers(model)[0][1]

    output = model(torch.randn(2, 4, 6))

    assert output.shape == (2, 4, 6)
    assert router.routing_enabled is False
    assert router.last_router_indices is None
    assert router.last_router_weights is None

    set_active_expert(model, None)
    model(torch.randn(2, 4, 6))
    assert router.routing_enabled is True
    assert router.last_router_indices is not None
