from __future__ import annotations

import torch
from torch import nn

from yylab.mole.inject import (
    MoLEConfig,
    attach_shared_router,
    begin_prompt_route,
    end_prompt_route,
    inject_mole,
    mole_layers,
    set_stage,
    shared_router,
)
from yylab.mole.layer import MoLELinear, SharedPromptRouter
from yylab.mole.losses import collect_router_logits
from yylab.mole.store import apply_mole_weights, load_mole_weights, save_mole


class TinyPromptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, 16)
        self.q_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids):
        return self.v_proj(torch.tanh(self.q_proj(self.embed(input_ids))))


def _config() -> MoLEConfig:
    return MoLEConfig(
        num_experts=3,
        rank=4,
        alpha=8.0,
        top_k=1,
        target_modules=("q_proj", "v_proj"),
        router_type="shared_prompt",
        router_hidden_size=8,
        router_max_tokens=4,
    )


def test_shared_prompt_route_is_one_coherent_adapter_and_one_trainable_gate():
    torch.manual_seed(4)
    model = TinyPromptModel()
    inject_mole(model, _config())
    router = shared_router(model)
    assert router is not None
    with torch.no_grad():
        for _, layer in mole_layers(model):
            for expert in layer.experts:
                expert.lora_B.normal_(0, 0.1)
        router.network[-1].weight.zero_()
        router.network[-1].bias.copy_(torch.tensor([-3.0, 5.0, -2.0]))

    ids = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])
    shared_output = model(ids)
    assert router.last_router_indices is not None
    assert torch.equal(router.last_router_indices, torch.ones((2, 1), dtype=torch.long))
    assert len(collect_router_logits(model)) == 1

    for _, layer in mole_layers(model):
        layer.set_active_expert(1)
    forced_output = model(ids)
    assert torch.allclose(shared_output, forced_output, atol=1e-6)

    for _, layer in mole_layers(model):
        layer.set_active_expert(None)
    set_stage(model, "router")
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable
    assert all(name.startswith("_mole_shared_router.") for name in trainable)
    model(ids).sum().backward()
    assert all(value.grad is not None for value in router.parameters())
    assert sum(value.grad.abs().sum().item() for value in router.parameters()) > 0


def test_shared_router_strict_artifact_roundtrip(tmp_path):
    torch.manual_seed(9)
    base = TinyPromptModel()
    base_state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    cfg = _config()
    inject_mole(base, cfg)
    with torch.no_grad():
        for _, layer in mole_layers(base):
            for expert in layer.experts:
                expert.lora_B.normal_(0, 0.05)
        router = shared_router(base)
        assert router is not None
        router.network[-1].bias.copy_(torch.tensor([4.0, -2.0, -3.0]))
    ids = torch.tensor([[3, 4, 5, 6]])
    expected = base(ids).detach()
    save_mole(tmp_path, base, cfg, base_ref="tiny")

    fresh = TinyPromptModel()
    fresh.load_state_dict(base_state)
    inject_mole(fresh, cfg)
    state = load_mole_weights(tmp_path)
    assert any(key.startswith("shared_router.") for key in state)
    assert apply_mole_weights(fresh, state) == 6
    assert torch.allclose(fresh(ids), expected, atol=1e-6)

    incomplete = dict(state)
    incomplete.pop(next(key for key in incomplete if key.startswith("shared_router.")))
    try:
        apply_mole_weights(fresh, incomplete)
        raise AssertionError("missing shared router tensor should fail strict validation")
    except ValueError as exc:
        assert "tensor set mismatch" in str(exc)


def test_legacy_bank_loads_strictly_before_shared_router_snapshot(tmp_path):
    torch.manual_seed(12)
    legacy_cfg = MoLEConfig(
        num_experts=2,
        rank=4,
        alpha=8.0,
        top_k=1,
        target_modules=("q_proj", "v_proj"),
    )
    legacy = TinyPromptModel()
    base_state = {name: value.detach().clone() for name, value in legacy.state_dict().items()}
    inject_mole(legacy, legacy_cfg)
    bank = tmp_path / "legacy"
    save_mole(bank, legacy, legacy_cfg, base_ref="tiny")

    training_model = TinyPromptModel()
    training_model.load_state_dict(base_state)
    inject_mole(training_model, legacy_cfg)
    assert apply_mole_weights(training_model, load_mole_weights(bank)) == 4
    shared_cfg = MoLEConfig.from_dict(legacy_cfg.to_dict())
    shared_cfg.router_type = "shared_prompt"
    shared_cfg.router_hidden_size = 8
    attach_shared_router(training_model, shared_cfg)
    snapshot = tmp_path / "shared"
    save_mole(snapshot, training_model, shared_cfg, base_ref="tiny")

    inference_model = TinyPromptModel()
    inference_model.load_state_dict(base_state)
    inject_mole(inference_model, shared_cfg)
    assert apply_mole_weights(inference_model, load_mole_weights(snapshot)) == 4


def test_generation_route_persists_until_sequence_ends():
    model = TinyPromptModel()
    inject_mole(model, _config())
    router = shared_router(model)
    assert router is not None
    begin_prompt_route(model)
    model(torch.tensor([[2, 3, 4, 5]]))
    first_logits = router.last_router_logits
    model(torch.tensor([[10, 11]]))
    assert router.last_router_logits is first_logits
    end_prompt_route(model)
    assert router.last_router_logits is None
    model(torch.tensor([[10, 11]]))
    assert router.last_router_logits is not None
    assert router.last_router_logits is not first_logits


def _reference_shared_dispatch(
    layer: MoLELinear,
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Pre-vectorization implementation, retained as a numerical oracle."""

    y = layer.base(x)
    add = torch.zeros_like(y)
    for expert_index, expert in enumerate(layer.experts):
        sample_weights = weights[:, expert_index]
        mask = sample_weights > 0
        if not torch.any(mask):
            continue
        contribution = expert(x[mask])
        scale_shape = (contribution.shape[0],) + (1,) * (contribution.ndim - 1)
        contribution = contribution * sample_weights[mask].to(contribution.dtype).reshape(
            scale_shape
        )
        add[mask] = add[mask] + contribution.to(add.dtype)
    return y + add


def test_grouped_shared_dispatch_matches_sparse_reference_outputs_and_gradients(monkeypatch):
    torch.manual_seed(31)
    router = SharedPromptRouter(hidden_size=5, num_experts=4, top_k=2, hidden_width=None)
    layer = MoLELinear(
        nn.Linear(5, 7, bias=False),
        num_experts=4,
        rank=3,
        alpha=6.0,
        top_k=2,
        shared_router=router,
    )
    with torch.no_grad():
        for expert in layer.experts:
            expert.lora_A.normal_(0, 0.2)
            expert.lora_B.normal_(0, 0.2)
    for parameter in layer.parameters():
        parameter.requires_grad_(False)

    indices = torch.tensor([[0, 3], [2, 1], [3, 2]])
    weights_data = torch.tensor(
        [
            [0.7, 0.0, 0.0, 0.3],
            [0.0, 0.4, 0.6, 0.0],
            [0.0, 0.0, 0.25, 0.75],
        ]
    )
    probe = torch.randn(3, 4, 7)

    reference_x = torch.randn(3, 4, 5, requires_grad=True)
    reference_weights = weights_data.clone().requires_grad_(True)
    reference = _reference_shared_dispatch(layer, reference_x, indices, reference_weights)
    (reference * probe).sum().backward()

    grouped_x = reference_x.detach().clone().requires_grad_(True)
    grouped_weights = weights_data.clone().requires_grad_(True)
    router.last_router_indices = indices
    router.last_router_weights = grouped_weights

    # The grouped path must not reintroduce a tensor-valued Python branch.
    def fail_tensor_branch(*_args, **_kwargs):
        raise AssertionError("shared dispatch called torch.any")

    monkeypatch.setattr(torch, "any", fail_tensor_branch)
    grouped = layer(grouped_x)
    (grouped * probe).sum().backward()

    assert torch.allclose(grouped, reference, atol=1e-6, rtol=1e-5)
    assert torch.allclose(grouped_x.grad, reference_x.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        grouped_weights.grad,
        reference_weights.grad,
        atol=1e-6,
        rtol=1e-5,
    )
