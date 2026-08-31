from __future__ import annotations

import torch
from torch import nn

from yylab.mole.layer import SharedTokenRouter
from yylab.mole.losses import router_usage_metrics
from yylab.train.planner import PlanInput, estimate_adapter_trainable_params, plan_training


def _qwen_joint(*, experts: int, rank: int, train_routers: bool = True) -> PlanInput:
    return PlanInput(
        params_b=0.6,
        vram_gb=12.0,
        seq_len=2048,
        method="auto",
        adapter_only=True,
        lora_rank=rank,
        num_experts=experts,
        hidden_size=1024,
        intermediate_size=3072,
        num_layers=28,
        num_attention_heads=16,
        num_kv_heads=8,
        head_dim=128,
        vocab_size=151_936,
        model_type="qwen3",
        train_token_routers=train_routers,
        fused_linear_ce=True,
        full_liger_kernels=True,
        prefer_cudnn_sdpa=True,
        sm_version="12.0",
    )


def test_qwen_e8_r8_joint_trainables_include_one_router_per_block():
    without_routers = estimate_adapter_trainable_params(_qwen_joint(experts=8, rank=8, train_routers=False))
    with_routers = estimate_adapter_trainable_params(_qwen_joint(experts=8, rank=8))

    assert without_routers == 40_370_176
    assert with_routers - without_routers == 28 * 1024 * 8 == 229_376
    assert with_routers == 40_599_552


def test_rank_expert_capacity_trade_keeps_lora_width_invariant():
    e8_r8_lora = estimate_adapter_trainable_params(_qwen_joint(experts=8, rank=8, train_routers=False))
    e4_r16_lora = estimate_adapter_trainable_params(_qwen_joint(experts=4, rank=16, train_routers=False))
    assert 8 * 8 == 4 * 16
    assert e8_r8_lora == e4_r16_lora == 40_370_176

    # Router width follows E rather than E*r. It is a small, explicit overhead
    # on top of the invariant expert rack, not hidden in the LoRA estimate.
    e8_r8_total = estimate_adapter_trainable_params(_qwen_joint(experts=8, rank=8))
    e4_r16_total = estimate_adapter_trainable_params(_qwen_joint(experts=4, rank=16))
    assert e8_r8_total - e4_r16_total == 28 * 1024 * (8 - 4)

    e8_plan = plan_training(_qwen_joint(experts=8, rank=8))
    e4_plan = plan_training(_qwen_joint(experts=4, rank=16))
    assert (e8_plan.micro_batch, e8_plan.grad_accum) == (8, 2)
    assert (e4_plan.micro_batch, e4_plan.grad_accum) == (8, 2)
    assert e8_plan.feasible and e4_plan.feasible


def test_joint_batch_projection_scales_to_available_vram_without_changing_effective_batch():
    roomy = plan_training(_qwen_joint(experts=8, rank=8))
    constrained_input = _qwen_joint(experts=8, rank=8)
    constrained_input.vram_gb = 7.0
    constrained = plan_training(constrained_input)
    long_context_input = _qwen_joint(experts=8, rank=8)
    long_context_input.seq_len = 4096
    long_context = plan_training(long_context_input)

    assert (roomy.micro_batch, roomy.grad_accum) == (8, 2)
    assert (constrained.micro_batch, constrained.grad_accum) == (4, 4)
    assert (long_context.micro_batch, long_context.grad_accum) == (4, 4)
    assert roomy.micro_batch * roomy.grad_accum == 16
    assert constrained.micro_batch * constrained.grad_accum == 16
    assert long_context.micro_batch * long_context.grad_accum == 16
    assert constrained.est_peak_gb < constrained_input.vram_gb * 0.88
    assert any("largest projected fit" in note for note in roomy.notes)


def test_full_cpu_optimizer_offload_reduces_estimated_vram_and_reports_ram_state():
    gpu_plan = plan_training(_qwen_joint(experts=16, rank=8))
    offload_input = _qwen_joint(experts=16, rank=8)
    offload_input.optimizer_state = "cpu_offload"
    offload_plan = plan_training(offload_input)

    assert offload_plan.est_peak_gb < gpu_plan.est_peak_gb
    assert offload_plan.offload is True
    assert any("FP32 master/moment state in RAM" in note for note in offload_plan.notes)


class RouterRack(nn.Module):
    def __init__(self, *routers: SharedTokenRouter) -> None:
        super().__init__()
        self.routers = nn.ModuleList(routers)


def test_router_usage_metrics_reports_balanced_token_routing():
    router = SharedTokenRouter(hidden_size=4, num_experts=4, top_k=2)
    router.last_router_indices = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]],
        dtype=torch.long,
    )
    router.last_router_weights = torch.tensor(
        [[0.75, 0.25], [0.75, 0.25], [0.75, 0.25], [0.75, 0.25]],
    )

    metrics = router_usage_metrics(RouterRack(router))

    assert metrics == {
        "router_entropy": 1.0,
        "dead_experts": 0,
        "expert_token_share": [0.25, 0.25, 0.25, 0.25],
    }


def test_router_usage_metrics_reports_dead_experts_across_blocks():
    first = SharedTokenRouter(hidden_size=4, num_experts=4, top_k=1)
    first.last_router_indices = torch.tensor([[0], [0]], dtype=torch.long)
    first.last_router_weights = torch.ones(2, 1)
    second = SharedTokenRouter(hidden_size=4, num_experts=4, top_k=1)
    second.last_router_indices = torch.tensor([[1], [1]], dtype=torch.long)
    second.last_router_weights = torch.ones(2, 1)

    metrics = router_usage_metrics(RouterRack(first, second))

    assert metrics["router_entropy"] == 0.5
    assert metrics["dead_experts"] == 2
    assert metrics["expert_token_share"] == [0.5, 0.5, 0.0, 0.0]


def test_router_usage_metrics_is_empty_before_first_route():
    router = SharedTokenRouter(hidden_size=4, num_experts=4, top_k=1)
    assert router_usage_metrics(RouterRack(router)) == {}
