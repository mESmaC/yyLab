from __future__ import annotations

import torch
from torch import nn

from .layer import MoLELinear, SharedPromptRouter, SharedTokenRouter


def load_balance_loss(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    num_experts = logits.shape[-1]
    probs = torch.softmax(logits.float(), dim=-1)
    _, top_idx = torch.topk(logits.float(), min(top_k, num_experts), dim=-1)
    frac = torch.zeros(num_experts, device=logits.device, dtype=torch.float32)
    ones = torch.ones(top_idx.numel(), device=logits.device, dtype=torch.float32)
    frac = frac.scatter_add_(0, top_idx.flatten(), ones)
    frac = frac / max(top_idx.numel(), 1)
    return num_experts * torch.sum(frac * probs.mean(dim=0))


def collect_router_logits(model: nn.Module) -> list[torch.Tensor]:
    out = []
    for m in model.modules():
        if isinstance(m, (SharedPromptRouter, SharedTokenRouter)) and m.last_router_logits is not None:
            out.append(m.last_router_logits)
        elif (
            isinstance(m, MoLELinear)
            and m.shared_router() is None
            and m.shared_token_router() is None
            and m.last_router_logits is not None
        ):
            out.append(m.last_router_logits)
    return out


def aux_loss(model: nn.Module, top_k: int, weight: float = 0.01) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for module in model.modules():
        if (
            isinstance(module, SharedTokenRouter)
            and module.adaptive
            and module.last_router_dense_weights is not None
        ):
            pool = torch.tensor(
                module.adaptive_expert_pool,
                dtype=torch.long,
                device=module.last_router_dense_weights.device,
            )
            active = module.last_router_dense_weights.index_select(-1, pool).float()
            importance = active.sum(dim=0) / active.sum().clamp_min(1e-12)
            # Every globally active lane executes, so a Switch assignment loss
            # would be constant. Penalize collapsed mixture importance instead.
            losses.append(len(pool) * importance.square().sum())
        elif (
            isinstance(module, (SharedPromptRouter, SharedTokenRouter))
            and module.last_router_logits is not None
        ):
            losses.append(load_balance_loss(module.last_router_logits, top_k))
        elif (
            isinstance(module, MoLELinear)
            and module.shared_router() is None
            and module.shared_token_router() is None
            and module.last_router_logits is not None
        ):
            losses.append(load_balance_loss(module.last_router_logits, top_k))
    if not losses:
        reference = next(model.parameters(), None)
        device = None if reference is None else reference.device
        return torch.zeros((), dtype=torch.float32, device=device)
    return weight * sum(losses) / len(losses)


def surprise_calibration_loss(
    model: nn.Module,
    task_nll: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Calibrate cheap per-lane surprise predictions to realized mixture NLL.

    Main language-model gradients decide which lane is useful. This auxiliary
    objective anchors the score's absolute scale in nats, allowing the runtime
    controller to compare it with teacher-forced prompt NLL.
    """

    losses: list[torch.Tensor] = []
    target = task_nll.detach().float()
    for module in model.modules():
        if not isinstance(module, SharedTokenRouter) or not module.adaptive:
            continue
        predicted = module.last_predicted_nll
        dense = module.last_router_dense_weights
        if predicted is None or dense is None:
            continue
        realized_prediction = (predicted.float() * dense.float()).sum(dim=-1).mean()
        losses.append(torch.nn.functional.smooth_l1_loss(realized_prediction, target))
    if not losses:
        return task_nll.new_zeros((), dtype=torch.float32)
    return float(weight) * sum(losses) / len(losses)


def adaptive_candidate_nll(model: nn.Module) -> torch.Tensor | None:
    """Return the latest globally aggregated predicted NLL for every lane."""

    scores = [
        module.last_predicted_nll.detach().float().mean(dim=0)
        for module in model.modules()
        if isinstance(module, SharedTokenRouter)
        and module.adaptive
        and module.last_predicted_nll is not None
    ]
    if not scores:
        return None
    return torch.stack(scores, dim=0).mean(dim=0)


def router_usage_metrics(model: nn.Module) -> dict:
    """Summarize the latest sparse token routes for live training metrics."""

    totals: torch.Tensor | None = None
    adaptive_pool: tuple[int, ...] | None = None
    predicted_nll: list[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, SharedTokenRouter):
            indices = module.last_router_indices
            selected_weights = module.last_router_weights
            if module.adaptive:
                adaptive_pool = module.adaptive_expert_pool
                if module.last_predicted_nll is not None:
                    predicted_nll.append(module.last_predicted_nll.detach().float().mean())
        elif isinstance(module, SharedPromptRouter):
            indices = module.last_router_indices
            dense_weights = module.last_router_weights
            selected_weights = (
                dense_weights.gather(-1, indices)
                if dense_weights is not None and indices is not None
                else None
            )
        else:
            continue
        if indices is None or selected_weights is None:
            continue
        if totals is None:
            totals = torch.zeros(
                module.num_experts,
                dtype=torch.float32,
                device=indices.device,
            )
        totals.scatter_add_(
            0,
            indices.detach().reshape(-1),
            selected_weights.detach().float().reshape(-1),
        )
    if totals is None or totals.sum() <= 0:
        return {}
    share = totals / totals.sum()
    if share.numel() > 1:
        entropy = -(share * share.clamp_min(1e-12).log()).sum() / torch.log(
            share.new_tensor(float(share.numel()))
        )
    else:
        entropy = share.new_tensor(1.0)
    bank_entropy = round(float(entropy.item()), 4)
    result = {
        "router_entropy": bank_entropy,
        "dead_experts": int((totals == 0).sum().item()),
        "expert_token_share": [round(float(value), 4) for value in share.tolist()],
    }
    if adaptive_pool is not None:
        active = torch.tensor(adaptive_pool, dtype=torch.long, device=totals.device)
        active_totals = totals.index_select(0, active)
        active_share = active_totals / active_totals.sum().clamp_min(1e-12)
        if active_share.numel() > 1:
            active_entropy = -(
                active_share * active_share.clamp_min(1e-12).log()
            ).sum() / torch.log(active_share.new_tensor(float(active_share.numel())))
        else:
            active_entropy = active_share.new_tensor(1.0)
        result.update(
            router_entropy=round(float(active_entropy.item()), 4),
            bank_router_entropy=bank_entropy,
            active_experts=len(adaptive_pool),
            inactive_experts=int(totals.numel() - len(adaptive_pool)),
            dead_experts=int((active_totals == 0).sum().item()),
            bank_unselected_experts=int((totals == 0).sum().item()),
        )
    if predicted_nll:
        result["predicted_nll"] = round(float(torch.stack(predicted_nll).mean().item()), 4)
    return result
