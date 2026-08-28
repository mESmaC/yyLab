from __future__ import annotations

import torch
from torch import nn

from .layer import MoLELinear


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
        if isinstance(m, MoLELinear) and m.last_router_logits is not None:
            out.append(m.last_router_logits)
    return out


def aux_loss(model: nn.Module, top_k: int, weight: float = 0.01) -> torch.Tensor:
    logits = collect_router_logits(model)
    if not logits:
        return torch.tensor(0.0)
    return weight * sum(load_balance_loss(lg, top_k) for lg in logits) / len(logits)
