from __future__ import annotations

import torch
from torch import nn


class LoRAExpert(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling


class TopKRouter(nn.Module):
    def __init__(self, in_features: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be within [1, num_experts]")
        self.in_features = in_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(in_features, num_experts, bias=False)
        nn.init.zeros_(self.gate.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x.float())
        top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(top_vals, dim=-1)
        return top_idx, weights, logits


class MoLELinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        num_experts: int,
        rank: int,
        alpha: float | None = None,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.in_features = in_f
        self.out_features = out_f
        self.num_experts = num_experts
        self.rank = rank
        self.top_k = min(top_k, num_experts)
        self.experts = nn.ModuleList(
            [
                LoRAExpert(
                    in_f,
                    out_f,
                    rank,
                    alpha if alpha is not None else rank * 2.0,
                    dtype=base.weight.dtype,
                    device=base.weight.device,
                )
                for _ in range(num_experts)
            ]
        )
        self.router = TopKRouter(in_f, num_experts, self.top_k)
        self.last_router_logits: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        xf = x.reshape(-1, self.in_features)
        idx, w, logits = self.router(xf)
        self.last_router_logits = logits
        tokens = xf.shape[0]
        add = torch.zeros(tokens, self.out_features, dtype=y.dtype, device=y.device)
        for e, expert in enumerate(self.experts):
            mask = (idx == e).any(dim=-1)
            if not torch.any(mask):
                continue
            we = torch.where(idx == e, w, torch.zeros_like(w)).sum(dim=-1)
            contrib = expert(xf[mask]) * we[mask].to(y.dtype).unsqueeze(-1)
            add[mask] = add[mask] + contrib
        return y + add.reshape(*x.shape[:-1], self.out_features)
