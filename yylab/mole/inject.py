from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .layer import MoLELinear

DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class MoLEConfig:
    num_experts: int = 4
    rank: int = 16
    alpha: float = 32.0
    top_k: int = 2
    target_modules: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TARGET_MODULES)

    def to_dict(self) -> dict:
        return {
            "num_experts": self.num_experts,
            "rank": self.rank,
            "alpha": self.alpha,
            "top_k": self.top_k,
            "target_modules": list(self.target_modules),
        }

    @classmethod
    def from_dict(cls, d: dict) -> MoLEConfig:
        return cls(
            num_experts=d.get("num_experts", 4),
            rank=d.get("rank", 16),
            alpha=d.get("alpha", 32.0),
            top_k=d.get("top_k", 2),
            target_modules=tuple(d.get("target_modules", DEFAULT_TARGET_MODULES)),
        )


def _parent_and_attr(model: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    return parent, parts[-1]


def inject_mole(
    model: nn.Module, cfg: MoLEConfig, expert_dtype: torch.dtype | None = None
) -> list[tuple[str, MoLELinear]]:
    replaced: list[tuple[str, MoLELinear]] = []
    targets = set(cfg.target_modules)
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            parent, attr = _parent_and_attr(model, name)
            mole = MoLELinear(module, cfg.num_experts, cfg.rank, cfg.alpha, cfg.top_k, expert_dtype=expert_dtype)
            setattr(parent, attr, mole)
            replaced.append((name, mole))
    return replaced


def mole_layers(model: nn.Module) -> list[tuple[str, MoLELinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, MoLELinear)]


def set_stage(model: nn.Module, stage: str) -> None:
    if stage not in {"experts", "router", "both", "none"}:
        raise ValueError(f"unknown stage: {stage}")
    for p in model.parameters():
        p.requires_grad_(False)
    if stage == "none":
        return
    for _, layer in mole_layers(model):
        if stage in {"experts", "both"}:
            for expert in layer.experts:
                for p in expert.parameters():
                    p.requires_grad_(True)
        if stage in {"router", "both"}:
            for p in layer.router.parameters():
                p.requires_grad_(True)


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
