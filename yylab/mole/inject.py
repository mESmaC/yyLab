from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .adaptive import AdaptiveOccupancyConfig, ExpertBudgetPlan, solve_expert_budget
from .layer import MoLELinear, SharedPromptRouter, SharedTokenRouter

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
    router_type: str = "per_layer"
    router_hidden_size: int | None = 256
    router_max_tokens: int = 256
    identity_expert: bool = False
    adaptive: AdaptiveOccupancyConfig = field(default_factory=AdaptiveOccupancyConfig)

    def __post_init__(self) -> None:
        if self.num_experts < 1 or self.rank < 1:
            raise ValueError("MoLE expert count and rank must be positive")
        route_count = self.num_experts + int(self.identity_expert)
        if not 1 <= self.top_k <= route_count:
            raise ValueError("MoLE top-k must be within the expert bank")
        if self.adaptive.enabled and self.identity_expert:
            raise ValueError("adaptive occupancy and the Soft LoRA identity route are separate policies")
        if self.adaptive.enabled and self.router_type != "per_token":
            raise ValueError("adaptive occupancy requires per-token routing")
        if self.adaptive.enabled:
            self.adaptive.validate(num_experts=self.num_experts)

    def to_dict(self) -> dict:
        return {
            "num_experts": self.num_experts,
            "rank": self.rank,
            "alpha": self.alpha,
            "top_k": self.top_k,
            "target_modules": list(self.target_modules),
            "router_type": self.router_type,
            "router_hidden_size": self.router_hidden_size,
            "router_max_tokens": self.router_max_tokens,
            "identity_expert": self.identity_expert,
            "adaptive": self.adaptive.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> MoLEConfig:
        if not isinstance(d, dict):
            raise TypeError("MoLE config must be a mapping")
        return cls(
            num_experts=d.get("num_experts", 4),
            rank=d.get("rank", 16),
            alpha=d.get("alpha", 32.0),
            top_k=d.get("top_k", 2),
            target_modules=tuple(d.get("target_modules", DEFAULT_TARGET_MODULES)),
            router_type=d.get("router_type", "per_layer"),
            router_hidden_size=d.get("router_hidden_size", 256),
            router_max_tokens=d.get("router_max_tokens", 256),
            identity_expert=bool(d.get("identity_expert", False)),
            adaptive=AdaptiveOccupancyConfig.from_dict(d.get("adaptive")),
        )


SHARED_ROUTER_MODULE = "_mole_shared_router"
TOKEN_ROUTER_MODULE = "_mole_token_router"
_BLOCK_LIST_NAMES = frozenset({"layers", "h", "blocks", "block"})


def shared_router(model: nn.Module) -> SharedPromptRouter | None:
    router = getattr(model, SHARED_ROUTER_MODULE, None)
    return router if isinstance(router, SharedPromptRouter) else None


def token_routers(model: nn.Module) -> list[tuple[str, SharedTokenRouter]]:
    """Return the once-per-transformer-block learned token routers."""

    return [(name, module) for name, module in model.named_modules() if isinstance(module, SharedTokenRouter)]


def _block_name_for_projection(name: str) -> str:
    """Find the decoder block owning a target projection.

    Hugging Face decoder families conventionally store blocks in ``layers``,
    ``h``, or ``blocks`` module lists. The fallback keeps custom architectures
    usable when their attention/MLP container is one level below the block.
    """

    parts = name.split(".")
    for index in range(len(parts) - 1):
        if parts[index] in _BLOCK_LIST_NAMES and parts[index + 1].isdigit():
            return ".".join(parts[: index + 2])
    if len(parts) >= 3:
        return ".".join(parts[:-2])
    raise ValueError(f"cannot identify the transformer block for projection: {name}")


def _module_at(model: nn.Module, dotted: str) -> nn.Module:
    module = model
    for part in dotted.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def attach_token_routers(
    model: nn.Module,
    cfg: MoLEConfig,
    projections: list[tuple[str, nn.Linear]],
    *,
    router_dtype: torch.dtype | None = None,
) -> dict[str, SharedTokenRouter]:
    """Register one learned per-token route on every adapted decoder block."""

    if cfg.router_type != "per_token":
        raise ValueError("block token routers require router_type='per_token'")
    grouped: dict[str, list[nn.Linear]] = {}
    for name, projection in projections:
        grouped.setdefault(_block_name_for_projection(name), []).append(projection)
    if not grouped:
        raise ValueError("per-token routing requires at least one target projection")

    configured_hidden = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    routers: dict[str, SharedTokenRouter] = {}
    handles = list(getattr(model, "_mole_token_router_hook_handles", ()))
    for block_name, block_projections in grouped.items():
        block = _module_at(model, block_name)
        existing = getattr(block, TOKEN_ROUTER_MODULE, None)
        if isinstance(existing, SharedTokenRouter):
            router = existing
            if router_dtype is not None and router.gate.weight.dtype != router_dtype:
                raise RuntimeError(
                    "existing shared token router dtype does not match the verified "
                    "base compute dtype"
                )
        else:
            hidden_size = configured_hidden or min(p.in_features for p in block_projections)
            reference = min(block_projections, key=lambda projection: projection.in_features)
            router = SharedTokenRouter(
                hidden_size=hidden_size,
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
                dtype=router_dtype,
                device=reference.weight.device,
                adaptive=cfg.adaptive.enabled,
                identity_expert=cfg.identity_expert,
            )
            block.add_module(TOKEN_ROUTER_MODULE, router)

            def prepare_route(_module, args, kwargs, *, routed=router) -> None:
                if not routed.routing_enabled:
                    return
                hidden_states = args[0] if args else kwargs.get("hidden_states")
                if not isinstance(hidden_states, torch.Tensor):
                    raise RuntimeError("token-routed decoder block received no hidden states")
                routed.prepare_route(hidden_states, force=True)

            handles.append(block.register_forward_pre_hook(prepare_route, with_kwargs=True))
        routers[block_name] = router
    object.__setattr__(model, "_mole_token_router_hook_handles", tuple(handles))
    return routers


def attach_shared_router(
    model: nn.Module,
    cfg: MoLEConfig,
    *,
    install_hooks: bool = True,
    router_dtype: torch.dtype | None = None,
) -> SharedPromptRouter:
    """Register one prompt router, optionally wiring it to input embeddings."""

    if cfg.router_type != "shared_prompt":
        raise ValueError("a shared router requires router_type='shared_prompt'")
    existing = shared_router(model)
    if existing is not None:
        if router_dtype is not None:
            existing_dtype = next(existing.parameters()).dtype
            if existing_dtype != router_dtype:
                raise RuntimeError(
                    "existing shared prompt router dtype does not match the verified "
                    "base compute dtype"
                )
        return existing
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_embeddings):
        raise TypeError("shared prompt routing requires model.get_input_embeddings()")
    embeddings = get_embeddings()
    weight = getattr(embeddings, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError("input embeddings must expose a rank-two weight tensor")
    router = SharedPromptRouter(
        hidden_size=weight.shape[1],
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        hidden_width=cfg.router_hidden_size,
        max_prompt_tokens=cfg.router_max_tokens,
        dtype=router_dtype,
        device=weight.device,
        identity_expert=cfg.identity_expert,
    )
    model.add_module(SHARED_ROUTER_MODULE, router)
    if install_hooks:

        def reset_for_forward(_module, _args) -> None:
            if not router.persistent_route:
                router.reset_route()

        def route_embeddings(_module, _args, output) -> None:
            if router.last_router_weights is None:
                router.prepare_route(output)

        handles = (
            model.register_forward_pre_hook(reset_for_forward),
            embeddings.register_forward_hook(route_embeddings),
        )
        object.__setattr__(model, "_mole_router_hook_handles", handles)
    for _, layer in mole_layers(model):
        layer.set_shared_router(router)
    return router


def begin_prompt_route(model: nn.Module) -> None:
    router = shared_router(model)
    if router is not None:
        router.begin_sequence()


def end_prompt_route(model: nn.Module) -> None:
    router = shared_router(model)
    if router is not None:
        router.end_sequence()


def _parent_and_attr(model: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    return parent, parts[-1]


def inject_mole(
    model: nn.Module,
    cfg: MoLEConfig,
    expert_dtype: torch.dtype | None = None,
    expert_device: torch.device | str | None = None,
) -> list[tuple[str, MoLELinear]]:
    if cfg.router_type not in {"per_layer", "per_token", "shared_prompt"}:
        raise ValueError(f"unknown router type: {cfg.router_type}")
    targets = set(cfg.target_modules)
    projections = [
        (name, module)
        for name, module in list(model.named_modules())
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets
    ]
    if not projections:
        return []
    projection_dtypes = {module.weight.dtype for _, module in projections}
    if expert_dtype is None:
        if len(projection_dtypes) != 1:
            raise ValueError(
                "mixed target projection dtypes require an explicit, verified MoLE compute dtype"
            )
        (expert_dtype,) = projection_dtypes
    if expert_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(
            "MoLE compute dtype must be one of FP16, BF16, or FP32; quantized bases "
            "must pass their verified activation compute dtype explicitly"
        )
    floating_projection_dtypes = {
        dtype
        for dtype in projection_dtypes
        if dtype in {torch.float16, torch.bfloat16, torch.float32}
    }
    if floating_projection_dtypes and floating_projection_dtypes != {expert_dtype}:
        raise ValueError(
            "MoLE compute dtype does not match the dense base projection dtype"
        )
    prompt_router = (
        attach_shared_router(model, cfg, router_dtype=expert_dtype)
        if cfg.router_type == "shared_prompt"
        else None
    )
    block_routers = (
        attach_token_routers(model, cfg, projections, router_dtype=expert_dtype)
        if cfg.router_type == "per_token"
        else {}
    )
    replaced: list[tuple[str, MoLELinear]] = []
    for name, module in projections:
        parent, attr = _parent_and_attr(model, name)
        mole = MoLELinear(
            module,
            cfg.num_experts,
            cfg.rank,
            cfg.alpha,
            cfg.top_k,
            expert_dtype=expert_dtype,
            expert_device=expert_device,
            shared_router=prompt_router,
            shared_token_router=(
                block_routers.get(_block_name_for_projection(name)) if block_routers else None
            ),
            adaptive=cfg.adaptive.enabled,
            identity_expert=cfg.identity_expert,
        )
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
    prompt_router = shared_router(model)
    block_routers = token_routers(model)
    for _, layer in mole_layers(model):
        if stage in {"experts", "both"}:
            for expert in layer.experts:
                for p in expert.parameters():
                    p.requires_grad_(True)
        if stage in {"router", "both"} and prompt_router is None and not block_routers:
            if layer.router is None:
                raise RuntimeError("MoLE layer is missing its trainable router")
            for p in layer.router.parameters():
                p.requires_grad_(True)
    if prompt_router is not None and stage in {"router", "both"}:
        prompt_router.requires_grad_(True)
    if block_routers and stage in {"router", "both"}:
        for _, router in block_routers:
            router.requires_grad_(True)


def set_active_expert(model: nn.Module, expert_index: int | None) -> None:
    """Force one whole-adapter route across every injected projection."""

    for _, layer in mole_layers(model):
        layer.set_active_expert(expert_index)
    for _, router in token_routers(model):
        router.routing_enabled = expert_index is None
        router.reset_route()


def set_trainable_expert(model: nn.Module, expert_index: int) -> None:
    """Freeze the complete rack, then optimize and force exactly one expert."""

    set_stage(model, "none")
    set_active_expert(model, expert_index)
    for _, layer in mole_layers(model):
        if not 0 <= expert_index < len(layer.experts):
            raise IndexError(f"expert index {expert_index} is out of range")
        layer.experts[expert_index].requires_grad_(True)


def reset_expert(model: nn.Module, expert_index: int = 0) -> None:
    """Reinitialize one resident expert for the next isolated training slice."""

    for _, layer in mole_layers(model):
        if not 0 <= expert_index < len(layer.experts):
            raise IndexError(f"expert index {expert_index} is out of range")
        layer.experts[expert_index].reset_parameters()


def set_adaptive_expert_pool(
    model: nn.Module,
    expert_ids: tuple[int, ...] | list[int],
) -> None:
    """Apply one coherent global active lane set to every block and projection."""

    pool = tuple(int(index) for index in expert_ids)
    routers = token_routers(model)
    if not routers:
        raise ValueError("adaptive occupancy requires shared block token routers")
    for _, router in routers:
        router.set_adaptive_expert_pool(pool)
    for _, layer in mole_layers(model):
        layer.set_active_experts(pool)


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def resolve_adaptive_budget(model: nn.Module, cfg: MoLEConfig) -> ExpertBudgetPlan:
    """Fit a v4 expert rack to the exact base and selected projection shapes."""

    if not cfg.adaptive.enabled:
        raise ValueError("cannot resolve an adaptive budget for fixed routing")
    if mole_layers(model):
        raise ValueError("adaptive budget must be resolved before MoLE injection")
    targets = set(cfg.target_modules)
    projections = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets
    ]
    if not projections:
        raise ValueError("adaptive expert budget found no matching target projections")
    rank_width = sum(module.in_features + module.out_features for _, module in projections)
    block_names = {_block_name_for_projection(name) for name, _ in projections}
    configured_hidden = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    if configured_hidden < 1:
        configured_hidden = min(module.in_features for _, module in projections)
    # One H→E gate plus one learned predicted-NLL bias per expert and block.
    router_parameters_per_expert = len(block_names) * (configured_hidden + 1)
    logical_counter = getattr(model, "num_parameters", None)
    if callable(logical_counter):
        # Transformers accounts for the logical shape hidden behind packed
        # Params4bit storage. Summing physical parameter tensors would count
        # the local Qwen NF4 base as 375.8M instead of its true 596.0M weights
        # and violate the architecture's 0.5%-of-total contract.
        base_parameters = int(logical_counter())
    else:
        base_parameters = sum(parameter.numel() for parameter in model.parameters())
    if base_parameters < 1:
        raise ValueError("adaptive expert budget found no base-model parameters")
    plan = solve_expert_budget(
        base_parameters=base_parameters,
        expert_rank_width=rank_width,
        router_parameters_per_expert=router_parameters_per_expert,
        target_total_multiplier=cfg.adaptive.target_total_multiplier,
        expert_footprint_fraction=cfg.adaptive.expert_footprint_fraction,
        minimum_experts=cfg.adaptive.minimum_experts,
    )
    cfg.num_experts = plan.num_experts
    cfg.rank = plan.rank
    cfg.top_k = min(cfg.num_experts, cfg.adaptive.minimum_experts)
    cfg.adaptive.base_parameter_count = plan.base_parameters
    cfg.adaptive.router_parameter_count = plan.router_parameters
    cfg.adaptive.expert_parameter_count = plan.parameters_per_expert
    cfg.adaptive.resolved_total_parameter_count = plan.resolved_total_parameters
    cfg.adaptive.resolved_expert_footprint_fraction = plan.expert_footprint_fraction
    cfg.adaptive.validate(num_experts=cfg.num_experts)
    return plan
