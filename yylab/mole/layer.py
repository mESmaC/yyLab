from __future__ import annotations

import weakref

import torch
import torch.nn.functional as F
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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling


class TopKRouter(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_experts: int,
        top_k: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be within [1, num_experts]")
        self.in_features = in_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(in_features, num_experts, bias=False, device=device)
        nn.init.zeros_(self.gate.weight)

    def forward(
        self,
        x: torch.Tensor,
        *,
        top_k: int | None = None,
        allowed_experts: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x.float())
        selected_k = self.top_k if top_k is None else int(top_k)
        if not 1 <= selected_k <= self.num_experts:
            raise ValueError("runtime top-k must be within [1, num_experts]")
        selection_logits = logits
        if allowed_experts is not None:
            if len(set(allowed_experts)) != len(allowed_experts):
                raise ValueError("allowed expert indices must be unique")
            if len(allowed_experts) < selected_k:
                raise ValueError("runtime top-k exceeds the resident expert set")
            if any(not 0 <= index < self.num_experts for index in allowed_experts):
                raise IndexError("allowed expert index is out of range")
            mask = torch.full_like(logits, float("-inf"))
            allowed = torch.tensor(allowed_experts, dtype=torch.long, device=logits.device)
            mask.index_copy_(
                -1,
                allowed,
                logits.index_select(-1, allowed),
            )
            selection_logits = mask
        top_vals, top_idx = torch.topk(selection_logits, selected_k, dim=-1)
        if selected_k == 1:
            # softmax over a one-element top-k set is identically one, which
            # disconnects the task loss from the gate.  Preserve a unit
            # forward weight while differentiating through the selected
            # probability from the complete expert distribution.
            selected_probability = torch.softmax(logits, dim=-1).gather(-1, top_idx)
            weights = torch.ones_like(selected_probability) + (
                selected_probability - selected_probability.detach()
            )
        else:
            weights = torch.softmax(top_vals, dim=-1)
        return top_idx, weights, logits


class SharedTokenRouter(TopKRouter):
    """Choose one learned sparse route per token for a transformer block.

    The router is registered once on its owning block.  Adapted projections
    keep weak references to it and reuse the cached route, which keeps the
    expert slot coherent across attention and MLP projections while avoiding
    a separate gate network for every linear layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        device: torch.device | None = None,
        adaptive: bool = False,
    ) -> None:
        super().__init__(hidden_size, num_experts, top_k, device=device)
        # Legacy per-projection gates remain zero-initialized for artifact
        # compatibility.  A new block router needs a tiny asymmetric start so
        # equal logits do not deterministically collapse every token onto the
        # same top-k experts before the load-balancing signal can act.
        nn.init.normal_(self.gate.weight, mean=0.0, std=1e-3)
        self.hidden_size = hidden_size
        self.adaptive = bool(adaptive)
        if self.adaptive:
            # softplus(1.854...) ~= 2 nats before the learned utility term.
            self.surprise_bias = nn.Parameter(
                torch.full((num_experts,), 1.8545866, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("surprise_bias", None)
        self.last_router_logits: torch.Tensor | None = None
        self.last_router_weights: torch.Tensor | None = None
        self.last_router_indices: torch.Tensor | None = None
        self.last_router_dense_weights: torch.Tensor | None = None
        self.last_predicted_nll: torch.Tensor | None = None
        self._cached_token_count: int | None = None
        self._cached_device: torch.device | None = None
        # Forced-expert evaluation can bypass the otherwise unconditional
        # decoder-block pre-hook without unregistering it. This is runtime
        # state only and never changes the artifact tensor contract.
        self.routing_enabled = True
        self._adaptive_expert_pool: tuple[int, ...] = tuple(range(min(top_k, num_experts)))

    def reset_route(self) -> None:
        self.last_router_logits = None
        self.last_router_weights = None
        self.last_router_indices = None
        self.last_router_dense_weights = None
        self.last_predicted_nll = None
        self._cached_token_count = None
        self._cached_device = None

    def set_adaptive_expert_pool(self, expert_ids: tuple[int, ...] | list[int]) -> None:
        if not self.adaptive:
            raise RuntimeError("fixed token router has no adaptive expert pool")
        pool = tuple(int(index) for index in expert_ids)
        if not pool:
            raise ValueError("adaptive expert pool cannot be empty")
        if len(set(pool)) != len(pool):
            raise ValueError("adaptive expert pool contains duplicate indices")
        if any(not 0 <= index < self.num_experts for index in pool):
            raise IndexError("adaptive expert pool index is out of range")
        self._adaptive_expert_pool = pool
        self.reset_route()

    @property
    def adaptive_expert_pool(self) -> tuple[int, ...]:
        return self._adaptive_expert_pool

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.adaptive:
            return super().forward(x)
        raw_utility = self.gate(x.float())
        assert self.surprise_bias is not None
        predicted_nll = F.softplus(self.surprise_bias - raw_utility)
        selection_utility = -predicted_nll
        pool = self._adaptive_expert_pool
        masked = torch.full_like(selection_utility, float("-inf"))
        allowed = torch.tensor(pool, dtype=torch.long, device=x.device)
        masked.index_copy_(
            -1,
            allowed,
            selection_utility.index_select(-1, allowed),
        )
        active_k = len(pool)
        top_values, indices = torch.topk(masked, active_k, dim=-1)
        weights = torch.softmax(top_values, dim=-1)
        self.last_predicted_nll = predicted_nll
        # Higher logits retain their traditional "more useful" meaning while
        # the calibrated positive quantity is exposed separately as NLL.
        return indices, weights, selection_utility

    def prepare_route(
        self,
        hidden_states: torch.Tensor,
        *,
        force: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and cache the owning block's flattened token routes."""

        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"shared token router expected states ending in hidden width {self.hidden_size}")
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        if force or self.last_router_weights is None:
            indices, weights, logits = self(flat_states)
            self.last_router_indices = indices
            self.last_router_weights = weights
            self.last_router_logits = logits
            self.last_router_dense_weights = torch.zeros_like(logits).scatter(
                -1,
                indices,
                weights.to(dtype=logits.dtype),
            )
            self._cached_token_count = flat_states.shape[0]
            self._cached_device = flat_states.device
        assert self.last_router_indices is not None
        assert self.last_router_weights is not None
        return self.last_router_indices, self.last_router_weights

    def cached_dense_weights(self, activation: torch.Tensor) -> torch.Tensor:
        self.cached_route(activation)
        if self.last_router_dense_weights is None:
            raise RuntimeError("shared token router did not cache dense route weights")
        return self.last_router_dense_weights

    def cached_route(self, activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the route prepared at block entry for a projection activation."""

        if activation.ndim < 2:
            raise RuntimeError("shared token routing expects token-shaped activations")
        if self.last_router_indices is None or self.last_router_weights is None:
            raise RuntimeError("shared token route is not prepared; prepare it from the block input first")
        token_count = activation.numel() // activation.shape[-1]
        if token_count != self._cached_token_count:
            raise RuntimeError("shared token route token count does not match the projection activation")
        if activation.device != self._cached_device:
            raise RuntimeError("shared token route and projection are on different devices")
        return self.last_router_indices, self.last_router_weights


class SharedPromptRouter(nn.Module):
    """Choose one sparse expert mixture for a complete prompt.

    The router is registered once on the top-level model.  Every MoLE
    projection keeps only a weak reference to it, so its parameters and route
    graph are not duplicated hundreds of times in ``state_dict`` or autograd.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        hidden_width: int | None = 256,
        max_prompt_tokens: int = 256,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_experts < 1:
            raise ValueError("router dimensions must be positive")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be within [1, num_experts]")
        if hidden_width is not None and hidden_width < 1:
            raise ValueError("router hidden width must be positive or None")
        if max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be positive")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_width = hidden_width
        self.max_prompt_tokens = max_prompt_tokens
        if hidden_width is None:
            self.network = nn.Sequential(
                nn.LayerNorm(hidden_size, device=device),
                nn.Linear(hidden_size, num_experts, device=device),
            )
        else:
            self.network = nn.Sequential(
                nn.LayerNorm(hidden_size, device=device),
                nn.Linear(hidden_size, hidden_width, device=device),
                nn.SiLU(),
                nn.Linear(hidden_width, num_experts, device=device),
            )
        self.last_router_logits: torch.Tensor | None = None
        self.last_router_weights: torch.Tensor | None = None
        self.last_router_indices: torch.Tensor | None = None
        self.persistent_route = False

    def reset_route(self) -> None:
        self.last_router_logits = None
        self.last_router_weights = None
        self.last_router_indices = None

    def begin_sequence(self) -> None:
        """Keep the first prompt decision throughout cached generation."""

        self.reset_route()
        self.persistent_route = True

    def end_sequence(self) -> None:
        self.persistent_route = False
        self.reset_route()

    def prompt_features(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"shared router expected hidden width {self.hidden_size}, got {hidden_states.shape[-1]}"
            )
        if hidden_states.ndim == 2:
            if attention_mask is not None:
                raise ValueError("attention_mask requires sequence-shaped hidden states")
            return hidden_states.float()
        if hidden_states.ndim != 3:
            raise ValueError("shared prompt routing expects [batch, sequence, hidden] states")
        hidden_states = hidden_states[:, : self.max_prompt_tokens]
        if attention_mask is None:
            return hidden_states.float().mean(dim=1)
        mask = attention_mask[:, : hidden_states.shape[1]]
        if mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must match the routed prompt tokens")
        mask = mask.to(device=hidden_states.device, dtype=torch.float32)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (hidden_states.float() * mask.unsqueeze(-1)).sum(dim=1) / count

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.hidden_size:
            raise ValueError(f"router features must have shape [batch, {self.hidden_size}]")
        return self.network(features.float())

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.logits_from_features(self.prompt_features(hidden_states, attention_mask))
        probabilities = torch.softmax(logits.float(), dim=-1)
        top_values, indices = torch.topk(probabilities, self.top_k, dim=-1)
        if self.top_k == 1:
            # A literal p/p normalization is exactly one and has zero
            # derivative, so a top-1 router could never learn from task loss.
            # Keep the coherent expert's forward weight at one while using the
            # selected softmax probability as a straight-through gradient path.
            top_values = torch.ones_like(top_values) + (top_values - top_values.detach())
        else:
            top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = torch.zeros_like(probabilities).scatter(-1, indices, top_values)
        return indices, weights, logits

    def prepare_route(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        force: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if force or self.last_router_weights is None:
            indices, weights, logits = self(hidden_states, attention_mask)
            self.last_router_indices = indices
            self.last_router_weights = weights
            self.last_router_logits = logits
        assert self.last_router_indices is not None
        assert self.last_router_weights is not None
        return self.last_router_indices, self.last_router_weights


class MoLELinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        num_experts: int,
        rank: int,
        alpha: float | None = None,
        top_k: int = 2,
        expert_dtype: torch.dtype | None = None,
        expert_device: torch.device | str | None = None,
        shared_router: SharedPromptRouter | None = None,
        shared_token_router: SharedTokenRouter | None = None,
        adaptive: bool = False,
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
        self.adaptive = bool(adaptive)
        dtype = expert_dtype or base.weight.dtype
        if dtype in (torch.uint8, torch.int8):
            dtype = torch.bfloat16
        self.experts = nn.ModuleList(
            [
                LoRAExpert(
                    in_f,
                    out_f,
                    rank,
                    alpha if alpha is not None else rank * 2.0,
                    dtype=dtype,
                    device=(
                        torch.device(expert_device)
                        if expert_device is not None
                        else base.weight.device
                    ),
                )
                for _ in range(num_experts)
            ]
        )
        # Keep the legacy per-projection gate present for artifact compatibility.
        # A shared router overrides it at runtime without changing the expert-bank keys.
        self.router = (
            None
            if self.adaptive and shared_token_router is not None
            else TopKRouter(in_f, num_experts, self.top_k).to(device=base.weight.device)
        )
        object.__setattr__(
            self,
            "_shared_router_ref",
            None if shared_router is None else weakref.ref(shared_router),
        )
        object.__setattr__(
            self,
            "_shared_token_router_ref",
            None if shared_token_router is None else weakref.ref(shared_token_router),
        )
        self.last_router_logits: torch.Tensor | None = None
        self.active_expert: int | None = None
        self.resident_expert_ids: tuple[int, ...] = tuple(range(self.top_k))
        self.register_buffer("_packed_active_a", None, persistent=False)
        self.register_buffer("_packed_active_b", None, persistent=False)
        self._packed_active_key: tuple[int, ...] | None = None

    def shared_router(self) -> SharedPromptRouter | None:
        ref = self._shared_router_ref
        return None if ref is None else ref()

    def set_shared_router(self, router: SharedPromptRouter | None) -> None:
        if router is not None and router.num_experts != self.num_experts:
            raise ValueError("shared router expert count does not match the MoLE layer")
        object.__setattr__(
            self,
            "_shared_router_ref",
            None if router is None else weakref.ref(router),
        )

    def shared_token_router(self) -> SharedTokenRouter | None:
        ref = self._shared_token_router_ref
        return None if ref is None else ref()

    def set_shared_token_router(self, router: SharedTokenRouter | None) -> None:
        if router is not None and router.num_experts != self.num_experts:
            raise ValueError("shared token router expert count does not match the MoLE layer")
        if router is not None and router.top_k != self.top_k:
            raise ValueError("shared token router top-k does not match the MoLE layer")
        object.__setattr__(
            self,
            "_shared_token_router_ref",
            None if router is None else weakref.ref(router),
        )

    def set_active_expert(self, expert_index: int | None) -> None:
        if expert_index is not None and not 0 <= expert_index < self.num_experts:
            raise IndexError(f"expert index {expert_index} is out of range")
        self.active_expert = expert_index
        self.last_router_logits = None

    def set_active_experts(self, expert_ids: tuple[int, ...] | list[int]) -> None:
        if not self.adaptive:
            raise RuntimeError("fixed MoLE layer has no adaptive active set")
        resident = tuple(int(index) for index in expert_ids)
        if not resident:
            raise ValueError("adaptive active set cannot be empty")
        if len(set(resident)) != len(resident):
            raise ValueError("adaptive active set contains duplicate indices")
        if any(not 0 <= index < self.num_experts for index in resident):
            raise IndexError("adaptive active expert index is out of range")
        if resident == self.resident_expert_ids:
            return
        self.resident_expert_ids = resident
        self._packed_active_a = None
        self._packed_active_b = None
        self._packed_active_key = None

    def set_resident_experts(self, expert_ids: tuple[int, ...] | list[int]) -> None:
        """Compatibility alias for the original adaptive prototype API."""

        self.set_active_experts(expert_ids)

    def _shared_prompt_delta(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        prompt_weights: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Apply the selected whole-adapter experts without synchronizing the GPU.

        Routing is sparse over samples rather than tokens.  Grouping by the
        (usually very small) top-k route slots avoids the old loop over every
        expert, whose ``torch.any(mask)`` branch copied a CUDA scalar back to
        Python once per expert and projection.  The expert bank is packed per
        projection, but only the selected top-k contributions are evaluated;
        no dense sequence-sized activation over every expert is constructed.
        """

        batched_x = x if x.ndim >= 3 else x.unsqueeze(0)
        if prompt_weights.shape[0] != batched_x.shape[0]:
            raise RuntimeError("shared router batch does not match the MoLE activation batch")
        if indices.shape != (batched_x.shape[0], self.top_k):
            raise RuntimeError("shared router indices do not match the MoLE activation batch and top-k")

        batch_size = batched_x.shape[0]
        flat_x = batched_x.reshape(batch_size, -1, self.in_features)
        packed_a = torch.stack([expert.lora_A for expert in self.experts], dim=0)
        packed_b = torch.stack([expert.lora_B for expert in self.experts], dim=0)
        selected_weights = prompt_weights.gather(-1, indices)
        add: torch.Tensor | None = None

        # top_k is a static, small configuration value (normally 1 or 2).
        # This loop contains no tensor-valued condition and therefore does not
        # force a device synchronization.
        for route_slot in range(self.top_k):
            selected = indices[:, route_slot]
            selected_a = packed_a.index_select(0, selected)
            selected_b = packed_b.index_select(0, selected)
            low_rank = torch.bmm(flat_x, selected_a.transpose(1, 2))
            contribution = torch.bmm(low_rank, selected_b.transpose(1, 2))
            scale = selected_weights[:, route_slot].to(contribution.dtype).view(batch_size, 1, 1)
            contribution = contribution * (scale * self.experts[0].scaling)
            contribution = contribution.to(output_dtype)
            add = contribution if add is None else add + contribution

        assert add is not None
        return add.reshape(*batched_x.shape[:-1], self.out_features)

    def _token_delta(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        route_weights: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Apply learned top-k routes in packed low-rank space.

        A direct dynamic-weight implementation gathers one complete LoRA A/B
        pair for every token route.  Besides being expensive, that makes the
        saved activation scale as ``tokens * top_k * (input + output) * rank``.
        Evaluating every expert output first is worse: it creates a
        ``tokens * experts * output`` tensor.

        LoRA's factorization lets us route between the two projections.  The
        first GEMM produces ``tokens * experts * rank`` channels, the sparse
        top-k gate masks those channels, and the second GEMM reduces the whole
        rack into the output.  Thus routing is fully vectorized, has no CUDA
        scalar readback or per-expert launch, and its activation size follows
        total adapter rank.  This is particularly useful when trading rank for
        more experts (for example E4/r16 and E8/r8 have the same routed width).
        """

        if x.ndim != 2 or x.shape[-1] != self.in_features:
            raise RuntimeError("token routing expects flattened [tokens, input] states")
        if indices.shape != (x.shape[0], self.top_k):
            raise RuntimeError("token router indices must have shape [tokens, top_k]")
        if route_weights.shape != indices.shape:
            raise RuntimeError("token router weights must match its indices")

        # Concatenation preserves the existing per-expert Parameter/module
        # layout (and therefore every artifact key) while presenting each side
        # of the rack as one GEMM to the accelerator.
        packed_a = torch.cat(tuple(expert.lora_A for expert in self.experts), dim=0)
        packed_b = torch.cat(tuple(expert.lora_B.transpose(0, 1) for expert in self.experts), dim=0)
        total_rank = self.num_experts * self.rank
        low_rank = (x @ packed_a.transpose(0, 1)).reshape(x.shape[0], self.num_experts, self.rank)

        sparse_gate = low_rank.new_zeros((x.shape[0], self.num_experts))
        sparse_gate.scatter_(1, indices, route_weights.to(dtype=low_rank.dtype))
        routed_low_rank = (low_rank * sparse_gate.unsqueeze(-1)).reshape(x.shape[0], total_rank)
        delta = routed_low_rank @ packed_b
        delta = delta * self.experts[0].scaling
        return delta.to(output_dtype)

    def _adaptive_token_delta(
        self,
        x: torch.Tensor,
        dense_route_weights: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute only the globally active expert lanes with two packed GEMMs.

        The fixed v1-v3 path packs the complete E*r rack. Adaptive v4 packs
        only K active lanes, so both GEMM width and saved low-rank activations
        scale with actual occupancy. Eval-mode packs are cached for the active
        prefix; training rebuilds them so gradients reach the live parameters.
        """

        if x.ndim != 2 or x.shape[-1] != self.in_features:
            raise RuntimeError("adaptive token routing expects [tokens, input] states")
        if dense_route_weights.shape != (x.shape[0], self.num_experts):
            raise RuntimeError("adaptive dense route weights have the wrong shape")
        for expert_index in self.resident_expert_ids:
            expert = self.experts[expert_index]
            if expert.lora_A.is_meta or expert.lora_A.device != x.device:
                raise RuntimeError(
                    f"adaptive expert {expert_index} is not resident on {x.device}"
                )
        active_key = self.resident_expert_ids
        packed_a = self._packed_active_a
        packed_b = self._packed_active_b
        cache_valid = (
            not self.training
            and self._packed_active_key == active_key
            and packed_a is not None
            and packed_b is not None
            and packed_a.device == x.device
        )
        if not cache_valid:
            packed_a = torch.cat(
                [self.experts[index].lora_A for index in active_key],
                dim=0,
            )
            packed_b = torch.cat(
                [self.experts[index].lora_B for index in active_key],
                dim=1,
            )
            if not self.training:
                self._packed_active_a = packed_a.detach()
                self._packed_active_b = packed_b.detach()
                self._packed_active_key = active_key
        active_count = len(active_key)
        low_rank = (x @ packed_a.t()).reshape(x.shape[0], active_count, self.rank)
        active_indices = torch.tensor(active_key, dtype=torch.long, device=x.device)
        active_weights = dense_route_weights.index_select(-1, active_indices)
        weighted = low_rank * active_weights.to(low_rank.dtype).unsqueeze(-1)
        delta = weighted.reshape(x.shape[0], active_count * self.rank) @ packed_b.t()
        return (delta * self.experts[active_key[0]].scaling).to(output_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.active_expert is not None:
            # Stage-1 training follows molor's isolated-adapter lifecycle: one
            # coherent whole-adapter expert is forced across every projection.
            # No per-token router graph or inactive expert activation is built.
            delta = self.experts[self.active_expert](x)
            return y + delta.to(y.dtype)
        shared = self.shared_router()
        if shared is not None:
            indices, prompt_weights = shared.prepare_route(x)
            batched_y = y if y.ndim >= 3 else y.unsqueeze(0)
            add = self._shared_prompt_delta(x, indices, prompt_weights, batched_y.dtype)
            routed = batched_y + add
            return routed if x.ndim >= 3 else routed.squeeze(0)

        xf = x.reshape(-1, self.in_features)
        token_router = self.shared_token_router()
        if token_router is not None:
            idx, w = token_router.cached_route(x)
            if self.adaptive:
                dense = token_router.cached_dense_weights(x)
                add = self._adaptive_token_delta(xf, dense, y.dtype)
                return y + add.reshape(*x.shape[:-1], self.out_features)
            add = self._token_delta(xf, idx, w, y.dtype)
            return y + add.reshape(*x.shape[:-1], self.out_features)

        if self.router is None:
            raise RuntimeError("MoLE layer has no compatible router")
        idx, w, logits = self.router(xf)
        self.last_router_logits = logits
        add = self._token_delta(xf, idx, w, y.dtype)
        return y + add.reshape(*x.shape[:-1], self.out_features)
