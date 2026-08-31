from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .adaptive import AdaptiveOccupancyController, ControllerCheckpoint, OccupancyDecision
from .inject import MoLEConfig, mole_layers, shared_router, token_routers
from .store import (
    SHARED_ROUTER_PREFIX,
    TOKEN_ROUTER_PREFIX,
    LazyExpertStore,
)

TelemetryCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class ResidencySnapshot:
    resident_experts: tuple[int, ...]
    active_experts: tuple[int, ...]
    resident_bytes: int
    lane_loads: int
    h2d_transfers: int
    h2d_bytes: int
    decode_started: bool


def _parameter_from_tensor(tensor: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(tensor.contiguous(), requires_grad=False)


def _meta_parameter_like(parameter: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(
        torch.empty(tuple(parameter.shape), dtype=parameter.dtype, device="meta"),
        requires_grad=False,
    )


def apply_adaptive_routing_weights(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> int:
    """Strictly load only router tensors into a meta-injected v4 model.

    ``apply_mole_weights`` intentionally validates and copies a complete legacy
    expert bank. Adaptive inference keeps those parameters on ``meta`` until a
    request establishes its physical residency set, so its routing state needs
    a small, separately strict loader.
    """

    targets: list[tuple[str, nn.Module, str]] = []
    expected: dict[str, torch.Tensor] = {}
    for name, layer in mole_layers(model):
        if layer.shared_token_router() is None and layer.router is not None:
            prefix = f"{name.replace('.', '__')}.router."
            targets.append((prefix, layer.router, name))
            for parameter_name, tensor in layer.router.state_dict().items():
                expected[f"{prefix}{parameter_name}"] = tensor
    for name, router in token_routers(model):
        prefix = f"{TOKEN_ROUTER_PREFIX}{name}."
        targets.append((prefix, router, name))
        for parameter_name, tensor in router.state_dict().items():
            expected[f"{prefix}{parameter_name}"] = tensor
    prompt_router = shared_router(model)
    if prompt_router is not None:
        targets.append((SHARED_ROUTER_PREFIX, prompt_router, "shared prompt router"))
        for parameter_name, tensor in prompt_router.state_dict().items():
            expected[f"{SHARED_ROUTER_PREFIX}{parameter_name}"] = tensor

    actual_keys = set(state)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"adaptive routing tensor set mismatch; missing={missing[:3]}, extra={extra[:3]}")
    for key, reference in expected.items():
        if tuple(state[key].shape) != tuple(reference.shape):
            raise ValueError(
                f"adaptive routing tensor shape mismatch for {key}: "
                f"got {tuple(state[key].shape)}, expected {tuple(reference.shape)}"
            )

    loaded = 0
    for prefix, module, label in targets:
        module_state = {
            key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)
        }
        if not module_state:
            raise ValueError(f"adaptive routing state is missing {label}")
        module.load_state_dict(module_state, strict=True)
        loaded += len(module_state)
    return loaded


class AdaptiveExpertRuntime:
    """Request-scoped occupancy over a request-fixed physical expert pool."""

    def __init__(
        self,
        model: nn.Module,
        config: MoLEConfig,
        store: LazyExpertStore,
        *,
        device: torch.device | str,
        prompt_probe_tokens: int = 128,
        resident_limit: int | None = None,
    ) -> None:
        if not config.adaptive.enabled:
            raise ValueError("adaptive runtime requires an enabled v4 MoLE config")
        if int(store.payload.get("format_version", 0)) != 4:
            raise ValueError("adaptive runtime requires a MoLE v4 artifact")
        if store.num_experts != config.num_experts:
            raise ValueError("adaptive runtime expert count does not match its artifact")
        if prompt_probe_tokens < 1:
            raise ValueError("prompt NLL probe window must be positive")
        self.model = model
        self.config = config
        self.store = store
        self.device = torch.device(device)
        self.prompt_probe_tokens = int(prompt_probe_tokens)
        self.resident_limit = None if resident_limit is None else int(resident_limit)
        self.controller = AdaptiveOccupancyController(config.adaptive, config.num_experts)
        self._layers = mole_layers(model)
        self._routers = token_routers(model)
        if not self._layers or not self._routers:
            raise ValueError("adaptive runtime requires injected MoLE layers and token routers")
        if any(not layer.adaptive for _, layer in self._layers):
            raise ValueError("adaptive runtime found a fixed-routing MoLE layer")
        if any(not router.adaptive for _, router in self._routers):
            raise ValueError("adaptive runtime found a fixed token router")

        self._resident_ids: tuple[int, ...] = ()
        self._active_ids: tuple[int, ...] = ()
        self._request_order: tuple[int, ...] = self._expert_order()
        self._decode_started = False
        self._decode_initial_count: int | None = None
        self._decode_checkpoint: ControllerCheckpoint | None = None
        self._decode_attempts = 0
        self._cache_retries = 0
        self._lane_loads = 0
        self._h2d_transfers = 0
        self._h2d_bytes = 0
        self._prompt_nlls: list[float] = []
        self._routing_probe_nll: float | None = None
        self._predicted_nll_by_expert: list[float] = []
        self._residency_reranked = False
        self._kv_reserve_bytes = 0
        self._last_decision: OccupancyDecision | None = None
        self._callback: TelemetryCallback | None = None
        self._lock = threading.RLock()

        store.validate_model(model)
        apply_adaptive_routing_weights(model, store.routing_weights())

    @property
    def resident_experts(self) -> tuple[int, ...]:
        return self._resident_ids

    @property
    def active_experts(self) -> tuple[int, ...]:
        return self._active_ids

    @property
    def decode_started(self) -> bool:
        return self._decode_started

    def set_telemetry_callback(self, callback: TelemetryCallback | None) -> None:
        self._callback = callback

    def _emit(self, phase: str, **extra: object) -> None:
        callback = self._callback
        if callback is None:
            return
        payload = {"phase": phase, **self.telemetry(), **extra}
        try:
            callback(payload)
        except Exception:
            # Telemetry is observational and must never terminate generation.
            return

    def _expert_order(self) -> tuple[int, ...]:
        preferred = tuple(self.config.adaptive.preferred_experts)
        remaining = tuple(index for index in range(self.config.num_experts) if index not in set(preferred))
        return (*preferred, *remaining)

    def _resident_capacity(self) -> int:
        configured = self.controller.configured_maximum
        if self.resident_limit is not None:
            return min(configured, max(0, self.resident_limit))
        if self.device.type != "cuda":
            return configured
        free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
        # Existing resident lanes are recoverable budget when a new request
        # replaces or shrinks the pool.
        reusable = len(self._resident_ids) * self.store.bytes_per_expert
        reserve = self.config.adaptive.vram_reserve_bytes + self._kv_reserve_bytes
        available = int(free_bytes) + reusable - reserve
        if self.store.bytes_per_expert <= 0:
            raise ValueError("adaptive artifact does not declare a positive lane size")
        return min(configured, max(0, available // self.store.bytes_per_expert))

    def _move_lane_tensor(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        value = tensor.to(dtype=dtype)
        non_blocking = False
        if self.device.type == "cuda":
            if self.config.adaptive.pin_host_memory and value.device.type == "cpu":
                try:
                    value = value.pin_memory()
                    non_blocking = True
                except RuntimeError:
                    non_blocking = False
            self._h2d_transfers += 1
            self._h2d_bytes += value.numel() * value.element_size()
        return value.to(device=self.device, non_blocking=non_blocking)

    def _evict_expert(self, expert_index: int) -> None:
        for _name, layer in self._layers:
            expert = layer.experts[expert_index]
            if not expert.lora_A.is_meta:
                expert.lora_A = _meta_parameter_like(expert.lora_A)
            if not expert.lora_B.is_meta:
                expert.lora_B = _meta_parameter_like(expert.lora_B)

    def _load_expert(self, expert_index: int) -> None:
        state = self.store.expert_weights(expert_index)
        expected: set[str] = set()
        for name, layer in self._layers:
            key = name.replace(".", "__")
            a_key = f"{key}.expert_{expert_index}.lora_A"
            b_key = f"{key}.expert_{expert_index}.lora_B"
            expected.update((a_key, b_key))
            expert = layer.experts[expert_index]
            a = self._move_lane_tensor(state[a_key], expert.lora_A.dtype)
            b = self._move_lane_tensor(state[b_key], expert.lora_B.dtype)
            expert.lora_A = _parameter_from_tensor(a)
            expert.lora_B = _parameter_from_tensor(b)
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise ValueError(
                f"adaptive expert {expert_index} tensor set mismatch; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        self._lane_loads += 1

    def establish_residency(
        self,
        expert_order: tuple[int, ...] | list[int] | None = None,
        *,
        resident_count: int | None = None,
    ) -> tuple[int, ...]:
        """Materialize the request pool; forbidden after decoding begins."""

        with self._lock:
            if self._decode_started:
                raise RuntimeError("expert residency is fixed after decoding starts")
            order = self._expert_order() if expert_order is None else tuple(expert_order)
            if len(order) != self.config.num_experts or set(order) != set(
                range(self.config.num_experts)
            ):
                raise ValueError("expert residency order must be a permutation of every expert")
            capacity = self._resident_capacity()
            self.controller.set_memory_maximum(capacity)
            if resident_count is None:
                selected_count = capacity
            else:
                selected_count = int(resident_count)
                if not self.controller.minimum <= selected_count <= capacity:
                    raise ValueError("requested resident count is outside the memory bounds")
            selected = order[:selected_count]
            selected_set = set(selected)
            for expert_index in self._resident_ids:
                if expert_index not in selected_set:
                    self._evict_expert(expert_index)
            already = set(self._resident_ids)
            for expert_index in selected:
                if expert_index not in already:
                    self._load_expert(expert_index)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            self._resident_ids = selected
            self._request_order = order
            self._set_active_count(self.controller.current_experts)
            return selected

    def _prompt_ranked_expert_order(self) -> tuple[int, ...]:
        """Rank lanes by mean router-predicted NLL over prompt tokens/layers."""

        predictions: list[torch.Tensor] = []
        for _name, router in self._routers:
            values = router.last_predicted_nll
            if values is None:
                continue
            if values.shape[-1] != self.config.num_experts:
                raise RuntimeError("adaptive router returned an unexpected predicted-NLL width")
            predictions.append(values.detach().float().reshape(-1, self.config.num_experts))
        if not predictions:
            self._predicted_nll_by_expert = []
            return self._expert_order()

        scores = torch.cat(predictions, dim=0).mean(dim=0).cpu().tolist()
        self._predicted_nll_by_expert = [float(score) for score in scores]
        prior = {expert: index for index, expert in enumerate(self._expert_order())}
        return tuple(
            sorted(
                range(self.config.num_experts),
                key=lambda expert: (
                    not math.isfinite(scores[expert]),
                    scores[expert] if math.isfinite(scores[expert]) else float("inf"),
                    prior[expert],
                ),
            )
        )

    def _restore_active_count(self, count: int, *, reason: str) -> None:
        self.controller.restore_active_experts(count, reason=reason)
        self._set_active_count(count)

    def _set_active_count(self, count: int) -> None:
        count = int(count)
        if not self._resident_ids:
            raise RuntimeError("adaptive expert residency has not been established")
        if not self.controller.minimum <= count <= len(self._resident_ids):
            raise ValueError("active expert count is outside the resident occupancy bounds")
        active = self._resident_ids[:count]
        for _name, layer in self._layers:
            # The layer API names the set by physical residency; in adaptive
            # execution it is also the exact set whose LoRA lanes execute.
            layer.set_resident_experts(active)
        for _name, router in self._routers:
            router.set_adaptive_expert_pool(active)
        self._active_ids = active

    def snapshot(self) -> ResidencySnapshot:
        with self._lock:
            return ResidencySnapshot(
                resident_experts=self._resident_ids,
                active_experts=self._active_ids,
                resident_bytes=len(self._resident_ids) * self.store.bytes_per_expert,
                lane_loads=self._lane_loads,
                h2d_transfers=self._h2d_transfers,
                h2d_bytes=self._h2d_bytes,
                decode_started=self._decode_started,
            )

    def prompt_nll(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> float:
        """Teacher-force a bounded prompt window without creating a KV cache."""

        if self._decode_started:
            raise RuntimeError("prompt NLL probing is unavailable after decoding starts")
        sequence = int(input_ids.shape[-1])
        if sequence < 2:
            return 0.0
        count = min(self.prompt_probe_tokens, sequence - 1)
        start = sequence - count - 1
        positions = torch.arange(start, sequence - 1, device=input_ids.device)
        with torch.inference_mode():
            try:
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=positions,
                )
            except TypeError as exc:
                if "logits_to_keep" not in str(exc):
                    raise
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        logits = output.logits if hasattr(output, "logits") else output[0]
        if logits.shape[1] == sequence:
            logits = logits.index_select(1, positions)
        elif logits.shape[1] != count:
            raise RuntimeError("prompt NLL model returned an unexpected logits window")
        targets = input_ids.index_select(1, positions + 1)
        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        )
        if attention_mask is not None:
            valid = attention_mask.index_select(1, positions + 1).reshape(-1).bool()
            losses = losses[valid]
        return 0.0 if losses.numel() == 0 else float(losses.mean().item())

    def prepare_request(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        telemetry_callback: TelemetryCallback | None = None,
        kv_reserve_bytes: int = 0,
    ) -> dict[str, object]:
        with self._lock:
            if self._decode_started:
                raise RuntimeError("cannot prepare a new request while decoding")
            if int(kv_reserve_bytes) < 0:
                raise ValueError("KV reserve must be non-negative")
            self.set_telemetry_callback(telemetry_callback)
            self._kv_reserve_bytes = int(kv_reserve_bytes)
            self.controller.reset_sequence()
            self._prompt_nlls = []
            self._routing_probe_nll = None
            self._predicted_nll_by_expert = []
            self._residency_reranked = False
            self._decode_initial_count = None
            self._decode_checkpoint = None
            self._decode_attempts = 0
            self._cache_retries = 0
            self._last_decision = None
            self.establish_residency(
                resident_count=(
                    self.controller.minimum
                    if self.config.adaptive.prefill_nll_probe
                    else None
                )
            )
            self._emit("resident")
            if not self.config.adaptive.prefill_nll_probe:
                return self.telemetry()

            # The first teacher-forced pass is a routing probe. Adaptive token
            # routers expose a predicted NLL for every lane even though only
            # the current active prefix executes. Rank the physical pool from
            # those full-E scores, materialize any replacements, then rerun the
            # prompt so occupancy decisions use the prompt-specific pool.
            routing_probe_nll = self.prompt_nll(input_ids, attention_mask)
            self._prompt_nlls.append(routing_probe_nll)
            self._routing_probe_nll = routing_probe_nll
            ranked_order = self._prompt_ranked_expert_order()
            ranked_residency = self.establish_residency(ranked_order)
            static_residency = self._expert_order()[: len(ranked_residency)]
            self._residency_reranked = ranked_residency != static_residency
            self._emit(
                "routing_probe",
                prompt_nll=routing_probe_nll,
                predicted_nll_by_expert=list(self._predicted_nll_by_expert),
                ranked_experts=list(ranked_order),
                residency_reranked=self._residency_reranked,
            )

            accepted_nll = self.prompt_nll(input_ids, attention_mask)
            self._prompt_nlls.append(accepted_nll)
            proposal = self.controller.observe(accepted_nll, "nll")
            self._last_decision = proposal
            accepted_count = proposal.previous_experts
            self._set_active_count(proposal.active_experts)
            self._emit(
                "prompt_probe",
                prompt_nll=accepted_nll,
                marginal_nll_gain=None,
                decision=asdict(proposal),
            )

            recomputations = 0
            while proposal.action == "expand":
                if recomputations >= self.config.adaptive.max_recomputations:
                    reason = "prompt recomputation budget exhausted before proposal evaluation"
                    self._restore_active_count(accepted_count, reason=reason)
                    self._emit("prompt_rollback", reason=reason)
                    break

                # ``observe`` has proposed and selected the next K. Evaluate
                # exactly that K before another observation is allowed to
                # mutate occupancy again.
                candidate_count = proposal.active_experts
                self._set_active_count(candidate_count)
                candidate_nll = self.prompt_nll(input_ids, attention_mask)
                self._prompt_nlls.append(candidate_nll)
                recomputations += 1
                evaluation = self.controller.evaluate_recomputation(
                    accepted_nll,
                    candidate_nll,
                )
                self._emit(
                    "prompt_probe",
                    prompt_nll=candidate_nll,
                    marginal_nll_gain=evaluation.marginal_gain_nats,
                    decision=asdict(proposal),
                    recomputation=asdict(evaluation),
                )
                if not evaluation.accepted:
                    self._restore_active_count(accepted_count, reason=evaluation.reason)
                    self._emit(
                        "prompt_rollback",
                        rejected_expert_count=candidate_count,
                        reason=evaluation.reason,
                    )
                    break

                accepted_count = candidate_count
                accepted_nll = candidate_nll
                proposal = self.controller.observe(accepted_nll, "nll")
                self._last_decision = proposal
                if proposal.action == "expand":
                    self._set_active_count(proposal.active_experts)
                    continue
                if proposal.action == "contract":
                    # A contraction here is another unmeasured proposal. Keep
                    # the last explicitly accepted prefix for decode.
                    self._restore_active_count(
                        accepted_count,
                        reason="retained last evaluated prompt occupancy",
                    )
                    break
                self._set_active_count(accepted_count)
            return self.telemetry()

    def begin_decode(self) -> None:
        with self._lock:
            if not self._resident_ids:
                raise RuntimeError("adaptive request must establish residency before decoding")
            if self._decode_started:
                raise RuntimeError("adaptive decoding already started")
            self._decode_initial_count = len(self._active_ids)
            self._decode_checkpoint = self.controller.checkpoint_sequence_state()
            self._decode_attempts = 1
            self._cache_retries = 0
            for _name, router in self._routers:
                router.reset_route()
            self._decode_started = True
            self._emit("decode_start")

    def restart_decode_attempt(self, *, reason: str = "generation retry") -> None:
        """Restore accepted logical occupancy without changing physical lanes."""

        with self._lock:
            if (
                not self._decode_started
                or self._decode_initial_count is None
                or self._decode_checkpoint is None
            ):
                raise RuntimeError("adaptive decoding has not started")
            before = (self._lane_loads, self._h2d_transfers, self._h2d_bytes)
            self.controller.restore_sequence_state(self._decode_checkpoint, reason=reason)
            self._set_active_count(self._decode_initial_count)
            for _name, router in self._routers:
                router.reset_route()
            after = (self._lane_loads, self._h2d_transfers, self._h2d_bytes)
            if after != before:
                raise RuntimeError("decode retry attempted to transfer expert weights")
            self._decode_attempts += 1
            self._cache_retries += 1
            self._emit("decode_retry", retry_reason=reason)

    def end_decode(self) -> None:
        with self._lock:
            if self._decode_started:
                self._decode_started = False
                self._emit("decode_end")

    def observe_entropy(self, scores: torch.Tensor) -> OccupancyDecision:
        """Update K after a completed forward; the change affects the next one."""

        with self._lock:
            if not self._decode_started:
                raise RuntimeError("decode entropy is only valid during generation")
            logits = scores.float()
            log_probabilities = torch.log_softmax(logits, dim=-1)
            probabilities = log_probabilities.exp()
            terms = torch.where(
                probabilities > 0,
                probabilities * log_probabilities,
                torch.zeros_like(probabilities),
            )
            entropy = float((-terms.sum(dim=-1).mean()).item())
            decision = self.controller.observe(entropy, "entropy")
            self._last_decision = decision
            self._set_active_count(decision.active_experts)
            self._emit("decode_decision", entropy=entropy, decision=asdict(decision))
            return decision

    def logits_processor(self) -> AdaptiveEntropyLogitsProcessor:
        return AdaptiveEntropyLogitsProcessor(self)

    def telemetry(self) -> dict[str, object]:
        with self._lock:
            snapshot = self.snapshot()
            return {
                **self.controller.telemetry(),
            "resident_experts": list(snapshot.resident_experts),
            "resident_expert_count": len(snapshot.resident_experts),
            "resident_capacity": self.controller.maximum,
            "resident_fraction": round(
                len(snapshot.resident_experts)
                * self.config.adaptive.effective_expert_footprint_fraction,
                6,
            ),
            "resident_bytes": snapshot.resident_bytes,
            "bank_experts": self.config.num_experts,
            "bank_parameter_fraction": round(
                self.config.num_experts
                * self.config.adaptive.effective_expert_footprint_fraction,
                6,
            ),
                "active_expert_ids": list(snapshot.active_experts),
                "lane_loads": snapshot.lane_loads,
                "h2d_transfers": snapshot.h2d_transfers,
                "h2d_bytes": snapshot.h2d_bytes,
                "decode_started": snapshot.decode_started,
                "prompt_nll": self._prompt_nlls[-1] if self._prompt_nlls else None,
                "prompt_probe_count": len(self._prompt_nlls),
                "routing_probe_nll": self._routing_probe_nll,
                "predicted_nll_by_expert": list(self._predicted_nll_by_expert),
                "request_expert_order": list(self._request_order),
                "residency_reranked": self._residency_reranked,
                "decode_attempts": self._decode_attempts,
                "cache_retries": self._cache_retries,
                "kv_reserve_bytes": self._kv_reserve_bytes,
                "effective_vram_reserve_bytes": (
                    self.config.adaptive.vram_reserve_bytes + self._kv_reserve_bytes
                ),
            }


class AdaptiveEntropyLogitsProcessor:
    """Observe current logits and update occupancy for the subsequent forward."""

    def __init__(self, runtime: AdaptiveExpertRuntime) -> None:
        self.runtime = runtime

    def __call__(self, _input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self.runtime.observe_entropy(scores)
        return scores
