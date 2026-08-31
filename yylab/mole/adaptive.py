from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass, field
from typing import Literal

SignalKind = Literal["nll", "entropy"]


@dataclass
class AdaptiveOccupancyConfig:
    """Serializable contract for surprise-gated expert occupancy.

    Fractions use the complete resolved model (base + routers + every expert)
    as their denominator. One expert therefore represents one fixed parameter
    quantum, independent of the number of experts in the bank.
    """

    enabled: bool = False
    score_semantics: str = "predicted_nll"
    target_total_multiplier: float = 1.25
    expert_footprint_fraction: float = 0.005
    min_active_fraction: float = 0.01
    max_active_fraction: float = 0.10
    surprise_ema_decay: float = 0.95
    surprise_ema_warmup: int = 32
    expand_margin_nats: float = 0.0
    contract_margin_nats: float = 0.05
    expand_patience: int = 1
    contract_patience: int = 8
    min_marginal_nll_gain_nats: float = 0.01
    max_recomputations: int = 4
    calibration_weight: float = 0.05
    nested_k_probe_every: int = 4
    exploration_experts: int = 1
    prefill_nll_probe: bool = True
    vram_reserve_bytes: int = 1_073_741_824
    pin_host_memory: bool = True
    baseline_nll: float | None = None
    preferred_experts: tuple[int, ...] = field(default_factory=tuple)
    base_parameter_count: int = 0
    router_parameter_count: int = 0
    expert_parameter_count: int = 0
    resolved_total_parameter_count: int = 0
    resolved_expert_footprint_fraction: float | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self, *, num_experts: int | None = None) -> None:
        finite_values = {
            "target_total_multiplier": self.target_total_multiplier,
            "expert_footprint_fraction": self.expert_footprint_fraction,
            "min_active_fraction": self.min_active_fraction,
            "max_active_fraction": self.max_active_fraction,
            "surprise_ema_decay": self.surprise_ema_decay,
            "expand_margin_nats": self.expand_margin_nats,
            "contract_margin_nats": self.contract_margin_nats,
            "min_marginal_nll_gain_nats": self.min_marginal_nll_gain_nats,
            "calibration_weight": self.calibration_weight,
        }
        if self.baseline_nll is not None:
            finite_values["baseline_nll"] = self.baseline_nll
        if self.resolved_expert_footprint_fraction is not None:
            finite_values["resolved_expert_footprint_fraction"] = (
                self.resolved_expert_footprint_fraction
            )
        for name, value in finite_values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"adaptive occupancy {name} must be finite")
        if self.score_semantics != "predicted_nll":
            raise ValueError("adaptive routers must declare predicted_nll score semantics")
        if self.target_total_multiplier <= 1.0:
            raise ValueError("adaptive total parameter budget must exceed the frozen base")
        if not 0 < self.expert_footprint_fraction <= 1.0:
            raise ValueError("expert footprint fraction must be within (0, 1]")
        if not 0 < self.min_active_fraction <= self.max_active_fraction <= 0.10:
            raise ValueError("adaptive occupancy must satisfy 0 < min <= max <= 10%")
        for name, fraction in (
            ("minimum", self.min_active_fraction),
            ("maximum", self.max_active_fraction),
        ):
            quanta = fraction / self.expert_footprint_fraction
            if not math.isclose(quanta, round(quanta), rel_tol=0.0, abs_tol=1e-7):
                raise ValueError(f"{name} occupancy must be an integer expert quantum")
        if not 0 <= self.surprise_ema_decay < 1:
            raise ValueError("surprise EMA decay must be within [0, 1)")
        if self.surprise_ema_warmup < 1:
            raise ValueError("surprise EMA warmup must be positive")
        if self.expand_patience < 1 or self.contract_patience < 1:
            raise ValueError("adaptive patience values must be positive")
        if self.expand_margin_nats < 0 or self.contract_margin_nats < 0:
            raise ValueError("adaptive surprise margins must be non-negative")
        if self.min_marginal_nll_gain_nats < 0:
            raise ValueError("minimum marginal NLL gain must be non-negative")
        if self.max_recomputations < 0:
            raise ValueError("maximum recomputations must be non-negative")
        if self.calibration_weight < 0:
            raise ValueError("surprise calibration weight must be non-negative")
        if self.nested_k_probe_every < 1:
            raise ValueError("nested-k probe cadence must be positive")
        if self.exploration_experts < 0:
            raise ValueError("exploration expert count must be non-negative")
        if self.vram_reserve_bytes < 0:
            raise ValueError("VRAM reserve must be non-negative")
        if self.baseline_nll is not None and self.baseline_nll < 0:
            raise ValueError("saved NLL baseline must be non-negative")
        counts = (
            self.base_parameter_count,
            self.router_parameter_count,
            self.expert_parameter_count,
            self.resolved_total_parameter_count,
        )
        if any(int(value) < 0 for value in counts):
            raise ValueError("resolved adaptive parameter counts must be non-negative")
        if self.resolved_expert_footprint_fraction is not None and not (
            0 < self.resolved_expert_footprint_fraction <= 1
        ):
            raise ValueError("resolved expert footprint fraction must be within (0, 1]")
        if any(index < 0 for index in self.preferred_experts):
            raise ValueError("preferred expert indices must be non-negative")
        if len(set(self.preferred_experts)) != len(self.preferred_experts):
            raise ValueError("preferred expert indices must be unique")
        if num_experts is not None:
            if num_experts < self.minimum_experts:
                raise ValueError(
                    f"adaptive bank needs at least {self.minimum_experts} experts, got {num_experts}"
                )
            if any(index >= num_experts for index in self.preferred_experts):
                raise ValueError("preferred expert index exceeds the expert bank")
            if self.resolved_total_parameter_count:
                expected_total = (
                    self.base_parameter_count
                    + self.router_parameter_count
                    + num_experts * self.expert_parameter_count
                )
                if expected_total != self.resolved_total_parameter_count:
                    raise ValueError(
                        "resolved adaptive parameter counts violate the artifact budget"
                    )

    @property
    def minimum_experts(self) -> int:
        return max(
            1,
            math.ceil(
                self.min_active_fraction / self.effective_expert_footprint_fraction - 1e-9
            ),
        )

    def maximum_experts(self, num_experts: int) -> int:
        configured = max(
            self.minimum_experts,
            math.floor(
                self.max_active_fraction / self.effective_expert_footprint_fraction + 1e-9
            ),
        )
        return min(int(num_experts), configured)

    @property
    def effective_expert_footprint_fraction(self) -> float:
        return float(
            self.resolved_expert_footprint_fraction or self.expert_footprint_fraction
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["preferred_experts"] = list(self.preferred_experts)
        return payload

    @classmethod
    def from_dict(cls, value: dict | None) -> AdaptiveOccupancyConfig:
        if not value:
            return cls()
        if not isinstance(value, dict):
            raise TypeError("adaptive occupancy config must be a mapping")
        fields = cls.__dataclass_fields__
        kwargs = {name: raw for name, raw in value.items() if name in fields}
        kwargs["preferred_experts"] = tuple(kwargs.get("preferred_experts") or ())
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class ExpertBudgetPlan:
    requested_total_parameters: int
    resolved_total_parameters: int
    base_parameters: int
    router_parameters: int
    num_experts: int
    rank: int
    parameters_per_expert: int
    expert_rank_width: int
    expert_footprint_fraction: float
    requested_expert_footprint_fraction: float
    budget_error_fraction: float

    @property
    def expert_bank_parameters(self) -> int:
        return self.num_experts * self.parameters_per_expert

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "expert_bank_parameters": self.expert_bank_parameters,
        }


def solve_expert_budget(
    *,
    base_parameters: int,
    expert_rank_width: int,
    router_parameters_per_expert: int,
    target_total_multiplier: float = 1.25,
    expert_footprint_fraction: float = 0.005,
    minimum_experts: int = 2,
    maximum_experts: int = 196,
    maximum_rank: int = 256,
) -> ExpertBudgetPlan:
    """Resolve integer E/rank values against an exact total-parameter budget."""

    base_parameters = int(base_parameters)
    expert_rank_width = int(expert_rank_width)
    router_parameters_per_expert = int(router_parameters_per_expert)
    if base_parameters < 1 or expert_rank_width < 1:
        raise ValueError("base parameters and expert rank width must be positive")
    if router_parameters_per_expert < 0:
        raise ValueError("router parameters per expert must be non-negative")
    if not math.isfinite(target_total_multiplier) or target_total_multiplier <= 1:
        raise ValueError("target total multiplier must be finite and greater than one")
    if not math.isfinite(expert_footprint_fraction) or not 0 < expert_footprint_fraction < 1:
        raise ValueError("expert footprint fraction must be within (0, 1)")
    if not 1 <= minimum_experts <= maximum_experts:
        raise ValueError("invalid expert-count search interval")
    if maximum_rank < 1:
        raise ValueError("maximum rank must be positive")

    requested_total = round(base_parameters * target_total_multiplier)
    desired_expert = requested_total * expert_footprint_fraction
    rank_center = max(1, min(maximum_rank, round(desired_expert / expert_rank_width)))
    best: tuple[float, ExpertBudgetPlan] | None = None
    for experts in range(minimum_experts, maximum_experts + 1):
        router_parameters = router_parameters_per_expert * experts
        remaining = requested_total - base_parameters - router_parameters
        if remaining <= 0:
            continue
        fill_rank = round(remaining / (experts * expert_rank_width))
        ranks = {
            rank_center,
            fill_rank,
            rank_center - 1,
            rank_center + 1,
            fill_rank - 1,
            fill_rank + 1,
        }
        for rank in ranks:
            if not 1 <= rank <= maximum_rank:
                continue
            per_expert = expert_rank_width * rank
            resolved_total = base_parameters + router_parameters + experts * per_expert
            actual_fraction = per_expert / resolved_total
            budget_error = abs(resolved_total - requested_total) / requested_total
            footprint_error = abs(actual_fraction - expert_footprint_fraction) / (
                expert_footprint_fraction
            )
            score = budget_error * 2.0 + footprint_error
            plan = ExpertBudgetPlan(
                requested_total_parameters=requested_total,
                resolved_total_parameters=resolved_total,
                base_parameters=base_parameters,
                router_parameters=router_parameters,
                num_experts=experts,
                rank=rank,
                parameters_per_expert=per_expert,
                expert_rank_width=expert_rank_width,
                expert_footprint_fraction=actual_fraction,
                requested_expert_footprint_fraction=expert_footprint_fraction,
                budget_error_fraction=(resolved_total - requested_total) / requested_total,
            )
            if best is None or score < best[0]:
                best = (score, plan)
    if best is None:
        raise ValueError("no integer expert architecture fits the requested parameter budget")
    return best[1]


@dataclass(frozen=True, slots=True)
class OccupancyDecision:
    action: Literal["hold", "expand", "contract"]
    previous_experts: int
    active_experts: int
    signal: float
    baseline: float | None
    signal_kind: SignalKind
    reason: str


@dataclass(frozen=True, slots=True)
class RecomputationDecision:
    accepted: bool
    should_stop: bool
    previous_nll: float
    candidate_nll: float
    marginal_gain_nats: float
    reason: str


@dataclass(frozen=True, slots=True)
class ControllerCheckpoint:
    current_experts: int
    nll_ema: float | None
    entropy_ema: float | None
    nll_observations: int
    entropy_observations: int
    high_count: int
    low_count: int
    stats: tuple[tuple[str, int], ...]
    last_signal: float | None
    last_signal_kind: SignalKind | None
    last_reason: str
    last_marginal_gain: float | None


class AdaptiveOccupancyController:
    """Thread-safe hysteretic controller over fixed 0.5%-style expert quanta."""

    def __init__(self, config: AdaptiveOccupancyConfig, num_experts: int) -> None:
        config.validate(num_experts=num_experts)
        self.config = config
        self.num_experts = int(num_experts)
        self.minimum = config.minimum_experts
        self.configured_maximum = config.maximum_experts(num_experts)
        self.memory_maximum = self.configured_maximum
        self.current_experts = self.minimum
        self._ema: dict[SignalKind, float | None] = {
            "nll": config.baseline_nll,
            "entropy": None,
        }
        self._observations: dict[SignalKind, int] = {
            "nll": config.surprise_ema_warmup if config.baseline_nll is not None else 0,
            "entropy": 0,
        }
        self._high_count = 0
        self._low_count = 0
        self._lock = threading.RLock()
        self._stats = {
            "expansions": 0,
            "contractions": 0,
            "recomputations": 0,
            "memory_caps": 0,
            "observations": 0,
            "active_sum": 0,
            "active_peak": self.current_experts,
        }
        self.last_signal: float | None = None
        self.last_signal_kind: SignalKind | None = None
        self.last_reason = "minimum occupancy"
        self.last_marginal_gain: float | None = None

    @property
    def maximum(self) -> int:
        return max(self.minimum, min(self.configured_maximum, self.memory_maximum))

    def set_memory_maximum(self, value: int) -> None:
        with self._lock:
            value = int(value)
            if value < self.minimum:
                raise MemoryError(
                    f"available memory holds {value} experts, below the required "
                    f"{self.minimum}-expert minimum"
                )
            self.memory_maximum = min(self.configured_maximum, value)
            if self.current_experts > self.maximum:
                self.current_experts = self.maximum
                self._stats["memory_caps"] += 1

    def reset_sequence(self) -> None:
        with self._lock:
            self.current_experts = self.minimum
            self._high_count = 0
            self._low_count = 0
            self.last_signal = None
            self.last_signal_kind = None
            self.last_reason = "minimum occupancy"
            self.last_marginal_gain = None
            self._stats.update(
                expansions=0,
                contractions=0,
                recomputations=0,
                memory_caps=0,
                observations=0,
                active_sum=0,
                active_peak=self.minimum,
            )

    def restore_active_experts(self, count: int, *, reason: str) -> None:
        """Restore an accepted occupancy after a rejected speculative change."""

        with self._lock:
            count = int(count)
            if not self.minimum <= count <= self.maximum:
                raise ValueError("restored expert count is outside the occupancy bounds")
            self.current_experts = count
            self._high_count = 0
            self._low_count = 0
            self.last_reason = str(reason)

    def checkpoint_sequence_state(self) -> ControllerCheckpoint:
        """Capture all mutable sequence policy state for a safe decode retry."""

        with self._lock:
            return ControllerCheckpoint(
                current_experts=self.current_experts,
                nll_ema=self._ema["nll"],
                entropy_ema=self._ema["entropy"],
                nll_observations=self._observations["nll"],
                entropy_observations=self._observations["entropy"],
                high_count=self._high_count,
                low_count=self._low_count,
                stats=tuple(sorted(self._stats.items())),
                last_signal=self.last_signal,
                last_signal_kind=self.last_signal_kind,
                last_reason=self.last_reason,
                last_marginal_gain=self.last_marginal_gain,
            )

    def restore_sequence_state(
        self,
        checkpoint: ControllerCheckpoint,
        *,
        reason: str,
    ) -> None:
        """Restore a checkpoint after an aborted generation attempt."""

        if not isinstance(checkpoint, ControllerCheckpoint):
            raise TypeError("adaptive controller checkpoint has the wrong type")
        with self._lock:
            if not self.minimum <= checkpoint.current_experts <= self.maximum:
                raise ValueError("checkpoint expert count is outside the occupancy bounds")
            self.current_experts = checkpoint.current_experts
            self._ema = {"nll": checkpoint.nll_ema, "entropy": checkpoint.entropy_ema}
            self._observations = {
                "nll": checkpoint.nll_observations,
                "entropy": checkpoint.entropy_observations,
            }
            self._high_count = checkpoint.high_count
            self._low_count = checkpoint.low_count
            self._stats = dict(checkpoint.stats)
            self.last_signal = checkpoint.last_signal
            self.last_signal_kind = checkpoint.last_signal_kind
            self.last_reason = str(reason)
            self.last_marginal_gain = checkpoint.last_marginal_gain

    def baseline(self, kind: SignalKind) -> float | None:
        with self._lock:
            return self._ema[kind]

    def _update_ema(self, signal: float, kind: SignalKind) -> None:
        previous = self._ema[kind]
        self._ema[kind] = (
            signal
            if previous is None
            else self.config.surprise_ema_decay * previous
            + (1.0 - self.config.surprise_ema_decay) * signal
        )
        self._observations[kind] += 1

    def observe(self, signal: float, kind: SignalKind) -> OccupancyDecision:
        signal = float(signal)
        if not math.isfinite(signal) or signal < 0:
            raise ValueError("adaptive surprise signal must be finite and non-negative")
        with self._lock:
            previous_k = self.current_experts
            baseline = self._ema[kind]
            warmed = self._observations[kind] >= self.config.surprise_ema_warmup
            action: Literal["hold", "expand", "contract"] = "hold"
            reason = "EMA warmup"
            if baseline is not None and warmed:
                if signal >= baseline + self.config.expand_margin_nats:
                    self._high_count += 1
                    self._low_count = 0
                    if (
                        self._high_count >= self.config.expand_patience
                        and self.current_experts < self.maximum
                    ):
                        self.current_experts += 1
                        self._high_count = 0
                        self._stats["expansions"] += 1
                        action = "expand"
                        reason = "surprise at or above expansion threshold"
                    elif self.current_experts >= self.maximum:
                        reason = (
                            "memory occupancy cap"
                            if self.memory_maximum < self.configured_maximum
                            else "configured occupancy cap"
                        )
                    else:
                        reason = "expansion patience"
                elif signal <= baseline - self.config.contract_margin_nats:
                    self._low_count += 1
                    self._high_count = 0
                    if (
                        self._low_count >= self.config.contract_patience
                        and self.current_experts > self.minimum
                    ):
                        self.current_experts -= 1
                        self._low_count = 0
                        self._stats["contractions"] += 1
                        action = "contract"
                        reason = "sustained low surprise"
                    else:
                        reason = "contraction hysteresis"
                else:
                    self._high_count = 0
                    self._low_count = 0
                    reason = "inside hysteresis band"
            self._update_ema(signal, kind)
            self.last_signal = signal
            self.last_signal_kind = kind
            self.last_reason = reason
            self._stats["observations"] += 1
            self._stats["active_sum"] += self.current_experts
            self._stats["active_peak"] = max(
                self._stats["active_peak"], self.current_experts
            )
            return OccupancyDecision(
                action=action,
                previous_experts=previous_k,
                active_experts=self.current_experts,
                signal=signal,
                baseline=baseline,
                signal_kind=kind,
                reason=reason,
            )

    def record_recomputation(self) -> None:
        with self._lock:
            self._stats["recomputations"] += 1

    def evaluate_recomputation(
        self,
        previous_nll: float,
        candidate_nll: float,
    ) -> RecomputationDecision:
        """Accept only useful prompt recomputations and expose the knee stop.

        A newly added lane is not assumed to help: selected-softmax
        renormalization can make the mixture worse. Callers must restore the
        previous active prefix when ``accepted`` is false.
        """

        previous_nll = float(previous_nll)
        candidate_nll = float(candidate_nll)
        if not all(math.isfinite(value) and value >= 0 for value in (previous_nll, candidate_nll)):
            raise ValueError("recomputed NLL values must be finite and non-negative")
        with self._lock:
            self._stats["recomputations"] += 1
            gain = previous_nll - candidate_nll
            self.last_marginal_gain = gain
            if gain < 0:
                accepted = False
                should_stop = True
                reason = "candidate expansion worsened NLL"
            elif gain < self.config.min_marginal_nll_gain_nats:
                accepted = False
                should_stop = True
                reason = "marginal NLL gain below the configured knee"
            else:
                accepted = True
                should_stop = False
                reason = "candidate expansion improved NLL"
            self.last_reason = reason
            return RecomputationDecision(
                accepted=accepted,
                should_stop=should_stop,
                previous_nll=previous_nll,
                candidate_nll=candidate_nll,
                marginal_gain_nats=gain,
                reason=reason,
            )

    def telemetry(self) -> dict[str, int | float | str | None]:
        with self._lock:
            observations = max(1, self._stats["observations"])
            return {
                "routing_mode": "adaptive_surprise",
                "active_experts": self.current_experts,
                "minimum_experts": self.minimum,
                "maximum_experts": self.maximum,
                "active_fraction": round(
                    self.current_experts
                    * self.config.effective_expert_footprint_fraction,
                    6,
                ),
                "mean_active_experts": round(
                    self._stats["active_sum"] / observations, 3
                ),
                "peak_active_experts": self._stats["active_peak"],
                "signal_kind": self.last_signal_kind,
                "surprise": self.last_signal,
                "surprise_ema": (
                    self._ema[self.last_signal_kind]
                    if self.last_signal_kind is not None
                    else None
                ),
                "expansions": self._stats["expansions"],
                "contractions": self._stats["contractions"],
                "recomputations": self._stats["recomputations"],
                "marginal_nll_gain_nats": self.last_marginal_gain,
                "memory_caps": self._stats["memory_caps"],
                "observations": self._stats["observations"],
                "stop_reason": self.last_reason,
            }
