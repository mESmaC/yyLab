"""yyLab bridge for native SPALMER base pretraining.

This module deliberately keeps SPALMER optional at yyLab import time.  Model
construction, CUDA discovery, optimizer creation, and training only happen
inside :func:`run_spalmer_base`, which is called by the isolated worker after a
user starts a run.

The scale plan is always recomputed from the selected tokenizer's real
vocabulary size.  A plan submitted by the UI is evidence for replay, not an
authority over the construction inputs.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .. import db
from ..artifacts import checkpoint_dir
from ..tokenizer_registry import get_tokenizer, resolve_tokenizer_artifact
from .base_config import normalize_training_token_budget
from .data import (
    CORPUS_MIX_CONCATENATE,
    CORPUS_MIX_TOKEN_BALANCED,
    DEFAULT_SHUFFLE_BUFFER,
    DEFAULT_SHUFFLE_BUFFER_BYTES,
    DEFAULT_TEXT_CHUNK_CHARS,
    OBJECTIVE_CAUSAL_LM,
    PackedBatches,
    corpus_training_preflight,
    iter_corpus_docs,
    normalize_corpus_ids,
    normalize_corpus_mix,
    normalize_source_splits,
    normalize_training_objective,
)

MODEL_FAMILY = "spalmer"
MANIFEST_SCHEMA_VERSION = 1
TOKENIZER_CONTRACT_VERSION = 1

_EXPERT_RUNTIME_FIELDS = frozenset(
    {
        "expert_execution",
        "expert_qat_backend",
        "expert_promotion_format",
        "expert_fake_quantization",
        "expert_stochastic_rounding",
        "potentiation_budget",
        "potentiation_ema_decay",
        "potentiation_warmup_steps",
        "potentiation_hold_steps",
        "potentiation_hysteresis",
        "router_score_transform",
        "residency_increment",
        "residency_min_gain",
        "initializer_range",
    }
)
_EXPERT_PLAN_EVIDENCE_FIELDS = frozenset(
    {
        "num_experts",
        "active_experts",
        "min_active_experts",
        "max_active_experts",
        "min_resident_experts",
        "max_resident_experts",
    }
)
_MODEL_RUNTIME_FIELDS = frozenset(
    {
        "ple_stochastic_rounding",
        "ple_sparse_gradients",
        "ple_gate_init",
        "ple_reference_max_numel",
        "norm_eps",
        "initializer_range",
        "surprise_ema_decay",
    }
)


class CorpusExhausted(RuntimeError):
    """Raised before a partial gradient-accumulation window is executed."""


@dataclass(frozen=True, slots=True)
class PreparedSpalmerRun:
    """Validated construction inputs shared by the worker and checkpoint code."""

    config: dict[str, Any]
    tokenizer: Any
    plan: Any
    bundle: Any
    training: Any


def _load_spalmer_api() -> SimpleNamespace:
    """Import the optional sibling package only at an execution boundary."""

    try:
        from spalmer import save_checkpoint
        from spalmer.attention.backends import fla_available
        from spalmer.experiment.planning import (
            ATXYScaleConfig,
            DirectionalScaleConfig,
            ScaleSearchSpace,
            plan_configuration,
        )
        from spalmer.presets import assert_accounting_matches_plan, build_configs
        from spalmer.tokenizer import HFTokenizerAdapter
        from spalmer.training import (
            CausalBatch,
            ExperimentTrainer,
            TrainingConfig,
            seed_everything,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "SPALMER base training requires the optional 'spalmer' package; "
            "install yyLab with its SPALMER training dependency"
        ) from exc
    return SimpleNamespace(
        ATXYScaleConfig=ATXYScaleConfig,
        CausalBatch=CausalBatch,
        DirectionalScaleConfig=DirectionalScaleConfig,
        ExperimentTrainer=ExperimentTrainer,
        HFTokenizerAdapter=HFTokenizerAdapter,
        ScaleSearchSpace=ScaleSearchSpace,
        TrainingConfig=TrainingConfig,
        assert_accounting_matches_plan=assert_accounting_matches_plan,
        build_configs=build_configs,
        fla_available=fla_available,
        plan_configuration=plan_configuration,
        save_checkpoint=save_checkpoint,
        seed_everything=seed_everything,
    )


def _mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return dict(value)


def _bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled", ""}:
        return False
    raise ValueError(f"expected a Boolean value, got {value!r}")


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return parsed


def _finite_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return parsed


def _normalize_string_list(value: object, name: str) -> list[str]:
    if value is None:
        values: Iterable[object] = ()
    elif isinstance(value, str):
        values = (value,)
    elif isinstance(value, Iterable):
        values = value
    else:
        raise TypeError(f"{name} must be a string or iterable of strings")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            raise TypeError(f"every {name} entry must be a string")
        normalized = item.strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _normalize_optional_component(
    value: object,
    *,
    name: str,
    default_enabled: bool,
) -> dict[str, Any]:
    if value is None:
        return {"enabled": default_enabled}
    if isinstance(value, bool):
        return {"enabled": value}
    component = _mapping(value, name)
    component["enabled"] = _bool(component.get("enabled"), default=default_enabled)
    return component


def normalize_spalmer_config(
    config: Mapping[str, Any],
    *,
    require_runtime: bool = False,
) -> dict[str, Any]:
    """Canonicalize and validate one replayable SPALMER base-training recipe."""

    cfg = dict(config)
    family = str(
        cfg.get("model_family") or cfg.get("architecture_type") or ""
    ).strip().casefold()
    if family != MODEL_FAMILY:
        raise ValueError("SPALMER base training requires model_family='spalmer'")
    cfg["model_family"] = MODEL_FAMILY

    objective = normalize_training_objective(cfg.get("objective"))
    if objective != OBJECTIVE_CAUSAL_LM:
        raise ValueError("SPALMER base pretraining currently supports only causal_lm")
    cfg["objective"] = objective
    method = str(cfg.get("method") or "full").strip().casefold()
    if method != "full":
        raise ValueError("SPALMER scratch initialization supports full-parameter training only")
    cfg["method"] = method

    initialization = _mapping(cfg.get("initialization"), "initialization")
    mode = str(initialization.get("mode") or "scratch").strip().casefold()
    if mode != "scratch":
        raise ValueError("the first SPALMER base-training path supports scratch initialization only")
    tokenizer_ref = str(
        initialization.get("tokenizer_ref")
        or cfg.get("tokenizer_ref")
        or _mapping(cfg.get("spalmer"), "spalmer").get("tokenizer_ref")
        or ""
    ).strip()
    if not tokenizer_ref:
        raise ValueError("SPALMER scratch training requires a registered or direct HF tokenizer")
    initialization.update(mode="scratch", tokenizer_ref=tokenizer_ref)
    cfg["initialization"] = initialization
    cfg["tokenizer_ref"] = tokenizer_ref

    corpora = normalize_corpus_ids(cfg.get("corpora", cfg.get("corpus")))
    if not corpora:
        raise ValueError("SPALMER base training requires at least one corpus")
    cfg["corpora"] = corpora
    cfg["domains"] = _normalize_string_list(cfg.get("domains"), "domains") or None
    cfg["splits"] = normalize_source_splits(
        cfg.get("splits", cfg.get("source_splits"))
    ) or None
    cfg["corpus_mix"] = normalize_corpus_mix(cfg.get("corpus_mix"))
    cfg["cross_corpus_dedup"] = _bool(
        cfg.get("cross_corpus_dedup"), default=True
    )

    weights = _mapping(cfg.get("corpus_weights"), "corpus_weights")
    normalized_weights: dict[str, float] = {}
    for corpus_id, value in weights.items():
        key = str(corpus_id)
        if key not in corpora:
            raise ValueError(f"corpus weight references an unselected corpus: {key!r}")
        normalized_weights[key] = _finite_float(
            value, f"corpus_weights[{key!r}]", minimum=0.0
        )
        if normalized_weights[key] == 0:
            raise ValueError("corpus weights must be positive")
    cfg["corpus_weights"] = normalized_weights
    source_ratio = cfg.get("source_max_ratio")
    cfg["source_max_ratio"] = (
        None
        if source_ratio in {None, "", 0, 0.0}
        else _finite_float(source_ratio, "source_max_ratio", minimum=0.0, maximum=1.0)
    )
    if cfg["source_max_ratio"] == 0:
        cfg["source_max_ratio"] = None
    if cfg["corpus_mix"] != CORPUS_MIX_TOKEN_BALANCED and (
        normalized_weights or cfg["source_max_ratio"] is not None
    ):
        raise ValueError("corpus weights and source_max_ratio require token-balanced mixing")
    if len(corpora) < 2:
        cfg["source_max_ratio"] = None
    elif (
        cfg["source_max_ratio"] is not None
        and cfg["source_max_ratio"] < 1 / len(corpora)
    ):
        raise ValueError("source_max_ratio is smaller than the feasible equal share")

    spalmer = _mapping(cfg.get("spalmer"), "spalmer")
    target = spalmer.get("target_parameters", cfg.get("target_parameters"))
    if target is None:
        raise ValueError("SPALMER requires an inclusive total target_parameters value")
    spalmer["target_parameters"] = _positive_int(target, "target_parameters")

    raw_experts = _mapping(
        spalmer.get("experts", cfg.get("experts")), "spalmer.experts"
    )
    explicit_experts = spalmer.get(
        "num_experts", cfg.get("num_experts", raw_experts.get("num_experts"))
    )
    if explicit_experts is not None:
        expert_count = _positive_int(explicit_experts, "num_experts")
        if expert_count < 2:
            raise ValueError("SPALMER requires at least two experts")
        spalmer["num_experts"] = expert_count

    precision = _mapping(cfg.get("precision"), "precision")
    weight_format = str(
        spalmer.get("expert_weight_format")
        or precision.get("expert_weight_format")
        or cfg.get("expert_weight_format")
        or "mxfp4"
    ).strip().casefold()
    if weight_format not in {"mxfp4", "nvfp4"}:
        raise ValueError("expert_weight_format must be 'mxfp4' or 'nvfp4'")
    spalmer["expert_weight_format"] = weight_format
    attention_backend = str(
        spalmer.get("attention_backend") or cfg.get("attention_backend") or "auto"
    ).strip().casefold()
    if attention_backend not in {"auto", "fla", "reference"}:
        raise ValueError("attention_backend must be 'auto', 'fla', or 'reference'")
    spalmer["attention_backend"] = attention_backend
    spalmer["require_fla"] = _bool(
        spalmer.get("require_fla"),
        default=not _bool(spalmer.get("allow_reference_attention"), default=False),
    )

    directional = _normalize_optional_component(
        spalmer.get("directional", cfg.get("directional")),
        name="directional",
        default_enabled=True,
    )
    if directional["enabled"]:
        directional["num_feature_groups"] = _positive_int(
            directional.get("num_feature_groups", 4), "directional.num_feature_groups"
        )
        if directional["num_feature_groups"] < 2:
            raise ValueError("directional.num_feature_groups must be at least two")
        directional["lateral_rank"] = _positive_int(
            directional.get("lateral_rank", 16), "directional.lateral_rank"
        )
    spalmer["directional"] = directional

    atxy = _normalize_optional_component(
        spalmer.get("atxy", cfg.get("atxy")),
        name="atxy",
        default_enabled=False,
    )
    if atxy["enabled"]:
        for field in (
            "value_dim",
            "a_cardinality",
            "t_cardinality",
            "x_cardinality",
            "y_cardinality",
        ):
            if field not in atxy:
                raise ValueError(f"enabled ATXY requires {field}")
            atxy[field] = _positive_int(atxy[field], f"atxy.{field}")
        atxy["injection_layer"] = _positive_int(
            atxy.get("injection_layer", 0),
            "atxy.injection_layer",
            allow_zero=True,
        )
    spalmer["atxy"] = atxy

    expert_plan_evidence = {
        field: raw_experts[field]
        for field in _EXPERT_PLAN_EVIDENCE_FIELDS
        if field in raw_experts
    }
    experts = {
        field: raw_experts[field]
        for field in _EXPERT_RUNTIME_FIELDS
        if field in raw_experts
    }
    for field in _EXPERT_RUNTIME_FIELDS:
        if field in spalmer and field not in experts:
            experts[field] = spalmer[field]
        elif field in cfg and field not in experts:
            experts[field] = cfg[field]
    qat_backend = str(
        experts.get("expert_qat_backend")
        or spalmer.get("expert_qat_backend")
        or cfg.get("expert_qat_backend")
        or "auto"
    ).strip().casefold()
    if qat_backend not in {"auto", "reference", "native"}:
        raise ValueError("expert_qat_backend must be 'auto', 'reference', or 'native'")
    experts["expert_qat_backend"] = qat_backend
    unknown_experts = sorted(
        set(raw_experts) - _EXPERT_RUNTIME_FIELDS - _EXPERT_PLAN_EVIDENCE_FIELDS
    )
    if unknown_experts:
        raise ValueError("unsupported SPALMER expert overrides: " + ", ".join(unknown_experts))
    spalmer["expert_qat_backend"] = qat_backend
    spalmer["experts"] = experts
    if expert_plan_evidence:
        spalmer["expert_plan_evidence"] = expert_plan_evidence

    model_overrides = _mapping(spalmer.get("model"), "spalmer.model")
    for field in _MODEL_RUNTIME_FIELDS:
        if field in spalmer and field not in model_overrides:
            model_overrides[field] = spalmer[field]
    unknown_model = sorted(set(model_overrides) - _MODEL_RUNTIME_FIELDS)
    if unknown_model:
        raise ValueError("unsupported SPALMER model overrides: " + ", ".join(unknown_model))
    spalmer["model"] = model_overrides

    submitted_plan = _mapping(spalmer.get("plan"), "spalmer.plan")
    fingerprint = str(
        spalmer.get("plan_fingerprint")
        or cfg.get("plan_fingerprint")
        or submitted_plan.get("fingerprint")
        or ""
    ).strip()
    if fingerprint:
        spalmer["plan_fingerprint"] = fingerprint
    if submitted_plan:
        spalmer["plan"] = submitted_plan
    cfg["spalmer"] = spalmer
    cfg["target_parameters"] = spalmer["target_parameters"]

    training = _mapping(cfg.get("training"), "training")
    training_aliases = {
        "seq_len": "sequence_length",
        "micro_batch": "micro_batch_size",
        "grad_accum": "gradient_accumulation_steps",
        "max_steps": "max_steps",
        "seed": "seed",
        "log_every": "telemetry_interval_steps",
        "save_every": "checkpoint_interval_steps",
        "lr": "learning_rate",
        "min_lr_ratio": "min_learning_rate_ratio",
        "warmup_steps": "warmup_steps",
        "weight_decay": "weight_decay",
        "beta1": "adam_beta1",
        "beta2": "adam_beta2",
        "eps": "adam_epsilon",
        "grad_clip": "gradient_clip",
        "auxiliary_loss_weight": "auxiliary_loss_weight",
        "surprise_calibration_weight": "surprise_calibration_weight",
    }
    for root_name, training_name in training_aliases.items():
        if root_name not in cfg and training_name in training:
            cfg[root_name] = training[training_name]
    for name, expected in (
        ("compute_dtype", "bfloat16"),
        ("parameter_dtype", "bfloat16"),
        ("optimizer_state_dtype", "float32"),
    ):
        value = str(training.get(name) or expected).strip().casefold()
        if value != expected:
            raise ValueError(f"SPALMER training requires {name}={expected!r}")
    optimizer_state_offload = str(
        training.get(
            "optimizer_state_offload",
            cfg.get("optimizer_state_offload", "cpu"),
        )
        or "cpu"
    ).strip().casefold()
    if optimizer_state_offload != "cpu":
        raise ValueError("SPALMER yyLab training requires optimizer_state_offload='cpu'")
    cfg["optimizer_state_offload"] = optimizer_state_offload
    for name in (
        "device",
        "require_cuda",
        "stochastic_parameter_rounding",
        "optimizer_update_chunk_size",
        "fused_adamw",
        "deterministic_algorithms",
    ):
        if name not in cfg and name in training:
            cfg[name] = training[name]

    integer_defaults = {
        "seq_len": 512,
        "micro_batch": 1,
        "grad_accum": 1,
        "epochs": 1,
        "shuffle_buffer": DEFAULT_SHUFFLE_BUFFER,
        "shuffle_buffer_bytes": DEFAULT_SHUFFLE_BUFFER_BYTES,
        "tokenize_chunk": 8,
        "max_text_chars": DEFAULT_TEXT_CHUNK_CHARS,
        "log_every": 10,
    }
    for field, default in integer_defaults.items():
        cfg[field] = _positive_int(cfg.get(field, default), field)
    cfg["seed"] = _positive_int(cfg.get("seed", 42), "seed", allow_zero=True)
    if cfg.get("save_every") not in {None, "", 0, 0.0}:
        cfg["save_every"] = _positive_int(cfg["save_every"], "save_every")
    else:
        cfg["save_every"] = None
    if cfg.get("max_steps") not in {None, "", 0, 0.0}:
        cfg["max_steps"] = _positive_int(cfg["max_steps"], "max_steps")
    else:
        cfg["max_steps"] = 0
    cfg = normalize_training_token_budget(cfg)
    if require_runtime:
        _positive_int(cfg.get("max_steps", 0), "max_steps")
        if not str(cfg.get("out_checkpoint_id") or "").strip():
            raise ValueError("SPALMER base training requires out_checkpoint_id")
    activation_checkpointing = _bool(
        cfg.get("activation_checkpointing", training.get("activation_checkpointing")),
        default=False,
    )
    if _bool(cfg.get("grad_ckpt"), default=False) or activation_checkpointing:
        raise ValueError("SPALMER activation checkpointing is not implemented")
    if _bool(cfg.get("torch_compile"), default=False):
        raise ValueError("torch.compile is not validated for SPALMER base training")
    cfg["grad_ckpt"] = False
    cfg["activation_checkpointing"] = False
    cfg["torch_compile"] = False
    return cfg


def resolve_spalmer_tokenizer(
    config: Mapping[str, Any],
    *,
    api: SimpleNamespace | None = None,
) -> Any:
    """Resolve a yyLab registry ID, local HF artifact, or direct HF identifier."""

    cfg = dict(config)
    initialization = _mapping(cfg.get("initialization"), "initialization")
    reference = str(initialization.get("tokenizer_ref") or cfg.get("tokenizer_ref") or "").strip()
    if not reference:
        raise ValueError("SPALMER tokenizer reference is empty")
    row = get_tokenizer(reference)
    if row:
        source = str(resolve_tokenizer_artifact(reference))
    else:
        candidate = Path(reference).expanduser()
        source = str(candidate.resolve()) if candidate.exists() else reference
    symbols = api or _load_spalmer_api()
    return symbols.HFTokenizerAdapter.from_pretrained(
        source,
        revision=initialization.get("tokenizer_revision"),
        local_files_only=_bool(initialization.get("local_files_only"), default=False),
        trust_remote_code=_bool(initialization.get("trust_remote_code"), default=False),
    )


def _scale_components(config: Mapping[str, Any], api: SimpleNamespace) -> tuple[Any, Any]:
    spalmer = _mapping(config.get("spalmer"), "spalmer")
    directional_values = _mapping(spalmer.get("directional"), "spalmer.directional")
    directional = None
    if _bool(directional_values.get("enabled"), default=True):
        directional = api.DirectionalScaleConfig(
            num_feature_groups=int(directional_values.get("num_feature_groups", 4)),
            lateral_rank=int(directional_values.get("lateral_rank", 16)),
        )
    atxy_values = _mapping(spalmer.get("atxy"), "spalmer.atxy")
    atxy = None
    if _bool(atxy_values.get("enabled"), default=False):
        atxy = api.ATXYScaleConfig(
            value_dim=int(atxy_values["value_dim"]),
            a_cardinality=int(atxy_values["a_cardinality"]),
            t_cardinality=int(atxy_values["t_cardinality"]),
            x_cardinality=int(atxy_values["x_cardinality"]),
            y_cardinality=int(atxy_values["y_cardinality"]),
            injection_layer=int(atxy_values.get("injection_layer", 0)),
        )
    return directional, atxy


def plan_spalmer_configuration(
    config: Mapping[str, Any],
    vocab_size: int,
    *,
    api: SimpleNamespace | None = None,
) -> Any:
    """Recompute the sole authoritative SPALMER plan and verify UI evidence."""

    cfg = normalize_spalmer_config(config)
    symbols = api or _load_spalmer_api()
    spalmer = cfg["spalmer"]
    directional, atxy = _scale_components(cfg, symbols)
    kwargs: dict[str, Any] = {
        "expert_weight_format": spalmer["expert_weight_format"],
        "directional": directional,
        "atxy": atxy,
    }
    if "num_experts" in spalmer:
        kwargs["search_space"] = symbols.ScaleSearchSpace(
            num_experts=int(spalmer["num_experts"])
        )
    plan = symbols.plan_configuration(
        int(spalmer["target_parameters"]),
        _positive_int(vocab_size, "vocab_size"),
        **kwargs,
    )
    submitted = str(spalmer.get("plan_fingerprint") or "")
    if submitted and submitted != str(plan.fingerprint):
        raise ValueError(
            "SPALMER plan fingerprint mismatch after resolving the tokenizer vocabulary: "
            f"submitted={submitted}, authoritative={plan.fingerprint}"
        )
    return plan


def build_spalmer_bundle(
    config: Mapping[str, Any],
    tokenizer: Any,
    plan: Any,
    *,
    api: SimpleNamespace | None = None,
) -> Any:
    """Create construction configs without allocating a model."""

    symbols = api or _load_spalmer_api()
    cfg = normalize_spalmer_config(config)
    spalmer = cfg["spalmer"]
    identity = tokenizer.identity
    return symbols.build_configs(
        plan,
        tokenizer_version=TOKENIZER_CONTRACT_VERSION,
        tokenizer_fingerprint=str(identity.fingerprint),
        attention_backend=str(spalmer["attention_backend"]),
        experts_overrides=dict(spalmer.get("experts") or {}),
        model_overrides=dict(spalmer.get("model") or {}),
    )


def build_spalmer_training_config(
    config: Mapping[str, Any],
    *,
    api: SimpleNamespace | None = None,
) -> Any:
    """Translate yyLab runtime controls to SPALMER's strict training config."""

    cfg = normalize_spalmer_config(config, require_runtime=True)
    symbols = api or _load_spalmer_api()
    optimizer = _mapping(cfg.get("optimizer"), "optimizer")
    scheduler = _mapping(cfg.get("scheduler"), "scheduler")
    max_steps = int(cfg["max_steps"])
    save_every = cfg.get("save_every")
    return symbols.TrainingConfig(
        max_steps=max_steps,
        micro_batch_size=int(cfg["micro_batch"]),
        sequence_length=int(cfg["seq_len"]),
        gradient_accumulation_steps=int(cfg["grad_accum"]),
        learning_rate=float(optimizer.get("lr", cfg.get("lr", 3e-4))),
        min_learning_rate_ratio=float(
            scheduler.get("min_lr_ratio", cfg.get("min_lr_ratio", 0.1))
        ),
        warmup_steps=int(scheduler.get("warmup_steps", cfg.get("warmup_steps", 0))),
        weight_decay=float(optimizer.get("weight_decay", cfg.get("weight_decay", 0.1))),
        adam_beta1=float(optimizer.get("beta1", cfg.get("beta1", 0.9))),
        adam_beta2=float(optimizer.get("beta2", cfg.get("beta2", 0.95))),
        adam_epsilon=float(optimizer.get("eps", cfg.get("eps", 1e-8))),
        gradient_clip=(
            None if cfg.get("grad_clip") is None else float(cfg.get("grad_clip", 1.0))
        ),
        auxiliary_loss_weight=float(cfg.get("auxiliary_loss_weight", 0.01)),
        surprise_calibration_weight=float(
            cfg.get("surprise_calibration_weight", 0.05)
        ),
        device=str(cfg.get("device") or "cuda"),
        require_cuda=_bool(cfg.get("require_cuda"), default=True),
        compute_dtype="bfloat16",
        parameter_dtype="bfloat16",
        optimizer_state_dtype="float32",
        optimizer_state_offload=str(cfg.get("optimizer_state_offload") or "cpu"),
        stochastic_parameter_rounding=_bool(
            cfg.get("stochastic_parameter_rounding"), default=True
        ),
        optimizer_update_chunk_size=int(cfg.get("optimizer_update_chunk_size", 1_048_576)),
        fused_adamw=str(cfg.get("fused_adamw") or "auto"),
        activation_checkpointing=False,
        deterministic_algorithms=_bool(
            cfg.get("deterministic_algorithms"), default=False
        ),
        seed=int(cfg["seed"]),
        # Validation is not yet part of the yyLab corpus bridge.  A value above
        # max_steps keeps the engine's positive-interval invariant explicit.
        validation_interval_steps=max_steps + 1,
        checkpoint_interval_steps=(int(save_every) if save_every else max_steps + 1),
        telemetry_interval_steps=int(cfg["log_every"]),
    )


class YYLabSPALMERBatchSource:
    """Stateful, deterministic adapter from yyLab packed rows to CausalBatch.

    A complete gradient-accumulation window is prefetched before its first
    microbatch is returned.  Finite corpora therefore stop between optimizer
    steps instead of leaving a partially accumulated gradient behind.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        factory: Callable[[], PackedBatches],
        *,
        causal_batch_type: type,
        micro_batch_size: int,
        sequence_length: int,
        accumulation_steps: int,
    ) -> None:
        self._factory = factory
        self._causal_batch_type = causal_batch_type
        self.micro_batch_size = _positive_int(micro_batch_size, "micro_batch_size")
        self.sequence_length = _positive_int(sequence_length, "sequence_length")
        self.accumulation_steps = _positive_int(accumulation_steps, "accumulation_steps")
        self.consumed_batches = 0
        self.delivered_sequences = 0
        self.delivered_input_tokens = 0
        self.delivered_target_tokens = 0
        self.discarded_incomplete_batches = 0
        self._packed: PackedBatches | None = None
        self._iterator: Any = None
        self._prefetched: deque[dict[str, Any]] = deque()
        self._exhausted = False

    @property
    def packed_batches(self) -> PackedBatches:
        if self._packed is None:
            self._packed = self._factory()
            self._iterator = iter(self._packed)
        return self._packed

    @property
    def docs_seen(self) -> int:
        return int(getattr(self.packed_batches, "docs_seen", 0))

    @property
    def tokens_seen(self) -> int:
        return int(getattr(self.packed_batches, "tokens_seen", 0))

    def _next_raw(self) -> dict[str, Any]:
        self.packed_batches
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise CorpusExhausted("selected corpora contain no further complete sequences") from exc

    def _fill_accumulation_window(self) -> None:
        if self._exhausted:
            raise CorpusExhausted("selected corpora are exhausted")
        remaining = self.accumulation_steps - (
            self.consumed_batches % self.accumulation_steps
        )
        fetched: list[dict[str, Any]] = []
        try:
            for _ in range(remaining):
                fetched.append(self._next_raw())
        except CorpusExhausted as exc:
            self.discarded_incomplete_batches += len(fetched)
            self._exhausted = True
            raise CorpusExhausted(
                "corpus exhausted before a complete gradient-accumulation window"
            ) from exc
        self._prefetched.extend(fetched)

    @staticmethod
    def _shape(raw: Mapping[str, Any]) -> tuple[int, int]:
        rows = raw.get("input_ids")
        if not isinstance(rows, list) or not rows:
            raise ValueError("packed batch contains no input_ids rows")
        width = len(rows[0])
        if width < 2 or any(not isinstance(row, list) or len(row) != width for row in rows):
            raise ValueError("packed input_ids must be a non-empty rectangular matrix")
        return len(rows), width

    def _record_delivered(self, raw: Mapping[str, Any]) -> None:
        rows, width = self._shape(raw)
        self.consumed_batches += 1
        self.delivered_sequences += rows
        self.delivered_input_tokens += rows * width
        self.delivered_target_tokens += rows * (width - 1)

    def next_batch(self, *, batch_size: int, sequence_length: int) -> Any:
        if int(batch_size) != self.micro_batch_size:
            raise ValueError(
                f"trainer requested batch_size={batch_size}, expected {self.micro_batch_size}"
            )
        if int(sequence_length) != self.sequence_length:
            raise ValueError(
                "trainer requested sequence_length="
                f"{sequence_length}, expected {self.sequence_length}"
            )
        if not self._prefetched:
            self._fill_accumulation_window()
        raw = self._prefetched.popleft()
        rows, width = self._shape(raw)
        if width != self.sequence_length + 1:
            raise ValueError(
                f"SPALMER needs input width {self.sequence_length + 1}, received {width}"
            )
        import torch

        input_ids = torch.as_tensor(raw["input_ids"], dtype=torch.long)
        if input_ids.shape != (rows, width):
            raise ValueError("packed input_ids changed shape during tensor conversion")
        self._record_delivered(raw)
        return self._causal_batch_type(input_ids=input_ids)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "consumed_batches": self.consumed_batches,
            "micro_batch_size": self.micro_batch_size,
            "sequence_length": self.sequence_length,
            "accumulation_steps": self.accumulation_steps,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", -1)) != self.STATE_VERSION:
            raise ValueError("unsupported yyLab SPALMER batch-source state version")
        expected = {
            "micro_batch_size": self.micro_batch_size,
            "sequence_length": self.sequence_length,
            "accumulation_steps": self.accumulation_steps,
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if int(state.get(key, -1)) != value
        }
        if mismatches:
            raise ValueError(f"batch-source state is incompatible: {mismatches}")
        consumed = _positive_int(
            state.get("consumed_batches", 0), "consumed_batches", allow_zero=True
        )
        self.consumed_batches = 0
        self.delivered_sequences = 0
        self.delivered_input_tokens = 0
        self.delivered_target_tokens = 0
        self.discarded_incomplete_batches = 0
        self._packed = self._factory()
        self._iterator = iter(self._packed)
        self._prefetched.clear()
        self._exhausted = False
        for _ in range(consumed):
            try:
                raw = next(self._iterator)
            except StopIteration as exc:
                raise ValueError("batch-source state exceeds the deterministic corpus stream") from exc
            self._record_delivered(raw)

    def data_stats(self) -> dict[str, Any]:
        stats_fn = getattr(self.packed_batches, "data_stats", None)
        stats = dict(stats_fn() if callable(stats_fn) else {})
        tokens = dict(stats.get("tokens") or {})
        tokens.update(
            forwarded_model_slots=self.delivered_input_tokens,
            forwarded_real_inputs=self.delivered_input_tokens,
            padding_inputs=0,
            model_target_slots=self.delivered_target_tokens,
            optimizer_supervised_targets=self.delivered_target_tokens,
            ignored_target_slots=0,
        )
        stats["tokens"] = tokens
        stats["sequences"] = self.delivered_sequences
        stats["consumed_batches"] = self.consumed_batches
        stats["discarded_incomplete_accumulation_batches"] = (
            self.discarded_incomplete_batches
        )
        return stats


def build_spalmer_batch_source(
    config: Mapping[str, Any],
    tokenizer: Any,
    *,
    causal_batch_type: type,
) -> YYLabSPALMERBatchSource:
    """Build a replayable yyLab corpus stream with SPALMER's N+1 input width."""

    cfg = normalize_spalmer_config(config, require_runtime=True)
    sequence_length = int(cfg["seq_len"])
    micro_batch = int(cfg["micro_batch"])
    accumulation_steps = int(cfg["grad_accum"])
    max_batches = int(cfg["max_steps"]) * accumulation_steps
    special = tokenizer.special_tokens
    eos_token_id = getattr(special, "eos_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(special, "eod_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(special, "pad_id", None)
    if eos_token_id is None:
        raise ValueError("SPALMER causal packing requires tokenizer EOS, EOD, or pad identity")

    def tokenize(texts: list[str]) -> list[list[int]]:
        raw_tokenizer = getattr(tokenizer, "tokenizer", None)
        if callable(raw_tokenizer):
            try:
                encoded = raw_tokenizer(
                    texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    verbose=False,
                )
                if isinstance(encoded, Mapping) and "input_ids" in encoded:
                    return [list(map(int, row)) for row in encoded["input_ids"]]
            except TypeError:
                # Some compatible HF-style objects do not accept `verbose`.
                encoded = raw_tokenizer(
                    texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )
                if isinstance(encoded, Mapping) and "input_ids" in encoded:
                    return [list(map(int, row)) for row in encoded["input_ids"]]
        return [tokenizer.encode(text, add_special_tokens=False) for text in texts]

    def factory() -> PackedBatches:
        def documents():
            for epoch in range(int(cfg["epochs"])):
                yield from iter_corpus_docs(
                    cfg["corpora"],
                    cfg.get("domains"),
                    shuffle_buffer=int(cfg["shuffle_buffer"]),
                    shuffle_buffer_bytes=int(cfg["shuffle_buffer_bytes"]),
                    seed=int(cfg["seed"]) + epoch,
                    corpus_mix=str(cfg.get("corpus_mix") or CORPUS_MIX_CONCATENATE),
                    splits=cfg.get("splits"),
                    cross_corpus_dedup=bool(cfg["cross_corpus_dedup"]),
                    corpus_weights=cfg.get("corpus_weights") or None,
                    source_max_ratio=cfg.get("source_max_ratio"),
                    annotate_source=True,
                )

        return PackedBatches(
            documents(),
            tokenize,
            # CausalBatch uses one extra token as the final next-token target.
            seq_len=sequence_length + 1,
            eos_token_id=int(eos_token_id),
            micro_batch=micro_batch,
            max_batches=max_batches,
            tokenize_chunk=int(cfg["tokenize_chunk"]),
            max_text_chars=int(cfg["max_text_chars"]),
        )

    return YYLabSPALMERBatchSource(
        factory,
        causal_batch_type=causal_batch_type,
        micro_batch_size=micro_batch,
        sequence_length=sequence_length,
        accumulation_steps=accumulation_steps,
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        _jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _append_json_line(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_jsonable(value), allow_nan=False, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
        stream.flush()


def _metric_value(metrics: Any, name: str, default: Any = None) -> Any:
    if isinstance(metrics, Mapping):
        return metrics.get(name, default)
    return getattr(metrics, name, default)


def _metric_payload(metrics: Any, *, max_steps: int) -> dict[str, Any]:
    objective = float(_metric_value(metrics, "objective", 0.0))
    step = int(_metric_value(metrics, "step", 0))
    tokens_seen = int(_metric_value(metrics, "tokens_seen", 0))
    elapsed = float(_metric_value(metrics, "elapsed_seconds", 0.0) or 0.0)
    throughput = tokens_seen / max(elapsed, 1e-6)
    max_group_load = _metric_value(metrics, "max_expert_group_load")
    max_padding_amplification = _metric_value(
        metrics,
        "max_expert_group_padding_amplification",
    )
    max_global_max_counterfactual_amplification = _metric_value(
        metrics,
        "max_expert_group_global_max_counterfactual_padding_amplification",
    )
    message = f"{throughput:,.1f} tok/s"
    if max_group_load is not None and max_padding_amplification is not None:
        message += (
            f" · expert group max={int(max_group_load):,}"
            f" · bucket padding={float(max_padding_amplification):.3f}x"
        )
    if max_global_max_counterfactual_amplification is not None:
        message += (
            " · legacy global-max="
            f"{float(max_global_max_counterfactual_amplification):.3f}x"
        )
    return {
        "schema_version": 1,
        "stage": "SPALMER base pretraining",
        "step": step,
        "stage_step": step,
        "stage_steps": int(max_steps),
        "max_steps": int(max_steps),
        "value": objective,
        "loss": objective,
        "aux": _metric_value(metrics, "auxiliary_loss"),
        "metric_label": "loss",
        "tokens": tokens_seen,
        "tokens_seen": tokens_seen,
        "tok_per_s": throughput,
        "elapsed_s": elapsed,
        "msg": message,
        "objective": objective,
        "model_loss": float(_metric_value(metrics, "model_loss", objective)),
        "auxiliary_loss": _metric_value(metrics, "auxiliary_loss"),
        "surprise_calibration_loss": _metric_value(
            metrics, "surprise_calibration_loss"
        ),
        "predictive_entropy": _metric_value(metrics, "predictive_entropy"),
        "learning_rate": _metric_value(metrics, "learning_rate"),
        "gradient_norm": _metric_value(metrics, "gradient_norm"),
        "elapsed_seconds": elapsed,
        "promoted_experts": list(_metric_value(metrics, "promoted_experts", ()) or ()),
        "average_surprise": _metric_value(metrics, "average_surprise"),
        "max_expert_group_load": (
            None if max_group_load is None else int(max_group_load)
        ),
        "max_expert_group_padding_amplification": (
            None
            if max_padding_amplification is None
            else float(max_padding_amplification)
        ),
        "max_expert_group_global_max_counterfactual_padding_amplification": (
            None
            if max_global_max_counterfactual_amplification is None
            else float(max_global_max_counterfactual_amplification)
        ),
    }


class _TelemetryWriter:
    def __init__(self, run_directory: Path, *, max_steps: int) -> None:
        self.run_directory = run_directory
        self.max_steps = max_steps
        self.last_step = 0

    def record(self, metrics: Any) -> None:
        payload = _metric_payload(metrics, max_steps=self.max_steps)
        step = int(payload["step"])
        if step <= self.last_step:
            return
        self.last_step = step
        _append_json_line(self.run_directory / "metrics.jsonl", payload)
        _write_json_atomic(self.run_directory / "progress.json", payload)
        print(
            "SPALMER "
            f"step={step}/{self.max_steps} loss={float(payload['loss']):.4f} "
            f"tokens={int(payload['tokens_seen']):,} "
            f"surprise={float(payload.get('average_surprise') or 0.0):.4f} "
            f"group_max={payload.get('max_expert_group_load')} "
            "bucket_padding_amp="
            f"{float(payload.get('max_expert_group_padding_amplification') or 0.0):.3f}x "
            "legacy_global_max_amp="
            f"{float(payload.get('max_expert_group_global_max_counterfactual_padding_amplification') or 0.0):.3f}x",
            flush=True,
        )


def _parameter_accounting_payload(accounting: Any) -> dict[str, Any]:
    payload = _jsonable(accounting)
    if not isinstance(payload, dict):
        payload = {}
    payload.update(
        total=int(getattr(accounting, "total", payload.get("total", 0)) or 0),
        expert_pool=int(
            getattr(accounting, "expert_pool", payload.get("expert_pool", 0)) or 0
        ),
        shared_base=int(
            getattr(accounting, "shared_base", payload.get("shared_base", 0)) or 0
        ),
        resident_parameters=int(
            getattr(accounting, "resident_parameters", payload.get("resident_parameters", 0))
            or 0
        ),
        per_token_active_parameters=int(
            getattr(
                accounting,
                "per_token_active_parameters",
                payload.get("per_token_active_parameters", 0),
            )
            or 0
        ),
    )
    return payload


def _optimizer_placement_payload(optimizer: Any) -> dict[str, Any]:
    reporter = getattr(optimizer, "state_placement", None)
    if not callable(reporter):
        raise RuntimeError("SPALMER optimizer does not expose physical state placement")
    payload = _jsonable(reporter())
    if not isinstance(payload, dict):
        raise RuntimeError("SPALMER optimizer returned invalid state-placement telemetry")
    dense = payload.get("dense")
    if not isinstance(dense, dict) or dense.get("policy") != "cpu":
        raise RuntimeError("SPALMER FP32 optimizer moments are not configured for CPU offload")
    return payload


def _validate_initialized_optimizer_offload(payload: Mapping[str, Any]) -> None:
    dense = payload.get("dense")
    if not isinstance(dense, Mapping):
        raise RuntimeError("SPALMER dense optimizer placement telemetry is missing")
    problems: list[str] = []
    if not bool(dense.get("initialized")):
        problems.append("optimizer moments were not initialized")
    if str(dense.get("moment_dtype") or "") != "float32":
        problems.append("optimizer moments are not FP32")
    devices = [str(value) for value in dense.get("devices") or []]
    if devices != ["cpu"]:
        problems.append(f"optimizer moment devices are {devices or ['missing']}, not CPU")
    parameter_devices = [str(value) for value in dense.get("parameter_devices") or []]
    if not parameter_devices or any(not value.startswith("cuda") for value in parameter_devices):
        problems.append("BF16 master parameters are not accelerator-resident")
    parameter_dtypes = [str(value) for value in dense.get("parameter_dtypes") or []]
    if parameter_dtypes != ["bfloat16"]:
        problems.append(
            f"master parameter dtypes are {parameter_dtypes or ['missing']}, not BF16"
        )
    if int(dense.get("moment_bytes") or 0) <= 0:
        problems.append("optimizer reports no physical moment storage")
    if problems:
        raise RuntimeError(
            "SPALMER optimizer-state offload placement proof failed: " + "; ".join(problems)
        )


def _tokenized_view_manifest(
    batches: YYLabSPALMERBatchSource,
    *,
    sequence_length: int,
) -> dict[str, Any]:
    batcher = batches.data_stats()
    tokens = dict(batcher.get("tokens") or {})
    rendered = int(tokens.get("rendered_token_positions") or 0)
    replay = int(tokens.get("packing_replay_inputs") or 0)
    forwarded_real = int(tokens.get("forwarded_real_inputs") or 0)
    forwarded_model = int(tokens.get("forwarded_model_slots") or 0)
    covered = min(rendered, max(0, forwarded_real - replay))
    padding = int(tokens.get("padding_inputs") or 0)
    targets = int(tokens.get("model_target_slots") or 0)
    supervised = int(tokens.get("optimizer_supervised_targets") or 0)
    ignored = int(tokens.get("ignored_target_slots") or 0)
    unforwarded = max(0, rendered - covered)
    tokens["unforwarded_rendered_positions"] = unforwarded
    return {
        "schema_version": 1,
        "kind": "yylab_tokenized_training_view",
        "objective": OBJECTIVE_CAUSAL_LM,
        "sequence_length": int(sequence_length),
        "input_width": int(sequence_length) + 1,
        "documents": {
            key: value
            for key, value in batcher.items()
            if key.endswith("docs") or key in {"docs_seen", "failure_reasons"}
        },
        "examples": {
            "sequences_forwarded": batches.delivered_sequences,
            "discarded_incomplete_accumulation_batches": (
                batches.discarded_incomplete_batches
            ),
        },
        "tokens": tokens,
        "per_corpus": dict(batcher.get("per_corpus") or {}),
        "reconciliation": {
            "model_slots_equal_real_plus_padding": forwarded_model
            == forwarded_real + padding,
            # Packed causal pretraining has no replay or padding; a final
            # unconsumed suffix is retained as unforwarded rendered content.
            "real_inputs_do_not_exceed_rendered": forwarded_real
            <= rendered + replay,
            "rendered_equal_covered_plus_unforwarded": rendered
            == covered + unforwarded,
            "targets_equal_supervised_plus_ignored": targets == supervised + ignored,
        },
        "docs_seen": int(batcher.get("docs_seen") or batches.docs_seen),
        "rendered_tokens": int(batcher.get("rendered_tokens") or batches.tokens_seen),
        "optimizer_supervised_tokens": supervised,
    }


def _validate_fast_attention(config: Mapping[str, Any], api: SimpleNamespace) -> None:
    cfg = _mapping(config.get("spalmer"), "spalmer")
    backend = str(cfg.get("attention_backend") or "auto")
    require_cuda = _bool(config.get("require_cuda"), default=True)
    require_fla = _bool(cfg.get("require_fla"), default=True)
    if backend == "reference" or not require_cuda or not require_fla:
        return
    available = getattr(api, "fla_available", None)
    if not callable(available) or not bool(available()):
        raise RuntimeError(
            "SPALMER GPU pretraining requires the optimized FLA KDA backend; "
            "install a compatible fla-core build (>=0.4), or explicitly select "
            "reference attention and allow_reference_attention for correctness-only runs"
        )


def prepare_spalmer_run(
    config: Mapping[str, Any],
    *,
    api: SimpleNamespace | None = None,
) -> PreparedSpalmerRun:
    """Resolve tokenizer, plan, construction bundle, and runtime config."""

    cfg = normalize_spalmer_config(config, require_runtime=True)
    symbols = api or _load_spalmer_api()
    tokenizer = resolve_spalmer_tokenizer(cfg, api=symbols)
    plan = plan_spalmer_configuration(cfg, int(tokenizer.vocab_size), api=symbols)
    bundle = build_spalmer_bundle(cfg, tokenizer, plan, api=symbols)
    training = build_spalmer_training_config(cfg, api=symbols)
    _validate_fast_attention(cfg, symbols)
    return PreparedSpalmerRun(cfg, tokenizer, plan, bundle, training)


def _save_bundle_checkpoint(
    destination: Path,
    *,
    api: SimpleNamespace,
    trainer: Any,
    prepared: PreparedSpalmerRun,
    metrics: Any,
    data_provenance: Mapping[str, Any],
    run_id: str,
) -> Path:
    metadata = {
        "model_family": MODEL_FAMILY,
        "run_id": run_id,
        "plan": prepared.plan.to_dict(),
        "plan_fingerprint": prepared.plan.fingerprint,
        "tokenizer": prepared.tokenizer.identity.to_dict(),
        "training_config": prepared.training.to_dict(),
        "training": (
            None
            if metrics is None
            else _metric_payload(metrics, max_steps=int(prepared.training.max_steps))
        ),
        "batch_source": dict(trainer.batches.state_dict()),
        "data_provenance": dict(data_provenance),
    }
    return api.save_checkpoint(
        destination,
        trainer.model,
        prepared.tokenizer,
        kda_config=prepared.bundle.kda,
        mla_config=prepared.bundle.mla,
        experts_config=prepared.bundle.experts,
        directional_config=prepared.bundle.directional,
        atxy_config=prepared.bundle.atxy,
        metadata=metadata,
    )


def run_spalmer_base(config: Mapping[str, Any], rd: Path) -> dict[str, Any]:
    """Execute one user-requested native SPALMER base-pretraining run."""

    rd = Path(rd)
    rd.mkdir(parents=True, exist_ok=True)
    cfg = normalize_spalmer_config(config, require_runtime=True)
    preflight = corpus_training_preflight(
        cfg["corpora"],
        cfg.get("domains"),
        corpus_mix=cfg["corpus_mix"],
        splits=cfg.get("splits"),
        cross_corpus_dedup=bool(cfg["cross_corpus_dedup"]),
        corpus_weights=cfg.get("corpus_weights") or None,
        source_max_ratio=cfg.get("source_max_ratio"),
    )
    if preflight["errors"]:
        raise ValueError("; ".join(preflight["errors"]))
    for warning in preflight["warnings"]:
        print(f"data warning: {warning}", flush=True)
    data_provenance = dict(preflight["provenance"])
    data_provenance["objective"] = OBJECTIVE_CAUSAL_LM

    api = _load_spalmer_api()
    prepared = prepare_spalmer_run(cfg, api=api)
    plan = prepared.plan
    print(
        "SPALMER plan: "
        f"target={int(plan.target_parameters):,}; actual={int(plan.parameters.total):,}; "
        f"error={float(plan.relative_error):+.2%}; vocab={prepared.tokenizer.vocab_size:,}; "
        f"experts={plan.config.num_experts}; format={plan.config.expert_weight_format}",
        flush=True,
    )

    api.seed_everything(
        int(prepared.training.seed),
        deterministic_algorithms=bool(prepared.training.deterministic_algorithms),
    )
    import torch

    destination_device = torch.device(prepared.training.device)
    if destination_device.type == "cuda":
        # Construct directly on the destination to avoid a second, transient
        # full-sized host copy before ExperimentTrainer takes ownership.
        with torch.device(destination_device):
            model = prepared.bundle.build()
    else:
        model = prepared.bundle.build()
    accounting = model.parameter_accounting()
    api.assert_accounting_matches_plan(plan, accounting)
    accounting_payload = _parameter_accounting_payload(accounting)

    batches = build_spalmer_batch_source(
        cfg,
        prepared.tokenizer,
        causal_batch_type=api.CausalBatch,
    )
    output = checkpoint_dir(str(cfg["out_checkpoint_id"]))
    output.mkdir(parents=True, exist_ok=True)
    telemetry = _TelemetryWriter(rd, max_steps=int(prepared.training.max_steps))

    def periodic_checkpoint(trainer: Any, metrics: Any) -> None:
        placement = _optimizer_placement_payload(trainer.optimizer)
        _validate_initialized_optimizer_offload(placement)
        step = int(_metric_value(metrics, "step", trainer.progress.completed_steps))
        step_dir = output / "steps" / f"step-{step:08d}"
        _save_bundle_checkpoint(
            step_dir / "spalmer.pt",
            api=api,
            trainer=trainer,
            prepared=prepared,
            metrics=metrics,
            data_provenance=data_provenance,
            run_id=rd.name,
        )
        _write_json_atomic(step_dir / "yylab_recipe.json", cfg)

    trainer = api.ExperimentTrainer(
        model,
        batches,
        prepared.training,
        on_checkpoint=periodic_checkpoint,
        on_telemetry=telemetry.record,
    )
    optimizer_placement = _optimizer_placement_payload(trainer.optimizer)
    dense_placement = optimizer_placement["dense"]
    pinned_state = dense_placement.get("cpu_pinned")
    host_memory = (
        "pinned when available"
        if pinned_state is None
        else ("pinned" if pinned_state else "pageable")
    )
    print(
        "SPALMER optimizer: BF16 accelerator masters; FP32 moments CPU-offloaded; "
        f"update_chunk={int(prepared.training.optimizer_update_chunk_size):,}; "
        f"host_memory={host_memory}",
        flush=True,
    )
    last_metrics: Any = None
    exhausted = False
    optimizer_offload_verified = False
    try:
        for metrics in trainer.run():
            last_metrics = metrics
            if not optimizer_offload_verified:
                # The configured policy is only a promise until Adam creates
                # its moments on the first completed update.  Prove and
                # persist the physical placement immediately instead of
                # waiting for a periodic checkpoint or the end of a long run.
                observed_placement = _optimizer_placement_payload(trainer.optimizer)
                _validate_initialized_optimizer_offload(observed_placement)
                first_step = int(
                    _metric_value(metrics, "step", trainer.progress.completed_steps)
                )
                _write_json_atomic(
                    rd / "optimizer_state_placement.json",
                    {
                        "schema_version": 1,
                        "verified_after_step": first_step,
                        "optimizer_state_placement": observed_placement,
                    },
                )
                dense = observed_placement["dense"]
                print(
                    "SPALMER optimizer placement verified: "
                    f"step={first_step}; parameters={dense.get('parameter_devices')}; "
                    f"parameter_dtype={dense.get('parameter_dtypes')}; "
                    f"moments={dense.get('devices')}/"
                    f"{dense.get('moment_dtype')}; pinned={dense.get('cpu_pinned')}; "
                    f"moment_bytes={int(dense.get('moment_bytes') or 0):,}",
                    flush=True,
                )
                optimizer_offload_verified = True
    except CorpusExhausted as exc:
        exhausted = True
        zero_grad = getattr(getattr(trainer, "optimizer", None), "zero_grad", None)
        if callable(zero_grad):
            try:
                zero_grad(set_to_none=True)
            except TypeError:
                zero_grad()
        print(f"SPALMER data exhausted cleanly: {exc}", flush=True)
    completed_steps = int(getattr(trainer.progress, "completed_steps", 0))
    tokens_seen = int(getattr(trainer.progress, "tokens_seen", 0))
    if completed_steps < 1 or last_metrics is None:
        raise RuntimeError(
            "selected SPALMER corpora produced no complete optimizer step; "
            "check shards, filters, sequence length, and accumulation size"
        )
    optimizer_placement = _optimizer_placement_payload(trainer.optimizer)
    _validate_initialized_optimizer_offload(optimizer_placement)
    if telemetry.last_step < int(_metric_value(last_metrics, "step", 0)):
        telemetry.record(last_metrics)
    final_progress = _metric_payload(
        last_metrics, max_steps=int(prepared.training.max_steps)
    )
    final_progress["msg"] = (
        "data exhausted before step cap; saving"
        if exhausted
        else "step cap reached; saving"
    )
    _write_json_atomic(rd / "progress.json", final_progress)

    training_view = _tokenized_view_manifest(
        batches,
        sequence_length=int(prepared.training.sequence_length),
    )
    data_provenance["training"] = training_view
    cfg["data_provenance"] = data_provenance
    _write_json_atomic(rd / "tokenized_view_manifest.json", training_view)

    checkpoint_path = _save_bundle_checkpoint(
        output / "spalmer.pt",
        api=api,
        trainer=trainer,
        prepared=prepared,
        metrics=last_metrics,
        data_provenance=data_provenance,
        run_id=rd.name,
    )
    result = {
        "steps": completed_steps,
        "tokens": tokens_seen,
        "data_exhausted": exhausted,
        "final": _metric_payload(last_metrics, max_steps=int(prepared.training.max_steps)),
        "optimizer_state_placement": optimizer_placement,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "checkpoint_format": "spalmer.prototype.checkpoint.v5",
        "checkpoint_file": "spalmer.pt",
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "plan": plan.to_dict(),
        "plan_fingerprint": plan.fingerprint,
        "tokenizer": prepared.tokenizer.identity.to_dict(),
        "model_config": prepared.bundle.model.to_dict(),
        "kda_config": _jsonable(prepared.bundle.kda),
        "mla_config": _jsonable(prepared.bundle.mla),
        "experts_config": _jsonable(prepared.bundle.experts),
        "directional_config": _jsonable(prepared.bundle.directional),
        "atxy_config": _jsonable(prepared.bundle.atxy),
        "training_config": prepared.training.to_dict(),
        "optimizer_state_placement": optimizer_placement,
        "parameter_accounting": accounting_payload,
        "data_provenance": data_provenance,
        "result": result,
        "exact_optimizer_resume": False,
    }
    _write_json_atomic(output / "spalmer_manifest.json", manifest)
    _write_json_atomic(output / "yylab_recipe.json", cfg)
    _write_json_atomic(output / "tokenized_view_manifest.json", training_view)

    actual_parameters = int(accounting_payload["total"] or plan.parameters.total)
    checkpoint_metadata = {
        "model_family": MODEL_FAMILY,
        "checkpoint_format": manifest["checkpoint_format"],
        "checkpoint_file": manifest["checkpoint_file"],
        "initialization": "scratch",
        "architecture": "SPALMER",
        "plan": plan.to_dict(),
        "plan_fingerprint": plan.fingerprint,
        "parameters": actual_parameters,
        "tokenizer": manifest["tokenizer"],
        "precision": {
            "expert_weights": plan.config.expert_weight_format,
            "expert_activations": prepared.bundle.experts.expert_activation_format,
            "master_weights": prepared.bundle.experts.expert_master_dtype,
            "expert_qat_backend": prepared.bundle.experts.expert_qat_backend,
            "optimizer_moments": "float32_cpu_offload",
        },
        "training_sequence_length": int(prepared.training.sequence_length),
        "data_provenance": data_provenance,
    }
    db.execute(
        "UPDATE checkpoints SET kind='base', source='local', ref=?, arch=?, params_b=?, "
        "path=?, hidden_size=?, num_layers=?, quant_locked='', meta_json=? WHERE id=?",
        (
            str(Path(checkpoint_path).resolve()),
            "SPALMER",
            round(actual_parameters / 1e9, 9),
            str(output.resolve()),
            int(plan.config.d_model),
            int(plan.config.n_layers),
            json.dumps(checkpoint_metadata, ensure_ascii=False, separators=(",", ":")),
            str(cfg["out_checkpoint_id"]),
        ),
    )
    return {
        "result": result,
        "model_family": MODEL_FAMILY,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "parameters": actual_parameters,
        "plan": plan.to_dict(),
        "plan_fingerprint": plan.fingerprint,
        "docs_seen": batches.docs_seen,
        "tokens": tokens_seen,
        "corpora": list(cfg["corpora"]),
        "corpus_mix": cfg["corpus_mix"],
        "domains": list(cfg.get("domains") or []),
        "splits": list(cfg.get("splits") or []),
        "data_provenance": data_provenance,
        "training_data": training_view,
    }


__all__ = [
    "CorpusExhausted",
    "PreparedSpalmerRun",
    "YYLabSPALMERBatchSource",
    "build_spalmer_batch_source",
    "build_spalmer_bundle",
    "build_spalmer_training_config",
    "normalize_spalmer_config",
    "plan_spalmer_configuration",
    "prepare_spalmer_run",
    "resolve_spalmer_tokenizer",
    "run_spalmer_base",
]
