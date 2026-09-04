from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from functools import lru_cache
from typing import Any

from nicegui import ui

from ... import db
from ...ingest.domains import DOMAINS
from ...tokenizer_registry import hub_preset, hub_preset_options, list_tokenizers
from ...train.data import (
    CORPUS_MIX_CONCATENATE,
    CORPUS_MIX_ROUND_ROBIN,
    CORPUS_MIX_TOKEN_BALANCED,
    corpus_training_preflight,
    normalize_corpus_mix,
    normalize_source_splits,
)
from ...workers.launch import launch_worker
from ..components import field_grid, form_section, hint
from ..shell import ACCENT, panel, shell
from .train_mole import corpus_options

SPALMER_RUN_KIND = "spalmer_base"
SPALMER_RECIPE_VERSION = 5
DEFAULT_SOURCE_SPLITS = ("train", "unspecified")
PARAMETER_TARGETS = {
    "10M": 10_000_000,
    "50M": 50_000_000,
    "100M": 100_000_000,
}
# Per-layer embedding backends.  Legacy fake-QAT stays the default so existing
# recipes, plans, and fingerprints are untouched; QR-PLE is an explicit choice.
PLE_BACKENDS = {
    "qr": "QR-PLE (BF16 compositional)",
    "fake_qat": "Legacy fake-QAT PLE",
}
DEFAULT_PLE_BACKEND = "fake_qat"
_PLE_BACKEND_NOTES = {
    "qr": (
        "QR-PLE: layer 0 keeps one exact BF16 input table; every later layer composes "
        "quotient/remainder lanes in BF16 with no fake quantization. The freed budget goes "
        "back into width, depth, and experts. QR checkpoints are scratch models (format v7) "
        "and cannot continue a fake-QAT parent."
    ),
    "fake_qat": (
        "Legacy PLE: every layer owns a wide four-bit fake-QAT lane table. Existing runs, "
        "plans, and fingerprints are unchanged."
    ),
}


def _ordered_values(value: str | Iterable[str] | None) -> list[str]:
    values = [value] if isinstance(value, str) else (value or [])
    selected: list[str] = []
    seen: set[str] = set()
    for raw in values:
        normalized = str(raw).strip()
        if normalized and normalized not in seen:
            selected.append(normalized)
            seen.add(normalized)
    return selected


def _resolve_target(scale: str, custom_target: int | float | None) -> int:
    if scale in PARAMETER_TARGETS:
        return PARAMETER_TARGETS[scale]
    if scale != "custom":
        raise ValueError(f"unknown SPALMER scale: {scale!r}")
    target = int(custom_target or 0)
    if target < 1:
        raise ValueError("custom target parameters must be positive")
    return target


def _spalmer_checkpoint_rows() -> list[dict[str, Any]]:
    """Registered base checkpoints whose metadata declares the SPALMER family."""

    rows = db.q(
        "SELECT id,name,params_b,hidden_size,num_layers,meta_json,created_at "
        "FROM checkpoints WHERE kind='base' ORDER BY created_at DESC"
    )
    selected: list[dict[str, Any]] = []
    for row in rows:
        try:
            meta = json.loads(row.get("meta_json") or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        family = str(meta.get("model_family") or meta.get("family") or "").strip().casefold()
        if family == "spalmer":
            selected.append(dict(row))
    return selected


def _parent_plan(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Read a recorded scale plan from run-produced or scan-registered metadata."""

    plan = meta.get("plan")
    if not isinstance(plan, Mapping):
        nested = meta.get("metadata")
        plan = nested.get("plan") if isinstance(nested, Mapping) else None
    return dict(plan) if isinstance(plan, Mapping) else {}


def _checkpoint_recurrence(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read a checkpoint's recurrent-core block from either metadata shape.

    Run-produced rows carry ``recurrence`` at the top of ``meta_json``; a bundle
    registered through the scanner carries ``recurrence_config`` there and the
    original block under the bundle's own ``metadata``.
    """

    meta: Any = row.get("meta_json", row) if isinstance(row, Mapping) else None
    if isinstance(meta, str):
        try:
            meta = json.loads(meta or "{}")
        except (TypeError, ValueError):
            return None
    if not isinstance(meta, Mapping):
        return None
    nested = meta.get("metadata")
    candidates = (
        meta.get("recurrence"),
        meta.get("recurrence_config"),
        nested.get("recurrence") if isinstance(nested, Mapping) else None,
    )
    for block in candidates:
        if not isinstance(block, Mapping) or not block:
            continue
        # A scratch recipe carries `{"enabled": False}` for a flat model.
        if block.get("enabled") is False:
            continue
        return dict(block)
    return None


def _checkpoint_ple(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read a checkpoint's PLE block from run-produced or scanner metadata.

    Run-produced rows carry ``ple`` at the top of ``meta_json``; a scanned bundle
    carries ``ple_backend`` there and ``model_config.ple_backend`` in its own
    metadata. Bundles predating the key are legacy fake-QAT and yield ``None``.
    """

    meta: Any = row.get("meta_json", row) if isinstance(row, Mapping) else None
    if isinstance(meta, str):
        try:
            meta = json.loads(meta or "{}")
        except (TypeError, ValueError):
            return None
    if not isinstance(meta, Mapping):
        return None
    nested = meta.get("metadata")
    candidates = (meta.get("ple"), nested.get("ple") if isinstance(nested, Mapping) else None)
    for block in candidates:
        if isinstance(block, Mapping) and block.get("backend"):
            return dict(block)
    model_config = meta.get("model_config")
    backend = meta.get("ple_backend") or (
        model_config.get("ple_backend") if isinstance(model_config, Mapping) else None
    )
    if backend:
        return {"backend": str(backend)}
    family = str(meta.get("model_family") or meta.get("family") or "").strip().casefold()
    architecture = str(meta.get("architecture") or "").strip().casefold()
    checkpoint_format = str(meta.get("checkpoint_format") or "").strip().casefold()
    if (
        family == "spalmer"
        or architecture == "spalmer"
        or checkpoint_format.startswith("spalmer.prototype.checkpoint")
    ):
        return {"backend": DEFAULT_PLE_BACKEND}
    return None


def _resolve_budget(
    *,
    token_budget: int,
    max_steps: int,
    sequence_length: int,
    micro_batch: int,
    gradient_accumulation: int,
) -> dict[str, int | str]:
    if sequence_length < 1 or micro_batch < 1 or gradient_accumulation < 1:
        raise ValueError("sequence length, micro batch, and gradient accumulation must be positive")
    requested_tokens = int(token_budget or 0)
    requested_steps = int(max_steps or 0)
    if requested_tokens < 0 or requested_steps < 0:
        raise ValueError("token budget and max steps cannot be negative")
    if requested_tokens and requested_steps:
        raise ValueError("choose either a training token budget or max steps, not both")
    if not requested_tokens and not requested_steps:
        raise ValueError("enter a training token budget or a maximum optimizer-step count")
    tokens_per_step = sequence_length * micro_batch * gradient_accumulation
    if requested_tokens:
        steps = math.ceil(requested_tokens / tokens_per_step)
        source = "token_budget"
    else:
        steps = requested_steps
        source = "max_steps"
    scheduled = steps * tokens_per_step
    return {
        "requested_training_tokens": requested_tokens,
        "max_steps": steps,
        "tokens_per_optimizer_step": tokens_per_step,
        "scheduled_training_tokens": scheduled,
        "token_budget_rounding": scheduled - requested_tokens if requested_tokens else 0,
        "step_cap_source": source,
    }


@lru_cache(maxsize=128)
def _cached_plan_json(
    target_parameters: int,
    vocab_size: int,
    expert_weight_format: str,
    ple_backend: str,
    directional_enabled: bool,
    directional_groups: int,
    directional_rank: int,
    atxy_enabled: bool,
    atxy_value_dim: int,
    atxy_a_cardinality: int,
    atxy_t_cardinality: int,
    atxy_x_cardinality: int,
    atxy_y_cardinality: int,
    atxy_injection_layer: int,
    recurrence_enabled: bool,
    recurrence_prelude: int,
    recurrence_coda: int,
    recurrence_default_steps: int,
    recurrence_latent_init_std: float,
) -> str:
    """Run the construction-free SPALMER planner behind a lazy import."""

    try:
        from spalmer.experiment.planning import (
            ATXYScaleConfig,
            DirectionalScaleConfig,
            RecurrenceScaleConfig,
            plan_configuration,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "SPALMER's planning package is unavailable. Install yyLab's SPALMER dependency "
            "before planning this run."
        ) from exc

    directional = (
        DirectionalScaleConfig(
            num_feature_groups=directional_groups,
            lateral_rank=directional_rank,
        )
        if directional_enabled
        else None
    )
    atxy = (
        ATXYScaleConfig(
            value_dim=atxy_value_dim,
            a_cardinality=atxy_a_cardinality,
            t_cardinality=atxy_t_cardinality,
            x_cardinality=atxy_x_cardinality,
            y_cardinality=atxy_y_cardinality,
            injection_layer=atxy_injection_layer,
        )
        if atxy_enabled
        else None
    )
    recurrence = (
        RecurrenceScaleConfig(
            prelude_layers=recurrence_prelude,
            coda_layers=recurrence_coda,
            default_steps=recurrence_default_steps,
            latent_init_std=recurrence_latent_init_std,
        )
        if recurrence_enabled
        else None
    )
    kwargs: dict[str, Any] = {}
    if ple_backend != DEFAULT_PLE_BACKEND:
        # Legacy previews keep the exact historical planner call so their
        # fingerprints match every stored run; only QR-PLE names its backend.
        kwargs["ple_backend"] = ple_backend
    plan = plan_configuration(
        target_parameters,
        vocab_size,
        expert_weight_format=expert_weight_format,
        directional=directional,
        atxy=atxy,
        recurrence=recurrence,
        **kwargs,
    )
    return json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":"))


def plan_spalmer_configuration(
    *,
    target_parameters: int,
    vocab_size: int,
    expert_weight_format: str,
    directional: dict[str, Any],
    atxy: dict[str, Any],
    recurrence: dict[str, Any] | None = None,
    ple_backend: str = DEFAULT_PLE_BACKEND,
) -> dict[str, Any]:
    """Return a JSON-safe plan without constructing a model or allocating weights."""

    target = int(target_parameters)
    vocabulary = int(vocab_size)
    if target < 1:
        raise ValueError("target parameters must be positive")
    if vocabulary < 1:
        raise ValueError("preview vocabulary must be positive")
    if expert_weight_format not in {"mxfp4", "nvfp4"}:
        raise ValueError("expert weights must use MXFP4 or NVFP4")
    backend = str(ple_backend or DEFAULT_PLE_BACKEND).strip().casefold()
    if backend not in PLE_BACKENDS:
        raise ValueError("PLE backend must be QR-PLE or legacy fake-QAT")
    directional_enabled = bool(directional.get("enabled"))
    atxy_enabled = bool(atxy.get("enabled"))
    recurrence_values = dict(recurrence or {})
    recurrence_enabled = bool(recurrence_values.get("enabled"))
    payload = _cached_plan_json(
        target,
        vocabulary,
        expert_weight_format,
        backend,
        directional_enabled,
        int(directional.get("num_feature_groups") or 0) if directional_enabled else 4,
        int(directional.get("lateral_rank") or 0) if directional_enabled else 16,
        atxy_enabled,
        int(atxy.get("value_dim") or 0) if atxy_enabled else 1,
        int(atxy.get("a_cardinality") or 0) if atxy_enabled else 1,
        int(atxy.get("t_cardinality") or 0) if atxy_enabled else 1,
        int(atxy.get("x_cardinality") or 0) if atxy_enabled else 1,
        int(atxy.get("y_cardinality") or 0) if atxy_enabled else 1,
        int(atxy.get("injection_layer") or 0),
        recurrence_enabled,
        int(recurrence_values.get("prelude_layers") or 1),
        int(recurrence_values.get("coda_layers") or 1),
        int(recurrence_values.get("default_steps") or 8),
        float(
            1.0
            if recurrence_values.get("latent_init_std") is None
            else recurrence_values["latent_init_std"]
        ),
    )
    return json.loads(payload)


@lru_cache(maxsize=8)
def attention_backend_status(backend: str) -> dict[str, Any]:
    """Inspect KDA backend availability without constructing a model."""

    requested = str(backend or "auto")
    if requested not in {"auto", "reference", "fla"}:
        raise ValueError(f"unknown attention backend: {requested!r}")
    try:
        from spalmer.attention.backends import fla_available

        fla_is_available = bool(fla_available())
    except (ImportError, ModuleNotFoundError) as exc:
        return {
            "requested_backend": requested,
            "available": requested != "fla",
            "fla_available": False,
            "detail": f"SPALMER attention capability check unavailable: {exc}",
        }
    if requested == "reference":
        detail = "Pure-PyTorch KDA reference path; correct but not the high-throughput CUDA lane."
        available = True
    elif requested == "fla":
        available = fla_is_available
        detail = (
            "fla-core KDA kernels are importable; the worker still requires CUDA."
            if available
            else "fla-core KDA kernels are not importable; strict FLA would fail."
        )
    else:
        # yyLab's GPU-first bridge deliberately refuses an implicit reference
        # fallback. The user can still choose reference explicitly.
        available = fla_is_available
        detail = (
            "Auto will select the available FLA KDA kernels on CUDA."
            if fla_is_available
            else (
                "Auto cannot start GPU pretraining because fla-core is not importable; "
                "choose PyTorch reference explicitly for a correctness-only run."
            )
        )
    return {
        "requested_backend": requested,
        "available": available,
        "fla_available": fla_is_available,
        "detail": detail,
    }


@lru_cache(maxsize=8)
def qat_backend_status(weight_format: str, backend: str) -> dict[str, Any]:
    """Return SPALMER's own W4A8 capability report without allocating tensors."""

    try:
        from spalmer.experts.qat import ExpertQATConfig, expert_qat_backend_status

        status = expert_qat_backend_status(
            ExpertQATConfig(
                weight_format=weight_format,
                activation_format="mxfp8",
                backend=backend,
            )
        )
    except (ImportError, ModuleNotFoundError) as exc:
        return {
            "requested_backend": backend,
            "selected_backend": None,
            "native_w4a8_available": False,
            "emulated": backend != "native",
            "torchao_available": False,
            "detail": f"SPALMER QAT capability check unavailable: {exc}",
        }
    return asdict(status)


def build_spalmer_run_config(
    *,
    checkpoint_name: str,
    tokenizer_ref: str,
    tokenizer_source: str,
    tokenizer_registry_id: str | None,
    tokenizer_preview_vocab_size: int,
    target_scale: str,
    target_parameters: int,
    resolved_plan: dict[str, Any],
    corpora: Iterable[str],
    domains: Iterable[str] | None,
    splits: Iterable[str] | None,
    corpus_mix: str,
    cross_corpus_dedup: bool,
    source_max_ratio: float | None,
    sequence_length: int,
    token_budget: int,
    max_steps: int,
    micro_batch: int,
    gradient_accumulation: int,
    learning_rate: float,
    minimum_lr_ratio: float,
    warmup_steps: int,
    weight_decay: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    seed: int,
    gradient_clip: float | None,
    auxiliary_loss_weight: float,
    surprise_calibration_weight: float,
    log_every: int,
    save_every: int,
    activation_checkpointing: bool,
    expert_weight_format: str,
    attention_backend: str,
    expert_qat_backend: str,
    directional: dict[str, Any],
    atxy: dict[str, Any],
    out_checkpoint_id: str,
    data_preflight_warnings: Iterable[str] = (),
    initialization_mode: str = "scratch",
    parent_checkpoint_id: str | None = None,
    epochs: int = 1,
    recurrence: dict[str, Any] | None = None,
    mean_recurrence: float | None = None,
    backprop_depth: int | None = None,
    recurrence_sigma: float | None = None,
    parent_recurrent: bool = False,
    ple_backend: str | None = None,
) -> dict[str, Any]:
    """Build the durable UI/worker boundary for one SPALMER scratch or continued run."""

    name = str(checkpoint_name or "").strip()
    reference = str(tokenizer_ref or "").strip()
    mode = str(initialization_mode or "scratch").strip().casefold()
    if mode not in {"scratch", "checkpoint"}:
        raise ValueError(f"unsupported initialization mode: {initialization_mode!r}")
    continuing = mode == "checkpoint"
    parent_id = str(parent_checkpoint_id or "").strip()
    if continuing and ple_backend is not None:
        raise ValueError(
            "continued SPALMER training inherits the PLE backend from the parent checkpoint"
        )
    resolved_ple_backend = str(ple_backend or DEFAULT_PLE_BACKEND).strip().casefold()
    if resolved_ple_backend not in PLE_BACKENDS:
        raise ValueError("PLE backend must be QR-PLE or legacy fake-QAT")
    if not name:
        raise ValueError("checkpoint name is required")
    if continuing and not parent_id:
        raise ValueError("choose a parent checkpoint to continue from")
    if not continuing and not reference:
        raise ValueError("choose a registered tokenizer or enter a tokenizer reference")
    if int(epochs) < 1:
        raise ValueError("epochs must be at least 1")
    selected_corpora = _ordered_values(corpora)
    if not selected_corpora:
        raise ValueError("choose at least one training corpus")
    normalized_domains = _ordered_values(domains)
    normalized_splits = normalize_source_splits(splits)
    mix = normalize_corpus_mix(corpus_mix)
    target = int(target_parameters)
    vocabulary = int(tokenizer_preview_vocab_size)
    fingerprint = ""
    if not continuing:
        if target < 1 or vocabulary < 1:
            raise ValueError("target parameters and tokenizer preview vocabulary must be positive")
        if int(resolved_plan.get("target_parameters") or 0) != target:
            raise ValueError("resolved SPALMER plan does not match the requested target")
        planned_config = dict(resolved_plan.get("config") or {})
        if int(planned_config.get("vocab_size") or 0) != vocabulary:
            raise ValueError("resolved SPALMER plan does not match the preview vocabulary")
        if str(planned_config.get("expert_weight_format") or "") != expert_weight_format:
            raise ValueError("resolved SPALMER plan does not match the expert weight format")
        planned_ple_backend = str(planned_config.get("ple_backend") or DEFAULT_PLE_BACKEND)
        if planned_ple_backend != resolved_ple_backend:
            raise ValueError("resolved SPALMER plan does not match the PLE backend")
        fingerprint = str(resolved_plan.get("fingerprint") or "")
        if len(fingerprint) != 64:
            raise ValueError("resolved SPALMER plan is missing its fingerprint")

    budget = _resolve_budget(
        token_budget=int(token_budget),
        max_steps=int(max_steps),
        sequence_length=int(sequence_length),
        micro_batch=int(micro_batch),
        gradient_accumulation=int(gradient_accumulation),
    )
    resolved_steps = int(budget["max_steps"])
    warmup = int(warmup_steps)
    if not 0 <= warmup < resolved_steps:
        raise ValueError("warmup steps must be in [0, max steps)")
    if float(learning_rate) <= 0:
        raise ValueError("learning rate must be positive")
    if not 0 <= float(minimum_lr_ratio) <= 1:
        raise ValueError("minimum learning-rate ratio must be in [0, 1]")
    if float(weight_decay) < 0:
        raise ValueError("weight decay cannot be negative")
    if not 0 <= float(beta1) < 1 or not 0 <= float(beta2) < 1:
        raise ValueError("Adam betas must be in [0, 1)")
    if float(epsilon) <= 0:
        raise ValueError("Adam epsilon must be positive")
    if int(seed) < 0:
        raise ValueError("seed cannot be negative")
    if gradient_clip is not None and float(gradient_clip) <= 0:
        raise ValueError("gradient clip must be positive or disabled")
    if float(auxiliary_loss_weight) < 0 or float(surprise_calibration_weight) < 0:
        raise ValueError("auxiliary loss weights cannot be negative")
    for label, value in (("log interval", log_every), ("save interval", save_every)):
        if int(value) < 1:
            raise ValueError(f"{label} must be positive")
    if activation_checkpointing:
        raise ValueError("SPALMER activation checkpointing is not implemented")

    attention_status = attention_backend_status(attention_backend)
    if attention_backend in {"auto", "fla"} and not attention_status["available"]:
        raise ValueError(str(attention_status["detail"]))
    qat_status = None
    if not continuing:
        qat_status = qat_backend_status(expert_weight_format, expert_qat_backend)
        if expert_qat_backend == "native" and not qat_status["native_w4a8_available"]:
            raise ValueError(str(qat_status["detail"]))

    recurrence_payload = dict(recurrence or {})
    recurrence_requested = bool(recurrence_payload.get("enabled"))
    training_recurrence_requested = any(
        value is not None
        for value in (mean_recurrence, backprop_depth, recurrence_sigma)
    )
    if continuing:
        if recurrence_requested:
            raise ValueError(
                "continued SPALMER training inherits the recurrent core from the "
                "parent checkpoint"
            )
        recurrent = bool(parent_recurrent)
    else:
        recurrent = recurrence_requested
        if not recurrent and training_recurrence_requested:
            raise ValueError(
                "recurrence training values require the recurrent-depth architecture"
            )
        if recurrent:
            prelude_layers = int(recurrence_payload.get("prelude_layers") or 0)
            coda_layers = int(recurrence_payload.get("coda_layers") or 0)
            default_steps = int(recurrence_payload.get("default_steps") or 0)
            if prelude_layers < 1 or coda_layers < 1:
                raise ValueError(
                    "the recurrent prelude and coda need at least one block each"
                )
            if default_steps < 1:
                raise ValueError("default inference steps must be positive")
            planned_layers = int(dict(resolved_plan.get("config") or {}).get("n_layers") or 0)
            if planned_layers <= prelude_layers + coda_layers:
                raise ValueError(
                    "the recurrent core needs at least one layer beyond the prelude "
                    "and coda"
                )
            if not isinstance(resolved_plan.get("recurrence"), Mapping):
                raise ValueError("resolved SPALMER plan is not recurrent")
    resolved_mean_recurrence: float | None = None
    resolved_backprop_depth: int | None = None
    resolved_recurrence_sigma: float | None = None
    if recurrent:
        resolved_mean_recurrence = float(8.0 if mean_recurrence is None else mean_recurrence)
        resolved_backprop_depth = int(4 if backprop_depth is None else backprop_depth)
        resolved_recurrence_sigma = float(
            0.5 if recurrence_sigma is None else recurrence_sigma
        )
        if resolved_mean_recurrence < 1:
            raise ValueError("mean recurrence must be at least 1")
        if resolved_backprop_depth < 1:
            raise ValueError("backprop depth must be at least 1")
        if resolved_recurrence_sigma < 0:
            raise ValueError("recurrence sigma cannot be negative")

    atxy_payload = dict(atxy)
    if atxy_payload.get("enabled"):
        for field in (
            "value_dim",
            "a_cardinality",
            "t_cardinality",
            "x_cardinality",
            "y_cardinality",
        ):
            if int(atxy_payload.get(field) or 0) < 1:
                raise ValueError(f"ATXY {field} must be positive when ATXY is enabled")

    source_cap = (
        float(source_max_ratio)
        if mix == CORPUS_MIX_TOKEN_BALANCED
        and len(selected_corpora) > 1
        and source_max_ratio is not None
        and float(source_max_ratio) > 0
        else None
    )
    if source_cap is not None and source_cap < 1 / len(selected_corpora):
        raise ValueError("maximum single-corpus share is below the feasible equal share")
    experts = {
        "num_experts": 200,
        "active_experts": 2,
        "min_active_experts": 2,
        "max_active_experts": 20,
        "min_resident_experts": 2,
        "max_resident_experts": 20,
        "expert_execution": "grouped",
        "expert_promotion_format": "mxfp8",
        "potentiation_budget": 2,
        "potentiation_ema_decay": 0.95,
        "potentiation_warmup_steps": 4,
        "potentiation_hold_steps": 8,
        "potentiation_hysteresis": 0.05,
        "residency_increment": 2,
        "residency_min_gain": 0.02,
    }
    spalmer = {
        "target_scale": target_scale,
        "target_parameters": target,
        "preview_vocab_size": vocabulary,
        "num_experts": 200,
        "expert_weight_format": expert_weight_format,
        "expert_activation_format": "mxfp8",
        "expert_master_dtype": "bfloat16",
        "attention_backend": attention_backend,
        "require_fla": attention_backend != "reference",
        "allow_reference_attention": attention_backend == "reference",
        "expert_qat_backend": expert_qat_backend,
        "ple_backend": resolved_ple_backend,
        "directional": dict(directional),
        "atxy": atxy_payload,
        "recurrence": (
            {
                "enabled": True,
                "prelude_layers": int(recurrence_payload.get("prelude_layers") or 1),
                "coda_layers": int(recurrence_payload.get("coda_layers") or 1),
                "default_steps": int(recurrence_payload.get("default_steps") or 8),
                "latent_init_std": 1.0,
            }
            if recurrence_requested
            else {"enabled": False}
        ),
        "experts": experts,
        "plan_fingerprint": fingerprint,
        "resolved_plan": resolved_plan,
        "capabilities": {
            "attention": attention_status,
            "expert_qat": qat_status,
        },
    }
    training = {
        "max_steps": resolved_steps,
        "micro_batch_size": int(micro_batch),
        "sequence_length": int(sequence_length),
        "gradient_accumulation_steps": int(gradient_accumulation),
        "learning_rate": float(learning_rate),
        "min_learning_rate_ratio": float(minimum_lr_ratio),
        "warmup_steps": warmup,
        "weight_decay": float(weight_decay),
        "adam_beta1": float(beta1),
        "adam_beta2": float(beta2),
        "adam_epsilon": float(epsilon),
        "gradient_clip": None if gradient_clip is None else float(gradient_clip),
        "auxiliary_loss_weight": float(auxiliary_loss_weight),
        "surprise_calibration_weight": float(surprise_calibration_weight),
        "device": "cuda",
        "require_cuda": True,
        "compute_dtype": "bfloat16",
        "parameter_dtype": "bfloat16",
        "optimizer_state_dtype": "float32",
        "optimizer_state_offload": "cpu",
        "stochastic_parameter_rounding": True,
        "optimizer_update_chunk_size": 1_048_576,
        "fused_adamw": "auto",
        "activation_checkpointing": False,
        "deterministic_algorithms": False,
        "seed": int(seed),
        # The yyLab corpus bridge does not yet have a validation stream.
        "validation_interval_steps": resolved_steps + 1,
        "checkpoint_interval_steps": int(save_every),
        "telemetry_interval_steps": int(log_every),
    }
    recurrence_training: dict[str, Any] = {}
    if recurrent:
        recurrence_training = {
            "mean_recurrence": resolved_mean_recurrence,
            "mean_backprop_depth": resolved_backprop_depth,
            "recurrence_sigma": resolved_recurrence_sigma,
            "recurrence_sampling": "poisson_lognormal",
        }
        training.update(recurrence_training)
    if continuing:
        initialization = {"mode": "checkpoint", "checkpoint_id": parent_id}
        tokenizer = {
            "source": "checkpoint",
            "registry_id": None,
            "ref": "",
            "preview_vocab_size": None,
        }
        spalmer_block: dict[str, Any] = {
            "attention_backend": attention_backend,
            "require_fla": attention_backend != "reference",
            "allow_reference_attention": attention_backend == "reference",
        }
        architecture_keys: dict[str, Any] = {}
    else:
        initialization = {"mode": "scratch", "tokenizer_ref": reference}
        tokenizer = {
            "source": tokenizer_source,
            "registry_id": tokenizer_registry_id,
            "ref": reference,
            "preview_vocab_size": vocabulary,
        }
        spalmer_block = spalmer
        architecture_keys = {
            "architecture": resolved_plan,
            "precision": {
                "recipe": "spalmer_w4a8_bf16_master_v1",
                "parameter_dtype": "bfloat16",
                "compute_dtype": "bfloat16",
                "expert_weight_format": expert_weight_format,
                "expert_activation_format": "mxfp8",
                "expert_master_dtype": "bfloat16",
                "expert_qat_backend": expert_qat_backend,
                **(
                    {
                        "ple_backend": "qr",
                        "ple_parameter_dtype": "bfloat16",
                        "ple_fake_quantization": False,
                    }
                    if resolved_ple_backend == "qr"
                    else {
                        "ple_backend": "fake_qat",
                        "ple_quant_bits": 4,
                        "ple_stochastic_rounding": True,
                    }
                ),
            },
        }
    return {
        "recipe_version": SPALMER_RECIPE_VERSION,
        "architecture_type": "spalmer_v1",
        "model_family": "spalmer",
        "method": "full",
        "objective": "causal_lm",
        "checkpoint_name": name,
        "output_name": name,
        "out_checkpoint_id": str(out_checkpoint_id),
        "initialization": initialization,
        "tokenizer": tokenizer,
        "spalmer": spalmer_block,
        **architecture_keys,
        "training": training,
        "corpora": selected_corpora,
        "domains": normalized_domains or None,
        "splits": normalized_splits or None,
        "corpus_mix": mix,
        "cross_corpus_dedup": bool(cross_corpus_dedup),
        "corpus_weights": {},
        "source_max_ratio": source_cap,
        "data_preflight_warnings": list(data_preflight_warnings),
        "seq_len": int(sequence_length),
        "micro_batch": int(micro_batch),
        "grad_accum": int(gradient_accumulation),
        "max_steps": resolved_steps,
        "epochs": int(epochs),
        "seed": int(seed),
        "shuffle_buffer": 128,
        "shuffle_buffer_bytes": 64 * 2**20,
        "tokenize_chunk": 8,
        "lr": float(learning_rate),
        "min_lr_ratio": float(minimum_lr_ratio),
        "warmup_steps": warmup,
        "weight_decay": float(weight_decay),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "eps": float(epsilon),
        "grad_clip": None if gradient_clip is None else float(gradient_clip),
        "auxiliary_loss_weight": float(auxiliary_loss_weight),
        "surprise_calibration_weight": float(surprise_calibration_weight),
        "log_every": int(log_every),
        "save_every": int(save_every),
        "activation_checkpointing": False,
        # Root aliases so a replayed recipe keeps the depth policy even when
        # only the flat keys are read.
        **(
            {
                "mean_recurrence": resolved_mean_recurrence,
                "mean_backprop_depth": resolved_backprop_depth,
                "recurrence_sigma": resolved_recurrence_sigma,
            }
            if recurrent
            else {}
        ),
        **budget,
    }


def _format_parameters(value: int) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"


@ui.page("/train/spalmer")
def train_spalmer_page() -> None:
    with shell("Train · SPALMER", "/train/spalmer"):
        ui.label("SPALMER base pretraining").classes("text-xl font-semibold")
        ui.label(
            "Plan and queue a real 10M–100M SPALMER checkpoint with per-layer PLE, "
            "3:1 KDA/MLA mixing, and a coherent 200-expert identity bank."
        ).classes("text-sm text-gray-400 -mt-2")

        tokenizer_rows = {
            row["id"]: row for row in list_tokenizers(ready_only=True, parsed=True)
        }
        tokenizer_choices = {
            tokenizer_id: f"{row['name']} · {int(row.get('vocab_size') or 0):,} tokens"
            for tokenizer_id, row in tokenizer_rows.items()
        }
        first_tokenizer_id = next(iter(tokenizer_choices), None)
        first_tokenizer_vocab = int(
            (tokenizer_rows.get(first_tokenizer_id) or {}).get("vocab_size") or 8192
        )
        corpora = corpus_options()
        parent_rows = {row["id"]: row for row in _spalmer_checkpoint_rows()}
        parent_choices = {
            checkpoint_id: (
                f"{row['name']} · "
                f"{_format_parameters(int(round(float(row.get('params_b') or 0) * 1e9)))}"
            )
            for checkpoint_id, row in parent_rows.items()
        }
        first_parent_id = next(iter(parent_choices), None)
        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
            with panel().classes("shrink-0").style("width: 720px; max-width: 100%"):
                form_section("1", "Initialization and output")
                checkpoint_name = (
                    ui.input("Checkpoint name", value="spalmer-10m")
                    .props("dense outlined")
                    .classes("w-full")
                )
                initialization_mode = (
                    ui.select(
                        {
                            "scratch": "Random weights (scratch)",
                            "checkpoint": "Continue from checkpoint",
                        },
                        value="scratch",
                        label="Initialization",
                    )
                    .props("dense outlined")
                    .classes("w-full")
                )
                parent_checkpoint = (
                    ui.select(
                        parent_choices,
                        value=first_parent_id,
                        label="Parent SPALMER checkpoint",
                    )
                    .props("dense outlined")
                    .classes("w-full")
                )
                parent_hint = hint(
                    "Architecture, expert numerics, and tokenizer are inherited from the parent. "
                    "Optimizer moments and the learning-rate schedule start fresh."
                )
                with ui.column().classes("w-full") as tokenizer_controls:
                    registered_tokenizer = (
                        ui.select(
                            tokenizer_choices,
                            value=first_tokenizer_id,
                            label="Registered tokenizer",
                        )
                        .props("dense outlined clearable")
                        .classes("w-full")
                    )
                    direct_tokenizer_ref = (
                        ui.input(
                            "Direct local / Hugging Face tokenizer reference",
                            placeholder="Optional; overrides the registered tokenizer",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    hub_preset_select = (
                        ui.select(
                            {"": "None", **hub_preset_options()},
                            value="",
                            label="Hugging Face preset (fills the direct reference)",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    hub_preset_note = ui.label("").classes("text-xs text-teal-300")
                    hub_preset_note.visible = False
                    preview_vocab = (
                        ui.number(
                            "Tokenizer vocabulary for planning",
                            value=first_tokenizer_vocab,
                            min=1,
                            max=10_000_000,
                            precision=0,
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    hint(
                        "The vocabulary is used only for the construction-free preview. The worker resolves "
                        "the tokenizer, replans with its actual vocabulary, and rejects a fingerprint mismatch. "
                        "A registered tokenizer's catalogued vocabulary is authoritative."
                    )

                ui.separator().classes("my-3")
                with ui.column().classes("w-full") as architecture_section:
                    form_section("2", "Architecture and execution")
                    with field_grid(3):
                        target_scale = (
                            ui.select(
                                {
                                    "10M": "10M total parameters",
                                    "50M": "50M total parameters",
                                    "100M": "100M total parameters",
                                    "custom": "Custom total target",
                                },
                                value="10M",
                                label="Parameter target",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        custom_target = (
                            ui.number(
                                "Custom exact parameters",
                                value=25_000_000,
                                min=1,
                                precision=0,
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        expert_weight_format = (
                            ui.select(
                                {
                                    "mxfp4": "MXFP4 expert weights",
                                    "nvfp4": "NVFP4 expert weights",
                                },
                                value="mxfp4",
                                label="Expert forward weights",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with field_grid(2):
                        ple_backend_select = (
                            ui.select(
                                dict(PLE_BACKENDS),
                                value=DEFAULT_PLE_BACKEND,
                                label="PLE backend",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        ple_backend_note = ui.label("").classes(
                            "text-xs text-gray-400 px-3 py-2 rounded-md w-full"
                        )
                        ple_backend_note.style(f"background: {ACCENT}10")
                    with field_grid(2):
                        attention_backend = (
                            ui.select(
                                {
                                    "auto": "Auto (FLA on CUDA when available)",
                                    "reference": "PyTorch reference",
                                    "fla": "FLA KDA (strict)",
                                },
                                value="auto",
                                label="KDA attention backend",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        expert_qat_backend = (
                            ui.select(
                                {
                                    "auto": "Auto (reference until native exists)",
                                    "reference": "Pure-Torch Q/DQ reference",
                                    "native": "Native mixed W4A8 (strict)",
                                },
                                value="auto",
                                label="Expert QAT backend",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    attention_status_label = ui.label("").classes(
                        "text-xs text-gray-400 px-3 py-2 rounded-md w-full"
                    )
                    attention_status_label.style(f"background: {ACCENT}10")
                    qat_status_label = ui.label("").classes(
                        "text-xs text-gray-400 px-3 py-2 rounded-md w-full"
                    )
                    qat_status_label.style(f"background: {ACCENT}10")

                    directional_enabled = ui.switch(
                        "Directional lateral/silencing branch", value=True
                    )
                    with ui.row().classes("w-full gap-3") as directional_controls:
                        directional_groups = (
                            ui.number("Feature groups", value=4, min=2, max=64, precision=0)
                            .props("dense outlined")
                            .classes("flex-1")
                        )
                        directional_rank = (
                            ui.number("Lateral rank", value=16, min=1, max=1024, precision=0)
                            .props("dense outlined")
                            .classes("flex-1")
                        )
                    atxy_enabled = ui.switch("Include ATXY module", value=False)
                    with ui.column().classes("w-full gap-2") as atxy_controls:
                        with field_grid(3):
                            atxy_value_dim = (
                                ui.number("ATXY value width", value=64, min=1, precision=0)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            atxy_a = (
                                ui.number("A cardinality", value=256, min=1, precision=0)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            atxy_t = (
                                ui.number("T cardinality", value=256, min=1, precision=0)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                        with field_grid(3):
                            atxy_x = (
                                ui.number("X cardinality", value=256, min=1, precision=0)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            atxy_y = (
                                ui.number("Y cardinality", value=256, min=1, precision=0)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            atxy_layer = (
                                ui.number("Injection layer", value=0, min=0, precision=0)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                        hint(
                            "This allocates the native ATXY component. Training only exercises it when the "
                            "worker receives numeric addresses, a mask, and a versioned ATXY store."
                        )
                    recurrence_enabled = ui.switch(
                        "Recurrent depth (Huginn-style latent reasoning)", value=False
                    )
                    with ui.column().classes("w-full gap-2") as recurrence_controls:
                        with field_grid(3):
                            recurrence_prelude = (
                                ui.number("Prelude blocks", value=1, min=1, precision=0)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            recurrence_coda = (
                                ui.number("Coda blocks", value=1, min=1, precision=0)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            recurrence_default_steps = (
                                ui.number(
                                    "Default inference steps", value=8, min=1, precision=0
                                )
                                .props("dense outlined")
                                .classes("w-full")
                            )
                        hint(
                            "The blocks between the prelude and the coda become one core that is "
                            "iterated r times per forward, so compute scales with r while the "
                            "parameter count only grows by the latent adapter. The checkpoint "
                            "records the default inference depth, and the paged expert cache is "
                            "re-staged every iteration — size it to the union of the core routes."
                        )
                    ui.label(
                        "Canonical expert policy: 200 coherent identities · training top-k 2 · "
                        "request residency 2–20 · grouped execution · BF16 master / "
                        "CPU-offloaded FP32 moments."
                    ).classes("text-xs text-violet-300")

                ui.separator().classes("my-3")
                form_section("3", "Training data")
                corpora_select = (
                    ui.select(
                        corpora,
                        value=[],
                        multiple=True,
                        label="Training corpora (one or more)",
                    )
                    .props("dense outlined use-chips clearable options-dense")
                    .classes("w-full")
                )
                with ui.row().classes("items-center gap-2 -mt-1"):
                    ui.button(
                        "Select all", on_click=lambda: set_corpora_selection(corpora)
                    ).props("dense flat no-caps size=sm")
                    ui.button("Clear", on_click=lambda: set_corpora_selection([])).props(
                        "dense flat no-caps size=sm"
                    )
                with field_grid(2):
                    domain_filter = (
                        ui.select(
                            DOMAINS,
                            value=[],
                            multiple=True,
                            label="Domain filter (optional)",
                        )
                        .props("dense outlined use-chips clearable")
                        .classes("w-full")
                    )
                    split_filter = (
                        ui.select(
                            {
                                "train": "Train",
                                "validation": "Validation",
                                "test": "Test",
                                "unspecified": "Unspecified / legacy",
                            },
                            value=list(DEFAULT_SOURCE_SPLITS),
                            multiple=True,
                            label="Source splits",
                        )
                        .props("dense outlined use-chips clearable options-dense")
                        .classes("w-full")
                    )
                with field_grid(2):
                    corpus_mix = (
                        ui.select(
                            {
                                CORPUS_MIX_TOKEN_BALANCED: "Token-balanced",
                                CORPUS_MIX_ROUND_ROBIN: "Document round-robin",
                                CORPUS_MIX_CONCATENATE: "Concatenate in chip order",
                            },
                            value=CORPUS_MIX_TOKEN_BALANCED,
                            label="Corpus mixing",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    source_ratio_cap = (
                        ui.number(
                            "Maximum single-corpus share (0 = off)",
                            value=0.5,
                            min=0,
                            max=1,
                            step=0.05,
                            precision=2,
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                cross_corpus_dedup = ui.checkbox(
                    "Deduplicate exact model-visible records across selected corpora", value=True
                )
                data_summary = ui.label("Choose one or more corpora.").classes(
                    "text-xs text-gray-400 px-3 py-2 rounded-md w-full"
                )
                data_summary.style(f"background: {ACCENT}10")
                data_messages = ui.label("").classes(
                    "text-xs text-amber-400 whitespace-pre-wrap"
                )
                data_messages.visible = False

                ui.separator().classes("my-3")
                form_section("4", "Training schedule")
                with field_grid(3):
                    sequence_length = (
                        ui.number("Sequence length", value=512, min=64, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    token_budget = (
                        ui.number(
                            "Training token budget (0 = use steps)",
                            value=10_000_000,
                            min=0,
                            precision=0,
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    max_steps = (
                        ui.number("Max steps (0 = use tokens)", value=0, min=0, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                with field_grid(3):
                    epochs = (
                        ui.number("Epochs (passes over the data)", value=1, min=1, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                with field_grid(3):
                    micro_batch = (
                        ui.number("Micro batch", value=1, min=1, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    grad_accum = (
                        ui.number("Gradient accumulation", value=8, min=1, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    learning_rate = (
                        ui.number("Learning rate", value=3e-4, min=1e-9, step=1e-5)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                with field_grid(3):
                    warmup_steps = (
                        ui.number("Warmup steps", value=100, min=0, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    weight_decay = (
                        ui.number("Weight decay", value=0.1, min=0, step=0.01)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    gradient_clip = (
                        ui.number("Gradient clip (0 = disabled)", value=1.0, min=0, step=0.1)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                with field_grid(3):
                    seed = (
                        ui.number("Seed", value=42, min=0, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    log_every = (
                        ui.number("Log every steps", value=10, min=1, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    save_every = (
                        ui.number("Save every steps", value=100, min=1, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                with ui.column().classes("w-full gap-2") as recurrence_training_controls:
                    with field_grid(3):
                        mean_recurrence = (
                            ui.number("Mean recurrence r̄", value=8.0, min=1, step=0.5)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        backprop_depth = (
                            ui.number("Backprop depth k", value=4, min=1, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        recurrence_sigma = (
                            ui.number("Recurrence sigma", value=0.5, min=0, step=0.1)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    hint(
                        "Each optimizer step samples one iteration count r from a log-normal "
                        "Poisson with mean r̄, and backpropagates through only the last k "
                        "iterations, so activation memory stays bounded while the model learns "
                        "to keep iterating. r̄ = 1 degenerates to a plain stack."
                    )
                with ui.expansion("Advanced optimizer and losses", icon="tune").classes(
                    "w-full"
                ):
                    with field_grid(3):
                        minimum_lr_ratio = (
                            ui.number("Minimum LR ratio", value=0.1, min=0, max=1, step=0.05)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        beta1 = (
                            ui.number("Adam beta1", value=0.9, min=0, max=0.9999, step=0.01)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        beta2 = (
                            ui.number("Adam beta2", value=0.95, min=0, max=0.99999, step=0.01)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with field_grid(3):
                        epsilon = (
                            ui.number("Adam epsilon", value=1e-8, min=1e-12, step=1e-8)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        auxiliary_loss_weight = (
                            ui.number("Router auxiliary loss", value=0.01, min=0, step=0.01)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        surprise_calibration_weight = (
                            ui.number("Surprise calibration loss", value=0.05, min=0, step=0.01)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    activation_checkpointing = ui.switch(
                        "Activation checkpointing (not implemented)", value=False
                    )
                    activation_checkpointing.enabled = False
                    hint(
                        "The current yyLab SPALMER bridge has no validation stream and does not "
                        "implement activation checkpointing. Both remain disabled for this run."
                    )
                budget_summary = ui.label("").classes("text-xs text-gray-400")

                ui.separator().classes("my-3")
                start_button = (
                    ui.button("Queue SPALMER pretraining", icon="play_arrow")
                    .props("no-caps unelevated")
                    .classes("w-full")
                )
                hint("Requires CUDA. The detached worker writes progress and logs to Runs.")

            with ui.column().classes("flex-1 gap-4 min-w-[380px]"):
                with panel():
                    ui.label("Construction-free plan").classes("text-base font-semibold")
                    plan_preview = ui.column().classes("w-full gap-2")
                with panel():
                    ui.label("Numerical contract").classes("text-base font-semibold")
                    ui.label(
                        "Only routed-expert forwards are W4A8: MXFP4 or NVFP4 weights and MXFP8 "
                        "activations. Persistent parameters and gradients are BF16; Adam moments "
                        "remain FP32. Legacy PLE uses four-bit stochastic fake quantization; "
                        "QR-PLE keeps its exact input table and codebooks in BF16."
                    ).classes("text-sm text-gray-300")
                    ui.label(
                        "The current routed-expert QAT lane is format-faithful fake "
                        "quantize/dequantize with a BF16 straight-through backward. Native packed "
                        "mixed W4A8 is shown unavailable until SPALMER supplies that kernel."
                    ).classes("text-xs text-amber-300 mt-2")

        def current_directional() -> dict[str, Any]:
            return {
                "enabled": bool(directional_enabled.value),
                "num_feature_groups": int(directional_groups.value or 0),
                "lateral_rank": int(directional_rank.value or 0),
            }

        def current_atxy() -> dict[str, Any]:
            return {
                "enabled": bool(atxy_enabled.value),
                "value_dim": int(atxy_value_dim.value or 0),
                "a_cardinality": int(atxy_a.value or 0),
                "t_cardinality": int(atxy_t.value or 0),
                "x_cardinality": int(atxy_x.value or 0),
                "y_cardinality": int(atxy_y.value or 0),
                "injection_layer": int(atxy_layer.value or 0),
            }

        def current_recurrence() -> dict[str, Any]:
            return {
                "enabled": bool(recurrence_enabled.value),
                "prelude_layers": int(recurrence_prelude.value or 1),
                "coda_layers": int(recurrence_coda.value or 1),
                "default_steps": int(recurrence_default_steps.value or 8),
                "latent_init_std": 1.0,
            }

        def current_ple_backend() -> str:
            return str(ple_backend_select.value or DEFAULT_PLE_BACKEND)

        def requested_target() -> int:
            return _resolve_target(str(target_scale.value or "10M"), custom_target.value)

        def selected_tokenizer() -> tuple[str, str, str | None]:
            direct = str(direct_tokenizer_ref.value or "").strip()
            if direct:
                return direct, "direct", None
            tokenizer_id = str(registered_tokenizer.value or "").strip()
            if not tokenizer_id:
                return "", "registered", None
            return tokenizer_id, "registered", tokenizer_id

        def continuing() -> bool:
            return str(initialization_mode.value or "scratch") == "checkpoint"

        def selected_parent() -> dict[str, Any] | None:
            return parent_rows.get(str(parent_checkpoint.value or ""))

        def parent_recurrence() -> dict[str, Any] | None:
            parent = selected_parent()
            return None if parent is None else _checkpoint_recurrence(parent)

        def recurrent_run() -> bool:
            if continuing():
                return parent_recurrence() is not None
            return bool(recurrence_enabled.value)

        def update_initialization_controls() -> None:
            parent_checkpoint.visible = continuing()
            parent_hint.visible = continuing()
            tokenizer_controls.visible = not continuing()
            architecture_section.visible = not continuing()
            recurrence_training_controls.visible = recurrent_run()
            update_plan_preview()

        def update_recurrence_controls() -> None:
            recurrence_controls.visible = bool(recurrence_enabled.value)
            recurrence_training_controls.visible = recurrent_run()
            update_plan_preview()

        def set_corpora_selection(value: str | Iterable[str] | None) -> None:
            corpora_select.value = _ordered_values(value)
            corpora_select.update()
            update_data_summary()

        def update_target_controls() -> None:
            custom_target.visible = target_scale.value == "custom"
            update_plan_preview()

        def update_ple_backend_controls() -> None:
            backend = current_ple_backend()
            ple_backend_note.text = _PLE_BACKEND_NOTES.get(backend, "")
            update_plan_preview()

        def update_directional_controls() -> None:
            directional_controls.visible = bool(directional_enabled.value)
            update_plan_preview()

        def update_atxy_controls() -> None:
            atxy_controls.visible = bool(atxy_enabled.value)
            update_plan_preview()

        def update_registered_tokenizer() -> None:
            tokenizer_id = str(registered_tokenizer.value or "")
            row = tokenizer_rows.get(tokenizer_id)
            if row and not str(direct_tokenizer_ref.value or "").strip():
                known_vocab = int(row.get("vocab_size") or 0)
                if known_vocab:
                    preview_vocab.value = known_vocab
                    preview_vocab.update()
            update_plan_preview()

        def apply_hub_preset() -> None:
            preset = hub_preset(str(hub_preset_select.value or ""))
            if preset is None:
                hub_preset_note.visible = False
                return
            direct_tokenizer_ref.value = preset.repo_id
            direct_tokenizer_ref.update()
            if preset.vocab_size:
                preview_vocab.value = preset.vocab_size
                preview_vocab.update()
            hub_preset_note.text = (
                f"{preset.note} The worker downloads it when the run starts and replans with the "
                "actual vocabulary. Import it under Tokenizers to make it a registered option with a "
                "verified, catalogued vocabulary."
            )
            hub_preset_note.visible = True
            update_plan_preview()

        def update_backend_status() -> None:
            attention = attention_backend_status(str(attention_backend.value or "auto"))
            attention_status_label.text = attention["detail"]
            attention_status_label.classes(
                replace=(
                    "text-xs text-red-300 px-3 py-2 rounded-md w-full"
                    if not attention["available"]
                    else "text-xs text-gray-400 px-3 py-2 rounded-md w-full"
                )
            )
            qat = qat_backend_status(
                str(expert_weight_format.value or "mxfp4"),
                str(expert_qat_backend.value or "auto"),
            )
            qat_status_label.text = str(qat["detail"])
            qat_status_label.classes(
                replace=(
                    "text-xs text-red-300 px-3 py-2 rounded-md w-full"
                    if expert_qat_backend.value == "native"
                    and not qat["native_w4a8_available"]
                    else "text-xs text-amber-300 px-3 py-2 rounded-md w-full"
                    if qat["emulated"]
                    else "text-xs text-green-300 px-3 py-2 rounded-md w-full"
                )
            )

        def update_plan_preview() -> None:
            plan_preview.clear()
            with plan_preview:
                if continuing():
                    parent = selected_parent()
                    if parent is None:
                        ui.label("Register a SPALMER checkpoint to continue from.").classes(
                            "text-sm text-red-300"
                        )
                        return
                    meta = json.loads(parent.get("meta_json") or "{}")
                    parent_plan = _parent_plan(meta)
                    parameters = int(round(float(parent.get("params_b") or 0) * 1e9))
                    ui.label(
                        f"Inherited from {parent['name']}: {_format_parameters(parameters)} "
                        f"parameters · d={int(parent.get('hidden_size') or 0):,} · "
                        f"{int(parent.get('num_layers') or 0)} layers"
                    ).classes("text-sm text-gray-200")
                    inherited_ple = _checkpoint_ple(parent)
                    if inherited_ple:
                        backend = str(inherited_ple.get("backend") or DEFAULT_PLE_BACKEND)
                        detail = PLE_BACKENDS.get(backend, backend)
                        if backend == "qr":
                            exact = int(inherited_ple.get("exact_input_parameters") or 0)
                            refresh = int(inherited_ple.get("qr_refresh_parameters") or 0)
                            detail += (
                                f" · exact input {_format_parameters(exact)}"
                                f" · QR refresh {_format_parameters(refresh)}"
                            )
                        ui.label(f"PLE backend: {detail}").classes("text-sm text-cyan-200")
                    inherited_recurrence = _checkpoint_recurrence(parent)
                    if inherited_recurrence:
                        steps = int(inherited_recurrence.get("default_steps") or 0)
                        prelude = int(inherited_recurrence.get("prelude_layers") or 0)
                        core = int(inherited_recurrence.get("core_layers") or 0)
                        coda = int(inherited_recurrence.get("coda_layers") or 0)
                        ui.label(
                            f"recurrent: prelude {prelude} · core {core} · coda {coda} · "
                            f"effective depth {prelude}+{core}×{steps}+{coda} = "
                            f"{prelude + core * steps + coda} at {steps} steps"
                        ).classes("text-sm text-teal-200")
                    ui.label(
                        f"Plan {str(parent_plan.get('fingerprint') or '')[:16]}… · tokenizer and "
                        "expert numerics come from the bundle."
                    ).classes("text-[11px] text-gray-500")
                    return
                try:
                    plan = plan_spalmer_configuration(
                        target_parameters=requested_target(),
                        vocab_size=int(preview_vocab.value or 0),
                        expert_weight_format=str(expert_weight_format.value or "mxfp4"),
                        directional=current_directional(),
                        atxy=current_atxy(),
                        recurrence=current_recurrence(),
                        ple_backend=current_ple_backend(),
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    ui.label(str(exc)).classes("text-sm text-red-300")
                    return
                config = dict(plan["config"])
                parameters = dict(plan["parameters"])
                memory = dict(plan["memory"])
                actual = int(parameters["total"])
                target = int(plan["target_parameters"])
                error = int(plan["parameter_error"])
                relative_error = float(plan["relative_error"])
                with ui.row().classes("w-full gap-2 flex-wrap"):
                    ui.badge(f"target {_format_parameters(target)}").props("outline color=primary")
                    ui.badge(f"planned {_format_parameters(actual)}").props("outline color=teal")
                    error_color = "green" if abs(relative_error) <= 0.02 else "orange"
                    ui.badge(
                        f"error {error:+,} ({relative_error:+.2%})"
                    ).props(f"outline color={error_color}")
                ple_plan = plan.get("ple") if isinstance(plan.get("ple"), Mapping) else None
                ple_label = (
                    f"QR-PLE · {int(config['ple_expansion'])} lanes"
                    if ple_plan is not None
                    else f"fake-QAT PLE ×{int(config['ple_expansion'])}"
                )
                ui.label(
                    f"d={int(config['d_model']):,} · {int(config['n_layers'])} layers · "
                    f"{int(config['num_heads'])} heads × "
                    f"{int(config['d_model']) // int(config['num_heads'])} · "
                    f"{ple_label}"
                ).classes("text-sm text-gray-200")
                if ple_plan is not None:
                    moduli = ", ".join(str(int(value)) for value in ple_plan.get("moduli") or [])
                    exact = int(ple_plan.get("exact_input_parameters") or 0)
                    refresh = int(ple_plan.get("qr_refresh_parameters") or 0)
                    ple_total = int(ple_plan.get("total_parameters") or 0)
                    ui.label(
                        f"QR-PLE: exact BF16 input {_format_parameters(exact)} · refresh "
                        f"codebooks {_format_parameters(refresh)} on "
                        f"{max(int(config['n_layers']) - 1, 0)} layers · lane moduli {moduli} · "
                        f"PLE total {_format_parameters(ple_total)}"
                    ).classes("text-sm text-cyan-200")
                ui.label(
                    f"200 experts × width {int(config['expert_width'])} · shared width "
                    f"{int(config['shared_width']):,} · KDA {int(config['n_layers']) * 3 // 4} / "
                    f"MLA {int(config['n_layers']) // 4}"
                ).classes("text-sm text-gray-300")
                recurrence_plan = plan.get("recurrence")
                if isinstance(recurrence_plan, Mapping):
                    core_mixers = dict(recurrence_plan.get("core_mixers") or {})
                    prelude = int(recurrence_plan.get("prelude_layers") or 0)
                    core = int(recurrence_plan.get("core_layers") or 0)
                    coda = int(recurrence_plan.get("coda_layers") or 0)
                    steps = int(recurrence_plan.get("default_steps") or 0)
                    depth = int(
                        recurrence_plan.get("effective_depth")
                        or prelude + core * steps + coda
                    )
                    adapter = int(
                        parameters.get("recurrence")
                        or parameters.get("recurrence_adapter")
                        or 0
                    )
                    ui.label(
                        f"recurrent: prelude {prelude} · core {core} "
                        f"({int(core_mixers.get('kda') or 0)} KDA / "
                        f"{int(core_mixers.get('mla') or 0)} MLA) · coda {coda} · "
                        f"effective depth {prelude}+{core}×{steps}+{coda} = {depth} "
                        f"at {steps} steps · adapter {_format_parameters(adapter)}"
                    ).classes("text-sm text-teal-200")
                ui.label(
                    f"BF16-master reference persistent state: "
                    f"{float(memory['reference_training_gib']):.3f} GiB; activations are not included."
                ).classes("text-xs text-gray-400")
                ple_rows: list[tuple[str, int]] = [
                    ("PLE total", int(parameters["ple_lookup"]) + int(parameters["ple_controls"])),
                ]
                if ple_plan is not None:
                    ple_rows.extend(
                        (
                            ("PLE exact input", int(ple_plan.get("exact_input_parameters") or 0)),
                            ("PLE QR refresh", int(ple_plan.get("qr_refresh_parameters") or 0)),
                        )
                    )
                with ui.grid(columns=3).classes("w-full gap-x-4 gap-y-1"):
                    for label, value in (
                        *ple_rows,
                        (
                            "Attention",
                            int(parameters["kda_mixers"]) + int(parameters["mla_mixers"]),
                        ),
                        ("Experts", int(parameters["expert_banks"])),
                        ("Shared", int(parameters["shared_channel"])),
                        ("Directional", int(parameters["directional"])),
                        ("ATXY", int(parameters["atxy"])),
                        ("Router", int(parameters["router"])),
                        ("Vocab head", int(parameters["lm_head"])),
                    ):
                        ui.label(f"{label}: {_format_parameters(value)}").classes(
                            "text-xs text-gray-500"
                        )
                ui.label(f"Plan {str(plan['fingerprint'])[:16]}…").classes(
                    "text-[11px] text-gray-500"
                )

        def update_data_summary() -> None:
            selected = _ordered_values(corpora_select.value)
            if not selected:
                data_summary.text = "Choose one or more corpora."
                data_messages.text = ""
                data_messages.visible = False
                update_budget_summary()
                return
            domains = _ordered_values(domain_filter.value)
            splits = normalize_source_splits(split_filter.value)
            mix = str(corpus_mix.value or CORPUS_MIX_TOKEN_BALANCED)
            cap = (
                float(source_ratio_cap.value or 0) or None
                if mix == CORPUS_MIX_TOKEN_BALANCED and len(selected) > 1
                else None
            )
            report = corpus_training_preflight(
                selected,
                domains or None,
                corpus_mix=mix,
                splits=splits,
                cross_corpus_dedup=bool(cross_corpus_dedup.value),
                source_max_ratio=cap,
            )
            provenance = report["provenance"]
            split_label = ", ".join(splits) if splits else "all"
            data_summary.text = (
                f"{len(selected)} corpora · {int(provenance['estimated_docs']):,} docs · "
                f"{int(provenance['estimated_chars']) / 1e9:.3f}B chars · {mix.replace('_', ' ')} "
                f"· splits {split_label}"
            )
            messages = [
                *(f"ERROR: {message}" for message in report["errors"]),
                *(f"WARNING: {message}" for message in report["warnings"]),
            ]
            data_messages.text = "\n".join(messages)
            data_messages.visible = bool(messages)
            update_budget_summary()

        def update_mix_controls() -> None:
            source_ratio_cap.enabled = (
                corpus_mix.value == CORPUS_MIX_TOKEN_BALANCED
                and len(_ordered_values(corpora_select.value)) > 1
            )
            update_data_summary()

        def update_budget_summary() -> None:
            try:
                budget = _resolve_budget(
                    token_budget=int(token_budget.value or 0),
                    max_steps=int(max_steps.value or 0),
                    sequence_length=int(sequence_length.value or 0),
                    micro_batch=int(micro_batch.value or 0),
                    gradient_accumulation=int(grad_accum.value or 0),
                )
            except (TypeError, ValueError) as exc:
                budget_summary.text = str(exc)
                return
            budget_summary.text = (
                f"{int(budget['max_steps']):,} optimizer steps × "
                f"{int(budget['tokens_per_optimizer_step']):,} tokens = "
                f"{int(budget['scheduled_training_tokens']):,} scheduled tokens; "
                f"the corpora are re-shuffled and replayed for {int(epochs.value or 1)} "
                "epoch(s) until the step cap is reached."
            )

        def on_start() -> None:
            selected_corpora = _ordered_values(corpora_select.value)
            parent = selected_parent() if continuing() else None
            if continuing() and parent is None:
                ui.notify("Choose a parent SPALMER checkpoint.", type="negative")
                return
            tokenizer_ref, tokenizer_source, tokenizer_registry_id = ("", "checkpoint", None)
            if not continuing():
                tokenizer_ref, tokenizer_source, tokenizer_registry_id = selected_tokenizer()
                if tokenizer_source == "registered":
                    tokenizer_row = tokenizer_rows.get(str(tokenizer_registry_id or ""))
                    if tokenizer_row is None:
                        ui.notify("Choose a ready registered tokenizer.", type="negative")
                        return
                    catalogued_vocab = int(tokenizer_row.get("vocab_size") or 0)
                    if catalogued_vocab and catalogued_vocab != int(preview_vocab.value or 0):
                        ui.notify(
                            f"Registered tokenizer vocabulary is {catalogued_vocab:,}; "
                            "reset the planning vocabulary before starting.",
                            type="negative",
                            multi_line=True,
                        )
                        return
            selected_domains = _ordered_values(domain_filter.value)
            selected_splits = normalize_source_splits(split_filter.value)
            mix = str(corpus_mix.value or CORPUS_MIX_TOKEN_BALANCED)
            cap = (
                float(source_ratio_cap.value or 0) or None
                if mix == CORPUS_MIX_TOKEN_BALANCED and len(selected_corpora) > 1
                else None
            )
            data_check = corpus_training_preflight(
                selected_corpora,
                selected_domains or None,
                corpus_mix=mix,
                splits=selected_splits,
                cross_corpus_dedup=bool(cross_corpus_dedup.value),
                source_max_ratio=cap,
            )
            if data_check["errors"]:
                ui.notify("; ".join(data_check["errors"]), type="negative", multi_line=True)
                return
            inherited_recurrence = parent_recurrence() if continuing() else None
            try:
                if continuing():
                    plan = _parent_plan(json.loads(parent.get("meta_json") or "{}"))
                else:
                    plan = plan_spalmer_configuration(
                        target_parameters=requested_target(),
                        vocab_size=int(preview_vocab.value or 0),
                        expert_weight_format=str(expert_weight_format.value or "mxfp4"),
                        directional=current_directional(),
                        atxy=current_atxy(),
                        recurrence=current_recurrence(),
                        ple_backend=current_ple_backend(),
                    )
                checkpoint_id = db.new_id("ckpt")
                config = build_spalmer_run_config(
                    checkpoint_name=str(checkpoint_name.value or ""),
                    tokenizer_ref=tokenizer_ref,
                    tokenizer_source=tokenizer_source,
                    tokenizer_registry_id=tokenizer_registry_id,
                    tokenizer_preview_vocab_size=int(preview_vocab.value or 0),
                    target_scale="inherited" if continuing() else str(target_scale.value or "10M"),
                    target_parameters=0 if continuing() else requested_target(),
                    resolved_plan=plan,
                    corpora=selected_corpora,
                    domains=selected_domains,
                    splits=selected_splits,
                    corpus_mix=mix,
                    cross_corpus_dedup=bool(cross_corpus_dedup.value),
                    source_max_ratio=cap,
                    sequence_length=int(sequence_length.value or 0),
                    token_budget=int(token_budget.value or 0),
                    max_steps=int(max_steps.value or 0),
                    micro_batch=int(micro_batch.value or 0),
                    gradient_accumulation=int(grad_accum.value or 0),
                    learning_rate=float(learning_rate.value or 0),
                    minimum_lr_ratio=float(minimum_lr_ratio.value or 0),
                    warmup_steps=int(warmup_steps.value or 0),
                    weight_decay=float(weight_decay.value or 0),
                    beta1=float(beta1.value or 0),
                    beta2=float(beta2.value or 0),
                    epsilon=float(epsilon.value or 0),
                    seed=int(seed.value or 0),
                    gradient_clip=float(gradient_clip.value or 0) or None,
                    auxiliary_loss_weight=float(auxiliary_loss_weight.value or 0),
                    surprise_calibration_weight=float(
                        surprise_calibration_weight.value or 0
                    ),
                    log_every=int(log_every.value or 0),
                    save_every=int(save_every.value or 0),
                    activation_checkpointing=bool(activation_checkpointing.value),
                    expert_weight_format=str(expert_weight_format.value or "mxfp4"),
                    attention_backend=str(attention_backend.value or "auto"),
                    expert_qat_backend=str(expert_qat_backend.value or "auto"),
                    directional=current_directional(),
                    atxy=current_atxy(),
                    out_checkpoint_id=checkpoint_id,
                    data_preflight_warnings=data_check["warnings"],
                    initialization_mode="checkpoint" if continuing() else "scratch",
                    parent_checkpoint_id=str(parent["id"]) if parent else None,
                    epochs=int(epochs.value or 1),
                    recurrence=(
                        {"enabled": False} if continuing() else current_recurrence()
                    ),
                    mean_recurrence=(
                        float(mean_recurrence.value or 0) if recurrent_run() else None
                    ),
                    backprop_depth=(
                        int(backprop_depth.value or 0) if recurrent_run() else None
                    ),
                    recurrence_sigma=(
                        float(recurrence_sigma.value or 0) if recurrent_run() else None
                    ),
                    parent_recurrent=bool(inherited_recurrence),
                    ple_backend=None if continuing() else current_ple_backend(),
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                ui.notify(str(exc), type="negative", multi_line=True)
                return

            if continuing():
                planned_config = {
                    "d_model": int(parent.get("hidden_size") or 0),
                    "n_layers": int(parent.get("num_layers") or 0),
                }
                planned_parameters = int(round(float(parent.get("params_b") or 0) * 1e9))
            else:
                planned_config = dict(plan["config"])
                planned_parameters = int(dict(plan["parameters"])["total"])
            pending_metadata = {
                "architecture_type": "spalmer_v1",
                "initialization": config["initialization"],
                "target_parameters": int(plan.get("target_parameters") or planned_parameters),
                "planned_parameters": planned_parameters,
                "plan_fingerprint": str(plan.get("fingerprint") or ""),
                "tokenizer": config["tokenizer"],
                "spalmer": config["spalmer"],
                "recurrence": (
                    inherited_recurrence
                    if continuing()
                    else config["spalmer"].get("recurrence")
                ),
            }
            run_id = db.new_id("run")
            now = db.now_iso()
            with db.transaction() as connection:
                connection.execute(
                    "INSERT INTO checkpoints(id,name,created_at,kind,source,ref,arch,params_b,"
                    "hidden_size,num_layers,meta_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        checkpoint_id,
                        config["checkpoint_name"],
                        now,
                        "pending",
                        "pending",
                        "",
                        "SPALMERCausalLM",
                        planned_parameters / 1_000_000_000,
                        int(planned_config["d_model"]),
                        int(planned_config["n_layers"]),
                        json.dumps(pending_metadata),
                    ),
                )
                connection.execute(
                    "INSERT INTO runs(id,kind,status,created_at,config_json) VALUES(?,?,?,?,?)",
                    (run_id, SPALMER_RUN_KIND, "queued", now, json.dumps(config)),
                )
            try:
                pid = launch_worker(run_id)
            except Exception as exc:  # noqa: BLE001
                db.execute(
                    "UPDATE runs SET status='failed',finished_at=?,error=? WHERE id=?",
                    (db.now_iso(), str(exc), run_id),
                )
                db.execute(
                    "DELETE FROM checkpoints WHERE id=? AND kind='pending'",
                    (checkpoint_id,),
                )
                ui.notify(f"Worker launch failed: {exc}", type="negative", multi_line=True)
                return
            db.execute(
                "UPDATE runs SET metrics_json=? WHERE id=?",
                (json.dumps({"pid": pid}), run_id),
            )
            ui.notify(
                f"SPALMER pretraining queued — {config['checkpoint_name']} "
                f"({config['max_steps']:,} steps, pid {pid}). See Runs.",
                type="positive",
                multi_line=True,
            )

        target_scale.on_value_change(lambda _: update_target_controls())
        registered_tokenizer.on_value_change(lambda _: update_registered_tokenizer())
        direct_tokenizer_ref.on_value_change(lambda _: update_plan_preview())
        hub_preset_select.on_value_change(lambda _: apply_hub_preset())
        preview_vocab.on_value_change(lambda _: update_plan_preview())
        expert_weight_format.on_value_change(
            lambda _: (update_backend_status(), update_plan_preview())
        )
        attention_backend.on_value_change(lambda _: update_backend_status())
        expert_qat_backend.on_value_change(lambda _: update_backend_status())
        ple_backend_select.on_value_change(lambda _: update_ple_backend_controls())
        directional_enabled.on_value_change(lambda _: update_directional_controls())
        atxy_enabled.on_value_change(lambda _: update_atxy_controls())
        recurrence_enabled.on_value_change(lambda _: update_recurrence_controls())
        for element in (
            custom_target,
            directional_groups,
            directional_rank,
            atxy_value_dim,
            atxy_a,
            atxy_t,
            atxy_x,
            atxy_y,
            atxy_layer,
            recurrence_prelude,
            recurrence_coda,
            recurrence_default_steps,
        ):
            element.on_value_change(lambda _: update_plan_preview())
        corpora_select.on_value_change(lambda _: update_mix_controls())
        domain_filter.on_value_change(lambda _: update_data_summary())
        split_filter.on_value_change(lambda _: update_data_summary())
        corpus_mix.on_value_change(lambda _: update_mix_controls())
        source_ratio_cap.on_value_change(lambda _: update_data_summary())
        cross_corpus_dedup.on_value_change(lambda _: update_data_summary())
        for element in (sequence_length, token_budget, max_steps, micro_batch, grad_accum):
            element.on_value_change(lambda _: update_budget_summary())
        start_button.on_click(on_start)
        initialization_mode.on_value_change(lambda _: update_initialization_controls())
        parent_checkpoint.on_value_change(lambda _: update_initialization_controls())
        epochs.on_value_change(lambda _: update_budget_summary())

        update_target_controls()
        update_ple_backend_controls()
        update_directional_controls()
        update_atxy_controls()
        update_recurrence_controls()
        update_backend_status()
        update_mix_controls()
        update_budget_summary()
        update_initialization_controls()
