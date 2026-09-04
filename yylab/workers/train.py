from __future__ import annotations

import argparse
import gc
import json
import math
import traceback
from pathlib import Path

from .. import db
from ..artifacts import adapter_dir, checkpoint_dir, read_json, router_dir, run_dir, write_json
from ..models.precision_capabilities import model_compute_dtype, require_dense_precision
from ..mole.adaptive import AdaptiveOccupancyConfig, AdaptiveOccupancyController
from ..mole.inject import (
    MoLEConfig,
    attach_shared_router,
    inject_mole,
    reset_expert,
    resolve_adaptive_budget,
    set_active_expert,
    set_adaptive_expert_pool,
    set_stage,
    set_trainable_expert,
)
from ..mole.losses import (
    adaptive_candidate_nll,
    aux_loss,
    router_usage_metrics,
    surprise_calibration_loss,
)
from ..mole.store import (
    apply_mole_weights,
    assemble_mole_experts,
    load_mole_config,
    load_mole_weights,
    save_isolated_expert,
    save_mole,
)
from ..tokenizer_registry import get_tokenizer, resolve_tokenizer_artifact
from ..train.base_config import (
    BASE_RECIPE_VERSION,
    ScratchModelSpec,
    build_hf_config,
    estimate_parameter_count,
    normalize_training_token_budget,
)
from ..train.compact_tokenizer import CompactTokenizerSpec, train_compact_tokenizer
from ..train.data import (
    CORPUS_MIX_CONCATENATE,
    CORPUS_MIX_TOKEN_BALANCED,
    DEFAULT_TEXT_CHUNK_CHARS,
    OBJECTIVE_CHAT_SFT,
    ChatSFTBatches,
    ExpertDataPlan,
    PackedBatches,
    allocate_expert_steps,
    corpus_training_preflight,
    estimate_steps,
    iter_corpus_docs,
    normalize_corpus_ids,
    normalize_corpus_mix,
    normalize_source_splits,
    normalize_training_objective,
    plan_expert_data,
)
from ..train.loop import LoopOpts, make_optimizer, optimizer_runtime_metadata, train_loop
from ..train.low_precision import LOW_PRECISION_METADATA_KEY, apply_qat, qat_metadata


def log(msg: str) -> None:
    print(msg, flush=True)


def benchmark_progress(rd: Path, stage: str, step: int, total: int, value: float, msg: str) -> None:
    """Publish benchmark progress using the same live contract as training."""

    metric_label = (
        "score"
        if stage == "quality benchmark"
        else ("loss" if stage == "training performance" else "progress")
    )
    write_json(
        rd / "progress.json",
        {
            "stage": stage,
            "step": step,
            "max_steps": total,
            "value": round(float(value), 4),
            "metric_label": metric_label,
            # Retain loss for older Runs-page clients that only know this field.
            "loss": round(float(value), 4),
            "msg": msg,
        },
    )


def run_benchmark_quality(cfg: dict, rd: Path) -> dict:
    from ..benchmark.quality import run_quality_benchmark

    resolved = dict(cfg)
    baseline_run_id = str(resolved.get("baseline_run_id") or "")
    if baseline_run_id:
        baseline_dir = run_dir(baseline_run_id)
        semantic_projection = baseline_dir / "semantic-summary.json"
        baseline_path = (
            semantic_projection if semantic_projection.is_file() else baseline_dir / "summary.json"
        )
        if not baseline_path.is_file():
            raise FileNotFoundError(f"baseline benchmark has no summary: {baseline_run_id}")
        resolved["baseline_summary_path"] = str(baseline_path)
    return run_quality_benchmark(
        resolved,
        rd,
        progress=lambda stage, step, total, value, msg: benchmark_progress(
            rd, stage, step, total, value, msg
        ),
    )


def run_benchmark_training_performance(cfg: dict, rd: Path) -> dict:
    from ..benchmark.performance import run_training_performance

    return run_training_performance(
        cfg,
        rd,
        progress=lambda stage, step, total, value, msg: benchmark_progress(
            rd, stage, step, total, value, msg
        ),
    )


def mark_training_artifacts_failed(run_kind: str, cfg: dict) -> None:
    """Fail only artifacts owned by a training run, never benchmark subjects."""

    if run_kind in {"mole_stage1", "router_stage2", "base"} and cfg.get("adapter_id"):
        db.execute("UPDATE adapters SET status='failed' WHERE id=?", (cfg["adapter_id"],))
    if run_kind == "router_stage2" and cfg.get("router_id"):
        db.execute("UPDATE routers SET status='failed' WHERE id=?", (cfg["router_id"],))
    if run_kind in {"base", "spalmer_base"} and cfg.get("out_checkpoint_id"):
        # The failed run remains the durable error record. Do not leave a
        # phantom checkpoint selectable as a training or inference input.
        db.execute(
            "DELETE FROM checkpoints WHERE id=? AND kind='pending'",
            (cfg["out_checkpoint_id"],),
        )


def load_base(ref: str, quant: str, **load_options):
    from ..models.loading import load_base as _load_base

    model, tokenizer, cuda, quant, notes = _load_base(ref, quant, **load_options)
    model.config.use_cache = False
    return model, tokenizer, cuda, quant, notes


def normalize_base_data_config(cfg: dict) -> dict:
    """Canonicalize old and new base data selections without mutating the run."""

    resolved = dict(cfg)
    raw_corpora = resolved.get("corpora")
    if raw_corpora is None:
        raw_corpora = resolved.get("corpus")
    corpora = normalize_corpus_ids(raw_corpora)
    if not corpora:
        raise ValueError("base training requires at least one corpus")
    for corpus_id in corpora:
        corpus = db.q1(
            "SELECT name,status,kind,verdict FROM corpora WHERE id=?",
            (corpus_id,),
        )
        if not corpus:
            raise ValueError(f"base training corpus does not exist: {corpus_id!r}")
        if str(corpus.get("kind") or "").casefold() == "rejected" or str(
            corpus.get("verdict") or ""
        ).casefold() == "rejected":
            raise ValueError(
                f"base training refuses rejected corpus {corpus.get('name') or corpus_id!r}"
            )
        if str(corpus.get("status") or "") != "ready":
            raise ValueError(
                f"base training requires ready corpora; {corpus.get('name') or corpus_id!r} "
                f"is {corpus.get('status') or 'unknown'}"
            )

    raw_domains = resolved.get("domains")
    if isinstance(raw_domains, str):
        raw_domains = [raw_domains]
    domains: list[str] = []
    seen_domains: set[str] = set()
    for value in raw_domains or ():
        domain = str(value).strip()
        if domain and domain not in seen_domains:
            domains.append(domain)
            seen_domains.add(domain)

    resolved["corpora"] = corpora
    resolved["domains"] = domains or None
    raw_splits = resolved.get("splits")
    if raw_splits is None:
        raw_splits = resolved.get("source_splits")
    resolved["splits"] = normalize_source_splits(raw_splits) or None
    resolved["corpus_mix"] = normalize_corpus_mix(resolved.get("corpus_mix"))
    resolved["cross_corpus_dedup"] = bool(resolved.get("cross_corpus_dedup", True))
    raw_weights = resolved.get("corpus_weights") or {}
    if not isinstance(raw_weights, dict):
        raise ValueError("corpus_weights must be an object keyed by corpus ID")
    normalized_weights: dict[str, float] = {}
    for corpus_id, weight in raw_weights.items():
        corpus_id = str(corpus_id)
        parsed_weight = float(weight)
        if corpus_id not in corpora:
            raise ValueError(f"corpus weight references an unselected corpus: {corpus_id!r}")
        if not math.isfinite(parsed_weight) or parsed_weight <= 0:
            raise ValueError("corpus weights must be finite and positive")
        normalized_weights[corpus_id] = parsed_weight
    resolved["corpus_weights"] = normalized_weights
    raw_source_cap = resolved.get("source_max_ratio")
    if raw_source_cap is None or raw_source_cap == "":
        resolved["source_max_ratio"] = None
    else:
        parsed_source_cap = float(raw_source_cap)
        if not math.isfinite(parsed_source_cap) or not 0 <= parsed_source_cap <= 1:
            raise ValueError("source_max_ratio must be 0 or a finite value in (0, 1]")
        resolved["source_max_ratio"] = parsed_source_cap or None
    if resolved["corpus_mix"] != CORPUS_MIX_TOKEN_BALANCED and (
        resolved["corpus_weights"] or resolved["source_max_ratio"] is not None
    ):
        raise ValueError("corpus weights and source_max_ratio require token-balanced mixing")
    if len(corpora) < 2:
        resolved["source_max_ratio"] = None
    elif (
        resolved["source_max_ratio"] is not None
        and resolved["source_max_ratio"] < 1 / len(corpora)
    ):
        raise ValueError(
            "source_max_ratio is smaller than the minimum feasible equal share "
            f"({1 / len(corpora):.3f})"
        )
    resolved["objective"] = normalize_training_objective(resolved.get("objective"))
    return resolved


def normalize_base_worker_config(cfg: dict) -> dict:
    """Normalize base data and re-derive any explicit token-budget step cap."""

    return normalize_training_token_budget(normalize_base_data_config(cfg))


def build_batches(
    cfg: dict,
    tokenizer,
    max_steps: int,
    *,
    domains: list[str] | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> PackedBatches | ChatSFTBatches:
    selected_domains = domains if domains is not None else cfg.get("domains") or None

    def docs():
        for epoch in range(max(1, int(cfg.get("epochs", 1)))):
            yield from iter_corpus_docs(
                cfg["corpora"],
                selected_domains,
                shuffle_buffer=int(cfg.get("shuffle_buffer", 128)),
                shuffle_buffer_bytes=int(cfg.get("shuffle_buffer_bytes", 64 * 2**20)),
                seed=int(cfg.get("seed", 42)) + epoch,
                shard_index=shard_index,
                num_shards=num_shards,
                corpus_mix=cfg.get("corpus_mix", CORPUS_MIX_CONCATENATE),
                splits=cfg.get("splits"),
                cross_corpus_dedup=bool(cfg.get("cross_corpus_dedup", True)),
                corpus_weights=cfg.get("corpus_weights") or None,
                source_max_ratio=cfg.get("source_max_ratio"),
                annotate_source=True,
            )

    if normalize_training_objective(cfg.get("objective")) == OBJECTIVE_CHAT_SFT:
        return ChatSFTBatches(
            docs(),
            tokenizer,
            seq_len=cfg["seq_len"],
            micro_batch=cfg["micro_batch"],
            max_batches=max_steps * cfg["grad_accum"],
        )

    def tokenize(texts: list[str]) -> list[list[int]]:
        # PackedBatches slices these raw encodings to cfg["seq_len"] before
        # model execution.  Transformers' model-length warning is therefore
        # misleading here: no un-packed sequence is ever forwarded.
        enc = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            verbose=False,
        )
        return enc["input_ids"]

    return PackedBatches(
        docs(),
        tokenize,
        seq_len=cfg["seq_len"],
        eos_token_id=tokenizer.eos_token_id,
        micro_batch=cfg["micro_batch"],
        max_batches=max_steps * cfg["grad_accum"],
        tokenize_chunk=int(cfg.get("tokenize_chunk", 8)),
        max_text_chars=int(cfg.get("max_text_chars", DEFAULT_TEXT_CHUNK_CHARS)),
    )


def tokenized_view_manifest(
    batches: PackedBatches | ChatSFTBatches,
    result: dict,
    *,
    objective: str,
    sequence_length: int,
) -> dict:
    """Reconcile source rendering, packing, and model-facing token exposure."""

    stats_fn = getattr(batches, "data_stats", None)
    batcher = stats_fn() if callable(stats_fn) else {}
    tokens = dict(batcher.get("tokens") or {})
    tokens.update(
        {
            "forwarded_model_slots": int(result.get("forwarded_model_tokens", 0)),
            "forwarded_real_inputs": int(result.get("forwarded_real_tokens", 0)),
            "padding_inputs": int(result.get("padding_tokens", 0)),
            "model_target_slots": int(result.get("model_target_slots", 0)),
            "optimizer_supervised_targets": int(result.get("supervised_tokens", 0)),
            "ignored_target_slots": int(result.get("ignored_target_slots", 0)),
        }
    )
    rendered = int(tokens.get("rendered_token_positions") or 0)
    covered = int(tokens.get("covered_rendered_positions") or 0)
    replay = int(tokens.get("packing_replay_inputs") or 0)
    forwarded_real = int(tokens["forwarded_real_inputs"])
    forwarded_model = int(tokens["forwarded_model_slots"])
    padding = int(tokens["padding_inputs"])
    target_slots = int(tokens["model_target_slots"])
    supervised = int(tokens["optimizer_supervised_targets"])
    ignored = int(tokens["ignored_target_slots"])
    unforwarded = max(0, rendered - covered)
    tokens["unforwarded_rendered_positions"] = unforwarded
    reconciliation = {
        "model_slots_equal_real_plus_padding": forwarded_model
        == forwarded_real + padding,
        "real_inputs_equal_covered_plus_replay": forwarded_real == covered + replay,
        "rendered_equal_covered_plus_unforwarded": rendered == covered + unforwarded,
        "targets_equal_supervised_plus_ignored": target_slots == supervised + ignored,
    }
    return {
        "schema_version": 1,
        "kind": "yylab_tokenized_training_view",
        "objective": objective,
        "sequence_length": int(sequence_length),
        "documents": {
            key: value
            for key, value in batcher.items()
            if key.endswith("docs") or key in {"docs_seen", "failure_reasons"}
        },
        "examples": {
            "rendered": int(batcher.get("examples_rendered") or 0),
            "sequences_forwarded": int(result.get("sequences_forwarded", 0)),
            "unsupervised_sequences_skipped": int(
                batcher.get("unsupervised_sequences_skipped") or 0
            ),
        },
        "tokens": tokens,
        "per_corpus": dict(batcher.get("per_corpus") or {}),
        "reconciliation": reconciliation,
        # Compact compatibility fields for older run/checkpoint consumers.
        "docs_seen": int(batcher.get("docs_seen") or getattr(batches, "docs_seen", 0)),
        "rendered_tokens": int(
            batcher.get("rendered_tokens") or getattr(batches, "tokens_seen", 0)
        ),
        "optimizer_supervised_tokens": supervised,
    }


def loop_opts(
    cfg: dict,
    rd: Path,
    cuda: bool,
    *,
    max_steps: int | None = None,
    stage: str = "train",
    step_offset: int = 0,
    total_steps: int | None = None,
) -> LoopOpts:
    return LoopOpts(
        micro_batch=cfg["micro_batch"],
        grad_accum=cfg["grad_accum"],
        max_steps=max_steps if max_steps is not None else cfg["max_steps"],
        log_every=cfg.get("log_every", 5),
        save_every=cfg.get("save_every"),
        device="cuda" if cuda else "cpu",
        run_dir=rd,
        stage=stage,
        step_offset=step_offset,
        total_steps=total_steps,
        prefer_cudnn_sdpa=cfg.get("cudnn_sdpa", True),
        grad_clip=float(cfg.get("grad_clip", 1.0)),
    )


def enforce_quant_lock(checkpoint_id: str, quant: str) -> None:
    """Validate an established lock without poisoning it before training succeeds."""

    row = db.q1("SELECT quant_locked FROM checkpoints WHERE id=?", (checkpoint_id,))
    if not row:
        return
    locked = row["quant_locked"]
    if locked and locked != quant:
        raise RuntimeError(
            f"Checkpoint quantization is locked to '{locked}'; adapter requested '{quant}'. "
            "Quant cannot drift between adapters on the same checkpoint."
        )


def commit_quant_lock(checkpoint_id: str, quant: str) -> None:
    """Pin quantization only after a complete adapter artifact exists."""

    enforce_quant_lock(checkpoint_id, quant)
    changed = db.execute(
        "UPDATE checkpoints SET quant_locked=? WHERE id=? AND (quant_locked='' OR quant_locked IS NULL OR quant_locked=?)",
        (quant, checkpoint_id, quant),
    )
    if changed != 1:
        raise RuntimeError("could not commit checkpoint quantization lock")


def enable_grad_ckpt(model) -> None:
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    enable_inputs = getattr(model, "enable_input_require_grads", None)
    if callable(enable_inputs):
        enable_inputs()


def prepare_kbit_training(model, quant: str) -> None:
    if quant not in {"nf4", "int8"}:
        return
    from peft import prepare_model_for_kbit_training

    prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)


def enable_fused_linear_ce(model, enabled: bool = True, *, full_kernels: bool = True) -> bool:
    """Install the optimized Qwen3 kernel set, including fused linear CE."""

    if not enabled or getattr(model.config, "model_type", None) != "qwen3":
        return False
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        from liger_kernel.transformers.model.qwen3 import lce_forward

        target = model
        get_base_model = getattr(model, "get_base_model", None)
        if callable(get_base_model):
            candidate = get_base_model()
            if getattr(candidate.config, "model_type", None) == "qwen3":
                target = candidate

        apply_liger_kernel_to_qwen3(
            rope=full_kernels,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=full_kernels,
            swiglu=full_kernels,
            model=target,
        )
        if getattr(target.forward, "__func__", None) is not lce_forward:
            raise RuntimeError("Liger did not install the fused Qwen3 forward")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Qwen3 memory-efficient loss setup failed; install a compatible liger-kernel runtime"
        ) from exc
    return True


def log_memory(label: str) -> None:
    parts: list[str] = []
    try:
        import psutil

        parts.append(f"RAM RSS={psutil.Process().memory_info().rss / 2**30:.2f} GB")
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        if torch.cuda.is_available():
            parts.append(f"CUDA allocated={torch.cuda.memory_allocated() / 2**30:.2f} GB")
            parts.append(f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GB")
    except Exception:  # noqa: BLE001
        pass
    if parts:
        log(f"memory [{label}]: " + ", ".join(parts))


def _warmup_steps(total_steps: int) -> int:
    """Keep a one-step expert from spending its only update at zero LR."""

    return 0 if total_steps <= 1 else max(1, total_steps // 20)


JOINT_PER_TOKEN_STRATEGY = "joint_per_token_v1"
ADAPTIVE_SURPRISE_STRATEGY = "joint_adaptive_surprise_v1"
SOFT_LORA_IDENTITY_STRATEGY = "joint_soft_lora_identity_v1"
SEQUENTIAL_EXPERT_STRATEGY = "sequential_single_resident_v1"


def _mole_training_strategy(cfg: dict) -> str:
    """Resolve MoLE stage-1 behavior without changing queued legacy runs.

    Older run configs predate the strategy field and were created with the
    isolated, sequential contract.  New run configs opt into joint learned
    routing explicitly, which keeps replaying an old queue deterministic.
    """

    strategy = str(cfg.get("training_strategy") or SEQUENTIAL_EXPERT_STRATEGY)
    if strategy not in {
        ADAPTIVE_SURPRISE_STRATEGY,
        JOINT_PER_TOKEN_STRATEGY,
        SOFT_LORA_IDENTITY_STRATEGY,
        SEQUENTIAL_EXPERT_STRATEGY,
    }:
        raise ValueError(f"unknown MoLE training strategy: {strategy}")
    return strategy


def _run_mole_joint_per_token(
    cfg: dict,
    rd: Path,
    *,
    adaptive: bool = False,
) -> dict:
    """Jointly optimize every LoRA expert and learned token router.

    The corpus is streamed once with the optional *global* domain filter from
    the run config.  It is deliberately never partitioned into per-expert
    domains or document shards: every token reaches the learned sparse gates.
    """

    import torch
    from transformers.optimization import get_cosine_schedule_with_warmup

    enforce_quant_lock(cfg["checkpoint_id"], cfg["quant"])
    model, tokenizer, cuda, quant, notes = load_base(cfg["checkpoint_ref"], cfg["quant"])
    for note in notes:
        log(f"note: {note}")
    prepare_kbit_training(model, quant)
    log_memory("joint MoLE base loaded")
    base_seed = int(cfg.get("seed", 42))
    torch.manual_seed(base_seed)
    if cuda:
        torch.cuda.manual_seed_all(base_seed)

    adaptive_config = AdaptiveOccupancyConfig()
    identity_expert = bool(cfg.get("identity_expert", False))
    if adaptive and identity_expert:
        raise ValueError("adaptive surprise and Soft LoRA identity routing cannot be combined")
    if adaptive:
        adaptive_payload = dict(cfg.get("adaptive") or {})
        adaptive_payload["enabled"] = True
        adaptive_config = AdaptiveOccupancyConfig.from_dict(adaptive_payload)
    requested_experts = max(
        int(cfg.get("num_experts") or adaptive_config.minimum_experts),
        adaptive_config.minimum_experts if adaptive else 1,
    )
    requested_rank = int(cfg.get("rank") or 8)
    requested_top_k = min(
        requested_experts + int(identity_expert),
        int(cfg.get("top_k") or adaptive_config.minimum_experts),
    )
    mole_cfg = MoLEConfig(
        num_experts=requested_experts,
        rank=requested_rank,
        alpha=cfg.get("alpha") or requested_rank * 2.0,
        top_k=requested_top_k,
        target_modules=tuple(cfg["target_modules"]),
        router_type="per_token",
        identity_expert=identity_expert,
        adaptive=adaptive_config,
    )
    budget_plan = None
    if adaptive:
        budget_plan = resolve_adaptive_budget(model, mole_cfg)
        if cfg.get("alpha") is None:
            mole_cfg.alpha = mole_cfg.rank * 2.0
        log(
            "adaptive architecture resolved: "
            f"E={mole_cfg.num_experts} r={mole_cfg.rank}; "
            f"lane={budget_plan.parameters_per_expert:,} params "
            f"({budget_plan.expert_footprint_fraction:.3%}); "
            f"active={mole_cfg.adaptive.minimum_experts}-"
            f"{mole_cfg.adaptive.maximum_experts(mole_cfg.num_experts)} lanes"
        )
    replaced = inject_mole(
        model,
        mole_cfg,
        expert_dtype=model_compute_dtype(quant),
    )
    if not replaced:
        raise RuntimeError("no target linear modules matched the requested MoLE architecture")
    set_active_expert(model, None)
    controller: AdaptiveOccupancyController | None = None
    training_state: dict[str, object] = {}
    if adaptive:
        controller = AdaptiveOccupancyController(mole_cfg.adaptive, mole_cfg.num_experts)
        preferred = list(mole_cfg.adaptive.preferred_experts)
        preferred.extend(index for index in range(mole_cfg.num_experts) if index not in preferred)
        if not mole_cfg.adaptive.preferred_experts:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(base_seed)
            preferred = torch.randperm(mole_cfg.num_experts, generator=generator).tolist()
        initial_pool = tuple(preferred[: controller.minimum])
        set_adaptive_expert_pool(model, initial_pool)
        training_state.update(order=preferred, active_pool=initial_pool)
    set_stage(model, "both")
    if cfg.get("grad_ckpt", True):
        enable_grad_ckpt(model)
    full_liger = cfg.get("liger_full_kernels", True)
    fused_ce = enable_fused_linear_ce(
        model,
        cfg.get("liger_fused_ce", True),
        full_kernels=full_liger,
    )

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    routing_label = (
        "adaptive surprise MoLE"
        if adaptive
        else ("Soft LoRA identity MoLE" if identity_expert else "joint per-token MoLE")
    )
    log(
        f"{routing_label}: E={mole_cfg.num_experts} r={mole_cfg.rank} "
        f"k={mole_cfg.top_k}; trainable={trainable:,}; global domains={cfg.get('domains') or 'all'}; "
        f"liger_full={fused_ce and full_liger}; cudnn_sdpa={cfg.get('cudnn_sdpa', True)}"
    )
    log_memory("joint experts and token routers injected")

    optimizer = make_optimizer(
        model,
        cfg["lr"],
        cuda=cuda,
        optimizer_state=cfg.get("optimizer_state", "auto"),
    )
    optimizer_resolution = optimizer_runtime_metadata(optimizer)
    optimizer_impl = optimizer_resolution["implementation"]
    log(
        f"optimizer requested=auto/{cfg.get('optimizer_state', 'auto')}; "
        f"resolved={optimizer_impl}/{optimizer_resolution['state']}; "
        f"moments={optimizer_resolution['state_precision']}; "
        f"master={optimizer_resolution['master_weight_precision']}"
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=_warmup_steps(cfg["max_steps"]),
        num_training_steps=cfg["max_steps"],
    )
    # build_batches applies cfg["domains"] once as a global corpus filter. It
    # never sees expert_plan here, so no expert receives a hand-authored slice.
    training_cfg = cfg
    if adaptive:
        training_cfg = {
            **cfg,
            "num_experts": mole_cfg.num_experts,
            "rank": mole_cfg.rank,
            "top_k": mole_cfg.top_k,
            "adaptive": mole_cfg.adaptive.to_dict(),
        }
    batches = build_batches(training_cfg, tokenizer, cfg["max_steps"])
    a_dir = adapter_dir(cfg["adapter_id"])
    strategy = (
        ADAPTIVE_SURPRISE_STRATEGY
        if adaptive
        else (SOFT_LORA_IDENTITY_STRATEGY if identity_expert else JOINT_PER_TOKEN_STRATEGY)
    )
    metadata = {
        "stage": "joint_experts_and_router",
        "training_strategy": strategy,
        "router_arch": (
            "global_surprise_lanes_v1"
            if adaptive
            else (
                "shared_per_block_soft_lora_identity_v1"
                if identity_expert
                else "shared_per_block_per_token_v1"
            )
        ),
        "global_domain_filter": sorted(cfg.get("domains") or []),
        "fused_linear_cross_entropy": fused_ce,
        "liger_full_kernels": fused_ce and full_liger,
        "cudnn_priority_sdpa": bool(cfg.get("cudnn_sdpa", True)),
        "optimizer_state": optimizer_resolution["state"],
        "optimizer_impl": optimizer_impl,
        "optimizer_resolution": optimizer_resolution,
    }
    if budget_plan is not None:
        metadata.update(
            adaptive_capacity=budget_plan.to_dict(),
            occupancy_semantics="incremental_expert_parameters_over_resolved_total",
            training_residency="full_bank_weights_resident_selected_lanes_execute",
        )

    def update_adaptive_pool(step: int, task_nll: torch.Tensor) -> None:
        assert controller is not None
        decision = controller.observe(float(task_nll.detach().item()), "nll")
        scores = adaptive_candidate_nll(model)
        if scores is None:
            return
        ordering = torch.argsort(scores, stable=True).cpu().tolist()
        active_count = decision.active_experts
        cadence = mole_cfg.adaptive.nested_k_probe_every
        if step % cadence == 0:
            span = controller.maximum - controller.minimum + 1
            active_count = controller.minimum + ((step // cadence - 1) % span)
        exploration = min(
            mole_cfg.adaptive.exploration_experts,
            active_count,
            mole_cfg.num_experts,
        )
        selected = ordering[: max(0, active_count - exploration)]
        cursor = (step * max(1, exploration)) % mole_cfg.num_experts
        if len(selected) < active_count:
            for offset in range(mole_cfg.num_experts):
                candidate = (cursor + offset) % mole_cfg.num_experts
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) >= active_count:
                    break
        pool = tuple(selected)
        set_adaptive_expert_pool(model, pool)
        training_state.update(order=ordering, active_pool=pool)
        mole_cfg.adaptive.baseline_nll = controller.baseline("nll")
        mole_cfg.adaptive.preferred_experts = tuple(ordering)

    def training_metrics() -> dict:
        metrics = router_usage_metrics(model)
        if controller is not None:
            metrics.update(
                adaptive_controller=controller.telemetry(),
                training_active_experts=len(training_state.get("active_pool", ())),
            )
        return metrics

    def save(_step: int) -> None:
        if controller is not None:
            metadata["adaptive_controller"] = controller.telemetry()
            metadata["preferred_experts"] = list(mole_cfg.adaptive.preferred_experts)
        save_mole(
            a_dir,
            model,
            mole_cfg,
            base_ref=cfg["checkpoint_ref"],
            extra_meta=metadata,
        )

    result = train_loop(
        model,
        batches,
        optimizer,
        scheduler,
        loop_opts(cfg, rd, cuda, stage=routing_label),
        aux_fn=lambda: aux_loss(model, mole_cfg.top_k, cfg.get("aux_weight", 0.01)),
        save_fn=save,
        metrics_fn=training_metrics,
        loss_aux_fn=(
            (
                lambda task_nll: surprise_calibration_loss(
                    model,
                    task_nll,
                    mole_cfg.adaptive.calibration_weight,
                )
            )
            if adaptive
            else None
        ),
        after_step_fn=update_adaptive_pool if adaptive else None,
    )
    if result["steps"] < 1:
        raise RuntimeError("joint per-token MoLE training produced no optimizer steps")
    save(0)
    commit_quant_lock(cfg["checkpoint_id"], quant)
    db.execute(
        "UPDATE adapters SET status='trained', path=?, domains_json=?, meta_json=?, "
        "num_experts=?, top_k=?, rank=? WHERE id=?",
        (
            str(a_dir),
            json.dumps(metadata["global_domain_filter"]),
            json.dumps(metadata),
            mole_cfg.num_experts,
            mole_cfg.top_k,
            mole_cfg.rank,
            cfg["adapter_id"],
        ),
    )
    return {
        "strategy": strategy,
        "result": result,
        "steps": result["steps"],
        "docs_seen": batches.docs_seen,
        "tokens": batches.tokens_seen,
    }


def run_mole(cfg: dict, rd: Path) -> dict:
    strategy = _mole_training_strategy(cfg)
    soft_identity = bool(cfg.get("identity_expert", False))
    if (strategy == SOFT_LORA_IDENTITY_STRATEGY) != soft_identity:
        raise ValueError("Soft LoRA strategy and identity_expert must be enabled together")
    if strategy in {
        ADAPTIVE_SURPRISE_STRATEGY,
        JOINT_PER_TOKEN_STRATEGY,
        SOFT_LORA_IDENTITY_STRATEGY,
    }:
        return _run_mole_joint_per_token(
            cfg,
            rd,
            adaptive=strategy == ADAPTIVE_SURPRISE_STRATEGY,
        )

    import torch
    from transformers.optimization import get_cosine_schedule_with_warmup

    enforce_quant_lock(cfg["checkpoint_id"], cfg["quant"])
    model, tokenizer, cuda, quant, notes = load_base(cfg["checkpoint_ref"], cfg["quant"])
    for n in notes:
        log(f"note: {n}")
    prepare_kbit_training(model, quant)
    log_memory("base loaded")

    final_cfg = MoLEConfig(
        num_experts=cfg["num_experts"],
        rank=cfg["rank"],
        alpha=cfg.get("alpha") or cfg["rank"] * 2.0,
        top_k=cfg["top_k"],
        target_modules=tuple(cfg["target_modules"]),
    )
    # Keep only one expert resident while stage 1 runs. The completed isolated
    # checkpoints are assembled into the E-expert bank after all slices finish.
    resident_cfg = MoLEConfig(
        num_experts=1,
        rank=final_cfg.rank,
        alpha=final_cfg.alpha,
        top_k=1,
        target_modules=final_cfg.target_modules,
    )
    replaced = inject_mole(model, resident_cfg, expert_dtype=model_compute_dtype(quant))
    if not replaced:
        raise RuntimeError("no target linear modules matched the requested MoLE architecture")
    set_trainable_expert(model, 0)
    if cfg.get("grad_ckpt", True):
        enable_grad_ckpt(model)
    full_liger = cfg.get("liger_full_kernels", True)
    fused_ce = enable_fused_linear_ce(
        model,
        cfg.get("liger_fused_ce", True),
        full_kernels=full_liger,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(
        f"isolated MoLE stage 1: final E={final_cfg.num_experts} r={final_cfg.rank} "
        f"k={final_cfg.top_k}; resident/trainable={trainable:,}; "
        f"liger_full={fused_ce and full_liger}; cudnn_sdpa={cfg.get('cudnn_sdpa', True)}"
    )
    log_memory("single expert injected")

    raw_plan = cfg.get("expert_plan")
    plans = (
        [ExpertDataPlan.from_dict(value) for value in raw_plan]
        if raw_plan
        else plan_expert_data(cfg["corpora"], final_cfg.num_experts, cfg.get("domains") or None)
    )
    if len(plans) != final_cfg.num_experts:
        raise ValueError("expert data plan does not match num_experts")
    step_budget = max(int(cfg["max_steps"]), final_cfg.num_experts)
    expert_steps = allocate_expert_steps(step_budget, plans)
    a_dir = adapter_dir(cfg["adapter_id"])
    parts_dir = a_dir / "experts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    expert_paths: list[Path] = []
    expert_results: list[dict] = []
    step_offset = 0
    docs_seen = 0
    packed_tokens = 0
    base_seed = int(cfg.get("seed", 42))

    for expert_index, (plan, max_steps) in enumerate(zip(plans, expert_steps, strict=True)):
        torch.manual_seed(base_seed + expert_index)
        if cuda:
            torch.cuda.manual_seed_all(base_seed + expert_index)
        reset_expert(model, 0)
        set_trainable_expert(model, 0)
        optimizer = make_optimizer(model, cfg["lr"], cuda=cuda)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=_warmup_steps(max_steps),
            num_training_steps=max_steps,
        )
        batches = build_batches(
            cfg,
            tokenizer,
            max_steps,
            domains=list(plan.domains) or None,
            shard_index=plan.shard_index,
            num_shards=plan.num_shards,
        )
        expert_path = parts_dir / f"expert_{expert_index:03d}.safetensors"

        def save_part(_step: int, path: Path = expert_path) -> None:
            save_isolated_expert(path, model)

        stage = f"expert {expert_index + 1}/{final_cfg.num_experts} · {plan.name}"
        log(
            f"training {stage}: step cap={max_steps}, domains={list(plan.domains)}, "
            f"document_shard={plan.shard_index}"
        )
        result = train_loop(
            model,
            batches,
            optimizer,
            scheduler,
            loop_opts(
                cfg,
                rd,
                cuda,
                max_steps=max_steps,
                stage=stage,
                step_offset=step_offset,
                total_steps=step_budget,
            ),
            save_fn=save_part,
        )
        if result.get("data_exhausted", result["steps"] < max_steps):
            log(
                f"{stage}: data slice exhausted after {result['steps']}/{max_steps} optimizer steps; "
                "the configured step count is a safety cap"
            )
        if result["steps"] < 1:
            raise RuntimeError(f"{stage} produced no optimizer steps")
        save_part(0)
        expert_paths.append(expert_path)
        expert_results.append({"index": expert_index, "plan": plan.to_dict(), **result})
        docs_seen += batches.docs_seen
        packed_tokens += batches.tokens_seen
        step_offset += result["steps"]
        del batches, scheduler, optimizer
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        log_memory(f"{stage} complete")

    set_active_expert(model, None)
    set_stage(model, "none")
    metadata = {
        "stage": "isolated_experts",
        "training_strategy": "sequential_single_resident_v1",
        "expert_plan": [plan.to_dict() for plan in plans],
        "fused_linear_cross_entropy": fused_ce,
        "liger_full_kernels": fused_ce and full_liger,
        "cudnn_priority_sdpa": bool(cfg.get("cudnn_sdpa", True)),
    }
    assemble_mole_experts(
        a_dir,
        model,
        final_cfg,
        expert_paths,
        base_ref=cfg["checkpoint_ref"],
        extra_meta=metadata,
    )
    commit_quant_lock(cfg["checkpoint_id"], quant)
    db.execute(
        "UPDATE adapters SET status='trained', path=?, domains_json=?, meta_json=? WHERE id=?",
        (
            str(a_dir),
            json.dumps(sorted({domain for plan in plans for domain in plan.domains})),
            json.dumps(metadata),
            cfg["adapter_id"],
        ),
    )
    total_result = {
        "steps": sum(result["steps"] for result in expert_results),
        "tokens": sum(result["tokens"] for result in expert_results),
        "elapsed_s": round(sum(result["elapsed_s"] for result in expert_results), 1),
    }
    return {
        "strategy": metadata["training_strategy"],
        "experts": expert_results,
        "result": total_result,
        "steps": total_result["steps"],
        "docs_seen": docs_seen,
        "tokens": packed_tokens,
    }


def run_router(cfg: dict, rd: Path) -> dict:
    from transformers.optimization import get_cosine_schedule_with_warmup

    a_row = db.q1("SELECT * FROM adapters WHERE id=?", (cfg["adapter_id"],))
    if not a_row:
        raise RuntimeError("adapter not found")
    if a_row["checkpoint_id"] != cfg["checkpoint_id"]:
        raise ValueError("adapter and router run reference different frozen checkpoints")
    quant = a_row["quant"]
    enforce_quant_lock(cfg["checkpoint_id"], quant)
    model, tokenizer, cuda, quant, notes = load_base(cfg["checkpoint_ref"], quant)
    for n in notes:
        log(f"note: {n}")
    prepare_kbit_training(model, quant)
    log_memory("router base loaded")

    saved = load_mole_config(a_row["path"], expected_base_ref=cfg["checkpoint_ref"])
    if saved.get("config", {}).get("router_type") == "per_token":
        raise ValueError(
            "this adapter already contains learned per-token block routers; "
            "legacy shared-prompt router training is not applicable"
        )
    mole_cfg = MoLEConfig.from_dict(saved["config"])
    # Load the legacy expert bank strictly before adding the new shared-router
    # namespace. Existing adapters and old per-layer router snapshots remain valid.
    adapter_dtype = model_compute_dtype(quant)
    inject_mole(model, mole_cfg, expert_dtype=adapter_dtype)
    n = apply_mole_weights(model, load_mole_weights(a_row["path"]))
    log(f"loaded adapter weights into {n} expert matrices")
    mole_cfg.router_type = "shared_prompt"
    mole_cfg.top_k = min(cfg.get("top_k") or mole_cfg.top_k, mole_cfg.num_experts)
    mole_cfg.router_hidden_size = int(cfg.get("router_hidden_size", 256))
    mole_cfg.router_max_tokens = int(cfg.get("router_max_tokens", 256))
    attach_shared_router(model, mole_cfg, router_dtype=adapter_dtype)
    set_active_expert(model, None)
    set_stage(model, "router")
    if cfg.get("grad_ckpt", True):
        enable_grad_ckpt(model)
    full_liger = cfg.get("liger_full_kernels", True)
    fused_ce = enable_fused_linear_ce(
        model,
        cfg.get("liger_fused_ce", True),
        full_kernels=full_liger,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(
        f"shared prompt router trainable={trainable:,}; liger_full={fused_ce and full_liger}; "
        f"cudnn_sdpa={cfg.get('cudnn_sdpa', True)}"
    )
    log_memory("router bank injected")

    optimizer = make_optimizer(model, cfg["lr"], cuda=cuda)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=_warmup_steps(cfg["max_steps"]),
        num_training_steps=cfg["max_steps"],
    )
    batches = build_batches(cfg, tokenizer, cfg["max_steps"])
    r_dir = router_dir(cfg["router_id"])

    def save(_step: int) -> None:
        save_mole(
            r_dir,
            model,
            mole_cfg,
            base_ref=cfg["checkpoint_ref"],
            extra_meta={"stage": "router", "router_arch": "shared_prompt_v1"},
        )

    result = train_loop(
        model,
        batches,
        optimizer,
        scheduler,
        loop_opts(cfg, rd, cuda),
        aux_fn=lambda: aux_loss(model, mole_cfg.top_k, cfg.get("aux_weight", 0.01)),
        save_fn=save,
    )
    master_summary_fn = getattr(optimizer, "master_state_summary", None)
    master_summary = master_summary_fn() if callable(master_summary_fn) else None
    if master_summary is not None:
        log(
            "persistent optimizer state: "
            + json.dumps(master_summary, sort_keys=True)
        )
        result["persistent_optimizer_state"] = master_summary
    save(0)
    db.execute("UPDATE routers SET status='trained', path=? WHERE id=?", (str(r_dir), cfg["router_id"]))
    return {"result": result, "docs_seen": batches.docs_seen, "tokens": batches.tokens_seen}


def _base_initialization(cfg: dict) -> dict:
    initialization = cfg.get("initialization")
    if isinstance(initialization, dict):
        return dict(initialization)
    # Version-one jobs always continued an existing checkpoint.
    return {
        "mode": "pretrained",
        "checkpoint_id": cfg.get("checkpoint_id"),
        "checkpoint_ref": cfg.get("checkpoint_ref"),
        "tokenizer_ref": cfg.get("checkpoint_ref"),
    }


def _precision_config(cfg: dict) -> dict:
    precision = cfg.get("precision")
    if isinstance(precision, dict):
        return dict(precision)
    return {
        "parameter_dtype": cfg.get("parameter_dtype", "bf16"),
        "qat": "none",
        "qat_group_size": 32,
        "activation_qat": "none",
    }


def _create_scratch_base(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    initialization = _base_initialization(cfg)
    compact_values = initialization.get("compact_tokenizer")
    tokenizer_note: str
    if isinstance(compact_values, dict):
        compact_values = dict(compact_values)
        compact_values.setdefault(
            "special_token_profile",
            "chat"
            if normalize_training_objective(cfg.get("objective")) == OBJECTIVE_CHAT_SFT
            else "causal",
        )
        compact_spec = CompactTokenizerSpec.from_dict(compact_values).validate()
        output_checkpoint_id = str(cfg.get("out_checkpoint_id") or "")
        if not output_checkpoint_id:
            raise ValueError("inline compact tokenizer requires an output checkpoint ID")
        tokenizer_path = checkpoint_dir(output_checkpoint_id) / "tokenizer-source"
        log(
            "training inline compact tokenizer: "
            f"vocab={compact_spec.vocab_size:,}; profile={compact_spec.special_token_profile}; "
            f"corpora={','.join(cfg['corpora'])}; domains={cfg.get('domains') or 'all'}; "
            f"splits={cfg.get('splits') or 'all'}"
        )

        def tokenizer_progress(documents: int, training_bytes: int) -> None:
            if documents == 1 or documents % 10_000 == 0:
                log(
                    "compact tokenizer intake: "
                    f"{documents:,} documents; {training_bytes / 2**20:,.1f} MiB"
                )

        documents = iter_corpus_docs(
            cfg["corpora"],
            cfg.get("domains"),
            shuffle_buffer=1,
            shuffle_buffer_bytes=1,
            seed=int(cfg.get("seed", 42)),
            corpus_mix=cfg.get("corpus_mix", CORPUS_MIX_CONCATENATE),
            splits=cfg.get("splits"),
        )
        tokenizer = train_compact_tokenizer(
            (str(document.get("text") or "") for document in documents),
            tokenizer_path,
            compact_spec,
            progress=tokenizer_progress,
        )
        initialization["compact_tokenizer"] = compact_spec.to_dict()
        cfg["initialization"] = initialization
        tokenizer_note = (
            f"trained inline {compact_spec.special_token_profile} byte-BPE tokenizer "
            f"with {len(tokenizer):,} entries"
        )
    else:
        tokenizer_ref = str(
            initialization.get("tokenizer_ref") or cfg.get("tokenizer_ref") or ""
        )
        if not tokenizer_ref:
            raise ValueError("scratch training requires a tokenizer source")
        if get_tokenizer(tokenizer_ref):
            tokenizer_ref = str(resolve_tokenizer_artifact(tokenizer_ref))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref)
        tokenizer_note = f"loaded scratch tokenizer from {tokenizer_ref}"
    if tokenizer.eos_token_id is None:
        raise ValueError("scratch tokenizer must define an EOS token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    precision = _precision_config(cfg)
    qat_mode = str(precision.get("qat") or "none")
    spec = ScratchModelSpec.from_dict(cfg.get("architecture"))
    spec.validate(
        tokenizer_vocab_size=len(tokenizer),
        train_sequence_length=int(cfg["seq_len"]),
        qat_mode=qat_mode,
    )
    config = build_hf_config(spec, tokenizer)
    resolved_spec = ScratchModelSpec.from_dict(config.yylab_scratch_architecture)
    tokenizer.model_max_length = resolved_spec.max_position_embeddings

    dtype_name = str(precision.get("parameter_dtype") or "bf16")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported parameter dtype: {dtype_name}")
    cuda = torch.cuda.is_available()
    dtype = dtype_map[dtype_name]
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if cuda else "cpu")
    require_dense_precision(dtype_name, device)
    # Construct directly on the destination device and requested dtype. This
    # avoids a second full-sized host model during random initialization.
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    model.config.use_cache = False
    model.config.yylab_training_recipe = {
        "version": BASE_RECIPE_VERSION,
        "initialization": initialization,
        "architecture": resolved_spec.to_dict(),
        "precision": precision,
        "objective": normalize_training_objective(cfg.get("objective")),
        "sequence_length": int(cfg["seq_len"]),
        "seed": seed,
    }


    notes = [
        f"randomly initialized {model.config.model_type} with "
        f"{estimate_parameter_count(resolved_spec):,} parameters on {device}",
        tokenizer_note,
        f"native context={resolved_spec.max_position_embeddings:,}; "
        f"training sequence={int(cfg['seq_len']):,}",
    ]
    low_precision = _requested_qat_metadata(
        precision,
        parameter_dtype=dtype_name,
        device=device,
    )
    if low_precision is not None:
        count = apply_qat(model, low_precision)
        if not low_precision.get("native_low_bit_gemm"):
            raise RuntimeError("low-precision metadata did not resolve to a native GEMM")
        native_note = "native block-scaled FP4 forward GEMMs"
        if low_precision.get("native_sparse_gemm"):
            sparse_note = "; compressed paired-4:8 cuSPARSELt execution"
        elif low_precision.get("sparsity") != "none":
            raise RuntimeError(
                "structured low-precision metadata did not resolve to a native compressed GEMM"
            )
        else:
            sparse_note = ""
        if (
            low_precision.get("backward_precision") == "mxfp8"
            and not low_precision.get("native_low_bit_backward")
        ):
            raise RuntimeError("MXFP8 backward did not resolve to a native GEMM")
        backward_note = (
            "; native block-32 MXFP8 dgrad/wgrad"
            if low_precision.get("native_low_bit_backward")
            else "; BF16 dgrad/wgrad"
        )
        notes.append(
            f"{qat_mode} QAT applied to {count} transformer Linear modules; "
            f"{native_note}{sparse_note}{backward_note}; BF16 shadow parameters retained"
        )
    return model, tokenizer, cuda, dtype_name, notes


_QAT_EXECUTION_FIELDS = (
    "format",
    "backend",
    "group_size",
    "activation",
    "parameter_dtype",
    "sparsity",
    "sparsity_backend",
    "backward_precision",
    "backward_backend",
)


def _requested_qat_metadata(
    precision: dict,
    *,
    parameter_dtype: str,
    device=None,
) -> dict | None:
    qat_mode = str(precision.get("qat") or "none")
    return qat_metadata(
        qat_mode,
        device=device,
        group_size=int(
            precision.get("qat_group_size")
            or (16 if qat_mode == "nvfp4" else 32)
        ),
        activation=str(
            precision.get("activation_qat")
            or ("nvfp4" if qat_mode == "nvfp4" else "none")
        ),
        parameter_dtype=parameter_dtype,
        sparsity=str(precision.get("sparsity") or "none"),
        backend=str(precision.get("qat_backend") or "auto"),
        sparsity_backend=str(precision.get("sparsity_backend") or "auto"),
        backward_precision=str(precision.get("backward_precision") or "bf16"),
        backward_backend=str(precision.get("backward_backend") or "auto"),
    )


def _effective_precision_recipe(
    precision: dict,
    metadata: dict | None,
    *,
    inherited: bool,
) -> dict:
    effective = dict(precision)
    effective["inherit_checkpoint_low_precision"] = bool(inherited)
    if metadata is None:
        effective["qat"] = "none"
        effective["resolved_low_precision"] = None
        return effective
    sparsity = str(metadata["sparsity"])
    backward_precision = str(metadata["backward_precision"])
    effective.update(
        {
            "parameter_dtype": str(metadata["parameter_dtype"]),
            "qat": str(metadata["format"]),
            "qat_group_size": int(metadata["group_size"]),
            "activation_qat": str(metadata["activation"]),
            "qat_backend": "native",
            "sparsity": sparsity,
            "sparsity_backend": "compressed" if sparsity != "none" else "auto",
            "backward_precision": backward_precision,
            "backward_backend": "native" if backward_precision == "mxfp8" else "auto",
            "resolved_low_precision": dict(metadata),
        }
    )
    return effective


def _configure_continued_low_precision(
    model,
    precision: dict,
    *,
    parameter_dtype: str,
    inherit_checkpoint_low_precision: bool,
) -> tuple[dict, str | None]:
    """Resolve the exact low-precision execution recipe for a loaded base."""

    requested = _requested_qat_metadata(
        precision,
        parameter_dtype=parameter_dtype,
        device=next(model.parameters()).device,
    )
    inherited_raw = getattr(
        getattr(model, "config", None),
        LOW_PRECISION_METADATA_KEY,
        None,
    )
    inherited = dict(inherited_raw) if isinstance(inherited_raw, dict) else None
    if inherit_checkpoint_low_precision:
        if inherited is None:
            raise ValueError(
                "checkpoint low-precision inheritance was requested, but the loaded "
                "checkpoint has no native low-precision recipe"
            )
        if requested is not None:
            mismatches = [
                field
                for field in _QAT_EXECUTION_FIELDS
                if requested.get(field) != inherited.get(field)
            ]
            if mismatches:
                raise ValueError(
                    "requested low-precision recipe conflicts with the inherited "
                    "checkpoint recipe: "
                    + ", ".join(mismatches)
                )
        effective = _effective_precision_recipe(
            precision,
            inherited,
            inherited=True,
        )
        return effective, "retained the checkpoint's verified native low-precision recipe"

    if inherited is not None:
        # load_base(..., restore_saved_qat=False) must remove this marker along
        # with the inherited QAT classes. Keeping it would reactivate a recipe
        # that this run explicitly declined.
        raise RuntimeError(
            "dense checkpoint loading retained an unrequested low-precision recipe"
        )
    if requested is None:
        return _effective_precision_recipe(precision, None, inherited=False), None

    count = apply_qat(model, requested)
    effective = _effective_precision_recipe(
        precision,
        requested,
        inherited=False,
    )
    return (
        effective,
        f"applied requested verified native {requested['format']} QAT to {count} Linear modules",
    )


def _make_base_optimizer(model, cfg: dict, *, cuda: bool):
    if not isinstance(cfg.get("optimizer"), dict):
        cfg["optimizer"] = {}
    optimizer_cfg = cfg["optimizer"]
    master_format = str(optimizer_cfg.get("master_weight_storage", "fp32") or "fp32")
    if master_format != "fp32":
        raise ValueError(
            "legacy potentiated/adaptive FP8 optimizer masters are not executable; "
            "use the FP32 CPU master"
        )
    retired_adaptive_fields = {
        "adaptive_max_promoted_fraction",
        "adaptive_promote_threshold",
        "adaptive_demote_threshold",
        "adaptive_demote_patience",
    }
    configured_retired_fields = sorted(retired_adaptive_fields.intersection(optimizer_cfg))
    if configured_retired_fields:
        raise ValueError(
            "retired adaptive optimizer-master settings are not executable: "
            + ", ".join(configured_retired_fields)
        )
    lr = float(optimizer_cfg.get("lr", cfg.get("lr", 5e-5)))
    betas = (
        float(optimizer_cfg.get("beta1", 0.9)),
        float(optimizer_cfg.get("beta2", 0.999)),
    )
    optimizer = make_optimizer(
        model,
        lr,
        weight_decay=float(optimizer_cfg.get("weight_decay", cfg.get("weight_decay", 0.0))),
        cuda=cuda,
        optimizer_state=str(optimizer_cfg.get("state", cfg.get("optimizer_state", "auto")) or "auto"),
        implementation=str(optimizer_cfg.get("implementation", "auto") or "auto"),
        betas=betas,
        eps=float(optimizer_cfg.get("eps", 1e-8)),
    )
    optimizer_cfg["resolved"] = optimizer_runtime_metadata(optimizer)
    return optimizer


def _make_base_scheduler(optimizer, cfg: dict):
    from transformers import get_scheduler

    scheduler_cfg = cfg.get("scheduler") if isinstance(cfg.get("scheduler"), dict) else {}
    name = str(scheduler_cfg.get("name", "cosine") or "cosine")
    max_steps = int(cfg["max_steps"])
    explicit_warmup = scheduler_cfg.get("warmup_steps")
    if explicit_warmup is not None and int(explicit_warmup) >= 0:
        warmup_steps = min(int(explicit_warmup), max_steps)
    elif "warmup_ratio" in scheduler_cfg:
        warmup_steps = min(round(max_steps * float(scheduler_cfg["warmup_ratio"])), max_steps)
    else:
        warmup_steps = _warmup_steps(max_steps)
    specific: dict = {}
    if name == "cosine_with_restarts":
        specific["num_cycles"] = int(scheduler_cfg.get("num_cycles", 1))
    return get_scheduler(
        name,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        scheduler_specific_kwargs=specific or None,
    ), warmup_steps


def run_base(cfg: dict, rd: Path) -> dict:
    import torch

    cfg = normalize_base_worker_config(cfg)
    data_preflight = corpus_training_preflight(
        cfg["corpora"],
        cfg.get("domains"),
        corpus_mix=cfg["corpus_mix"],
        splits=cfg.get("splits"),
        cross_corpus_dedup=cfg["cross_corpus_dedup"],
        corpus_weights=cfg.get("corpus_weights") or None,
        source_max_ratio=cfg.get("source_max_ratio"),
    )
    if data_preflight["errors"]:
        raise ValueError("; ".join(data_preflight["errors"]))
    for warning in data_preflight["warnings"]:
        log(f"data warning: {warning}")
    data_provenance = data_preflight["provenance"]
    objective = cfg["objective"]
    data_provenance["objective"] = objective
    log(
        "base data: "
        f"corpora={','.join(cfg['corpora'])}; "
        f"mix={cfg['corpus_mix']}; domains={cfg.get('domains') or 'all'}; "
        f"splits={cfg.get('splits') or 'all'}; "
        f"cross_dedup={cfg['cross_corpus_dedup']}; "
        f"source_max_ratio={cfg.get('source_max_ratio') or 'none'}; objective={objective}"
    )
    initialization = _base_initialization(cfg)
    mode = str(initialization.get("mode") or "pretrained")
    method = str(cfg.get("method") or "full")
    if mode == "scratch" and method != "full":
        raise ValueError("scratch initialization supports full-parameter training only")

    precision = _precision_config(cfg)
    requested_dtype = str(precision.get("parameter_dtype") or "bf16")
    requested_qat = str(precision.get("qat") or "none")
    inherit_saved_qat = bool(precision.get("inherit_checkpoint_low_precision", False))
    if method == "qlora" and (requested_qat != "none" or inherit_saved_qat):
        raise ValueError(
            "QLoRA base quantization cannot be combined with native QAT execution"
        )
    quant = str(cfg.get("quant") or "nf4") if method == "qlora" else requested_dtype
    if mode == "scratch":
        model, tokenizer, cuda, quant, notes = _create_scratch_base(cfg)
        initialization = _base_initialization(cfg)
    else:
        checkpoint_id = str(initialization.get("checkpoint_id") or cfg.get("checkpoint_id") or "")
        checkpoint_ref = str(initialization.get("checkpoint_ref") or cfg.get("checkpoint_ref") or "")
        if not checkpoint_ref:
            raise ValueError("continued base training requires a checkpoint reference")
        if method != "full":
            enforce_quant_lock(checkpoint_id, quant)
        model, tokenizer, cuda, quant, notes = load_base(
            checkpoint_ref,
            quant,
            restore_saved_qat=inherit_saved_qat,
        )
        precision, low_precision_note = _configure_continued_low_precision(
            model,
            precision,
            parameter_dtype=requested_dtype,
            inherit_checkpoint_low_precision=inherit_saved_qat,
        )
        cfg["precision"] = precision
        requested_dtype = str(precision["parameter_dtype"])
        if low_precision_note is not None:
            notes.append(low_precision_note)

    for note in notes:
        log(f"note: {note}")
    if method == "qlora":
        prepare_kbit_training(model, quant)
    log_memory("base training model loaded")

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    model.config.yylab_base_recipe_version = BASE_RECIPE_VERSION
    model.config.yylab_training_recipe = {
        "version": BASE_RECIPE_VERSION,
        "initialization": initialization,
        "architecture": getattr(model.config, "yylab_scratch_architecture", None),
        "precision": precision,
        "objective": objective,
        "sequence_length": int(cfg["seq_len"]),
        "seed": seed,
    }
    if method in {"lora", "qlora"}:
        from peft import LoraConfig, get_peft_model

        peft_cfg = LoraConfig(
            r=cfg.get("rank", 32),
            lora_alpha=cfg.get("alpha", 64),
            target_modules="all-linear",
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg, autocast_adapter_dtype=False)
        model.print_trainable_parameters()
    if cfg.get("grad_ckpt", True):
        enable_grad_ckpt(model)
    if cfg.get("allow_tf32", True) and cuda:
        torch.set_float32_matmul_precision("high")
    full_liger = cfg.get("liger_full_kernels", True)
    fused_ce = enable_fused_linear_ce(
        model,
        cfg.get("liger_fused_ce", True),
        full_kernels=full_liger,
    )
    log(
        f"base training init={mode}; dtype={requested_dtype}; "
        f"qat={precision.get('qat', 'none')}; sparsity={precision.get('sparsity', 'none')}; "
        f"liger_full={fused_ce and full_liger}; "
        f"cudnn_sdpa={cfg.get('cudnn_sdpa', True)}"
    )

    optimizer = _make_base_optimizer(model, cfg, cuda=cuda)
    scheduler, warmup_steps = _make_base_scheduler(optimizer, cfg)
    optimizer_cfg = cfg.get("optimizer") if isinstance(cfg.get("optimizer"), dict) else {}
    optimizer_resolution = optimizer_runtime_metadata(optimizer)
    write_json(
        rd / "optimizer_resolution.json",
        {
            "requested": {
                "implementation": str(optimizer_cfg.get("implementation", "auto") or "auto"),
                "state": str(optimizer_cfg.get("state", "auto") or "auto"),
            },
            "resolved": optimizer_resolution,
        },
    )
    # Keep the canonical run recipe replayable and retain exactly what auto chose.
    write_json(rd / "config.json", cfg)
    log(
        f"optimizer requested={optimizer_cfg.get('implementation', 'auto')}/"
        f"{optimizer_cfg.get('state', 'auto')}; resolved="
        f"{optimizer_resolution['implementation']}/{optimizer_resolution['state']}; "
        f"moments={optimizer_resolution['state_precision']}; "
        f"master={optimizer_resolution['master_weight_precision']}; "
        f"scheduler={type(scheduler).__name__}; warmup={warmup_steps} steps"
    )
    batches = build_batches(cfg, tokenizer, cfg["max_steps"])
    train_model = model
    if cfg.get("torch_compile", False):
        compile_mode = str(cfg.get("compile_mode") or "default")
        train_model = torch.compile(model, mode=compile_mode)
        log(f"torch.compile enabled ({compile_mode})")

    def save(step: int) -> None:
        if method == "full":
            root = checkpoint_dir(cfg["out_checkpoint_id"])
            out = root if step == 0 else root / "steps" / f"step-{step:08d}"
            out.mkdir(parents=True, exist_ok=True)
            previous_cache = getattr(model.config, "use_cache", False)
            model.config.use_cache = True
            try:
                model.save_pretrained(out)
                tokenizer.save_pretrained(out)
                write_json(out / "yylab_recipe.json", cfg)
            finally:
                model.config.use_cache = previous_cache
        else:
            out = adapter_dir(cfg["adapter_id"])
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out)
            write_json(out / "yylab_recipe.json", cfg)

    result = train_loop(
        train_model,
        batches,
        optimizer,
        scheduler,
        loop_opts(cfg, rd, cuda),
        save_fn=save,
    )
    if int(result.get("steps", 0)) < 1:
        if objective == OBJECTIVE_CHAT_SFT:
            stats_fn = getattr(batches, "data_stats", None)
            stats = stats_fn() if callable(stats_fn) else {}
            raise RuntimeError(
                "chat SFT found no parseable transcript with supervised assistant tokens; "
                "expected structured messages or line-start system:/user:/assistant: blocks "
                f"compatible with the tokenizer chat template (docs scanned="
                f"{stats.get('docs_seen', 0)}, parseable={stats.get('parseable_docs', 0)}, "
                f"unrenderable={stats.get('unrenderable_docs', 0)}, failures="
                f"{stats.get('failure_reasons', {})})"
            )
        raise RuntimeError(
            "selected base-training corpora produced no complete training sequences; "
            "check corpus shards, domain/split filters, and sequence length"
        )
    training_data_stats = tokenized_view_manifest(
        batches,
        result,
        objective=objective,
        sequence_length=int(cfg["seq_len"]),
    )
    data_provenance["training"] = training_data_stats
    cfg["data_provenance"] = data_provenance
    write_json(rd / "tokenized_view_manifest.json", training_data_stats)
    master_summary_fn = getattr(optimizer, "master_state_summary", None)
    master_summary = master_summary_fn() if callable(master_summary_fn) else None
    if master_summary is not None:
        log("persistent optimizer state: " + json.dumps(master_summary, sort_keys=True))
        result["persistent_optimizer_state"] = master_summary
    save(0)

    if method == "full":
        out = checkpoint_dir(cfg["out_checkpoint_id"])
        write_json(out / "tokenized_view_manifest.json", training_data_stats)
        params_raw = sum(parameter.numel() for parameter in model.parameters())
        architecture = getattr(model.config, "yylab_scratch_architecture", None)
        checkpoint_meta = {
            "recipe_version": BASE_RECIPE_VERSION,
            "initialization": mode,
            "derived_from": (
                initialization.get("checkpoint_ref") or cfg.get("checkpoint_ref")
                if mode != "scratch"
                else None
            ),
            "tokenizer_ref": initialization.get("tokenizer_ref"),
            "compact_tokenizer": initialization.get("compact_tokenizer"),
            "architecture": architecture,
            "precision": precision,
            "optimizer": optimizer_cfg,
            "objective": objective,
            "training_sequence_length": int(cfg["seq_len"]),
            "parameters": params_raw,
            "data_provenance": data_provenance,
        }
        db.execute(
            "UPDATE checkpoints SET kind='base', source='local', ref=?, arch=?, params_b=?, path=?,"
            " hidden_size=?, num_layers=?, meta_json=? WHERE id=?",
            (
                str(out),
                (getattr(model.config, "architectures", None) or [type(model).__name__])[0],
                round(params_raw / 1e9, 6),
                str(out),
                int(getattr(model.config, "hidden_size", 0) or 0),
                int(getattr(model.config, "num_hidden_layers", 0) or 0),
                json.dumps(checkpoint_meta),
                cfg["out_checkpoint_id"],
            ),
        )
    else:
        write_json(adapter_dir(cfg["adapter_id"]) / "tokenized_view_manifest.json", training_data_stats)
        checkpoint_id = str(initialization.get("checkpoint_id") or cfg.get("checkpoint_id") or "")
        commit_quant_lock(checkpoint_id, quant)
        adapter_row = db.q1("SELECT meta_json FROM adapters WHERE id=?", (cfg["adapter_id"],))
        try:
            adapter_meta = json.loads((adapter_row or {}).get("meta_json") or "{}")
        except (TypeError, ValueError):
            adapter_meta = {}
        adapter_meta["data_provenance"] = data_provenance
        adapter_meta["objective"] = objective
        db.execute(
            "UPDATE adapters SET status='trained', path=?, corpora_json=?, meta_json=? WHERE id=?",
            (
                str(adapter_dir(cfg["adapter_id"])),
                json.dumps(cfg["corpora"]),
                json.dumps(adapter_meta),
                cfg["adapter_id"],
            ),
        )
    return {
        "result": result,
        "docs_seen": batches.docs_seen,
        "tokens": batches.tokens_seen,
        "initialization": mode,
        "objective": objective,
        "parameter_dtype": requested_dtype,
        "qat": precision.get("qat", "none"),
        "sparsity": precision.get("sparsity", "none"),
        "master_weight_storage": optimizer_resolution["master_weight_precision"],
        "optimizer_resolution": optimizer_resolution,
        "persistent_optimizer_state": master_summary,
        "corpora": list(cfg["corpora"]),
        "corpus_mix": cfg["corpus_mix"],
        "domains": list(cfg.get("domains") or []),
        "splits": list(cfg.get("splits") or []),
        "data_provenance": data_provenance,
        "training_data": training_data_stats,
    }


def _calib_ids(ref: str, corpora: list[str], seq_len: int = 512, n_seqs: int = 8) -> list[list[int]]:
    from transformers import AutoTokenizer

    from ..train.data import iter_corpus_docs

    tokenizer = AutoTokenizer.from_pretrained(ref)
    carry: list[int] = []
    seqs: list[list[int]] = []
    eos = tokenizer.eos_token_id
    for doc in iter_corpus_docs(corpora, None, shuffle_buffer=128):
        ids = tokenizer(doc["text"], add_special_tokens=False)["input_ids"]
        carry.extend(ids)
        if eos is not None:
            carry.append(eos)
        while len(carry) >= seq_len and len(seqs) < n_seqs:
            seqs.append(carry[:seq_len])
            carry = carry[seq_len:]
        if len(seqs) >= n_seqs:
            break
    if carry and len(seqs) < n_seqs:
        seqs.append(carry[:seq_len])
    return seqs


def run_export(cfg: dict, rd: Path) -> dict:
    from ..artifacts import checkpoint_dir
    from ..export.merge import merge_lora, merge_mole
    from ..models.loading import scan_checkpoint

    out = checkpoint_dir(cfg["out_checkpoint_id"])
    calibration_ids: list[list[int]] | None = None
    source_router_type: str | None = None
    export_mode = "merged_lora_v1"
    if cfg["adapter_arch"] == "lora":
        merge_lora(cfg["checkpoint_ref"], cfg["adapter_path"], out)
    else:
        artifact = load_mole_config(
            Path(cfg["adapter_path"]),
            expected_base_ref=cfg["checkpoint_ref"],
        )
        source_router_type = str(artifact.get("config", {}).get("router_type") or "per_layer")
        joint_per_token = source_router_type == "per_token"
        if joint_per_token and cfg.get("router_path"):
            raise ValueError(
                "joint per-token adapters already contain block routers; "
                "a legacy shared-prompt router snapshot cannot be selected for export"
            )
        export_mode = "calibrated_static_mole_v1" if joint_per_token else "static_mole_merge_v1"
        if joint_per_token and not cfg.get("corpora"):
            raise ValueError(
                "joint per-token export requires calibration corpora; "
                "dynamic token routing cannot be preserved in a merged checkpoint"
            )
        if cfg.get("corpora"):
            calibration_ids = _calib_ids(cfg["checkpoint_ref"], cfg["corpora"])
            if joint_per_token and not calibration_ids:
                raise RuntimeError("calibration corpora produced no token sequences")
            log(
                f"calibrating static expert weights on {len(calibration_ids)} sequences; "
                "the exported checkpoint will not preserve dynamic routing"
            )
        merge_mole(
            cfg["checkpoint_ref"],
            cfg["adapter_path"],
            out,
            router_path=None if joint_per_token else cfg.get("router_path"),
            calib_ids=calibration_ids,
        )

    export_metadata = {
        "exported_from": cfg["checkpoint_ref"],
        "adapter": cfg.get("adapter_path"),
        "adapter_arch": cfg["adapter_arch"],
        "export_mode": export_mode,
        "source_router_type": source_router_type,
        "dynamic_token_routing_preserved": False if source_router_type == "per_token" else None,
        "calibration": {
            "performed": calibration_ids is not None,
            "sequence_count": len(calibration_ids or []),
            "corpora": list(cfg.get("corpora") or []),
        },
    }
    model_config_path = out / "config.json"
    model_config = read_json(model_config_path)
    model_config["yylab_export"] = export_metadata
    write_json(model_config_path, model_config)
    write_json(out / "yylab_export.json", export_metadata)

    info = scan_checkpoint(out)
    import json as _json

    db.execute(
        "UPDATE checkpoints SET kind='base', source='local', ref=?, arch=?, params_b=?, path=?,"
        " hidden_size=?, num_layers=?, meta_json=? WHERE id=?",
        (
            str(out),
            info["arch"],
            info["params_b"],
            str(out),
            info["hidden_size"],
            info["num_layers"],
            _json.dumps(export_metadata),
            cfg["out_checkpoint_id"],
        ),
    )
    return {
        "params_b": info["params_b"],
        "arch": info["arch"],
        "export": export_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_id = args.run

    run = db.q1("SELECT * FROM runs WHERE id=?", (run_id,))
    if not run:
        raise SystemExit(f"run {run_id} not found")
    cfg = json.loads(run["config_json"] or "{}")
    rd = run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "config.json", cfg)

    db.execute(
        "UPDATE runs SET status='running', started_at=?, log_path=? WHERE id=?",
        (db.now_iso(), str(rd / "train.log"), run_id),
    )
    log(f"yylab worker: run={run_id} kind={run['kind']}")

    try:
        if run["kind"] in {"base", "spalmer_base"}:
            cfg = normalize_base_worker_config(cfg)
            # The run-directory recipe is the canonical replay payload even for
            # legacy rows that stored one scalar corpus ID.
            write_json(rd / "config.json", cfg)
        if not cfg.get("max_steps") and run["kind"] in {
            "mole_stage1",
            "router_stage2",
            "base",
            "spalmer_base",
        }:
            cfg["max_steps"] = estimate_steps(
                cfg["corpora"],
                cfg.get("domains"),
                cfg.get("epochs", 1),
                cfg["seq_len"],
                cfg["micro_batch"],
                cfg["grad_accum"],
                cfg.get("token_stats_key"),
                splits=cfg.get("splits"),
            )
            log(f"estimated {cfg['max_steps']} optimizer steps")

        if run["kind"] == "mole_stage1":
            summary = run_mole(cfg, rd)
        elif run["kind"] == "router_stage2":
            summary = run_router(cfg, rd)
        elif run["kind"] == "base":
            summary = run_base(cfg, rd)
        elif run["kind"] == "spalmer_base":
            from ..train.spalmer import run_spalmer_base

            summary = run_spalmer_base(cfg, rd)
        elif run["kind"] == "export":
            summary = run_export(cfg, rd)
        elif run["kind"] == "benchmark_quality":
            summary = run_benchmark_quality(cfg, rd)
        elif run["kind"] == "benchmark_train_perf":
            summary = run_benchmark_training_performance(cfg, rd)
        else:
            raise RuntimeError(f"unknown run kind: {run['kind']}")
        db.execute(
            "UPDATE runs SET status='done', finished_at=?, metrics_json=? WHERE id=?",
            (db.now_iso(), json.dumps(summary), run_id),
        )
        log("done")
    except Exception as e:  # noqa: BLE001
        err = f"{e}\n{traceback.format_exc()}"
        log(err)
        db.execute(
            "UPDATE runs SET status='failed', finished_at=?, error=? WHERE id=?",
            (db.now_iso(), err[-4000:], run_id),
        )
        mark_training_artifacts_failed(run["kind"], cfg)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
