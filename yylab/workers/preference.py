"""Dedicated offline-DPO worker.

This stays separate from the general training worker so preference optimization
has an explicit run kind and failure boundary.  The trainable policy is a PEFT
LoRA adapter; disabling that adapter supplies the frozen reference policy.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .. import db
from ..artifacts import adapter_dir, run_dir, write_json
from ..train.preference import (
    PREFERENCE_OBJECTIVE,
    PREFERENCE_RUN_KIND,
    DPOConfig,
    EncodedPreferencePair,
    PreferenceCollator,
    dpo_objective,
    load_preference_dataset,
    optimizer_steps,
    preference_dataset_stats,
    sequence_log_probabilities,
)


def log(message: str) -> None:
    print(message, flush=True)


def _chunks(values: Sequence[int], size: int) -> Iterator[list[int]]:
    for offset in range(0, len(values), size):
        yield list(values[offset : offset + size])


def _write_progress(
    rd: Path,
    *,
    stage: str,
    step: int,
    total: int,
    value: float,
    label: str,
    message: str,
) -> None:
    write_json(
        rd / "progress.json",
        {
            "stage": stage,
            "step": int(step),
            "max_steps": int(total),
            "value": round(float(value), 6),
            "metric_label": label,
            "loss": round(float(value), 6),
            "msg": message,
        },
    )


def _append_metric(rd: Path, metric: dict[str, object]) -> None:
    with (rd / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metric, ensure_ascii=False, allow_nan=False) + "\n")


def _model_device(model: nn.Module) -> torch.device:
    embeddings = getattr(model, "get_input_embeddings", lambda: None)()
    weight = getattr(embeddings, "weight", None)
    if isinstance(weight, Tensor):
        return weight.device
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("preference policy has no parameters") from exc


def _on_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _disable_adapter(model: nn.Module) -> AbstractContextManager[Any]:
    disable = getattr(model, "disable_adapter", None)
    if not callable(disable):
        raise RuntimeError("DPO reference requires a PEFT model with disable_adapter()")
    context = disable()
    if not hasattr(context, "__enter__") or not hasattr(context, "__exit__"):
        raise RuntimeError("PEFT disable_adapter() did not return a context manager")
    return context


def precompute_reference_logps(
    model: nn.Module,
    pairs: Sequence[EncodedPreferencePair],
    collator: PreferenceCollator,
    *,
    micro_batch: int,
    device: torch.device,
    progress: Any | None = None,
) -> Tensor:
    """Cache base-policy chosen/rejected log probabilities on CPU."""

    was_training = model.training
    model.eval()
    rows: list[Tensor] = []
    completed = 0
    with _disable_adapter(model), torch.inference_mode():
        for indices in _chunks(list(range(len(pairs))), micro_batch):
            batch = _on_device(collator.collate([pairs[index] for index in indices]), device)
            rows.append(sequence_log_probabilities(model, batch).detach().float().cpu())
            completed += len(indices)
            if progress is not None:
                progress(completed, len(pairs))
    model.train(was_training)
    if not rows:
        raise ValueError("cannot precompute reference scores for an empty dataset")
    return torch.cat(rows)


def _micro_batches(pair_count: int, config: DPOConfig) -> list[list[int]]:
    generator = torch.Generator(device="cpu")
    batches: list[list[int]] = []
    for epoch in range(config.epochs):
        generator.manual_seed(config.seed + epoch)
        order = torch.randperm(pair_count, generator=generator).tolist()
        batches.extend(_chunks(order, config.micro_batch))
    return batches


def train_dpo_policy(
    model: nn.Module,
    pairs: Sequence[EncodedPreferencePair],
    collator: PreferenceCollator,
    reference_logps: Tensor,
    config: DPOConfig,
    rd: Path,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    """Optimize a prepared PEFT policy and publish optimizer-step metrics."""

    expected_rows = len(pairs) * 2
    if reference_logps.shape != (expected_rows,):
        raise ValueError(
            f"reference cache has {reference_logps.numel()} rows; expected {expected_rows}"
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("DPO policy has no trainable adapter parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    micro_batches = _micro_batches(len(pairs), config)
    total_steps = optimizer_steps(len(pairs), config)
    warmup_steps = min(total_steps, math.ceil(total_steps * config.warmup_ratio))
    from transformers import get_scheduler

    scheduler = get_scheduler(
        "cosine",
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    last: dict[str, float | int] = {}
    optimizer_step = 0
    last_logged_step = 0
    for group_offset in range(0, len(micro_batches), config.grad_accum):
        if optimizer_step >= total_steps:
            break
        group = micro_batches[group_offset : group_offset + config.grad_accum]
        group_pairs = sum(len(indices) for indices in group)
        totals = {
            "loss": 0.0,
            "preference_accuracy": 0.0,
            "reward_margin": 0.0,
            "chosen_reward": 0.0,
            "rejected_reward": 0.0,
        }
        for indices in group:
            weight = len(indices) / group_pairs
            batch = _on_device(collator.collate([pairs[index] for index in indices]), device)
            flat_indices = [row for index in indices for row in (2 * index, 2 * index + 1)]
            refs = reference_logps[flat_indices].to(device)
            policy = sequence_log_probabilities(model, batch)
            metrics = dpo_objective(
                policy,
                refs,
                beta=config.beta,
                label_smoothing=config.label_smoothing,
            )
            loss = metrics.loss * weight
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite DPO loss")
            loss.backward()
            for name in totals:
                value = metrics.loss if name == "loss" else getattr(metrics, name)
                totals[name] += float(value.detach()) * weight
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite DPO gradient norm")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        last = {
            "step": optimizer_step,
            **{name: round(value, 8) for name, value in totals.items()},
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "gradient_norm": float(gradient_norm),
            "pairs": group_pairs,
        }
        if optimizer_step == 1 or optimizer_step % config.log_every == 0:
            _append_metric(rd, last)
            _write_progress(
                rd,
                stage="offline DPO",
                step=optimizer_step,
                total=total_steps,
                value=totals["loss"],
                label="loss",
                message=(
                    f"preference accuracy {totals['preference_accuracy']:.1%} · "
                    f"reward margin {totals['reward_margin']:.4f}"
                ),
            )
            log(
                f"step {optimizer_step}/{total_steps} loss={totals['loss']:.6f} "
                f"accuracy={totals['preference_accuracy']:.3f} "
                f"margin={totals['reward_margin']:.5f}"
            )
            last_logged_step = optimizer_step
    if not last:
        raise RuntimeError("DPO training produced no optimizer steps")
    if last_logged_step != optimizer_step:
        _append_metric(rd, last)
        _write_progress(
            rd,
            stage="offline DPO",
            step=optimizer_step,
            total=total_steps,
            value=float(last["loss"]),
            label="loss",
            message=(
                f"preference accuracy {float(last['preference_accuracy']):.1%} · "
                f"reward margin {float(last['reward_margin']):.4f}"
            ),
        )
    return last


def _enforce_quant_lock(checkpoint_id: str, quant: str) -> None:
    row = db.q1("SELECT quant_locked FROM checkpoints WHERE id=?", (checkpoint_id,))
    if not row:
        raise ValueError(f"checkpoint {checkpoint_id!r} no longer exists")
    locked = str(row.get("quant_locked") or "")
    if locked and locked != quant:
        raise RuntimeError(
            f"Checkpoint quantization is locked to {locked!r}; DPO requested {quant!r}."
        )


def _validate_run_registry(config: DPOConfig) -> None:
    checkpoint = db.q1("SELECT ref FROM checkpoints WHERE id=?", (config.checkpoint_id,))
    if not checkpoint or checkpoint["ref"] != config.checkpoint_ref:
        raise ValueError("DPO config does not match its registered base checkpoint")
    adapter = db.q1(
        "SELECT checkpoint_id, arch, quant, rank, status FROM adapters WHERE id=?",
        (config.adapter_id,),
    )
    if not adapter:
        raise ValueError(f"preference adapter {config.adapter_id!r} no longer exists")
    expected = (
        adapter["checkpoint_id"] == config.checkpoint_id
        and adapter["arch"] == "lora"
        and adapter["quant"] == config.quant
        and int(adapter["rank"]) == config.rank
        and adapter["status"] == "training"
    )
    if not expected:
        raise ValueError("DPO config does not match its registered training adapter")


def _commit_adapter_artifact(config: DPOConfig, output: Path, metadata: dict[str, object]) -> None:
    """Atomically register the completed artifact and its checkpoint quant lock."""

    with db.transaction() as connection:
        checkpoint = connection.execute(
            "SELECT quant_locked FROM checkpoints WHERE id=?",
            (config.checkpoint_id,),
        ).fetchone()
        if checkpoint is None:
            raise ValueError(f"checkpoint {config.checkpoint_id!r} no longer exists")
        locked = str(checkpoint["quant_locked"] or "")
        if locked and locked != config.quant:
            raise RuntimeError(
                f"Checkpoint quantization changed to {locked!r} while DPO was running."
            )
        lock_update = connection.execute(
            "UPDATE checkpoints SET quant_locked=? WHERE id=? "
            "AND (quant_locked='' OR quant_locked IS NULL OR quant_locked=?)",
            (config.quant, config.checkpoint_id, config.quant),
        )
        adapter_update = connection.execute(
            "UPDATE adapters SET status='trained', path=?, meta_json=? "
            "WHERE id=? AND status='training'",
            (str(output), json.dumps(metadata), config.adapter_id),
        )
        if lock_update.rowcount != 1 or adapter_update.rowcount != 1:
            raise RuntimeError("could not atomically register completed DPO adapter")


def _prepare_policy(config: DPOConfig) -> tuple[nn.Module, Any, bool, list[str]]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from ..models.loading import load_base

    model, tokenizer, cuda, _quant, notes = load_base(config.checkpoint_ref, config.quant)
    model.config.use_cache = False
    if config.quant in {"nf4", "int8"}:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora, autocast_adapter_dtype=False)
    if config.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        enable_inputs = getattr(model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()
    return model, tokenizer, cuda, list(notes)


def run_preference_dpo(config: DPOConfig, rd: Path) -> dict[str, object]:
    """Run strict offline DPO and persist a standard yyLab LoRA artifact."""

    _validate_run_registry(config)
    _enforce_quant_lock(config.checkpoint_id, config.quant)
    dataset = load_preference_dataset(
        config.dataset_path,
        data_format=config.dataset_format,
    )
    examples = dataset.examples
    canonical_checksum = str(dataset.provenance["canonical_sha256"])
    if config.dataset_checksum and canonical_checksum != config.dataset_checksum:
        raise RuntimeError(
            "preference dataset changed after UI validation: canonical checksum "
            f"{canonical_checksum} != expected {config.dataset_checksum}"
        )
    source_stats = preference_dataset_stats(examples)
    log(
        f"validated {len(examples)} unique preference pairs from {config.dataset_path} "
        f"({dataset.provenance['source_rows']} source rows, "
        f"{dataset.provenance['duplicate_pairs']} exact duplicates removed)"
    )
    _write_progress(
        rd,
        stage="loading policy",
        step=0,
        total=len(examples),
        value=0,
        label="pairs",
        message="loading base checkpoint and attaching LoRA policy",
    )
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model, tokenizer, cuda, notes = _prepare_policy(config)
    model_context = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    if model_context and config.max_length > model_context:
        raise ValueError(
            f"DPO max_length {config.max_length} exceeds model context {model_context}"
        )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    log(
        f"policy parameters: {trainable_parameters:,} trainable LoRA / "
        f"{total_parameters:,} loaded total"
    )
    collator = PreferenceCollator(tokenizer, max_length=config.max_length)
    pairs = [collator.encode(example) for example in examples]
    token_stats = {
        "chosen_target_tokens": sum(pair.chosen.target_tokens for pair in pairs),
        "rejected_target_tokens": sum(pair.rejected.target_tokens for pair in pairs),
        "prompt_tokens_truncated": sum(
            pair.chosen.prompt_tokens_truncated + pair.rejected.prompt_tokens_truncated
            for pair in pairs
        ),
        "target_tokens_truncated": sum(
            pair.chosen.target_tokens_truncated + pair.rejected.target_tokens_truncated
            for pair in pairs
        ),
    }
    if token_stats["target_tokens_truncated"]:
        log(
            "warning: max_length truncated "
            f"{token_stats['target_tokens_truncated']:,} assistant target tokens"
        )
    device = _model_device(model)
    reference_update_every = max(config.micro_batch, math.ceil(len(pairs) / 100))

    def reference_progress(done: int, total: int) -> None:
        if done != total and done > config.micro_batch and done % reference_update_every:
            return
        _write_progress(
            rd,
            stage="reference scoring",
            step=done,
            total=total,
            value=done,
            label="pairs",
            message="caching frozen base-policy log probabilities",
        )

    reference_logps = precompute_reference_logps(
        model,
        pairs,
        collator,
        micro_batch=config.micro_batch,
        device=device,
        progress=reference_progress,
    )
    final_metrics = train_dpo_policy(
        model,
        pairs,
        collator,
        reference_logps,
        config,
        rd,
        device=device,
    )
    output = adapter_dir(config.adapter_id)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    artifact_metadata = {
        "format": "yylab_preference_adapter_v1",
        "objective": PREFERENCE_OBJECTIVE,
        "algorithm": "dpo",
        "reference_policy": "base_with_adapter_disabled",
        "base_checkpoint_id": config.checkpoint_id,
        "base_checkpoint_ref": config.checkpoint_ref,
        "dataset_path": str(Path(config.dataset_path).expanduser().resolve()),
        "dataset_format": config.dataset_format,
        "dataset_provenance": dict(dataset.provenance),
        "config": config.to_dict(),
        "dataset": {**source_stats, **token_stats},
        "parameters": {
            "trainable": trainable_parameters,
            "loaded_total": total_parameters,
        },
        "final_metrics": final_metrics,
        "notes": notes,
    }
    write_json(output / "yylab_preference.json", artifact_metadata)
    _commit_adapter_artifact(config, output, artifact_metadata)
    return artifact_metadata


def execute_run(run_id: str) -> dict[str, object]:
    run = db.q1("SELECT * FROM runs WHERE id=?", (run_id,))
    if not run:
        raise ValueError(f"run {run_id!r} not found")
    if run["kind"] != PREFERENCE_RUN_KIND:
        raise ValueError(
            f"preference worker requires run kind {PREFERENCE_RUN_KIND!r}, got {run['kind']!r}"
        )
    config = DPOConfig.from_dict(json.loads(run["config_json"] or "{}"))
    rd = run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "config.json", config.to_dict())
    db.execute(
        "UPDATE runs SET status='running', started_at=?, log_path=? WHERE id=?",
        (db.now_iso(), str(rd / "train.log"), run_id),
    )
    log(f"yylab preference worker: run={run_id} objective={PREFERENCE_OBJECTIVE}")
    return run_preference_dpo(config, rd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_id = args.run
    try:
        summary = execute_run(run_id)
        db.execute(
            "UPDATE runs SET status='done', finished_at=?, metrics_json=? WHERE id=?",
            (db.now_iso(), json.dumps(summary), run_id),
        )
        log("done")
    except Exception as exc:  # noqa: BLE001
        error = f"{exc}\n{traceback.format_exc()}"
        log(error)
        db.execute(
            "UPDATE runs SET status='failed', finished_at=?, error=? WHERE id=?",
            (db.now_iso(), error[-4000:], run_id),
        )
        run = db.q1("SELECT config_json FROM runs WHERE id=?", (run_id,))
        try:
            config = json.loads((run or {}).get("config_json") or "{}")
        except (TypeError, ValueError):
            config = {}
        if config.get("adapter_id"):
            db.execute(
                "UPDATE adapters SET status='failed' WHERE id=? AND status='training'",
                (config["adapter_id"],),
            )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
