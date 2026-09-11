from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from nicegui import ui

from ... import db
from ...artifacts import run_dir
from ...ingest.domains import DOMAINS
from ...tokenizer_registry import list_tokenizers
from ...train.data import (
    CORPUS_MIX_TOKEN_BALANCED,
    corpus_training_preflight,
    normalize_corpus_ids,
    normalize_corpus_mix,
    normalize_source_splits,
    normalize_training_objective,
)
from ...workers.launch import launch_worker
from ..components import field_grid, form_section, hint
from ..shell import panel, shell
from .train_mole import corpus_options

SPRINT_RUN_KIND = "sprint"
SPRINT_BENCHMARK_KIND = "sprint_benchmark"
SPLIT_OPTIONS = ["train", "validation", "test", "unspecified"]
EXECUTION_PRESETS = {
    "cpu_offload": {"parameter_dtype": "bfloat16", "optimizer_state_offload": "cpu"},
    "gpu": {"parameter_dtype": "float32", "optimizer_state_offload": "none"},
}
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
ATTENTION_OPTIONS = {"auto": "Auto · let PyTorch choose", "cudnn": "cuDNN", "efficient": "Memory-efficient",
                     "flash": "Flash attention", "math": "Math"}
SHAPE_PROFILES = {"throughput": "Throughput (wider, fewer layers)", "balanced": "Original proportions"}
CPU_OPTIMIZER_KERNELS = {"auto": "Auto", "native": "Native", "compiled": "Compiled"}
LOSS_OPTIONS = {
    "auto": "Auto · use the model's default choice",
    "torch": "PyTorch · higher GPU memory use",
    "liger": "Liger · lower GPU memory use (CUDA)",
}


def _positive_integer(value: Any, name: str) -> int:
    try:
        parsed = int(value)
        if isinstance(value, bool) or parsed != float(value) or parsed < 1:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    return parsed


def sprint_performance_config(
    *, cpu_threads: int | str | None = None, offload_bucket_mb: int = 32, attention_backend: str = "auto",
    cpu_optimizer_kernel: str = "auto", parameter_dtype: str = "bfloat16", optimizer_state_offload: str = "cpu",
) -> dict[str, Any]:
    if attention_backend not in ATTENTION_OPTIONS:
        raise ValueError("Unsupported attention implementation")
    if cpu_optimizer_kernel not in CPU_OPTIMIZER_KERNELS:
        raise ValueError("CPU optimizer update must be Auto, Native, or Compiled")
    if cpu_optimizer_kernel == "compiled" and (parameter_dtype, optimizer_state_offload) != ("bfloat16", "cpu"):
        raise ValueError("Compiled CPU optimizer updates require BF16 weights and CPU optimizer mode")
    result = {"offload_bucket_mb": _positive_integer(offload_bucket_mb, "Optimizer transfer size"),
              "attention_backend": attention_backend, "cpu_optimizer_kernel": cpu_optimizer_kernel}
    if cpu_threads is not None and str(cpu_threads).strip():
        result["cpu_threads"] = _positive_integer(cpu_threads, "CPU threads")
    return result


def _optional_integer_sweep(values: str | list[int] | None, name: str) -> list[int] | None:
    if values is None or values == "" or values == []:
        return None
    if isinstance(values, str):
        if not values.strip():
            return None
        values = [value.strip() for value in values.split(",")]
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be comma-separated positive integers")
    return list(dict.fromkeys(_positive_integer(value, name) for value in values))


def sprint_execution_config(
    *, parameter_dtype: str = "bfloat16", optimizer_state_offload: str = "cpu",
    gradient_checkpointing: bool = False,
) -> dict[str, Any]:
    if (parameter_dtype, optimizer_state_offload) not in {("bfloat16", "cpu"), ("float32", "none")}:
        raise ValueError("Choose BF16 weights with CPU optimizer state, or FP32 weights with device-resident optimizer state")
    if not isinstance(gradient_checkpointing, bool):
        raise ValueError("Gradient checkpointing must be Boolean")
    return {"parameter_dtype": parameter_dtype, "optimizer_state_offload": optimizer_state_offload,
            "gradient_checkpointing": gradient_checkpointing}


def sprint_memory_estimate(parameters: int, *, parameter_dtype: str = "bfloat16",
                           optimizer_state_offload: str = "cpu") -> dict[str, int]:
    """Persistent CUDA training state, excluding activations and staging buffers."""
    sprint_execution_config(parameter_dtype=parameter_dtype, optimizer_state_offload=optimizer_state_offload)
    if parameters < 0:
        raise ValueError("Parameter count cannot be negative")
    offloaded = optimizer_state_offload == "cpu"
    return {"gpu_bytes": parameters * (4 if offloaded else 16),
            "cpu_bytes": parameters * (12 if offloaded else 0)}


@lru_cache(maxsize=64)
def _planned_config(target: int, vocab_size: int, context: int, profile: str) -> str:
    from ...models.sprint import plan_sprint_model

    return json.dumps(
        plan_sprint_model(target, vocab_size, max_position_embeddings=context, profile=profile).to_dict(),
        sort_keys=True,
    )


def resolve_sprint_shape(
    *,
    vocab_size: int,
    context_length: int,
    target_parameters: int = 0,
    model_config: Mapping[str, Any] | None = None,
    profile: str = "throughput",
    loss_backend: str | None = None,
) -> dict[str, Any]:
    """Plan and count an architecture without constructing model weights."""
    from ...models.sprint import SprintConfig, estimate_parameter_count

    if loss_backend is not None and loss_backend not in LOSS_OPTIONS:
        raise ValueError("Choose a supported loss implementation")
    if target_parameters:
        if profile not in SHAPE_PROFILES:
            raise ValueError("Choose a supported model sizing profile")
        config = SprintConfig(**json.loads(_planned_config(
            int(target_parameters), int(vocab_size), int(context_length), profile,
        )))
    else:
        config = SprintConfig(**{
            **dict(model_config or {}),
            "vocab_size": int(vocab_size),
            "max_position_embeddings": int(context_length),
        })
    if loss_backend is not None:
        config.loss_backend = loss_backend
    config.validate()
    count = estimate_parameter_count(config)
    return {
        "model_config": config.to_dict(),
        "parameters": count,
        "target_parameters": int(target_parameters),
        **({"shape_profile": profile} if target_parameters else {}),
    }


def resolve_sprint_budget(
    *, requested_tokens: int, max_steps: int, seq_len: int, micro_batch: int, grad_accum: int,
) -> dict[str, int | str]:
    """Match the trainer's complete-row cap, including a partial final batch."""
    if min(seq_len, micro_batch, grad_accum) < 1:
        raise ValueError("Sequence length, microbatch, and accumulation must be positive")
    if requested_tokens < 0 or max_steps < 0 or bool(requested_tokens) == bool(max_steps):
        raise ValueError("Choose either a positive token budget or a positive step limit")
    if requested_tokens and requested_tokens < seq_len:
        raise ValueError("Token budget must cover at least one complete sequence")
    rows_per_step = micro_batch * grad_accum
    max_rows = requested_tokens // seq_len if requested_tokens else max_steps * rows_per_step
    scheduled = max_rows * seq_len
    return {
        "requested_training_tokens": requested_tokens,
        "tokens_per_optimizer_step": seq_len * rows_per_step,
        "max_steps": (max_rows + rows_per_step - 1) // rows_per_step,
        "max_rows": max_rows,
        "scheduled_training_tokens": scheduled,
        "token_budget_rounding": scheduled - requested_tokens if requested_tokens else 0,
        "unused_token_budget": requested_tokens - scheduled if requested_tokens else 0,
        "step_cap_source": "token_budget" if requested_tokens else "max_steps",
    }


def build_sprint_run_config(values: Mapping[str, Any] | None = None, **overrides) -> dict[str, Any]:
    """Build a validated, replayable speed-first recipe for the isolated worker."""
    raw = {**dict(values or {}), **overrides}
    name = str(raw.get("checkpoint_name") or "").strip()
    if not name:
        raise ValueError("Enter a checkpoint name")
    corpora = normalize_corpus_ids(raw.get("corpora"))
    if not corpora:
        raise ValueError("Select at least one ready corpus")
    init = dict(raw.get("initialization") or {"mode": "scratch"})
    mode = str(init.get("mode") or "scratch")
    if mode not in {"scratch", "checkpoint", "resume"}:
        raise ValueError("Initialization must be scratch, checkpoint, or resume")
    tokenizer_ref = str(raw.get("tokenizer_ref") or init.get("tokenizer_ref") or "").strip()
    if mode == "scratch" and not tokenizer_ref:
        raise ValueError("Select a ready tokenizer")
    if mode == "checkpoint" and not (init.get("checkpoint_ref") or init.get("path")):
        raise ValueError("Select a Sprint checkpoint to continue")
    resume_from = str(raw.get("resume_from") or "").strip()
    if mode == "resume" and not resume_from:
        raise ValueError("Enter the Sprint checkpoint directory to resume")
    if mode == "scratch":
        init["tokenizer_ref"] = tokenizer_ref
    tokens, steps = int(raw.get("token_budget") or 0), int(raw.get("max_steps") or 0)
    if tokens < 0 or steps < 0 or bool(tokens) == bool(steps):
        raise ValueError("Choose either a positive token budget or a positive step limit")
    sequence = int(raw.get("seq_len", 1024))
    if sequence < 2:
        raise ValueError("Sequence length must be at least 2")
    micro, accum = int(raw.get("micro_batch", 4)), int(raw.get("grad_accum", 1))
    budget = resolve_sprint_budget(
        requested_tokens=tokens, max_steps=steps, seq_len=sequence, micro_batch=micro, grad_accum=accum
    )
    model_config = dict(raw.get("model_config") or {})
    if mode == "scratch":
        from ...models.sprint import SprintConfig

        model = SprintConfig(**model_config)
        model.validate()
        if sequence > model.max_position_embeddings:
            raise ValueError("Training sequence length exceeds the model context")
        model_config = model.to_dict()
    mix = normalize_corpus_mix(raw.get("corpus_mix", CORPUS_MIX_TOKEN_BALANCED))
    weights = {str(key): float(value) for key, value in dict(raw.get("corpus_weights") or {}).items()}
    if any(key not in corpora or not math.isfinite(value) or value <= 0 for key, value in weights.items()):
        raise ValueError("Corpus weights must be positive finite numbers for selected corpora")
    cap = float(raw.get("source_max_ratio") or 0)
    if not math.isfinite(cap) or not 0 <= cap <= 1:
        raise ValueError("Source ratio cap must be between 0 and 1")
    if (weights or cap) and mix != CORPUS_MIX_TOKEN_BALANCED:
        raise ValueError("Corpus weights and source ratio caps require token-balanced mixing")
    if cap and cap < 1 / len(corpora):
        raise ValueError("Source ratio cap is too small for the number of selected corpora")
    splits = normalize_source_splits(raw.get("splits", ["train", "unspecified"]))
    eval_splits = normalize_source_splits(raw.get("eval_splits", ["validation"]))
    eval_interval = int(raw.get("eval_interval", 0))
    if eval_interval and not eval_splits:
        raise ValueError("Select held-out source splits for evaluation")
    if eval_interval and (not splits or set(splits) & set(eval_splits)):
        raise ValueError("Training and evaluation source splits must be disjoint")
    device = str(raw.get("device") or "auto")
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("Device must be auto, cuda, or cpu")
    compile_mode = str(raw.get("compile_mode") or "default")
    if compile_mode not in COMPILE_MODES:
        raise ValueError("Unsupported compilation mode")
    execution = sprint_execution_config(
        parameter_dtype=str(raw.get("parameter_dtype", "bfloat16")),
        optimizer_state_offload=str(raw.get("optimizer_state_offload", "cpu")),
        gradient_checkpointing=raw.get("gradient_checkpointing", raw.get("activation_checkpointing", False)),
    )
    cfg = {
        "recipe_version": 1,
        "architecture_type": "sprint",
        "checkpoint_name": name,
        "out_checkpoint_id": str(raw.get("out_checkpoint_id") or ""),
        "initialization": init,
        "resume_from": resume_from,
        "tokenizer_ref": tokenizer_ref,
        "tokenizer_id": str(raw.get("tokenizer_id") or ""),
        "model_config": model_config,
        "target_parameters": int(raw.get("target_parameters") or 0),
        "corpora": corpora,
        "domains": list(dict.fromkeys(raw.get("domains") or [])),
        "splits": splits,
        "objective": normalize_training_objective(raw.get("objective")),
        "corpus_mix": mix,
        "corpus_weights": weights,
        "cross_corpus_dedup": bool(raw.get("cross_corpus_dedup", True)),
        "source_max_ratio": cap or None,
        "seq_len": sequence,
        "micro_batch": micro,
        "grad_accum": accum,
        "token_budget": tokens,
        **budget,
        "lr": float(raw.get("lr", 3e-4)),
        "weight_decay": float(raw.get("weight_decay", 0.1)),
        "warmup_steps": int(raw.get("warmup_steps", 100)),
        "min_lr_ratio": float(raw.get("min_lr_ratio", 0.1)),
        "grad_clip": float(raw.get("grad_clip", 1.0)),
        "seed": int(raw.get("seed", 42)),
        "device": device,
        "torch_compile": bool(raw.get("torch_compile", True)),
        "compile_mode": compile_mode,
        **execution,
        **sprint_performance_config(cpu_threads=raw.get("cpu_threads"),
                                    offload_bucket_mb=raw.get("offload_bucket_mb", 32),
                                    attention_backend=raw.get("attention_backend", "auto"),
                                    cpu_optimizer_kernel=raw.get("cpu_optimizer_kernel", "auto"),
                                    parameter_dtype=execution["parameter_dtype"],
                                    optimizer_state_offload=execution["optimizer_state_offload"]),
        "activation_checkpointing": execution["gradient_checkpointing"],
        "token_cache": bool(raw.get("token_cache", False)),
        "data_prefetch_batches": int(raw.get("data_prefetch_batches", 4)),
        "eval_interval": eval_interval,
        "eval_steps": int(raw.get("eval_steps", 10)),
        "eval_splits": eval_splits,
        "save_interval": int(raw.get("save_interval", 1000)),
        "log_interval": int(raw.get("log_interval", 20)),
    }
    for key in ("lr", "weight_decay", "min_lr_ratio", "grad_clip"):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if cfg["lr"] == 0 or cfg["min_lr_ratio"] > 1:
        raise ValueError("Learning rate must be positive; minimum LR ratio must be between 0 and 1")
    for key in ("warmup_steps", "eval_interval", "save_interval", "data_prefetch_batches"):
        if cfg[key] < 0:
            raise ValueError(f"{key} cannot be negative")
    if cfg["log_interval"] < 1 or cfg["eval_steps"] < 1:
        raise ValueError("Logging interval and evaluation steps must be positive")
    if mode == "scratch" and cfg["target_parameters"]:
        profile = str(raw.get("shape_profile") or "throughput")
        if profile not in SHAPE_PROFILES:
            raise ValueError("Choose a supported model sizing profile")
        cfg["shape_profile"] = profile
    return cfg


def build_sprint_benchmark_config(
    *, model_config: Mapping[str, Any], seq_len: int, batch_sizes: str,
    compile_modes: list[str], effective_batch: int, warmup_steps: int,
    measure_steps: int, device: str = "auto", parameter_dtype: str = "bfloat16",
    optimizer_state_offload: str = "cpu", gradient_checkpointing: bool = False,
    cpu_threads: int | str | None = None, offload_bucket_mb: int = 32, attention_backend: str = "auto",
    cpu_thread_counts: str | list[int] | None = None,
    offload_bucket_sizes_mb: str | list[int] | None = None, attention_backends: list[str] | None = None,
    shape_profile: str | None = None,
    cpu_optimizer_kernel: str = "auto",
) -> dict[str, Any]:
    from ...models.sprint import SprintConfig

    model = SprintConfig(**dict(model_config))
    model.validate()
    try:
        batches = list(dict.fromkeys(int(value.strip()) for value in batch_sizes.split(",")))
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter microbatches as comma-separated positive integers") from exc
    if not batches or any(value < 1 or effective_batch % value for value in batches):
        raise ValueError("Every microbatch must be positive and divide the effective batch")
    if effective_batch < 1 or warmup_steps < 1 or measure_steps < 1:
        raise ValueError("Effective batch, warmup steps, and measurement steps must be positive")
    if not compile_modes or set(compile_modes) - {"eager", *COMPILE_MODES}:
        raise ValueError("Choose at least one supported execution mode")
    if not 2 <= seq_len <= model.max_position_embeddings:
        raise ValueError("Benchmark sequence length must fit the model context")
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("Device must be auto, cuda, or cpu")
    config = {
        "model_config": model.to_dict(), "seq_len": int(seq_len), "batch_sizes": batches,
        "compile_modes": list(dict.fromkeys(compile_modes)), "effective_batch": int(effective_batch),
        "warmup_steps": int(warmup_steps), "measure_steps": int(measure_steps), "device": device,
        **sprint_execution_config(parameter_dtype=parameter_dtype, optimizer_state_offload=optimizer_state_offload,
                                  gradient_checkpointing=gradient_checkpointing),
        **sprint_performance_config(cpu_threads=cpu_threads, offload_bucket_mb=offload_bucket_mb,
                                    attention_backend=attention_backend, cpu_optimizer_kernel=cpu_optimizer_kernel,
                                    parameter_dtype=parameter_dtype, optimizer_state_offload=optimizer_state_offload),
    }
    for name, values in (("cpu_thread_counts", cpu_thread_counts), ("offload_bucket_sizes_mb", offload_bucket_sizes_mb)):
        parsed = _optional_integer_sweep(values, name)
        if parsed:
            config[name] = parsed
    if optimizer_state_offload != "cpu" and len(config.get("offload_bucket_sizes_mb", [])) > 1:
        raise ValueError("Comparing optimizer transfer sizes requires CPU optimizer mode")
    if attention_backends:
        if not isinstance(attention_backends, (list, tuple)) or any(value not in ATTENTION_OPTIONS for value in attention_backends):
            raise ValueError("Choose supported attention implementations")
        config["attention_backends"] = list(dict.fromkeys(attention_backends))
    if shape_profile is not None:
        if shape_profile not in SHAPE_PROFILES:
            raise ValueError("Choose a supported model sizing profile")
        config["shape_profile"] = shape_profile
    return config


def sprint_checkpoint_rows() -> list[dict[str, Any]]:
    rows = db.q("SELECT * FROM checkpoints WHERE kind IN ('base','sprint') ORDER BY created_at DESC")
    selected = []
    for row in rows:
        try:
            meta = json.loads(row.get("meta_json") or "{}")
        except (ValueError, TypeError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        if (
            row.get("kind") == "sprint" or row.get("arch") == "SprintForCausalLM"
            or meta.get("model_family") == "sprint" or meta.get("architecture_type") == "sprint"
        ):
            selected.append(dict(row))
    return selected


def latest_sprint_recommendation(
    model_config: Mapping[str, Any], sequence_length: int, *, device: str = "auto",
    parameter_dtype: str = "bfloat16", optimizer_state_offload: str = "cpu",
    gradient_checkpointing: bool = False,
) -> dict[str, Any]:
    """Return a completed measurement for this architecture, never a different model."""
    import torch

    selected = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    # Isolated workers start on logical CUDA device zero unless an index is explicit.
    selected_index = selected.index if selected.index is not None else 0
    expected_execution = sprint_execution_config(
        parameter_dtype=parameter_dtype, optimizer_state_offload=optimizer_state_offload,
        gradient_checkpointing=gradient_checkpointing,
    )
    selected_hardware = ""
    if selected.type == "cuda" and torch.cuda.is_available():
        try:
            selected_hardware = " ".join(torch.cuda.get_device_name(selected_index).casefold().split())
        except (RuntimeError, AssertionError):
            pass  # Device type/index remain mandatory even if metadata is unavailable.
    shape_keys = (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "max_position_embeddings", "tie_word_embeddings", "mlp_type", "loss_backend", "loss_chunk_size",
    )
    expected = tuple(model_config.get(key) for key in shape_keys)
    for row in db.q(
        "SELECT id,metrics_json FROM runs WHERE kind=? AND status='done' ORDER BY finished_at DESC",
        (SPRINT_BENCHMARK_KIND,),
    ):
        try:
            report = json.loads(row.get("metrics_json") or "{}")
            if not isinstance(report, dict) or not report.get("recommended"):
                path = run_dir(row["id"]) / "benchmark.json"
                report = json.loads(path.read_text("utf-8")) if path.is_file() else {}
            measured_model = report.get("model_config") or {}
            if tuple(measured_model.get(key) for key in shape_keys) != expected:
                continue
            if int(report.get("sequence_length") or 0) != int(sequence_length):
                continue
            measured_execution = report.get("execution") or {}
            if any(measured_execution.get(key) != value for key, value in expected_execution.items()):
                continue
            measured_device = torch.device(str(report.get("device") or "cpu"))
            if not report.get("device") or measured_device.type != selected.type:
                continue
            if selected.type == "cuda" and (measured_device.index or 0) != selected_index:
                continue
            measured_hardware = " ".join(str(report.get("hardware") or "").casefold().split())
            if selected_hardware and measured_hardware and measured_hardware != selected_hardware:
                continue
            recommended = dict(report.get("recommended") or {})
            if any(key in recommended and recommended[key] != value for key, value in expected_execution.items()):
                continue
            if int(recommended.get("micro_batch") or 0) < 1 or int(recommended.get("grad_accum") or 0) < 1:
                continue
            if recommended.get("torch_compile") and recommended.get("compile_mode") not in COMPILE_MODES:
                continue
            recommended.update(sprint_performance_config(
                cpu_threads=recommended.get("cpu_threads", report.get("torch_cpu_threads")),
                offload_bucket_mb=recommended.get("offload_bucket_mb", 32),
                attention_backend=recommended.get("attention_backend", measured_execution.get("attention_backend", "auto")),
                cpu_optimizer_kernel=recommended.get("cpu_optimizer_kernel", measured_execution.get("cpu_optimizer_kernel", "native")),
                parameter_dtype=parameter_dtype, optimizer_state_offload=optimizer_state_offload,
            ))
            return {"run_id": row["id"], "recommended": recommended,
                    "tokens_per_second": report.get("best_supervised_tokens_per_second")}
        except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
            continue
    raise ValueError("No completed benchmark matches this model, sequence length, device, and optimizer/memory settings. Run a benchmark first.")


def queue_sprint_job(config: Mapping[str, Any], *, benchmark: bool = False) -> str:
    """Create durable run/output records together and clean up failed launches."""
    cfg = dict(config)
    run_id, now = db.new_id("run"), db.now_iso()
    checkpoint_id = "" if benchmark else str(cfg.get("out_checkpoint_id") or db.new_id("ckpt"))
    if checkpoint_id:
        cfg["out_checkpoint_id"] = checkpoint_id
    model = dict(cfg.get("model_config") or {})
    kind = SPRINT_BENCHMARK_KIND if benchmark else SPRINT_RUN_KIND
    with db.transaction() as connection:
        if checkpoint_id:
            connection.execute(
                "INSERT INTO checkpoints(id,name,created_at,kind,source,ref,arch,hidden_size,"
                "num_layers,meta_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (checkpoint_id, cfg["checkpoint_name"], now, "pending", "pending", "",
                 "SprintForCausalLM", int(model.get("hidden_size") or 0),
                 int(model.get("num_hidden_layers") or 0), json.dumps({
                     "architecture_type": "sprint", "model_config": model,
                     "initialization": cfg.get("initialization"),
                     **({"shape_profile": cfg["shape_profile"]} if cfg.get("shape_profile") else {}),
                 })),
            )
        connection.execute(
            "INSERT INTO runs(id,kind,status,created_at,config_json) VALUES(?,?,?,?,?)",
            (run_id, kind, "queued", now, json.dumps(cfg)),
        )
    try:
        pid = launch_worker(run_id)
    except Exception as exc:
        db.execute("UPDATE runs SET status='failed',finished_at=?,error=? WHERE id=?",
                   (db.now_iso(), str(exc), run_id))
        if checkpoint_id:
            db.execute("DELETE FROM checkpoints WHERE id=? AND kind='pending'", (checkpoint_id,))
        raise
    db.execute(
        "UPDATE runs SET metrics_json=json_set(COALESCE(NULLIF(metrics_json,''),'{}'),'$.pid',?) WHERE id=?",
        (pid, run_id),
    )
    return run_id


@ui.page("/train/sprint")
def train_sprint_page() -> None:
    with shell("Train · Sprint", "/train/sprint"):
        ui.label("Sprint").classes("text-2xl font-semibold")
        ui.label("A dense language model built for training throughput.").classes("text-gray-400")
        hint("Fused projections, SDPA attention, tied embeddings, BF16 weights and FP32 CPU optimizer state, "
             "and reusable packed data. Measure batch and compilation settings on your hardware.")
        tokenizers = {row["id"]: row for row in list_tokenizers(ready_only=True)}
        checkpoints = {row["id"]: row for row in sprint_checkpoint_rows()}
        with panel():
            form_section("1", "Model and tokenizer")
            with field_grid(3):
                name = ui.input("Checkpoint name", value="sprint-run").classes("w-full")
                initialization = ui.select({"scratch": "New model", "checkpoint": "Continue checkpoint",
                                            "resume": "Resume optimizer state"}, value="scratch",
                                           label="Initialization").classes("w-full")
                tokenizer = ui.select({key: f"{row['name']} · {int(row['vocab_size']):,} tokens"
                                       for key, row in tokenizers.items()}, label="Registered tokenizer",
                                      value=next(iter(tokenizers), None)).classes("w-full")
            if not tokenizers:
                hint("Import or train a tokenizer on the Tokenizers page before creating a new model.")
            with field_grid(2):
                parent = ui.select({key: row["name"] for key, row in checkpoints.items()},
                                   label="Sprint checkpoint").classes("w-full")
                resume = ui.input("Resume checkpoint directory", placeholder="Run checkpoint or saved step directory")
            parent.bind_visibility_from(initialization, "value", backward=lambda value: value == "checkpoint")
            resume.bind_visibility_from(initialization, "value", backward=lambda value: value == "resume")
            tokenizer.bind_visibility_from(initialization, "value", backward=lambda value: value == "scratch")
            hint("Continue starts a fresh optimizer and accepts a new training budget. Resume restores an "
                 "interrupted run's optimizer, scheduler, and data position using its original recipe and budget.")
            with ui.column().classes("w-full") as shape_controls:
                with field_grid(4):
                    sizing = ui.select({"target": "Target parameter count", "manual": "Manual architecture"},
                                       value="target", label="Model size")
                    target = ui.number("Target parameters", value=100_000_000, min=1, step=1_000_000, format="%.0f")
                    context = ui.number("Maximum context", value=2048, min=2, step=128, format="%.0f")
                    benchmark_vocab = ui.number("Benchmark vocabulary", value=8192, min=1, step=128, format="%.0f")
                benchmark_vocab.bind_visibility_from(tokenizer, "value", backward=lambda value: not value)
                hint("A benchmark can use a synthetic vocabulary without a registered tokenizer. "
                     "Training a new model requires a registered tokenizer.")
                target.bind_visibility_from(sizing, "value", backward=lambda value: value == "target")
                with ui.column().classes("w-full") as target_profile_controls:
                    shape_profile = ui.select(SHAPE_PROFILES, value="throughput", label="Model proportions").classes("w-full")
                    hint("The throughput profile favors measured speed with wider, fewer layers at 10M parameters "
                         "and above. Smaller models keep their original proportions. Convergence and final model "
                         "quality for the new shapes have not been validated.")
                target_profile_controls.bind_visibility_from(sizing, "value", backward=lambda value: value == "target")
                with field_grid(4) as manual:
                    width = ui.number("Width", value=768, min=8, step=64, format="%.0f")
                    layers = ui.number("Layers", value=12, min=1, format="%.0f")
                    heads = ui.number("Attention heads", value=12, min=1, format="%.0f")
                    intermediate = ui.number("MLP width", value=2048, min=8, step=64, format="%.0f")
                manual.bind_visibility_from(sizing, "value", backward=lambda value: value == "manual")
                loss_backend = ui.select(LOSS_OPTIONS, value="auto", label="Loss implementation").classes("w-full")
                hint("PyTorch keeps the full output scores in memory. Liger uses a fused loss to reduce GPU memory, "
                     "and requires CUDA and Liger kernels. Benchmark both for your model; lower memory use does not "
                     "guarantee higher throughput. This choice is saved with the model.")
                preview = ui.label().classes("text-sm text-gray-300")
            shape_controls.bind_visibility_from(initialization, "value", backward=lambda value: value == "scratch")

        with panel():
            form_section("2", "Training data", "Packed corpora are cached and reused across runs.")
            corpora = ui.select(corpus_options(), multiple=True, label="Corpora", value=[]).props("use-chips").classes("w-full")
            with field_grid(3):
                objective = ui.select({"causal_lm": "Pretraining", "chat_sft": "Chat SFT · assistant tokens"},
                                      value="causal_lm", label="Objective")
                domains = ui.select(DOMAINS, multiple=True, value=[], label="Domains · empty means all").props("use-chips")
                splits = ui.select(SPLIT_OPTIONS, multiple=True, value=["train", "unspecified"],
                                   label="Training source splits").props("use-chips")
            with ui.expansion("Corpus mixing and deduplication").classes("w-full"):
                with field_grid(3):
                    mix = ui.select({"token_balanced": "Balance source tokens", "round_robin": "Round robin",
                                     "concatenate": "Concatenate"}, value="token_balanced", label="Corpus mixing")
                    cap = ui.number("Maximum source ratio · 0 disables", value=0, min=0, max=1, step=0.05)
                    dedup = ui.checkbox("Deduplicate across corpora", value=True)
                weights = ui.input("Optional corpus weights (JSON)", value="{}",
                                   placeholder='{"corpus_id": 2.0}').classes("w-full")
                source_cache = ui.checkbox("Build or reuse the per-document token cache before packing", value=False)
                hint("Sprint always caches packed training rows. Enable this extra cache for repeated corpus "
                     "experiments; preparing the whole corpus adds first-run time and disk usage.")

        with panel():
            form_section("3", "Training budget and speed")
            with field_grid(4):
                sequence = ui.number("Training sequence length", value=1024, min=2, step=128, format="%.0f")
                micro = ui.number("Microbatch · sequences", value=4, min=1, format="%.0f")
                accum = ui.number("Gradient accumulation", value=8, min=1, format="%.0f")
                device = ui.select({"auto": "Auto · prefer CUDA", "cuda": "CUDA", "cpu": "CPU"}, value="auto", label="Device")
            execution_mode = ui.select(
                {"cpu_offload": "CPU optimizer · lower GPU memory (BF16 weights)",
                 "gpu": "GPU optimizer · throughput option (FP32 weights)"},
                value="cpu_offload", label="Weights and optimizer",
            ).classes("w-full")
            hint("CPU optimizer mode keeps FP32 master weights and Adam moments in host RAM. "
                 "Host updates and transfers can reduce throughput; benchmarks use the selected mode.")
            with field_grid(3):
                budget_mode = ui.select({"tokens": "Token budget", "steps": "Optimizer steps"}, value="tokens", label="Stop after")
                tokens = ui.number("Training tokens", value=1_000_000_000, min=1, step=1_000_000, format="%.0f")
                steps = ui.number("Total optimizer steps", value=10_000, min=1, format="%.0f")
            tokens.bind_visibility_from(budget_mode, "value", backward=lambda value: value == "tokens")
            steps.bind_visibility_from(budget_mode, "value", backward=lambda value: value == "steps")
            budget_preview = ui.label().classes("text-sm text-gray-300")
            with field_grid(3):
                lr = ui.number("Learning rate", value=3e-4, min=0, step=1e-4, format="%.6f")
                compile_enabled = ui.checkbox("Compile model", value=True)
                compile_mode = ui.select(list(COMPILE_MODES), value="default", label="Compilation mode")
            hint("Benchmark microbatches that fit memory. Compilation adds startup time; benchmark both modes "
                 "for small models or short runs. Token budgets stop at the last complete sequence; "
                 "the final optimizer step may use fewer sequences.")
            with ui.expansion("Optimizer and memory").classes("w-full"):
                with field_grid(3):
                    decay = ui.number("Weight decay", value=0.1, min=0)
                    warmup = ui.number("Warmup steps", value=100, min=0, format="%.0f")
                    minimum_lr = ui.number("Final / peak learning rate", value=0.1, min=0, max=1)
                    clip = ui.number("Gradient clipping · 0 disables", value=1, min=0)
                    seed = ui.number("Seed", value=42, min=0, format="%.0f")
                    prefetch = ui.number("Prefetch batches", value=4, min=0, format="%.0f")
                checkpointing = ui.checkbox("Activation checkpointing · reduces memory, adds computation", value=False)
                with field_grid(4):
                    cpu_threads = ui.input("CPU threads", value="", placeholder="Keep current setting")
                    offload_bucket = ui.number("Optimizer transfer size (MiB)", value=32, min=1, format="%.0f")
                    attention_backend = ui.select(ATTENTION_OPTIONS, value="auto", label="Attention implementation")
                    cpu_optimizer_kernel = ui.select(CPU_OPTIMIZER_KERNELS, value="auto", label="CPU optimizer update")
                cpu_optimizer_kernel.bind_visibility_from(execution_mode, "value", backward=lambda value: value == "cpu_offload")
                hint("Leave CPU threads empty to keep the current setting. Benchmark these controls together; "
                     "more threads or larger transfers are not always faster. Auto attention uses PyTorch's selection.")
                hint("Auto CPU optimizer updates use the compiled implementation when available and fall back to native. "
                     "Compiled requires BF16 weights and CPU optimizer mode.")

        with panel():
            form_section("4", "Evaluation and saving")
            with field_grid(3):
                log_interval = ui.number("Log every N steps", value=20, min=1, format="%.0f")
                save_interval = ui.number("Save every N steps · 0 means final only", value=1000, min=0, format="%.0f")
                eval_interval = ui.number("Evaluate every N steps · 0 disables", value=0, min=0, format="%.0f")
                eval_splits = ui.select(SPLIT_OPTIONS, multiple=True, value=["validation"], label="Evaluation source splits")
                eval_steps = ui.number("Evaluation batches", value=10, min=1, format="%.0f")
            hint("Evaluation requires held-out source splits. Final checkpoints and resumable training state are saved automatically.")

        def current_shape() -> dict[str, Any]:
            row = tokenizers.get(tokenizer.value or "")
            return resolve_sprint_shape(
                vocab_size=int(row["vocab_size"] if row else benchmark_vocab.value or 0),
                context_length=int(context.value or 0),
                target_parameters=int(target.value or 0) if sizing.value == "target" else 0,
                profile=str(shape_profile.value),
                loss_backend=str(loss_backend.value),
                model_config={"hidden_size": int(width.value or 0), "num_hidden_layers": int(layers.value or 0),
                              "num_attention_heads": int(heads.value or 0), "intermediate_size": int(intermediate.value or 0)},
            )

        def update_preview() -> None:
            try:
                plan = current_shape()
                model = plan["model_config"]
                state = sprint_memory_estimate(plan["parameters"], **EXECUTION_PRESETS[str(execution_mode.value)])
                if state["cpu_bytes"]:
                    memory = (f"CUDA state: {state['gpu_bytes'] / 2**30:,.2f} GiB for BF16 weights and gradients; "
                              f"host RAM: {state['cpu_bytes'] / 2**30:,.2f} GiB for FP32 master weights and Adam moments. "
                              "A 1B model uses about 4 GB GPU + 12 GB CPU state.")
                else:
                    memory = (f"CUDA FP32 Adam state: {state['gpu_bytes'] / 2**30:,.2f} GiB. "
                              "A 1B model uses about 16 GB GPU state.")
                preview.set_text(f"{plan['parameters']:,} parameters · {model['num_hidden_layers']} layers · "
                                 f"width {model['hidden_size']} · {model['num_attention_heads']} heads · tied embeddings · "
                                 f"{SHAPE_PROFILES[plan['shape_profile']] + ' · ' if plan.get('shape_profile') else ''}"
                                 f"{memory} Estimates exclude activations and transfer buffers.")
            except (ValueError, TypeError, RuntimeError) as exc:
                preview.set_text(str(exc))
            try:
                budget = resolve_sprint_budget(
                    requested_tokens=int(tokens.value or 0) if budget_mode.value == "tokens" else 0,
                    max_steps=int(steps.value or 0) if budget_mode.value == "steps" else 0,
                    seq_len=int(sequence.value or 0), micro_batch=int(micro.value or 0), grad_accum=int(accum.value or 0),
                )
                budget_preview.set_text(f"Up to {budget['tokens_per_optimizer_step']:,} input tokens / step · "
                                        f"{budget['max_steps']:,} steps · {budget['scheduled_training_tokens']:,} scheduled input tokens")
            except (TypeError, ValueError) as exc:
                budget_preview.set_text(str(exc))

        def queue_training() -> None:
            try:
                selected_mode = str(initialization.value)
                parent_row = checkpoints.get(parent.value or "", {})
                shape = current_shape() if selected_mode == "scratch" else {"model_config": {}, "target_parameters": 0}
                cfg = build_sprint_run_config(
                    checkpoint_name=name.value, tokenizer_ref=tokenizer.value if selected_mode == "scratch" else "",
                    tokenizer_id=tokenizer.value if selected_mode == "scratch" else "", **shape,
                    initialization={"mode": selected_mode, "checkpoint_id": parent_row.get("id"),
                                    "checkpoint_ref": parent_row.get("path") or parent_row.get("ref")},
                    resume_from=resume.value if selected_mode == "resume" else "", corpora=corpora.value,
                    domains=domains.value, splits=splits.value, objective=objective.value, corpus_mix=mix.value,
                    corpus_weights=json.loads(weights.value or "{}"), source_max_ratio=cap.value,
                    cross_corpus_dedup=dedup.value, token_cache=source_cache.value, seq_len=int(sequence.value or 0),
                    micro_batch=int(micro.value or 0), grad_accum=int(accum.value or 0),
                    token_budget=int(tokens.value or 0) if budget_mode.value == "tokens" else 0,
                    max_steps=int(steps.value or 0) if budget_mode.value == "steps" else 0, device=device.value,
                    lr=lr.value, torch_compile=compile_enabled.value, compile_mode=compile_mode.value,
                    weight_decay=decay.value, warmup_steps=warmup.value, min_lr_ratio=minimum_lr.value,
                    grad_clip=clip.value, seed=seed.value, data_prefetch_batches=prefetch.value,
                    **EXECUTION_PRESETS[str(execution_mode.value)],
                    cpu_threads=cpu_threads.value, offload_bucket_mb=offload_bucket.value,
                    attention_backend=attention_backend.value,
                    cpu_optimizer_kernel=cpu_optimizer_kernel.value if execution_mode.value == "cpu_offload" else "auto",
                    activation_checkpointing=checkpointing.value, log_interval=log_interval.value,
                    save_interval=save_interval.value, eval_interval=eval_interval.value,
                    eval_steps=eval_steps.value, eval_splits=eval_splits.value,
                )
                preflight = corpus_training_preflight(
                    cfg["corpora"], cfg["domains"] or None, corpus_mix=cfg["corpus_mix"], splits=cfg["splits"],
                    cross_corpus_dedup=cfg["cross_corpus_dedup"], corpus_weights=cfg["corpus_weights"] or None,
                    source_max_ratio=cfg["source_max_ratio"],
                )
                if preflight["errors"]:
                    raise ValueError("; ".join(preflight["errors"]))
                run_id = queue_sprint_job(cfg)
                ui.notify(f"Sprint training queued: {run_id}. Progress and throughput are on Runs.", type="positive")
            except Exception as exc:
                ui.notify(str(exc), type="negative", multi_line=True)

        ui.button("Start Sprint training", icon="play_arrow", on_click=queue_training).props("no-caps")

        with panel() as benchmark_panel:
            form_section("5", "Find your fastest settings", "Synthetic training benchmark · uses the architecture selected above.")
            hint("Compares forward, backward, and optimizer throughput with a fixed effective batch. "
                 "Uses the weights, optimizer placement, and activation checkpointing selected above. "
                 "This excludes corpus preparation and evaluates speed, not model quality.")
            with field_grid(3):
                batch_sizes = ui.input("Microbatches to compare", value="1,2,4,8")
                effective_batch = ui.number("Fixed effective batch", value=32, min=1, format="%.0f")
                benchmark_modes = ui.select(["eager", *COMPILE_MODES],
                                            multiple=True, value=["eager", "default"], label="Execution modes")
                benchmark_warmup = ui.number("Warmup steps per candidate", value=5, min=1, format="%.0f")
                measure = ui.number("Measured steps per candidate", value=20, min=1, format="%.0f")
            with ui.expansion("Compare CPU and attention settings").classes("w-full"):
                hint("Empty fields use the training settings above. Each combination is measured, so large sweeps take longer.")
                with field_grid(3):
                    benchmark_threads = ui.input("CPU thread counts", value="", placeholder="e.g. 4,8,12,20")
                    benchmark_buckets = ui.input("Optimizer transfer sizes (MiB)", value="", placeholder="e.g. 16,32,64")
                    benchmark_attention = ui.select(ATTENTION_OPTIONS, multiple=True, value=[], label="Attention implementations")

            def queue_benchmark() -> None:
                try:
                    shape = current_shape()
                    cfg = build_sprint_benchmark_config(
                        model_config=shape["model_config"], shape_profile=shape.get("shape_profile"),
                        seq_len=int(sequence.value or 0),
                        batch_sizes=str(batch_sizes.value or ""), compile_modes=list(benchmark_modes.value or []),
                        effective_batch=int(effective_batch.value or 0), warmup_steps=int(benchmark_warmup.value or 0),
                        measure_steps=int(measure.value or 0), device=str(device.value),
                        **EXECUTION_PRESETS[str(execution_mode.value)], gradient_checkpointing=checkpointing.value,
                        cpu_threads=cpu_threads.value, offload_bucket_mb=offload_bucket.value,
                        attention_backend=attention_backend.value, cpu_thread_counts=benchmark_threads.value,
                        offload_bucket_sizes_mb=benchmark_buckets.value, attention_backends=benchmark_attention.value,
                        cpu_optimizer_kernel=cpu_optimizer_kernel.value if execution_mode.value == "cpu_offload" else "auto",
                    )
                    run_id = queue_sprint_job(cfg, benchmark=True)
                    ui.notify(f"Sprint throughput benchmark queued: {run_id}. Results will appear on Runs.", type="positive")
                except Exception as exc:
                    ui.notify(str(exc), type="negative", multi_line=True)

            recommendation_status = ui.label().classes("text-sm text-gray-300")

            def apply_recommendation() -> None:
                try:
                    match = latest_sprint_recommendation(
                        current_shape()["model_config"], int(sequence.value or 0), device=str(device.value),
                        **EXECUTION_PRESETS[str(execution_mode.value)], gradient_checkpointing=checkpointing.value,
                    )
                    recommended = match["recommended"]
                    micro.set_value(int(recommended["micro_batch"]))
                    accum.set_value(int(recommended["grad_accum"]))
                    compile_enabled.set_value(bool(recommended["torch_compile"]))
                    compile_mode.set_value(str(recommended.get("compile_mode") or "default")
                                           if recommended["torch_compile"] else "default")
                    cpu_threads.set_value(str(recommended.get("cpu_threads") or ""))
                    offload_bucket.set_value(int(recommended["offload_bucket_mb"]))
                    attention_backend.set_value(str(recommended["attention_backend"]))
                    cpu_optimizer_kernel.set_value(str(recommended["cpu_optimizer_kernel"]))
                    rate = match.get("tokens_per_second")
                    rate_label = f" · {float(rate):,.0f} supervised tokens / second" if rate else ""
                    recommendation_status.set_text(f"Applied benchmark {match['run_id']}{rate_label}")
                    ui.notify("Measured batch, compilation, CPU, transfer, and attention settings applied to the training recipe.", type="positive")
                except Exception as exc:
                    ui.notify(str(exc), type="negative", multi_line=True)

            with ui.row().classes("gap-3 items-center"):
                ui.button("Benchmark training throughput", icon="speed", on_click=queue_benchmark).props("no-caps outline")
                ui.button("Apply latest matching benchmark", icon="tune", on_click=apply_recommendation).props("no-caps flat")
                ui.link("View runs and benchmark results", "/runs")
        benchmark_panel.bind_visibility_from(initialization, "value", backward=lambda value: value == "scratch")

        for element in (tokenizer, sizing, target, context, width, layers, heads, intermediate,
                        sequence, micro, accum, budget_mode, tokens, steps, benchmark_vocab, execution_mode,
                        shape_profile, loss_backend):
            element.on_value_change(lambda _: update_preview())
        update_preview()
