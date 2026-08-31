from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import orjson
from nicegui import ui

from ... import db
from ...ingest.domains import DOMAINS
from ...train.base_config import (
    BASE_RECIPE_VERSION,
    FAMILY_LABELS,
    MASTER_WEIGHT_LABELS,
    QAT_LABELS,
    SPARSITY_LABELS,
    TRAINING_RECIPE_PRESETS,
    ScratchModelSpec,
    estimate_parameter_count,
    fit_layers_to_parameter_target,
    projected_training_memory_gb,
)
from ...train.data import estimate_steps, tokenizer_stats_key
from ...train.low_precision import native_nvfp4_status, native_sparse_nvfp4_status
from ...train.mxfp8 import mxfp8_capability_status
from ...workers.launch import launch_worker
from ..components import field_grid, form_section, hint
from ..shell import ACCENT, panel, shell
from .train_mole import (
    _auto_method_quant,
    _checkpoint_shape,
    _make_plan,
    corpus_options,
    render_plan,
)

SCRATCH_PRESETS: dict[str, dict[str, int]] = {
    "qwen_40m": {
        "hidden_size": 512,
        "intermediate_size": 1536,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
    },
    "qwen_100m": {
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "head_dim": 64,
    },
    "qwen_250m": {
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 18,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 64,
    },
    "qwen_600m": {
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 64,
    },
}


def _checkpoint_config(row: dict | None) -> dict:
    if not row:
        return {}
    path = Path(str(row.get("ref") or "")) / "config.json"
    if path.is_file():
        try:
            return json.loads(path.read_text("utf-8"))
        except (OSError, TypeError, ValueError):
            pass
    try:
        metadata = json.loads(row.get("meta_json") or "{}")
        return dict(metadata.get("architecture") or {})
    except (TypeError, ValueError):
        return {}


def _tokenizer_vocab_hint(row: dict | None, explicit_vocab: int = 0) -> int:
    if int(explicit_vocab or 0) > 0:
        return int(explicit_vocab)
    config = _checkpoint_config(row)
    return int(config.get("vocab_size") or 0)


def _ordered_corpora(value: str | Iterable[str] | None) -> list[str]:
    """Normalize legacy scalar and multi-select values without losing their order."""

    values = [value] if isinstance(value, str) else (value or [])
    selected: list[str] = []
    seen: set[str] = set()
    for corpus_id in values:
        normalized = str(corpus_id).strip()
        if normalized and normalized not in seen:
            selected.append(normalized)
            seen.add(normalized)
    return selected


def _base_run_config(
    *,
    initialization: dict,
    method: str,
    architecture: dict | None,
    precision: dict,
    optimizer: dict,
    scheduler: dict,
    corpora: str | Iterable[str],
    domains: list[str] | None,
    seq_len: int,
    epochs: int,
    micro_batch: int,
    grad_accum: int,
    max_steps: int,
    seed: int,
    grad_clip: float,
    grad_ckpt: bool,
    log_every: int,
    save_every: int | None,
    runtime: dict,
) -> dict:
    """Build the current base recipe independently of NiceGUI widgets."""

    config = {
        "recipe_version": BASE_RECIPE_VERSION,
        "initialization": dict(initialization),
        "method": method,
        "precision": dict(precision),
        "optimizer": dict(optimizer),
        "scheduler": dict(scheduler),
        "corpora": _ordered_corpora(corpora),
        "corpus_mix": "round_robin",
        "domains": list(domains) if domains else None,
        "seq_len": int(seq_len),
        "epochs": int(epochs),
        "micro_batch": int(micro_batch),
        "grad_accum": int(grad_accum),
        "max_steps": int(max_steps),
        "seed": int(seed),
        "grad_clip": float(grad_clip),
        "grad_ckpt": bool(grad_ckpt),
        "log_every": int(log_every),
        "save_every": int(save_every) if save_every else None,
        "lr": float(optimizer["lr"]),
        "optimizer_state": str(optimizer["state"]),
        **runtime,
    }
    if architecture is not None:
        config["architecture"] = dict(architecture)
    checkpoint_id = initialization.get("checkpoint_id")
    checkpoint_ref = initialization.get("checkpoint_ref")
    if checkpoint_id:
        config["checkpoint_id"] = checkpoint_id
    if checkpoint_ref:
        config["checkpoint_ref"] = checkpoint_ref
    return config


@ui.page("/train/base")
def train_base_page():
    with shell("Train · Base", "/train/base"):
        ui.label("Raw or continued base-model training").classes("text-xl font-semibold")
        ui.label(
            "Initialize a decoder from random weights with a chosen architecture, or continue an existing "
            "checkpoint. Scratch runs train every parameter; LoRA/QLoRA remain continued-training modes."
        ).classes("text-sm text-gray-400 -mt-2")

        checkpoint_rows = {
            row["id"]: row for row in db.q("SELECT * FROM checkpoints WHERE kind!='pending' ORDER BY name")
        }
        checkpoint_options = {
            cid: f"{row['name']} ({row['params_b']:.3f} B)" if row["params_b"] else row["name"]
            for cid, row in checkpoint_rows.items()
        }
        first_checkpoint = next(iter(checkpoint_options), None)
        corpora = corpus_options()
        native_nvfp4, native_nvfp4_note = native_nvfp4_status()
        native_sparse_nvfp4, native_sparse_nvfp4_note = native_sparse_nvfp4_status(
            smoke=False
        )
        native_mxfp8, native_mxfp8_note = mxfp8_capability_status()

        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
            with panel().classes("shrink-0").style("width: 650px; max-width: 100%"):
                form_section("1", "Initialization")
                init_mode = ui.radio(
                    {
                        "scratch": "Random weights (scratch)",
                        "pretrained": "Existing checkpoint",
                    },
                    value="scratch",
                ).props("inline dense")
                output_name = (
                    ui.input("Output name", placeholder="auto").props("dense outlined").classes("w-full")
                )

                with ui.column().classes("w-full gap-2") as continued_box:
                    checkpoint_select = (
                        ui.select(
                            checkpoint_options,
                            value=first_checkpoint,
                            label="Checkpoint to continue",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    with field_grid(3):
                        method = (
                            ui.select(
                                {
                                    "auto": "Auto",
                                    "full": "Full parameters",
                                    "lora": "LoRA",
                                    "qlora": "QLoRA",
                                },
                                value="auto",
                                label="Update method",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        qlora_quant = (
                            ui.select(
                                {"nf4": "NF4", "int8": "INT8"},
                                value="nf4",
                                label="QLoRA storage",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        lora_rank = (
                            ui.number("LoRA rank", value=32, min=1, max=512, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )

                with ui.column().classes("w-full gap-2") as scratch_source_box:
                    tokenizer_select = (
                        ui.select(
                            checkpoint_options,
                            value=first_checkpoint,
                            label="Tokenizer from registered checkpoint",
                        )
                        .props("dense outlined clearable")
                        .classes("w-full")
                    )
                    tokenizer_override = (
                        ui.input(
                            "Tokenizer path / HuggingFace id override",
                            placeholder="Optional; overrides the registered selection",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    hint(
                        "Only the tokenizer is loaded from this source. Model weights are randomly initialized. "
                        "The model vocabulary cannot be smaller than the tokenizer."
                    )

                ui.separator().classes("my-3")
                with ui.column().classes("w-full gap-2") as architecture_box:
                    form_section("2", "Scratch architecture")
                    with field_grid(3):
                        preset = (
                            ui.select(
                                {
                                    "custom": "Custom",
                                    "qwen_40m": "~40M @ 32K vocab",
                                    "qwen_100m": "~100M @ 32K vocab",
                                    "qwen_250m": "~250M @ 32K vocab",
                                    "qwen_600m": "~600M @ Qwen vocab",
                                },
                                value="qwen_100m",
                                label="Shape preset",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        family = (
                            ui.select(FAMILY_LABELS, value="qwen3", label="Decoder family")
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        target_params_m = (
                            ui.number("Target params (M)", value=100, min=1, max=100000, precision=1)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    fit_target = ui.button("Fit layer count to target", icon="calculate").props(
                        "dense flat no-caps"
                    )
                    hint(
                        "The target is advisory: yyLab keeps your widths/head layout exact and chooses the "
                        "nearest whole layer count. The actual parameter count is shown before launch."
                    )

                    with field_grid(3):
                        vocab_size = (
                            ui.number("Model vocab (0 = tokenizer)", value=0, min=0, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        vocab_multiple = (
                            ui.select(
                                [1, 8, 16, 32, 64, 128, 256],
                                value=64,
                                label="Vocab padding multiple",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        tie_embeddings = ui.switch("Tie input/output embeddings", value=True)
                    with field_grid(3):
                        hidden_size = (
                            ui.number("Hidden size", value=768, min=8, max=65536, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        intermediate_size = (
                            ui.number("Intermediate size", value=2048, min=8, max=262144, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        num_layers = (
                            ui.number("Layers", value=12, min=1, max=512, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with field_grid(3):
                        num_heads = (
                            ui.number("Attention heads", value=12, min=1, max=512, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        num_kv_heads = (
                            ui.number("KV heads", value=4, min=1, max=512, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        head_dim = (
                            ui.number("Head dimension", value=64, min=8, max=512, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with field_grid(3):
                        native_context = (
                            ui.number("Native model context", value=8192, min=128, max=1048576, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        rope_theta = (
                            ui.number("RoPE theta", value=1_000_000, min=1, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        initializer_range = (
                            ui.number("Initialization std", value=0.02, min=0.000001, step=0.001)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with field_grid(3):
                        rms_eps = (
                            ui.number("RMS norm epsilon", value=1e-6, min=1e-12, step=1e-6)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        attention_dropout = (
                            ui.number("Attention dropout", value=0.0, min=0, max=0.99, step=0.01)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        attention_bias = ui.switch("Attention projection bias", value=False)

                ui.separator().classes("my-3")
                form_section("3", "Training data and token budget")
                corpora_select = (
                    ui.select(
                        corpora,
                        value=[],
                        multiple=True,
                        label="Training corpora (select one or more)",
                    )
                    .props("dense outlined use-chips clearable options-dense")
                    .classes("w-full")
                )
                with ui.row().classes("items-center gap-2 -mt-1"):
                    ui.button(
                        "Select all",
                        on_click=lambda: set_corpora_selection(corpora),
                    ).props("dense flat no-caps size=sm")
                    ui.button(
                        "Clear",
                        on_click=lambda: set_corpora_selection([]),
                    ).props("dense flat no-caps size=sm")
                    ui.label("Chip order is preserved in the training run.").classes(
                        "text-xs text-gray-500"
                    )
                domain_filter = (
                    ui.select(
                        DOMAINS,
                        multiple=True,
                        value=[],
                        label="Domain filter (optional)",
                    )
                    .props("dense outlined use-chips")
                    .classes("w-full")
                )
                data_summary = (
                    ui.label("0 corpora selected · choose one or more.")
                    .classes("text-xs text-gray-400 px-3 py-2 rounded-md w-full")
                    .style(f"background: {ACCENT}10")
                )
                with field_grid(3):
                    train_seq_len = (
                        ui.number("Training sequence length", value=2048, min=128, max=1048576, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    epochs = (
                        ui.number("Epochs", value=1, min=1, max=1000, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    max_steps_override = (
                        ui.number("Max optimizer steps (0 = data)", value=0, min=0, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                hint(
                    "Native model context is architectural capacity. The training sequence length is what the "
                    "run actually exercises; it must not exceed native context for scratch models."
                )

                ui.separator().classes("my-3")
                form_section("4", "Parameter state and QAT numerics")
                training_recipe = (
                    ui.select(
                        {
                            "custom": "Custom precision recipe",
                            "nvfp4_sparse_mxfp8_adaptive": (
                                "NVFP4 sparse forward + MXFP8 backward + adaptive FP8 state (recommended)"
                            ),
                            "nvfp4_paired_4to8": "NVFP4 + paired 4:8 sparsity-aware training (RTX Blackwell)",
                            "nvfp4_paired_4to8_fp8_master": (
                                "NVFP4 + paired 4:8 + potentiated FP8 master (experimental)"
                            ),
                        },
                        value="custom",
                        label="Training recipe preset",
                    )
                    .props("dense outlined")
                    .classes("w-full")
                )
                with field_grid(3):
                    parameter_dtype = (
                        ui.select(
                            {"bf16": "BF16 (recommended)", "fp16": "FP16", "fp32": "FP32"},
                            value="bf16",
                            label="Parameter dtype",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    qat_mode = (
                        ui.select(QAT_LABELS, value="none", label="QAT target")
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    qat_group = (
                        ui.select([16, 32, 64, 128], value=32, label="INT4 weight group")
                        .props("dense outlined")
                        .classes("w-full")
                    )
                activation_qat = (
                    ui.select(
                        {"none": "High-precision activations", "int8": "INT8 per-token activation QAT"},
                        value="none",
                        label="INT4 activation recipe",
                    )
                    .props("dense outlined")
                    .classes("w-full")
                )
                with field_grid(3):
                    qat_backend = (
                        ui.select(
                            {
                                "auto": "Auto (native NVFP4 when supported)",
                                "native": "Require native NVFP4 forward",
                                "emulated": "Force fake-quant emulation",
                            },
                            value="auto",
                            label="Low-bit backend",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    sparsity_pattern = (
                        ui.select(SPARSITY_LABELS, value="none", label="Weight sparsity")
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    sparsity_backend = (
                        ui.select(
                            {
                                "auto": "Auto sparse execution",
                                "compressed": "Require compressed cuSPARSELt",
                                "dense_masked": "Dense GEMM with paired mask",
                            },
                            value="auto",
                            label="Sparse execution",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                with field_grid(2):
                    backward_precision = (
                        ui.select(
                            {
                                "bf16": "BF16 dgrad / wgrad",
                                "mxfp8": "MXFP8 dgrad / wgrad",
                            },
                            value="bf16",
                            label="Backward GEMMs",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    backward_backend = (
                        ui.select(
                            {
                                "auto": "Auto MXFP8 backend",
                                "native": "Require native MXFP8 Tensor Cores",
                                "emulated": "MXFP8 format emulation",
                            },
                            value="auto",
                            label="MXFP8 backend",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                qat_truth = ui.label("").classes("text-xs text-amber-400")
                ui.label(
                    f"Native dense NVFP4 QAT lane: {'available' if native_nvfp4 else 'unavailable'} — "
                    f"{native_nvfp4_note}."
                ).classes("text-xs text-gray-500")
                ui.label(
                    f"Compressed sparse NVFP4 lane: {'available' if native_sparse_nvfp4 else 'unavailable'} — "
                    f"{native_sparse_nvfp4_note}."
                ).classes("text-xs text-gray-500")
                ui.label(
                    f"Native MXFP8 backward lane: {'available' if native_mxfp8 else 'unavailable'} — "
                    f"{native_mxfp8_note}."
                ).classes("text-xs text-gray-500")

                ui.separator().classes("my-3")
                form_section("5", "Optimization")
                with field_grid(3):
                    learning_rate = (
                        ui.number("Learning rate", value=3e-4, min=0, step=1e-5)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    optimizer_impl = (
                        ui.select(
                            {
                                "auto": "Auto implementation",
                                "torch_adamw": "Torch AdamW",
                                "fused_adamw": "Fused AdamW",
                                "adamw_8bit": "bitsandbytes AdamW8bit",
                                "paged_adamw_8bit": "Paged AdamW8bit",
                                "cpu_offload_adamw": "CPU-offload AdamW",
                            },
                            value="auto",
                            label="Optimizer",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    optimizer_state = (
                        ui.select(
                            {
                                "auto": "Auto placement",
                                "gpu": "GPU state",
                                "cpu_paged": "Paged / CPU-backed",
                                "cpu_offload": "Full CPU offload",
                            },
                            value="cpu_offload",
                            label="Optimizer state",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                master_weight_storage = (
                    ui.select(
                        MASTER_WEIGHT_LABELS,
                        value="fp32",
                        label="CPU master-weight storage",
                    )
                    .props("dense outlined")
                    .classes("w-full")
                )
                with field_grid(4) as adaptive_controls:
                    adaptive_max_percent = (
                        ui.number(
                            "Max promoted blocks (%)",
                            value=10.0,
                            min=0,
                            max=100,
                            step=1,
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    adaptive_promote_threshold = (
                        ui.number(
                            "Promotion threshold",
                            value=0.55,
                            min=0,
                            step=0.05,
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    adaptive_demote_threshold = (
                        ui.number(
                            "Demotion threshold",
                            value=0.20,
                            min=0,
                            step=0.05,
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    adaptive_demote_patience = (
                        ui.number(
                            "Stable steps before demotion",
                            value=2,
                            min=1,
                            max=255,
                            precision=0,
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                master_truth = ui.label("").classes("text-xs text-amber-400")
                with field_grid(3):
                    weight_decay = (
                        ui.number("Weight decay", value=0.1, min=0, step=0.01)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    beta1 = (
                        ui.number("Adam beta1", value=0.9, min=0, max=0.999999, step=0.01)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    beta2 = (
                        ui.number("Adam beta2", value=0.95, min=0, max=0.999999, step=0.001)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                with field_grid(3):
                    adam_eps = (
                        ui.number("Adam epsilon", value=1e-8, min=1e-12, step=1e-8)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    grad_clip = (
                        ui.number("Gradient clip", value=1.0, min=0, step=0.1)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    scheduler_name = (
                        ui.select(
                            {
                                "cosine": "Cosine",
                                "linear": "Linear",
                                "constant_with_warmup": "Constant + warmup",
                                "cosine_with_restarts": "Cosine restarts",
                            },
                            value="cosine",
                            label="LR schedule",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                with field_grid(3):
                    warmup_ratio = (
                        ui.number("Warmup ratio", value=0.03, min=0, max=1, step=0.01)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    micro_batch = (
                        ui.number("Micro batch", value=1, min=1, max=256, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    grad_accum = (
                        ui.number("Gradient accumulation", value=16, min=1, max=4096, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )

                with ui.expansion("Runtime and checkpointing", icon="tune").classes("w-full"):
                    with field_grid(3):
                        seed = (
                            ui.number("Seed", value=42, min=0, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        log_every = (
                            ui.number("Log every steps", value=5, min=1, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        save_every = (
                            ui.number("Save every steps (0 = final)", value=0, min=0, precision=0)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with field_grid(3):
                        grad_checkpointing = ui.switch("Gradient checkpointing", value=True)
                        cudnn_sdpa = ui.switch("Prefer cuDNN SDPA", value=True)
                        liger_kernels = ui.switch("Qwen3 Liger kernels", value=True)
                    with field_grid(3):
                        allow_tf32 = ui.switch("Allow TF32", value=True)
                        torch_compile = ui.switch("torch.compile", value=False)
                        compile_mode = (
                            ui.select(
                                {
                                    "default": "Compile: default",
                                    "reduce-overhead": "Compile: reduce overhead",
                                    "max-autotune": "Compile: max autotune",
                                },
                                value="default",
                                label="Compile mode",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )

                start_button = (
                    ui.button("Start base training", icon="play_arrow")
                    .props("no-caps unelevated")
                    .classes("w-full mt-4")
                )
                hint("Runs in a detached worker. The exact recipe is stored with the run and checkpoint.")

            with ui.column().classes("flex-1 gap-4 min-w-[340px]"):
                with panel():
                    ui.label("Preflight").classes("text-base font-semibold mb-2")
                    preflight_box = ui.column().classes("w-full gap-1")
                with panel():
                    ui.label("Precision is a set of independent decisions").classes(
                        "text-base font-semibold mb-2"
                    )
                    ui.label(
                        "Parameter dtype controls the saved trainable tensors. QAT controls the forward numerics. "
                        "Optimizer state controls VRAM/RAM placement. A packed deployment artifact is separate."
                    ).classes("text-sm text-gray-300")
                    ui.label(
                        "Compressed NVFP4 and MXFP8 change GEMM execution, while QAT checkpoints retain dense "
                        "shadow weights. CPU offload plus FP32, legacy potentiated, or adaptive FP8 master "
                        "storage controls persistent state headroom independently."
                    ).classes("text-xs text-amber-400 mt-2")

        def selected_checkpoint() -> dict | None:
            return checkpoint_rows.get(checkpoint_select.value)

        def selected_tokenizer_checkpoint() -> dict | None:
            return checkpoint_rows.get(tokenizer_select.value)

        def current_spec() -> ScratchModelSpec:
            return ScratchModelSpec(
                family=str(family.value or "qwen3"),
                vocab_size=int(vocab_size.value or 0),
                vocab_padding_multiple=int(vocab_multiple.value or 1),
                hidden_size=int(hidden_size.value or 0),
                intermediate_size=int(intermediate_size.value or 0),
                num_hidden_layers=int(num_layers.value or 0),
                num_attention_heads=int(num_heads.value or 0),
                num_key_value_heads=int(num_kv_heads.value or 0),
                head_dim=int(head_dim.value or 0),
                max_position_embeddings=int(native_context.value or 0),
                rope_theta=float(rope_theta.value or 0),
                tie_word_embeddings=bool(tie_embeddings.value),
                initializer_range=float(initializer_range.value or 0),
                rms_norm_eps=float(rms_eps.value or 0),
                attention_dropout=float(attention_dropout.value or 0),
                attention_bias=bool(attention_bias.value),
            )

        def vocab_hint() -> int:
            return _tokenizer_vocab_hint(
                selected_tokenizer_checkpoint(),
                int(vocab_size.value or 0),
            )

        def apply_preset() -> None:
            values = SCRATCH_PRESETS.get(str(preset.value or ""))
            if not values:
                return
            family.set_value("qwen3")
            hidden_size.set_value(values["hidden_size"])
            intermediate_size.set_value(values["intermediate_size"])
            num_layers.set_value(values["num_hidden_layers"])
            num_heads.set_value(values["num_attention_heads"])
            num_kv_heads.set_value(values["num_key_value_heads"])
            head_dim.set_value(values["head_dim"])
            update_preflight()

        def fit_parameter_target() -> None:
            try:
                hint_vocab = vocab_hint()
                if hint_vocab < 1:
                    raise ValueError(
                        "select a registered tokenizer with known vocab or enter a model vocab size"
                    )
                fitted = fit_layers_to_parameter_target(
                    current_spec(),
                    round(float(target_params_m.value or 0) * 1_000_000),
                    tokenizer_vocab_size=hint_vocab,
                )
                num_layers.set_value(fitted.num_hidden_layers)
                preset.set_value("custom")
                update_preflight()
            except ValueError as exc:
                ui.notify(str(exc), type="warning")

        def apply_training_recipe() -> None:
            values = TRAINING_RECIPE_PRESETS.get(str(training_recipe.value or "custom"), {})
            if not values:
                return
            parameter_dtype.set_value(str(values["parameter_dtype"]))
            qat_mode.set_value(str(values["qat"]))
            qat_backend.set_value(str(values["qat_backend"]))
            qat_group.set_value(int(values["qat_group_size"]))
            activation_qat.set_value(str(values["activation_qat"]))
            sparsity_pattern.set_value(str(values["sparsity"]))
            sparsity_backend.set_value(str(values.get("sparsity_backend", "auto")))
            backward_precision.set_value(str(values.get("backward_precision", "bf16")))
            backward_backend.set_value(str(values.get("backward_backend", "auto")))
            optimizer_state.set_value(str(values.get("optimizer_state", "cpu_offload")))
            optimizer_impl.set_value(str(values.get("optimizer_impl", "auto")))
            master_weight_storage.set_value(str(values.get("master_weight_storage", "fp32")))
            adaptive_max_percent.set_value(
                float(values.get("adaptive_max_promoted_fraction", 0.10)) * 100
            )
            adaptive_promote_threshold.set_value(
                float(values.get("adaptive_promote_threshold", 0.55))
            )
            adaptive_demote_threshold.set_value(
                float(values.get("adaptive_demote_threshold", 0.20))
            )
            adaptive_demote_patience.set_value(
                int(values.get("adaptive_demote_patience", 2))
            )
            update_qat_controls()

        def update_mode() -> None:
            scratch = init_mode.value == "scratch"
            scratch_source_box.visible = scratch
            architecture_box.visible = scratch
            continued_box.visible = not scratch
            qat_mode.enabled = scratch
            qat_group.enabled = scratch
            activation_qat.enabled = scratch
            training_recipe.enabled = scratch
            qat_backend.enabled = scratch
            sparsity_pattern.enabled = scratch
            sparsity_backend.enabled = scratch
            backward_precision.enabled = scratch
            backward_backend.enabled = scratch
            update_qat_controls()
            update_preflight()

        def update_qat_controls() -> None:
            qat = str(qat_mode.value or "none") if init_mode.value == "scratch" else "none"
            if qat == "ternary" and family.value != "bitnet":
                family.set_value("bitnet")
            elif family.value == "bitnet" and qat not in {"none", "ternary"}:
                qat_mode.set_value("none")
                qat = "none"
            if qat != "nvfp4" and sparsity_pattern.value != "none":
                sparsity_pattern.set_value("none")
            if qat != "nvfp4" and backward_precision.value != "bf16":
                backward_precision.set_value("bf16")
            qat_group.enabled = qat == "int4" and init_mode.value == "scratch"
            activation_qat.enabled = qat == "int4" and init_mode.value == "scratch"
            qat_backend.enabled = qat == "nvfp4" and init_mode.value == "scratch"
            sparsity_pattern.enabled = qat == "nvfp4" and init_mode.value == "scratch"
            sparse_enabled = (
                qat == "nvfp4"
                and str(sparsity_pattern.value or "none") != "none"
                and init_mode.value == "scratch"
            )
            sparsity_backend.enabled = sparse_enabled
            backward_precision.enabled = qat == "nvfp4" and init_mode.value == "scratch"
            backward_backend.enabled = (
                qat == "nvfp4"
                and backward_precision.value == "mxfp8"
                and init_mode.value == "scratch"
            )
            if qat == "none":
                qat_truth.text = (
                    "Dense forward numerics; parameter and optimizer choices determine state memory."
                )
            elif qat == "int4":
                qat_truth.text = (
                    "INT4 QAT simulates grouped weight quantization during forward. Shadow weights, gradients, "
                    "and optimizer state remain dense; the current Windows kernel is emulated."
                )
            elif qat == "nvfp4":
                backend = str(qat_backend.value or "auto")
                native_selected = backend == "native" or (backend == "auto" and native_nvfp4)
                execution = (
                    "Native packed E2M1/E4M3 block-scaled FP4 forward GEMMs are enabled; "
                    if native_selected and native_nvfp4
                    else "NVFP4 is using dense fake-quant emulation; "
                )
                sparse = str(sparsity_pattern.value or "none")
                sparse_request = str(sparsity_backend.value or "auto")
                compressed_selected = sparse != "none" and native_selected and (
                    sparse_request == "compressed"
                    or (sparse_request == "auto" and native_sparse_nvfp4)
                )
                if compressed_selected and native_sparse_nvfp4:
                    sparse_note = (
                        "paired 4:8 weights execute as compressed E2M1 through cuSPARSELt; "
                    )
                elif sparse != "none":
                    sparse_note = (
                        "paired 4:8 is enforced, but this selection uses a dense masked GEMM; "
                    )
                else:
                    sparse_note = ""
                backward_mode = str(backward_precision.value or "bf16")
                backward_request = str(backward_backend.value or "auto")
                native_backward_selected = backward_mode == "mxfp8" and (
                    backward_request == "native"
                    or (backward_request == "auto" and native_mxfp8)
                ) and parameter_dtype.value == "bf16"
                if native_backward_selected and native_mxfp8:
                    backward_note = "dgrad and wgrad use native block-32 MXFP8 Tensor Core GEMMs; "
                elif backward_mode == "mxfp8":
                    backward_note = "dgrad and wgrad use format-faithful MXFP8 emulation; "
                else:
                    backward_note = "dgrad and wgrad use BF16 GEMMs; "
                unavailable = (
                    f"Requested native mode is unavailable: {native_nvfp4_note}. "
                    if backend == "native" and not native_nvfp4
                    else ""
                )
                master_note = (
                    "The CPU master uses E4M3 with a sparse, adaptive potentiation overlay; "
                    if master_weight_storage.value == "fp8_e4m3_adaptive"
                    else (
                        "The CPU master uses legacy E4M3 with a dense potentiation residual; "
                        if master_weight_storage.value == "fp8_e4m3_potentiated"
                        else "The optimizer master remains FP32; "
                    )
                )
                qat_truth.text = (
                    unavailable
                    + execution
                    + sparse_note
                    + backward_note
                    + master_note
                    + "accelerator shadow parameters and Adam moments remain BF16/FP32 respectively."
                )
            else:
                qat_truth.text = (
                    "Ternary W1.58/A8 QAT is restricted to the BitNet architecture and remains experimental "
                    "with dense shadow weights."
                )
            adaptive_controls.visible = master_weight_storage.value == "fp8_e4m3_adaptive"
            if master_weight_storage.value == "fp8_e4m3_adaptive":
                master_truth.text = (
                    "Adaptive FP8 persistent master: every weight has a real E4M3 byte; only blocks whose "
                    "reconstruction or lost-update score crosses the promotion threshold receive a sparse "
                    "packed four-bit residual. Hysteresis demotes stable blocks. Adam moments remain FP32."
                )
            elif master_weight_storage.value == "fp8_e4m3_potentiated":
                master_truth.text = (
                    "Legacy potentiated FP8 master: persistent CPU state pays for a packed four-bit residual "
                    "for every weight (~1.625 bytes/parameter). Adam moments remain FP32."
                )
            else:
                master_truth.text = (
                    "FP32 CPU masters preserve every optimizer update; accelerator parameters may still be BF16."
                )
            update_preflight()

        def update_data_summary() -> None:
            selected = _ordered_corpora(corpora_select.value)
            if not selected:
                data_summary.text = "0 corpora selected · choose one or more."
                return
            domains = set(domain_filter.value or [])
            docs = chars = 0
            for corpus_id in selected:
                row = db.q1("SELECT stats_json FROM corpora WHERE id=?", (corpus_id,))
                if not row:
                    continue
                stats = orjson.loads(row["stats_json"] or b"{}")
                for domain, values in stats.get("domains", {}).items():
                    if domains and domain not in domains:
                        continue
                    docs += int(values.get("docs", 0))
                    chars += int(values.get("chars", 0))
            tag = f" · filter: {', '.join(sorted(domains))}" if domains else ""
            corpus_label = "corpus" if len(selected) == 1 else "corpora"
            data_summary.text = (
                f"{len(selected)} {corpus_label} selected{tag} · "
                f"{docs:,} docs · {chars / 1e9:.3f}B characters"
            )

        def set_corpora_selection(value: str | Iterable[str] | None) -> None:
            corpora_select.value = _ordered_corpora(value)
            corpora_select.update()
            update_data_summary()

        def update_preflight() -> None:
            preflight_box.clear()
            if init_mode.value != "scratch":
                render_plan(
                    preflight_box,
                    selected_checkpoint(),
                    int(train_seq_len.value or 2048),
                    str(method.value or "auto"),
                    str(qlora_quant.value or "nf4"),
                    rank=int(lora_rank.value or 32),
                    num_experts=1,
                    adapter_only=False,
                    isolated_expert_training=False,
                )
                return
            with preflight_box:
                try:
                    spec = current_spec()
                    hint_vocab = vocab_hint()
                    spec.validate(
                        tokenizer_vocab_size=hint_vocab or None,
                        train_sequence_length=int(train_seq_len.value or 0),
                        qat_mode=str(qat_mode.value or "none"),
                    )
                    qat = str(qat_mode.value or "none")
                    backend = str(qat_backend.value or "auto")
                    sparsity = str(sparsity_pattern.value or "none")
                    sparse_backend = str(sparsity_backend.value or "auto")
                    backward_mode = str(backward_precision.value or "bf16")
                    backward_execution = str(backward_backend.value or "auto")
                    if qat == "nvfp4" and backend == "native" and not native_nvfp4:
                        raise ValueError(f"native NVFP4 backend unavailable: {native_nvfp4_note}")
                    if sparsity != "none" and qat != "nvfp4":
                        raise ValueError("paired 4:8 sparsity requires NVFP4 QAT")
                    native_forward_selected = backend == "native" or (
                        backend == "auto" and native_nvfp4
                    )
                    if (
                        sparsity != "none"
                        and sparse_backend == "compressed"
                        and not native_forward_selected
                    ):
                        raise ValueError("compressed sparse execution requires native NVFP4")
                    if (
                        sparsity != "none"
                        and sparse_backend == "compressed"
                        and not native_sparse_nvfp4
                    ):
                        raise ValueError(
                            f"compressed sparse NVFP4 unavailable: {native_sparse_nvfp4_note}"
                        )
                    if backward_mode == "mxfp8" and qat != "nvfp4":
                        raise ValueError("MXFP8 backward requires NVFP4 QAT")
                    if (
                        backward_mode == "mxfp8"
                        and backward_execution == "native"
                        and not native_mxfp8
                    ):
                        raise ValueError(f"native MXFP8 unavailable: {native_mxfp8_note}")
                    if (
                        backward_mode == "mxfp8"
                        and backward_execution == "native"
                        and parameter_dtype.value != "bf16"
                    ):
                        raise ValueError("native MXFP8 backward requires BF16 parameters")
                    if hint_vocab < 1:
                        ui.label(
                            "Tokenizer vocab is unknown until worker load; enter model vocab for an exact preview."
                        ).classes("text-amber-400 text-sm")
                        return
                    parameters = estimate_parameter_count(spec, tokenizer_vocab_size=hint_vocab)
                    gpu_gb, cpu_gb = projected_training_memory_gb(
                        parameters,
                        parameter_dtype=str(parameter_dtype.value or "bf16"),
                        optimizer_state=str(optimizer_state.value or "gpu"),
                        master_weight_storage=str(master_weight_storage.value or "fp32"),
                        adaptive_max_promoted_fraction=float(
                            adaptive_max_percent.value or 0
                        )
                        / 100.0,
                    )
                    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                        ui.badge(f"{parameters / 1e6:,.1f}M actual params").props("color=primary")
                        ui.badge(
                            f"{spec.num_hidden_layers}L · H{spec.hidden_size} · I{spec.intermediate_size}"
                        ).props("outline color=grey")
                        ui.badge(
                            f"{spec.max_position_embeddings:,} native / {int(train_seq_len.value or 0):,} train"
                        ).props("outline color=teal")
                    ui.label(
                        f"Conservative parameter/gradient/Adam state projection: {gpu_gb:.2f} GB GPU"
                        + (f" + {cpu_gb:.2f} GB CPU" if cpu_gb else "")
                        + ". Activations, CUDA workspaces, and allocator reserve are additional."
                    ).classes("text-xs text-gray-400")
                    if qat != "none":
                        native_forward = qat == "nvfp4" and (
                            backend == "native" or (backend == "auto" and native_nvfp4)
                        )
                        native_backward = backward_mode == "mxfp8" and (
                            backward_execution == "native"
                            or (backward_execution == "auto" and native_mxfp8)
                        ) and parameter_dtype.value == "bf16"
                        ui.label(
                            f"{qat.upper()} QAT uses "
                            + ("native FP4 forward GEMMs" if native_forward else "emulated forward numerics")
                            + (
                                " and native MXFP8 backward GEMMs."
                                if native_backward
                                else " and BF16/emulated backward GEMMs."
                            )
                        ).classes("text-xs text-amber-400")
                    if sparsity != "none":
                        compressed = sparse_backend == "compressed" or (
                            sparse_backend == "auto" and native_sparse_nvfp4
                        )
                        ui.label(
                            "Paired 4:8 SAT gives 50% effective weight density. "
                            + (
                                "Forward GEMMs use compressed cuSPARSELt sparse MMA; saved checkpoints "
                                "still retain BF16 shadow weights for continued training."
                                if compressed
                                else "This selection uses a dense masked GEMM."
                            )
                        ).classes("text-xs text-amber-400")
                    master_storage = str(master_weight_storage.value or "fp32")
                    cpu_offloaded = (
                        optimizer_state.value == "cpu_offload"
                        or optimizer_impl.value == "cpu_offload_adamw"
                    )
                    if master_storage != "fp32" and not cpu_offloaded:
                        raise ValueError(
                            "potentiated FP8 master weights require Full CPU offload or CPU-offload AdamW"
                        )
                    if master_storage == "fp8_e4m3_adaptive":
                        promote = float(adaptive_promote_threshold.value or 0)
                        demote = float(adaptive_demote_threshold.value or 0)
                        if demote > promote:
                            raise ValueError(
                                "adaptive demotion threshold cannot exceed promotion threshold"
                            )
                        ui.label(
                            f"Adaptive potentiation can promote at most "
                            f"{float(adaptive_max_percent.value or 0):g}% of blocks; ordinary blocks "
                            "pay only for E4M3 + a block scale/counter, while Adam moments remain FP32."
                        ).classes("text-xs text-amber-400")
                    elif master_storage == "fp8_e4m3_potentiated":
                        ui.label(
                            "The FP8 base is not updated directly. Sub-ULP updates are retained on a packed "
                            "four-bit interpolation grid with stochastic rounding at the finer boundary."
                        ).classes("text-xs text-amber-400")
                except ValueError as exc:
                    ui.label(str(exc)).classes("text-red-400 text-sm")

        def resolve_pretrained_method() -> tuple[str, str, object | None]:
            selected_method = str(method.value or "auto")
            if selected_method != "auto":
                quant = str(qlora_quant.value or "nf4") if selected_method == "qlora" else "bf16"
                return selected_method, quant, None
            plan_method, plan_quant = _auto_method_quant(
                selected_checkpoint(), str(qlora_quant.value or "nf4")
            )
            plan = _make_plan(
                selected_checkpoint(),
                int(train_seq_len.value or 2048),
                plan_method,
                plan_quant,
                int(lora_rank.value or 32),
                1,
                adapter_only=False,
                isolated_expert_training=False,
            )
            if plan is not None:
                return plan.method, plan.quant, plan
            if plan_method == "auto":
                return "qlora", plan_quant, None
            return plan_method, plan_quant, None

        def on_start() -> None:
            problems: list[str] = []
            selected_corpora = _ordered_corpora(corpora_select.value)
            if not selected_corpora:
                problems.append("choose corpora")
            scratch = init_mode.value == "scratch"
            checkpoint = selected_checkpoint()
            tokenizer_row = selected_tokenizer_checkpoint()
            tokenizer_ref = (tokenizer_override.value or "").strip() or str(
                (tokenizer_row or {}).get("ref") or ""
            )
            if scratch and not tokenizer_ref:
                problems.append("choose or enter a tokenizer source")
            if not scratch and not checkpoint:
                problems.append("choose a checkpoint")
            if problems:
                ui.notify("Fix: " + ", ".join(problems), type="warning")
                return

            sequence_length = int(train_seq_len.value or 0)
            domains = list(domain_filter.value) if domain_filter.value else None
            architecture: dict | None = None
            auto_plan = None
            if scratch:
                spec = current_spec()
                try:
                    spec.validate(
                        tokenizer_vocab_size=vocab_hint() or None,
                        train_sequence_length=sequence_length,
                        qat_mode=str(qat_mode.value or "none"),
                    )
                    selected_qat = str(qat_mode.value or "none")
                    selected_backend = str(qat_backend.value or "auto")
                    selected_sparsity = str(sparsity_pattern.value or "none")
                    selected_sparse_backend = str(sparsity_backend.value or "auto")
                    selected_backward = str(backward_precision.value or "bf16")
                    selected_backward_backend = str(backward_backend.value or "auto")
                    if (
                        selected_qat == "nvfp4"
                        and selected_backend == "native"
                        and str(parameter_dtype.value or "bf16") != "bf16"
                    ):
                        raise ValueError("native NVFP4 QAT requires BF16 parameter dtype")
                    if selected_qat == "nvfp4" and selected_backend == "native" and not native_nvfp4:
                        raise ValueError(f"native NVFP4 backend unavailable: {native_nvfp4_note}")
                    if (
                        selected_sparsity != "none"
                        and selected_sparse_backend == "compressed"
                        and selected_backend == "emulated"
                    ):
                        raise ValueError("compressed sparse execution requires native NVFP4")
                    if (
                        selected_sparsity != "none"
                        and selected_sparse_backend == "compressed"
                        and not native_sparse_nvfp4
                    ):
                        raise ValueError(
                            f"compressed sparse NVFP4 unavailable: {native_sparse_nvfp4_note}"
                        )
                    if selected_backward == "mxfp8" and selected_qat != "nvfp4":
                        raise ValueError("MXFP8 backward requires NVFP4 QAT")
                    if (
                        selected_backward == "mxfp8"
                        and selected_backward_backend == "native"
                        and not native_mxfp8
                    ):
                        raise ValueError(f"native MXFP8 unavailable: {native_mxfp8_note}")
                    if (
                        selected_backward == "mxfp8"
                        and selected_backward_backend == "native"
                        and parameter_dtype.value != "bf16"
                    ):
                        raise ValueError("native MXFP8 backward requires BF16 parameters")
                    selected_master = str(master_weight_storage.value or "fp32")
                    selected_cpu_offload = (
                        optimizer_state.value == "cpu_offload"
                        or optimizer_impl.value == "cpu_offload_adamw"
                    )
                    if selected_master != "fp32" and not selected_cpu_offload:
                        raise ValueError("FP8 persistent masters require CPU-offload AdamW")
                    if (
                        selected_master == "fp8_e4m3_adaptive"
                        and float(adaptive_demote_threshold.value or 0)
                        > float(adaptive_promote_threshold.value or 0)
                    ):
                        raise ValueError(
                            "adaptive demotion threshold cannot exceed promotion threshold"
                        )
                except ValueError as exc:
                    ui.notify(str(exc), type="negative", multi_line=True)
                    return
                architecture = spec.to_dict()
                resolved_method = "full"
                resolved_quant = str(parameter_dtype.value or "bf16")
                initialization = {"mode": "scratch", "tokenizer_ref": tokenizer_ref}
                stats_key = tokenizer_stats_key(spec.family)
                base_name = (output_name.value or "").strip() or (
                    f"{spec.family}-{round(float(target_params_m.value or 0))}m-scratch"
                )
            else:
                resolved_method, resolved_quant, auto_plan = resolve_pretrained_method()
                if auto_plan is not None and not auto_plan.feasible:
                    ui.notify(
                        "The automatic continued-training plan does not pass the VRAM preflight. "
                        "Reduce sequence length or choose a smaller model.",
                        type="negative",
                        multi_line=True,
                    )
                    return
                if (
                    resolved_method != "full"
                    and checkpoint["quant_locked"]
                    and checkpoint["quant_locked"] != resolved_quant
                ):
                    ui.notify(
                        f"Checkpoint is quant-locked to {checkpoint['quant_locked']}.",
                        type="negative",
                    )
                    return
                initialization = {
                    "mode": "pretrained",
                    "checkpoint_id": checkpoint["id"],
                    "checkpoint_ref": checkpoint["ref"],
                    "tokenizer_ref": checkpoint["ref"],
                }
                stats_key = tokenizer_stats_key(_checkpoint_shape(checkpoint).get("model_type"))
                base_name = (output_name.value or "").strip() or (f"{checkpoint['name']}-{resolved_method}")

            batch = auto_plan.micro_batch if auto_plan is not None else int(micro_batch.value or 1)
            accumulation = auto_plan.grad_accum if auto_plan is not None else int(grad_accum.value or 1)
            epoch_count = int(epochs.value or 1)
            estimated_steps = estimate_steps(
                selected_corpora,
                domains,
                epoch_count,
                sequence_length,
                batch,
                accumulation,
                stats_key,
            )
            max_steps = int(max_steps_override.value or 0) or estimated_steps
            parameter_type = str(parameter_dtype.value or "bf16")
            if not scratch and resolved_method in {"lora", "qlora"}:
                parameter_type = "bf16"
            precision = {
                "parameter_dtype": parameter_type,
                "qat": str(qat_mode.value or "none") if scratch else "none",
                "qat_group_size": int(qat_group.value or 32),
                "activation_qat": str(activation_qat.value or "none"),
                "qat_backend": str(qat_backend.value or "auto") if scratch else "auto",
                "sparsity": str(sparsity_pattern.value or "none") if scratch else "none",
                "sparsity_backend": (
                    str(sparsity_backend.value or "auto") if scratch else "auto"
                ),
                "backward_precision": (
                    str(backward_precision.value or "bf16") if scratch else "bf16"
                ),
                "backward_backend": (
                    str(backward_backend.value or "auto") if scratch else "auto"
                ),
                "preset": str(training_recipe.value or "custom") if scratch else "custom",
            }
            optimizer = {
                "implementation": str(optimizer_impl.value or "auto"),
                "state": str(optimizer_state.value or "auto"),
                "lr": float(learning_rate.value or 0),
                "weight_decay": float(weight_decay.value or 0),
                "beta1": float(beta1.value or 0.9),
                "beta2": float(beta2.value or 0.999),
                "eps": float(adam_eps.value or 1e-8),
                "master_weight_storage": str(master_weight_storage.value or "fp32"),
                "adaptive_max_promoted_fraction": float(adaptive_max_percent.value or 0)
                / 100.0,
                "adaptive_promote_threshold": float(
                    adaptive_promote_threshold.value or 0
                ),
                "adaptive_demote_threshold": float(adaptive_demote_threshold.value or 0),
                "adaptive_demote_patience": int(adaptive_demote_patience.value or 1),
            }
            scheduler = {
                "name": str(scheduler_name.value or "cosine"),
                "warmup_ratio": float(warmup_ratio.value or 0),
            }
            runtime = {
                "liger_fused_ce": bool(liger_kernels.value),
                "liger_full_kernels": bool(liger_kernels.value),
                "cudnn_sdpa": bool(cudnn_sdpa.value),
                "allow_tf32": bool(allow_tf32.value),
                "torch_compile": bool(torch_compile.value),
                "compile_mode": str(compile_mode.value or "default"),
                "token_stats_key": stats_key,
                "quant": resolved_quant,
                "rank": int(lora_rank.value or 32),
                "alpha": int(lora_rank.value or 32) * 2,
            }
            config = _base_run_config(
                initialization=initialization,
                method=resolved_method,
                architecture=architecture,
                precision=precision,
                optimizer=optimizer,
                scheduler=scheduler,
                corpora=selected_corpora,
                domains=domains,
                seq_len=sequence_length,
                epochs=epoch_count,
                micro_batch=batch,
                grad_accum=accumulation,
                max_steps=max_steps,
                seed=int(seed.value or 0),
                grad_clip=float(grad_clip.value or 0),
                grad_ckpt=bool(grad_checkpointing.value),
                log_every=int(log_every.value or 1),
                save_every=int(save_every.value or 0) or None,
                runtime=runtime,
            )

            output_checkpoint_id: str | None = None
            adapter_id: str | None = None
            if resolved_method == "full":
                output_checkpoint_id = db.new_id("ckpt")
                db.execute(
                    "INSERT INTO checkpoints(id, name, created_at, kind, source, ref) VALUES(?,?,?,?,?,?)",
                    (
                        output_checkpoint_id,
                        base_name,
                        db.now_iso(),
                        "pending",
                        "pending",
                        "",
                    ),
                )
                config["out_checkpoint_id"] = output_checkpoint_id
            else:
                adapter_id = db.new_id("adp")
                db.execute(
                    "INSERT INTO adapters(id, name, created_at, checkpoint_id, arch, quant, rank, "
                    "corpora_json, status) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        adapter_id,
                        base_name,
                        db.now_iso(),
                        checkpoint["id"],
                        "lora",
                        resolved_quant,
                        int(lora_rank.value or 32),
                        json.dumps(selected_corpora),
                        "training",
                    ),
                )
                config["adapter_id"] = adapter_id

            run_id = db.new_id("run")
            db.execute(
                "INSERT INTO runs(id, kind, status, created_at, config_json) VALUES(?,?,?,?,?)",
                (run_id, "base", "queued", db.now_iso(), json.dumps(config)),
            )
            try:
                pid = launch_worker(run_id)
            except Exception as exc:  # noqa: BLE001
                db.execute(
                    "UPDATE runs SET status='failed', finished_at=?, error=? WHERE id=?",
                    (db.now_iso(), str(exc), run_id),
                )
                if output_checkpoint_id:
                    db.execute(
                        "DELETE FROM checkpoints WHERE id=? AND kind='pending'",
                        (output_checkpoint_id,),
                    )
                if adapter_id:
                    db.execute("UPDATE adapters SET status='failed' WHERE id=?", (adapter_id,))
                ui.notify(f"Worker launch failed: {exc}", type="negative", multi_line=True)
                return
            db.execute(
                "UPDATE runs SET metrics_json=? WHERE id=?",
                (json.dumps({"pid": pid}), run_id),
            )
            ui.notify(
                f"Base training started — {base_name} ({max_steps:,} step cap, pid {pid}).",
                type="positive",
            )

        init_mode.on_value_change(lambda _: update_mode())
        preset.on_value_change(lambda _: apply_preset())
        training_recipe.on_value_change(lambda _: apply_training_recipe())
        fit_target.on_click(fit_parameter_target)
        qat_mode.on_value_change(lambda _: update_qat_controls())
        qat_backend.on_value_change(lambda _: update_qat_controls())
        sparsity_pattern.on_value_change(lambda _: update_qat_controls())
        sparsity_backend.on_value_change(lambda _: update_qat_controls())
        backward_precision.on_value_change(lambda _: update_qat_controls())
        backward_backend.on_value_change(lambda _: update_qat_controls())
        master_weight_storage.on_value_change(lambda _: update_qat_controls())
        family.on_value_change(lambda _: update_qat_controls())
        corpora_select.on_value_change(lambda _: update_data_summary())
        domain_filter.on_value_change(lambda _: update_data_summary())
        start_button.on_click(on_start)
        for element in (
            checkpoint_select,
            tokenizer_select,
            vocab_size,
            vocab_multiple,
            tie_embeddings,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            native_context,
            rope_theta,
            initializer_range,
            rms_eps,
            attention_dropout,
            attention_bias,
            train_seq_len,
            method,
            qlora_quant,
            lora_rank,
            parameter_dtype,
            qat_group,
            activation_qat,
            qat_backend,
            sparsity_pattern,
            sparsity_backend,
            backward_precision,
            backward_backend,
            optimizer_state,
            optimizer_impl,
            master_weight_storage,
            adaptive_max_percent,
            adaptive_promote_threshold,
            adaptive_demote_threshold,
            adaptive_demote_patience,
        ):
            element.on_value_change(lambda _: update_preflight())
        update_mode()
        update_qat_controls()
        update_preflight()
