from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from nicegui import ui

from ... import db
from ...ingest.domains import DOMAINS
from ...models.bitsandbytes_capabilities import (
    available_bitsandbytes_quantizations,
    bitsandbytes_compute_precision,
)
from ...models.families import hf_compatible_checkpoints
from ...models.precision_capabilities import (
    available_dense_precisions,
    model_execution_device,
    require_model_precision,
)
from ...tokenizer_registry import list_tokenizers
from ...train.base_config import (
    BASE_RECIPE_VERSION,
    FAMILY_LABELS,
    QAT_LABELS,
    SPARSITY_LABELS,
    TRAINING_RECIPE_PRESETS,
    ScratchModelSpec,
    estimate_parameter_count,
    fit_scratch_to_parameter_target,
    projected_training_memory_gb,
    resolve_training_token_budget,
    tokenizer_embedding_parameter_floor,
)
from ...train.compact_tokenizer import CompactTokenizerSpec
from ...train.data import (
    CORPUS_MIX_CONCATENATE,
    CORPUS_MIX_ROUND_ROBIN,
    CORPUS_MIX_TOKEN_BALANCED,
    OBJECTIVE_CAUSAL_LM,
    OBJECTIVE_CHAT_SFT,
    corpus_training_preflight,
    estimate_steps,
    normalize_corpus_mix,
    normalize_source_splits,
    normalize_training_objective,
    tokenizer_stats_key,
)
from ...train.loop import optimizer_capability_status
from ...train.low_precision import (
    native_nvfp4_status,
    native_sparse_nvfp4_mxfp8_status,
    native_sparse_nvfp4_status,
)
from ...train.mxfp8 import native_mxfp8_status
from ...workers.launch import launch_worker
from ..components import field_grid, form_section, hint
from ..shell import ACCENT, panel, shell
from .train_mole import (
    _auto_method_quant,
    _checkpoint_qat_format,
    _checkpoint_shape,
    _make_plan,
    corpus_options,
    render_plan,
)

TOKENIZER_SOURCE_EXISTING = "existing"
TOKENIZER_SOURCE_COMPACT = "compact"
DEFAULT_BASE_SOURCE_SPLITS = ("train", "unspecified")
CHAT_SFT_SHUFFLE_BUFFER = 8192
CHAT_SFT_SHUFFLE_BUFFER_BYTES = 256 * 2**20
COMPACT_VOCAB_MINIMUM = 257
COMPACT_CHAT_VOCAB_MINIMUM = 264

_OPTIMIZER_LABELS = {
    "torch_adamw": "Torch AdamW (parameter-dtype moments)",
    "fused_adamw": "Fused AdamW (parameter-dtype GPU moments)",
    "adamw_8bit": "bitsandbytes AdamW8bit (INT8 GPU moments)",
    "paged_adamw_8bit": "Paged AdamW8bit (INT8; large moments CPU-paged)",
    "cpu_offload_adamw": "CPU-offload AdamW (FP32 CPU master + moments)",
}


def _available_optimizer_options(
    status: Mapping[str, tuple[bool, str]],
) -> dict[str, str]:
    options = {"auto": "Auto (execution-verified only)"}
    options.update(
        {
            implementation: label
            for implementation, label in _OPTIMIZER_LABELS.items()
            if status.get(implementation, (False, ""))[0]
        }
    )
    return options


def _validate_family_qat_selection(family: str, qat: str) -> None:
    if str(family).strip().casefold() == "bitnet" and qat != "none":
        raise ValueError("BitNet supports dense precision only; clear NVFP4 QAT")


def _compatible_optimizer_state_options(
    implementation: str,
    status: Mapping[str, tuple[bool, str]],
    *,
    cuda_available: bool,
) -> dict[str, str]:
    if implementation == "paged_adamw_8bit":
        return {"cpu_paged": "INT8 state (large buffers CPU-paged)"}
    if implementation == "cpu_offload_adamw":
        return {"cpu_offload": "Full FP32 CPU offload"}
    if implementation in {"torch_adamw", "fused_adamw", "adamw_8bit"}:
        return (
            {"gpu": "GPU-resident state"}
            if cuda_available
            else {"auto": "CPU-local state"}
        )
    options = {"auto": "Auto verified placement"}
    if cuda_available and any(
        status.get(candidate, (False, ""))[0]
        for candidate in ("torch_adamw", "fused_adamw", "adamw_8bit")
    ):
        options["gpu"] = "GPU-resident state"
    if status.get("paged_adamw_8bit", (False, ""))[0]:
        options["cpu_paged"] = "INT8 state (large buffers CPU-paged)"
    if status.get("cpu_offload_adamw", (False, ""))[0]:
        options["cpu_offload"] = "Full FP32 CPU offload"
    return options

SCRATCH_PRESETS: dict[str, dict[str, int]] = {
    "qwen_1m_compact": {
        "compact_tokenizer": 1,
        "compact_vocab_size": 2048,
        "vocab_size": 0,
        "vocab_padding_multiple": 64,
        "hidden_size": 64,
        "intermediate_size": 192,
        "num_hidden_layers": 18,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "target_params_m": 1,
    },
    "qwen_40m": {
        "vocab_size": 0,
        "hidden_size": 512,
        "intermediate_size": 1536,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
    },
    "qwen_100m": {
        "vocab_size": 0,
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "head_dim": 64,
    },
    "qwen_250m": {
        "vocab_size": 0,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 18,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 64,
    },
    "qwen_600m": {
        "vocab_size": 0,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 64,
    },
}


def _scratch_initialization(
    source: str,
    *,
    tokenizer_ref: str = "",
    compact_tokenizer: CompactTokenizerSpec | dict | None = None,
) -> dict:
    """Build a validated, replayable scratch-tokenizer initialization."""

    mode = str(source or TOKENIZER_SOURCE_EXISTING)
    if mode == TOKENIZER_SOURCE_COMPACT:
        spec = (
            compact_tokenizer
            if isinstance(compact_tokenizer, CompactTokenizerSpec)
            else CompactTokenizerSpec.from_dict(compact_tokenizer)
        )
        spec.validate()
        return {"mode": "scratch", "compact_tokenizer": spec.to_dict()}
    if mode != TOKENIZER_SOURCE_EXISTING:
        raise ValueError(f"unknown scratch tokenizer source: {mode}")
    ref = str(tokenizer_ref or "").strip()
    if not ref:
        raise ValueError("existing scratch tokenizer source requires a tokenizer reference")
    return {"mode": "scratch", "tokenizer_ref": ref}


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
    config = _checkpoint_config(row)
    tokenizer_vocab = int(config.get("vocab_size") or 0)
    # An explicit model vocabulary is not permission to undercut a known
    # tokenizer.  It is only a preview fallback when tokenizer metadata is
    # unavailable; the worker validates against len(tokenizer) again.
    return tokenizer_vocab or int(explicit_vocab or 0)


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


def _parameter_target_slug(target_parameters: int) -> str:
    """Format a requested size without collapsing distinct small targets."""

    target = int(target_parameters)
    if target < 1:
        raise ValueError("target parameter count must be positive")
    if target >= 100_000:
        whole, remainder = divmod(target, 1_000_000)
        fraction = f".{remainder:06d}".rstrip("0") if remainder else ""
        return f"{whole}{fraction}m"
    if target >= 1_000:
        whole, remainder = divmod(target, 1_000)
        fraction = f".{remainder:03d}".rstrip("0") if remainder else ""
        return f"{whole}{fraction}k"
    return str(target)


def _resolve_parameter_target(target_millions: float, exact_parameters: int = 0) -> int:
    """Resolve the exact integer target, with the raw field taking precedence."""

    exact = int(exact_parameters or 0)
    if exact < 0:
        raise ValueError("exact target parameter count cannot be negative")
    target = exact or round(float(target_millions or 0) * 1_000_000)
    if target < 1:
        raise ValueError("target parameter count must be positive")
    return target


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
    objective: str = OBJECTIVE_CAUSAL_LM,
    requested_training_tokens: int = 0,
    step_cap_source: str | None = None,
    splits: object = None,
    target_parameters: int | None = None,
    corpus_mix: str = CORPUS_MIX_ROUND_ROBIN,
    cross_corpus_dedup: bool = True,
    source_max_ratio: float | None = None,
    corpus_weights: Mapping[str, float] | None = None,
) -> dict:
    """Build the current base recipe independently of NiceGUI widgets."""

    budget = resolve_training_token_budget(
        requested_tokens=requested_training_tokens,
        seq_len=seq_len,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        max_steps=max_steps,
    )
    if step_cap_source:
        budget["step_cap_source"] = step_cap_source
    normalized_objective = normalize_training_objective(objective)
    selected_corpora = _ordered_corpora(corpora)
    normalized_mix = normalize_corpus_mix(corpus_mix)
    effective_source_cap = (
        float(source_max_ratio)
        if normalized_mix == CORPUS_MIX_TOKEN_BALANCED
        and len(selected_corpora) > 1
        and source_max_ratio is not None
        and source_max_ratio > 0
        else None
    )
    config = {
        "recipe_version": BASE_RECIPE_VERSION,
        "initialization": dict(initialization),
        "method": method,
        "objective": normalized_objective,
        "precision": dict(precision),
        "optimizer": dict(optimizer),
        "scheduler": dict(scheduler),
        "corpora": selected_corpora,
        "corpus_mix": normalized_mix,
        "cross_corpus_dedup": bool(cross_corpus_dedup),
        "source_max_ratio": effective_source_cap,
        "corpus_weights": {
            str(corpus_id): float(weight) for corpus_id, weight in (corpus_weights or {}).items()
        },
        "domains": list(domains) if domains else None,
        "splits": normalize_source_splits(splits) or None,
        "seq_len": int(seq_len),
        "epochs": int(epochs),
        "micro_batch": int(micro_batch),
        "grad_accum": int(grad_accum),
        "max_steps": int(budget["max_steps"]),
        "seed": int(seed),
        "grad_clip": float(grad_clip),
        "grad_ckpt": bool(grad_ckpt),
        "log_every": int(log_every),
        "save_every": int(save_every) if save_every else None,
        "lr": float(optimizer["lr"]),
        "optimizer_state": str(optimizer["state"]),
        **runtime,
        **budget,
    }
    if architecture is not None:
        config["architecture"] = dict(architecture)
    if target_parameters is not None:
        config["target_parameters"] = int(target_parameters)
    if normalized_objective == OBJECTIVE_CHAT_SFT:
        config.setdefault("shuffle_buffer", CHAT_SFT_SHUFFLE_BUFFER)
        config.setdefault("shuffle_buffer_bytes", CHAT_SFT_SHUFFLE_BUFFER_BYTES)
    checkpoint_id = initialization.get("checkpoint_id")
    checkpoint_ref = initialization.get("checkpoint_ref")
    if checkpoint_id:
        config["checkpoint_id"] = checkpoint_id
    if checkpoint_ref:
        config["checkpoint_ref"] = checkpoint_ref
    return config


@ui.page("/train/base")
def train_base_page():
    model_device = model_execution_device()
    bnb_quantizations = available_bitsandbytes_quantizations(model_device)
    dense_precisions = available_dense_precisions(model_device)
    default_parameter_dtype = next(iter(dense_precisions), None)
    optimizer_status = optimizer_capability_status(
        cuda=model_device.type == "cuda",
        device_index=model_device.index if model_device.type == "cuda" else None,
    )
    optimizer_options = _available_optimizer_options(optimizer_status)
    cuda_available = model_device.type == "cuda"
    default_optimizer_state = (
        "cpu_offload"
        if optimizer_status.get("cpu_offload_adamw", (False, ""))[0]
        else "auto"
    )
    bnb_quantization_options = {
        quantization: (
            f"{quantization.upper()} ({bitsandbytes_compute_precision(quantization).upper()} compute)"
        )
        for quantization in bnb_quantizations
    }
    continued_method_options = {"auto": "Auto"}
    if default_parameter_dtype is not None:
        continued_method_options.update(
            {
                "full": f"Full parameters ({default_parameter_dtype.upper()} default)",
                "lora": f"LoRA ({default_parameter_dtype.upper()} base default)",
            }
        )
    if bnb_quantizations:
        continued_method_options["qlora"] = "QLoRA"
    default_bnb_quantization = bnb_quantizations[0] if bnb_quantizations else None
    parameter_dtype_options = {
        precision: (
            f"{precision.upper()} (recommended)"
            if precision == default_parameter_dtype
            else precision.upper()
        )
        for precision in dense_precisions
    }
    with shell("Train · Base", "/train/base"):
        ui.label("Raw or continued base-model training").classes("text-xl font-semibold")
        ui.label(
            "Initialize a decoder from random weights with a chosen architecture, or continue an existing "
            "checkpoint. Scratch runs train every parameter; LoRA/QLoRA remain continued-training modes."
        ).classes("text-sm text-gray-400 -mt-2")

        checkpoint_rows = {
            row["id"]: row
            for row in hf_compatible_checkpoints(
                db.q("SELECT * FROM checkpoints WHERE kind!='pending' ORDER BY name")
            )
        }
        checkpoint_options = {
            cid: f"{row['name']} ({row['params_b']:.3f} B)" if row["params_b"] else row["name"]
            for cid, row in checkpoint_rows.items()
        }
        first_checkpoint = next(iter(checkpoint_options), None)
        registered_tokenizer_rows = {
            row["id"]: row for row in list_tokenizers(ready_only=True)
        }
        registered_tokenizer_options = {
            tokenizer_id: (
                f"{row['name']} ({int(row.get('vocab_size') or 0):,} tokens)"
            )
            for tokenizer_id, row in registered_tokenizer_rows.items()
        }
        corpora = corpus_options()
        native_nvfp4, native_nvfp4_note = native_nvfp4_status(model_device)
        native_sparse_nvfp4, native_sparse_nvfp4_note = native_sparse_nvfp4_status(
            model_device
        )
        native_mxfp8, native_mxfp8_note = native_mxfp8_status(model_device)
        if native_sparse_nvfp4 and native_mxfp8:
            native_sparse_mxfp8, native_sparse_mxfp8_note = (
                native_sparse_nvfp4_mxfp8_status(model_device)
            )
        else:
            native_sparse_mxfp8 = False
            native_sparse_mxfp8_note = "requires both sparse NVFP4 and MXFP8"
        available_qat_labels = {"none": QAT_LABELS["none"]}
        if native_nvfp4 and "bf16" in dense_precisions:
            available_qat_labels["nvfp4"] = QAT_LABELS["nvfp4"]
        available_sparsity_labels = {"none": SPARSITY_LABELS["none"]}
        if native_sparse_nvfp4:
            available_sparsity_labels["paired_4:8"] = SPARSITY_LABELS["paired_4:8"]
        available_backward_labels = {"bf16": "BF16 dgrad / wgrad"}
        if native_mxfp8:
            available_backward_labels["mxfp8"] = "MXFP8 dgrad / wgrad (native Tensor Cores)"
        available_recipe_labels = {"custom": "Custom native precision recipe"}
        if native_nvfp4 and native_sparse_nvfp4 and "bf16" in dense_precisions:
            available_recipe_labels["nvfp4_paired_4to8"] = (
                "Native NVFP4 + compressed paired 4:8"
            )
        if (
            native_nvfp4
            and native_sparse_mxfp8
            and "bf16" in dense_precisions
            and optimizer_status.get("cpu_offload_adamw", (False, ""))[0]
        ):
            available_recipe_labels["nvfp4_sparse_mxfp8_cpu_offload"] = (
                "Native NVFP4 sparse forward + MXFP8 backward + FP32 CPU offload"
            )

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
                                continued_method_options,
                                value="auto",
                                label="Update method",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        qlora_quant = (
                            ui.select(
                                bnb_quantization_options,
                                value=default_bnb_quantization,
                                label="QLoRA storage",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        qlora_quant.enabled = bool(bnb_quantizations)
                    if not bnb_quantizations:
                        hint(
                            "QLoRA is omitted because no bitsandbytes NF4/INT8 CUDA "
                            "forward/backward path passed its capability probe."
                        )
                    lora_rank = (
                        ui.number("LoRA rank", value=32, min=1, max=512, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )

                with ui.column().classes("w-full gap-2") as scratch_source_box:
                    tokenizer_source = (
                        ui.select(
                            {
                                TOKENIZER_SOURCE_EXISTING: "Use existing tokenizer",
                                TOKENIZER_SOURCE_COMPACT: "Train compact tokenizer from selected corpora",
                            },
                            value=TOKENIZER_SOURCE_EXISTING,
                            label="Scratch tokenizer source",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    with ui.column().classes("w-full gap-2") as existing_tokenizer_box:
                        registered_tokenizer_select = (
                            ui.select(
                                registered_tokenizer_options,
                                value=None,
                                label="Registered tokenizer artifact",
                            )
                            .props("dense outlined clearable")
                            .classes("w-full")
                        )
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
                                placeholder="Optional; overrides tokenizer artifact and checkpoint",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        hint(
                            "Choose a reusable tokenizer artifact, retain the legacy checkpoint tokenizer "
                            "source, or enter a direct path/HuggingFace id. Only the tokenizer is loaded; "
                            "model weights are random and model vocabulary cannot be smaller than it."
                        )
                    with ui.column().classes("w-full gap-2") as compact_tokenizer_box:
                        with field_grid(4):
                            compact_vocab_size = (
                                ui.number(
                                    "Compact vocabulary",
                                    value=2048,
                                    min=COMPACT_VOCAB_MINIMUM,
                                    max=1_000_000,
                                    precision=0,
                                )
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            compact_min_frequency = (
                                ui.number(
                                    "Minimum token frequency",
                                    value=2,
                                    min=1,
                                    precision=0,
                                )
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            compact_max_documents = (
                                ui.number(
                                    "Tokenizer document cap",
                                    value=100_000,
                                    min=1,
                                    precision=0,
                                )
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            compact_lowercase = ui.switch("Lowercase tokenizer training", value=False)
                        hint(
                            "The worker trains a deterministic byte-level BPE tokenizer from the selected "
                            "corpora before model construction. Complete byte coverage keeps code, math, and "
                            "multilingual UTF-8 representable. The minimum is 257 for causal LM and 264 "
                            "when chat-role tokens are required; 2048 is only the default, not a fixed size."
                        )

                ui.separator().classes("my-3")
                with ui.column().classes("w-full gap-2") as architecture_box:
                    form_section("2", "Scratch architecture")
                    with field_grid(4):
                        preset = (
                            ui.select(
                                {
                                    "custom": "Custom",
                                    "qwen_1m_compact": "~1M @ 2K compact vocab",
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
                            ui.number(
                                "Target params (M)",
                                value=100,
                                min=0.1,
                                max=100000,
                                step=0.001,
                                precision=3,
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        exact_target_params = (
                            ui.number(
                                "Exact target parameters (0 = M field)",
                                value=0,
                                min=0,
                                precision=0,
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    fit_target = ui.button("Fit architecture to target", icon="calculate").props(
                        "dense flat no-caps"
                    )
                    hint(
                        "Enter any target from 0.100M upward (for example 0.5M, 3M, 12M, or 27M), "
                        "or use the exact integer field for arbitrary counts. "
                        "The deterministic fitter may change width, MLP width, heads, KV heads, and depth. "
                        "It never reduces model vocabulary below the tokenizer; the exact embedding floor "
                        "and achieved parameter count are shown before launch."
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
                training_objective = (
                    ui.select(
                        {
                            OBJECTIVE_CAUSAL_LM: "Causal LM (pretrain / continued pretrain)",
                            OBJECTIVE_CHAT_SFT: "Chat SFT (assistant turns only)",
                        },
                        value=OBJECTIVE_CAUSAL_LM,
                        label="Training objective",
                    )
                    .props("dense outlined")
                    .classes("w-full")
                )
                hint(
                    "Chat SFT expects line-start system:, user:, and assistant: blocks. It renders each "
                    "transcript with the selected tokenizer's chat template and masks non-assistant tokens."
                )
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
                source_split_filter = (
                    ui.select(
                        {
                            "train": "Train",
                            "validation": "Validation",
                            "test": "Test",
                            "unspecified": "Unspecified / legacy",
                        },
                        multiple=True,
                        value=list(DEFAULT_BASE_SOURCE_SPLITS),
                        label="Source splits",
                    )
                    .props("dense outlined use-chips clearable options-dense")
                    .classes("w-full")
                )
                hint(
                    "Train + unspecified is the safe default: explicit validation/test shards are "
                    "excluded, while older corpora without split metadata remain usable. Clear all "
                    "choices to include every split."
                )
                with field_grid(2):
                    corpus_mix_strategy = (
                        ui.select(
                            {
                                CORPUS_MIX_TOKEN_BALANCED: "Token-balanced (recommended)",
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
                            "Maximum single-corpus share (0 = uncapped)",
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
                    "Deduplicate exact model-visible records across all selected corpora",
                    value=True,
                )
                hint(
                    "Token-balanced scheduling accounts for document size, so one corpus of giant assets "
                    "cannot dominate a capped run. The share cap stops a lone remaining source once its "
                    "character budget is reached; 0 disables the cap."
                )
                data_summary = (
                    ui.label("0 corpora selected · choose one or more.")
                    .classes("text-xs text-gray-400 px-3 py-2 rounded-md w-full")
                    .style(f"background: {ACCENT}10")
                )
                data_preflight_message = ui.label("").classes(
                    "text-xs text-amber-400 whitespace-pre-wrap"
                )
                data_preflight_message.visible = False
                with field_grid(4):
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
                    requested_training_tokens = (
                        ui.number(
                            "Training token budget (0 = steps / epochs)",
                            value=0,
                            min=0,
                            precision=0,
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    max_steps_override = (
                        ui.number("Max optimizer steps (0 = data)", value=0, min=0, precision=0)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                training_budget_summary = ui.label("").classes("text-xs text-amber-400")
                hint(
                    "Native model context is architectural capacity. The training sequence length is what the "
                    "run actually exercises; it must not exceed native context for scratch models. A nonzero "
                    "token budget overrides the step field and epoch-derived cap, rounding up to a whole "
                    "optimizer step."
                )

                ui.separator().classes("my-3")
                form_section("4", "Parameter state and QAT numerics")
                training_recipe = (
                    ui.select(
                        available_recipe_labels,
                        value="custom",
                        label="Training recipe preset",
                    )
                    .props("dense outlined")
                    .classes("w-full")
                )
                with field_grid(3):
                    parameter_dtype = (
                        ui.select(
                            parameter_dtype_options,
                            value=default_parameter_dtype,
                            label="Parameter dtype",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    qat_mode = (
                        ui.select(
                            available_qat_labels,
                            value="none",
                            label="Forward compute precision",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    sparsity_pattern = (
                        ui.select(
                            available_sparsity_labels,
                            value="none",
                            label="Native weight sparsity",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                with field_grid(1):
                    backward_precision = (
                        ui.select(
                            available_backward_labels,
                            value="bf16",
                            label="Backward GEMMs",
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
                ui.label(
                    "Sparse NVFP4 + MXFP8 combined lane: "
                    f"{'available' if native_sparse_mxfp8 else 'unavailable'} — "
                    f"{native_sparse_mxfp8_note}."
                ).classes("text-xs text-gray-500")
                ui.label(
                    "Unsupported precision modes are omitted. yyLab never substitutes emulated "
                    "quantization or a dense masked sparse path."
                ).classes("text-xs text-amber-400")

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
                            optimizer_options,
                            value="auto",
                            label="Optimizer / state precision",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    optimizer_state = (
                        ui.select(
                            _compatible_optimizer_state_options(
                                "auto",
                                optimizer_status,
                                cuda_available=cuda_available,
                            ),
                            value=default_optimizer_state,
                            label="Optimizer state",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                ui.label(
                    "Only implementations that completed a real optimizer update on this system are shown. "
                    "Full CPU offload uses an FP32 master and FP32 Adam moments."
                ).classes("text-xs text-amber-400")
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
                        "Parameter dtype controls the saved trainable tensors. Native low-precision selection "
                        "controls Tensor Core forward/backward execution. Optimizer state controls VRAM/RAM placement."
                    ).classes("text-sm text-gray-300")
                    ui.label(
                        "NVFP4, compressed paired 4:8, and MXFP8 are exposed only when this GPU and runtime "
                        "support their native kernels. Unsupported formats fail closed; no emulation is used."
                    ).classes("text-xs text-amber-400 mt-2")

        def selected_checkpoint() -> dict | None:
            return checkpoint_rows.get(checkpoint_select.value)

        def selected_tokenizer_checkpoint() -> dict | None:
            return checkpoint_rows.get(tokenizer_select.value)

        def selected_registered_tokenizer() -> dict | None:
            return registered_tokenizer_rows.get(registered_tokenizer_select.value)

        def existing_tokenizer_reference() -> str:
            override = str(tokenizer_override.value or "").strip()
            if override:
                return override
            registered = selected_registered_tokenizer()
            if registered:
                return str(registered["id"])
            return str((selected_tokenizer_checkpoint() or {}).get("ref") or "")

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

        def current_compact_tokenizer_spec() -> CompactTokenizerSpec:
            objective = normalize_training_objective(training_objective.value)
            spec = CompactTokenizerSpec(
                vocab_size=int(compact_vocab_size.value or 0),
                min_frequency=int(compact_min_frequency.value or 0),
                max_documents=int(compact_max_documents.value or 0),
                lowercase=bool(compact_lowercase.value),
                special_token_profile=(
                    "chat" if objective == OBJECTIVE_CHAT_SFT else "causal"
                ),
            )
            minimum = (
                COMPACT_CHAT_VOCAB_MINIMUM
                if objective == OBJECTIVE_CHAT_SFT
                else COMPACT_VOCAB_MINIMUM
            )
            if spec.vocab_size < minimum:
                raise ValueError(
                    f"compact tokenizer vocabulary must be at least {minimum} for "
                    f"{'chat SFT' if objective == OBJECTIVE_CHAT_SFT else 'causal LM'}"
                )
            return spec

        def requested_parameter_target() -> int:
            return _resolve_parameter_target(
                float(target_params_m.value or 0),
                int(exact_target_params.value or 0),
            )

        def vocab_hint() -> int:
            if tokenizer_source.value == TOKENIZER_SOURCE_COMPACT:
                return int(current_compact_tokenizer_spec().vocab_size)
            override = (tokenizer_override.value or "").strip()
            registered = selected_registered_tokenizer()
            if not override and registered:
                return int(registered.get("vocab_size") or int(vocab_size.value or 0))
            tokenizer_row = (
                {"ref": override, "meta_json": "{}"}
                if override
                else selected_tokenizer_checkpoint()
            )
            return _tokenizer_vocab_hint(
                tokenizer_row,
                int(vocab_size.value or 0),
            )

        def update_tokenizer_source() -> None:
            compact = tokenizer_source.value == TOKENIZER_SOURCE_COMPACT
            existing_tokenizer_box.visible = not compact
            compact_tokenizer_box.visible = compact
            update_preflight()

        def apply_preset() -> None:
            values = SCRATCH_PRESETS.get(str(preset.value or ""))
            if not values:
                return
            family.set_value("qwen3")
            if values.get("compact_tokenizer"):
                tokenizer_source.set_value(TOKENIZER_SOURCE_COMPACT)
                compact_vocab_size.set_value(values.get("compact_vocab_size", 2048))
            vocab_size.set_value(values.get("vocab_size", 0))
            vocab_multiple.set_value(values.get("vocab_padding_multiple", 64))
            hidden_size.set_value(values["hidden_size"])
            intermediate_size.set_value(values["intermediate_size"])
            num_layers.set_value(values["num_hidden_layers"])
            num_heads.set_value(values["num_attention_heads"])
            num_kv_heads.set_value(values["num_key_value_heads"])
            head_dim.set_value(values["head_dim"])
            if values.get("target_params_m"):
                target_params_m.set_value(values["target_params_m"])
                exact_target_params.set_value(0)
            update_preflight()

        def fit_parameter_target() -> None:
            try:
                hint_vocab = vocab_hint()
                if hint_vocab < 1:
                    raise ValueError(
                        "select a registered tokenizer with known vocab or enter a model vocab size"
                    )
                fitted = fit_scratch_to_parameter_target(
                    current_spec(),
                    requested_parameter_target(),
                    tokenizer_vocab_size=hint_vocab,
                    qat_mode=str(qat_mode.value or "none"),
                )
                vocab_size.set_value(fitted.vocab_size)
                hidden_size.set_value(fitted.hidden_size)
                intermediate_size.set_value(fitted.intermediate_size)
                num_layers.set_value(fitted.num_hidden_layers)
                num_heads.set_value(fitted.num_attention_heads)
                num_kv_heads.set_value(fitted.num_key_value_heads)
                head_dim.set_value(fitted.head_dim)
                preset.set_value("custom")
                update_preflight()
            except ValueError as exc:
                ui.notify(str(exc), type="warning")

        def update_optimizer_controls() -> None:
            implementation = str(optimizer_impl.value or "auto")
            options = _compatible_optimizer_state_options(
                implementation,
                optimizer_status,
                cuda_available=cuda_available,
            )
            optimizer_state.options = options
            if optimizer_state.value not in options:
                optimizer_state.set_value(next(iter(options)))
            optimizer_state.update()
            update_preflight()

        def apply_training_recipe() -> None:
            values = TRAINING_RECIPE_PRESETS.get(str(training_recipe.value or "custom"), {})
            if not values:
                return
            parameter_dtype.set_value(str(values["parameter_dtype"]))
            qat_mode.set_value(str(values["qat"]))
            sparsity_pattern.set_value(str(values["sparsity"]))
            backward_precision.set_value(str(values.get("backward_precision", "bf16")))
            optimizer_impl.set_value(str(values.get("optimizer_impl", "auto")))
            update_optimizer_controls()
            requested_state = str(values.get("optimizer_state", "cpu_offload"))
            if requested_state not in optimizer_state.options:
                raise ValueError(
                    f"training preset requests unavailable optimizer state {requested_state!r}"
                )
            optimizer_state.set_value(requested_state)
            update_qat_controls()

        def update_mode() -> None:
            scratch = init_mode.value == "scratch"
            scratch_source_box.visible = scratch
            architecture_box.visible = scratch
            continued_box.visible = not scratch
            if scratch:
                update_tokenizer_source()
            qat_mode.enabled = scratch
            training_recipe.enabled = scratch
            sparsity_pattern.enabled = scratch
            backward_precision.enabled = scratch
            update_qat_controls()
            update_preflight()
            update_training_budget_summary()

        def update_qat_controls() -> None:
            qat = str(qat_mode.value or "none") if init_mode.value == "scratch" else "none"
            bitnet = family.value == "bitnet"
            if qat != "nvfp4" and sparsity_pattern.value != "none":
                sparsity_pattern.set_value("none")
            if qat != "nvfp4" and backward_precision.value != "bf16":
                backward_precision.set_value("bf16")
            qat_mode.enabled = init_mode.value == "scratch"
            parameter_dtype.enabled = True
            sparsity_pattern.enabled = (
                qat == "nvfp4"
                and native_sparse_nvfp4
                and init_mode.value == "scratch"
            )
            backward_precision.enabled = qat == "nvfp4" and init_mode.value == "scratch"
            if bitnet and qat != "none":
                qat_truth.text = (
                    "BitNet supports dense precision only. The current NVFP4 selection is "
                    "invalid and must be cleared before launch."
                )
            elif qat == "none":
                qat_truth.text = (
                    "Dense forward numerics; parameter and optimizer choices determine state memory."
                )
            elif qat == "nvfp4":
                execution = "Native packed E2M1/E4M3 block-scaled FP4 forward GEMMs are enabled; "
                sparse = str(sparsity_pattern.value or "none")
                if sparse != "none":
                    sparse_note = (
                        "paired 4:8 weights execute as compressed E2M1 through cuSPARSELt; "
                    )
                else:
                    sparse_note = ""
                backward_mode = str(backward_precision.value or "bf16")
                if backward_mode == "mxfp8":
                    backward_note = "dgrad and wgrad use native block-32 MXFP8 Tensor Core GEMMs; "
                else:
                    backward_note = "dgrad and wgrad use BF16 GEMMs; "
                qat_truth.text = (
                    execution
                    + sparse_note
                    + backward_note
                    + "Optimizer state precision and placement follow the verified optimizer selection below."
                )
            update_preflight()

        def update_data_summary() -> None:
            selected = _ordered_corpora(corpora_select.value)
            selected_splits = normalize_source_splits(source_split_filter.value)
            if not selected:
                split_label = ", ".join(selected_splits) if selected_splits else "all"
                data_summary.text = (
                    f"0 corpora selected · splits: {split_label} · choose one or more."
                )
                data_preflight_message.text = ""
                data_preflight_message.visible = False
                update_training_budget_summary()
                return
            domains = sorted(set(domain_filter.value or []))
            mix = str(corpus_mix_strategy.value or CORPUS_MIX_TOKEN_BALANCED)
            cap = (
                float(source_ratio_cap.value or 0) or None
                if mix == CORPUS_MIX_TOKEN_BALANCED and len(selected) > 1
                else None
            )
            report = corpus_training_preflight(
                selected,
                domains or None,
                corpus_mix=mix,
                splits=selected_splits,
                cross_corpus_dedup=bool(cross_corpus_dedup.value),
                source_max_ratio=cap,
            )
            provenance = report["provenance"]
            docs = int(provenance["estimated_docs"])
            chars = int(provenance["estimated_chars"])
            catalogued = sum(
                corpus.get("catalog_status") == "cdas_catalogued"
                for corpus in provenance["corpora"]
            )
            domain_tag = f" · domains: {', '.join(domains)}" if domains else ""
            split_tag = f" · splits: {', '.join(selected_splits) if selected_splits else 'all'}"
            corpus_label = "corpus" if len(selected) == 1 else "corpora"
            data_summary.text = (
                f"{len(selected)} {corpus_label} selected{domain_tag}{split_tag} · "
                f"{docs:,} docs · {chars / 1e9:.3f}B characters · "
                f"CDAS {catalogued}/{len(selected)} · {mix.replace('_', ' ')} · "
                f"cross-dedup {'on' if cross_corpus_dedup.value else 'off'}"
            )
            messages = [
                *(f"ERROR: {message}" for message in report["errors"]),
                *(f"WARNING: {message}" for message in report["warnings"]),
            ]
            data_preflight_message.text = "\n".join(messages)
            data_preflight_message.visible = bool(messages)
            update_training_budget_summary()

        def update_mix_controls() -> None:
            source_ratio_cap.enabled = (
                corpus_mix_strategy.value == CORPUS_MIX_TOKEN_BALANCED
                and len(_ordered_corpora(corpora_select.value)) > 1
            )
            update_data_summary()

        def update_training_budget_summary() -> None:
            try:
                selected = _ordered_corpora(corpora_select.value)
                requested = int(requested_training_tokens.value or 0)
                explicit_steps = int(max_steps_override.value or 0)
                batch = int(micro_batch.value or 0)
                accumulation = int(grad_accum.value or 0)
                if init_mode.value != "scratch":
                    _, _, automatic_plan = resolve_pretrained_method()
                    if automatic_plan is not None:
                        batch = int(automatic_plan.micro_batch)
                        accumulation = int(automatic_plan.grad_accum)
                estimated = 0
                if not requested and not explicit_steps:
                    if not selected:
                        training_budget_summary.text = (
                            "Choose training data to estimate the epoch-derived token schedule."
                        )
                        return
                    model_type = (
                        str(family.value or "qwen3")
                        if init_mode.value == "scratch"
                        else str(_checkpoint_shape(selected_checkpoint()).get("model_type") or "")
                    )
                    estimated = estimate_steps(
                        selected,
                        list(domain_filter.value) if domain_filter.value else None,
                        int(epochs.value or 1),
                        int(train_seq_len.value or 0),
                        batch,
                        accumulation,
                        tokenizer_stats_key(model_type),
                        splits=source_split_filter.value,
                    )
                budget = resolve_training_token_budget(
                    requested_tokens=requested,
                    seq_len=int(train_seq_len.value or 0),
                    micro_batch=batch,
                    grad_accum=accumulation,
                    max_steps=explicit_steps,
                    estimated_steps=estimated,
                )
                source = str(budget["step_cap_source"])
                scheduled = int(budget["scheduled_training_tokens"])
                steps = int(budget["max_steps"])
                per_step = int(budget["tokens_per_optimizer_step"])
                if source == "token_budget":
                    rounding = int(budget["token_budget_rounding"])
                    training_budget_summary.text = (
                        f"Requested {requested:,} tokens -> {steps:,} optimizer steps x "
                        f"{per_step:,} = {scheduled:,} scheduled tokens "
                        f"(+{rounding:,} rounded to a whole step). Token budget overrides steps and epochs."
                    )
                elif source == "max_steps":
                    training_budget_summary.text = (
                        f"Explicit cap: {steps:,} optimizer steps x {per_step:,} = "
                        f"{scheduled:,} scheduled tokens."
                    )
                else:
                    training_budget_summary.text = (
                        f"Epoch-derived cap: {steps:,} optimizer steps x {per_step:,} = "
                        f"{scheduled:,} scheduled tokens."
                    )
            except (TypeError, ValueError) as exc:
                training_budget_summary.text = str(exc)

        def set_corpora_selection(value: str | Iterable[str] | None) -> None:
            corpora_select.value = _ordered_corpora(value)
            corpora_select.update()
            update_mix_controls()

        def update_preflight() -> None:
            preflight_box.clear()
            if init_mode.value != "scratch":
                selected_method = str(method.value or "auto")
                requested_quant = (
                    str(qlora_quant.value)
                    if selected_method in {"auto", "qlora"}
                    and qlora_quant.value is not None
                    else str(parameter_dtype.value or "")
                )
                render_plan(
                    preflight_box,
                    selected_checkpoint(),
                    int(train_seq_len.value or 2048),
                    selected_method,
                    requested_quant,
                    rank=int(lora_rank.value or 32),
                    num_experts=1,
                    dense_precision=str(parameter_dtype.value or ""),
                    adapter_only=False,
                    isolated_expert_training=False,
                )
                return
            with preflight_box:
                try:
                    spec = current_spec()
                    if tokenizer_source.value == TOKENIZER_SOURCE_COMPACT:
                        current_compact_tokenizer_spec().validate()
                    hint_vocab = vocab_hint()
                    qat = str(qat_mode.value or "none")
                    sparsity = str(sparsity_pattern.value or "none")
                    backward_mode = str(backward_precision.value or "bf16")
                    if qat == "nvfp4" and not native_nvfp4:
                        raise ValueError(f"native NVFP4 backend unavailable: {native_nvfp4_note}")
                    if qat == "nvfp4" and parameter_dtype.value != "bf16":
                        raise ValueError("native NVFP4 requires BF16 parameter storage")
                    if sparsity != "none" and qat != "nvfp4":
                        raise ValueError("paired 4:8 sparsity requires NVFP4 QAT")
                    if (
                        sparsity != "none"
                        and not native_sparse_nvfp4
                    ):
                        raise ValueError(
                            f"compressed sparse NVFP4 unavailable: {native_sparse_nvfp4_note}"
                        )
                    if backward_mode == "mxfp8" and qat != "nvfp4":
                        raise ValueError("MXFP8 backward requires NVFP4 QAT")
                    if (
                        backward_mode == "mxfp8"
                        and not native_mxfp8
                    ):
                        raise ValueError(f"native MXFP8 unavailable: {native_mxfp8_note}")
                    if (
                        backward_mode == "mxfp8"
                        and parameter_dtype.value != "bf16"
                    ):
                        raise ValueError("native MXFP8 backward requires BF16 parameters")
                    if (
                        sparsity != "none"
                        and backward_mode == "mxfp8"
                        and not native_sparse_mxfp8
                    ):
                        raise ValueError(
                            "sparse NVFP4 + MXFP8 combined path unavailable: "
                            f"{native_sparse_mxfp8_note}"
                        )
                    if hint_vocab < 1:
                        ui.label(
                            "Tokenizer vocab is unknown; enter model vocab or select a registered "
                            "tokenizer so the requested target can be fitted before launch."
                        ).classes("text-amber-400 text-sm")
                        return
                    embedding_floor = tokenizer_embedding_parameter_floor(
                        spec,
                        hint_vocab,
                        qat_mode=qat,
                    )
                    target_parameters = requested_parameter_target()
                    if target_parameters < embedding_floor:
                        raise ValueError(
                            f"Target {target_parameters:,} is below the tokenizer embedding floor "
                            f"{embedding_floor:,}; select a compact tokenizer or increase the target."
                        )
                    spec = fit_scratch_to_parameter_target(
                        spec,
                        target_parameters,
                        tokenizer_vocab_size=hint_vocab,
                        qat_mode=qat,
                    )
                    spec.validate(
                        tokenizer_vocab_size=hint_vocab,
                        train_sequence_length=int(train_seq_len.value or 0),
                        qat_mode=qat,
                    )
                    parameters = estimate_parameter_count(spec, tokenizer_vocab_size=hint_vocab)
                    gpu_gb, cpu_gb = projected_training_memory_gb(
                        parameters,
                        parameter_dtype=str(parameter_dtype.value or ""),
                        optimizer_state=str(optimizer_state.value or "gpu"),
                    )
                    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                        ui.badge(f"{parameters / 1e6:,.3f}M launch-fitted params").props(
                            "color=primary"
                        )
                        ui.badge(
                            f"{spec.num_hidden_layers}L · H{spec.hidden_size} · I{spec.intermediate_size}"
                        ).props("outline color=grey")
                        ui.badge(
                            f"{spec.max_position_embeddings:,} native / {int(train_seq_len.value or 0):,} train"
                        ).props("outline color=teal")
                    ui.label(
                        "Launch deterministically fits the saved architecture to the requested target; "
                        "the Fit button copies this preview into the editable shape fields."
                    ).classes("text-xs text-teal-400")
                    ui.label(
                        f"Tokenizer/model embedding floor: {embedding_floor / 1e6:,.3f}M parameters "
                        f"at the minimum valid width; vocabulary cannot be reduced below {hint_vocab:,}."
                    ).classes("text-xs text-amber-400")
                    if tokenizer_source.value == TOKENIZER_SOURCE_COMPACT:
                        compact_spec = current_compact_tokenizer_spec()
                        ui.label(
                            f"Compact tokenizer will be trained from up to "
                            f"{compact_spec.max_documents:,} selected-corpus documents at "
                            f"{compact_spec.vocab_size:,} vocabulary entries "
                            f"(min frequency {compact_spec.min_frequency}, "
                            f"lowercase={'yes' if compact_spec.lowercase else 'no'})."
                        ).classes("text-xs text-teal-400")
                    ui.label(
                        f"Conservative parameter/gradient/Adam state projection: {gpu_gb:.2f} GB GPU"
                        + (f" + {cpu_gb:.2f} GB CPU" if cpu_gb else "")
                        + ". Activations, CUDA workspaces, and allocator reserve are additional."
                    ).classes("text-xs text-gray-400")
                    if qat != "none":
                        ui.label(
                            f"{qat.upper()} uses native FP4 forward GEMMs"
                            + (
                                " and native MXFP8 backward GEMMs."
                                if backward_mode == "mxfp8"
                                else " and BF16 backward GEMMs."
                            )
                        ).classes("text-xs text-amber-400")
                    if sparsity != "none":
                        ui.label(
                            "Paired 4:8 gives 50% effective weight density. Forward GEMMs use compressed "
                            "cuSPARSELt sparse MMA; saved checkpoints retain BF16 parameters for continued training."
                        ).classes("text-xs text-amber-400")
                except ValueError as exc:
                    ui.label(str(exc)).classes("text-red-400 text-sm")

        def resolve_pretrained_method() -> tuple[str, str, object | None]:
            selected_method = str(method.value or "auto")
            if selected_method != "auto":
                quant = (
                    str(qlora_quant.value or "")
                    if selected_method == "qlora"
                    else str(parameter_dtype.value or "")
                )
                return selected_method, quant, None
            requested_auto_quant = (
                str(qlora_quant.value)
                if qlora_quant.value is not None
                else str(parameter_dtype.value or "")
            )
            plan_method, plan_quant = _auto_method_quant(
                selected_checkpoint(), requested_auto_quant
            )
            plan = _make_plan(
                selected_checkpoint(),
                int(train_seq_len.value or 2048),
                plan_method,
                plan_quant,
                int(lora_rank.value or 32),
                1,
                dense_precision=str(parameter_dtype.value or ""),
                adapter_only=False,
                isolated_expert_training=False,
            )
            if plan is not None:
                return plan.method, plan.quant, plan
            if plan_method == "auto":
                if plan_quant in bnb_quantizations:
                    return "qlora", plan_quant, None
                return "lora", plan_quant, None
            return plan_method, plan_quant, None

        def on_start() -> None:
            problems: list[str] = []
            selected_corpora = _ordered_corpora(corpora_select.value)
            if not selected_corpora:
                problems.append("choose corpora")
            scratch = init_mode.value == "scratch"
            checkpoint = selected_checkpoint()
            tokenizer_ref = existing_tokenizer_reference()
            compact_tokenizer_selected = (
                scratch and tokenizer_source.value == TOKENIZER_SOURCE_COMPACT
            )
            if not parameter_dtype.value:
                problems.append("choose an available parameter dtype")
            if scratch and not compact_tokenizer_selected and not tokenizer_ref:
                problems.append("choose or enter a tokenizer source")
            if not scratch and not checkpoint:
                problems.append("choose a checkpoint")
            if problems:
                ui.notify("Fix: " + ", ".join(problems), type="warning")
                return

            sequence_length = int(train_seq_len.value or 0)
            domains = list(domain_filter.value) if domain_filter.value else None
            source_splits = normalize_source_splits(source_split_filter.value)
            architecture: dict | None = None
            auto_plan = None
            inherited_qat_format = ""
            if scratch:
                spec = current_spec()
                try:
                    compact_spec = (
                        current_compact_tokenizer_spec().validate()
                        if compact_tokenizer_selected
                        else None
                    )
                    tokenizer_vocab = vocab_hint()
                    if tokenizer_vocab < 1:
                        raise ValueError(
                            "tokenizer vocabulary is unknown; enter model vocab or select a "
                            "registered tokenizer so yyLab can fit the requested parameter target"
                        )
                    selected_qat = str(qat_mode.value or "none")
                    _validate_family_qat_selection(str(family.value or ""), selected_qat)
                    target_parameters = requested_parameter_target()
                    spec = fit_scratch_to_parameter_target(
                        spec,
                        target_parameters,
                        tokenizer_vocab_size=tokenizer_vocab,
                        qat_mode=selected_qat,
                    )
                    spec.validate(
                        tokenizer_vocab_size=tokenizer_vocab,
                        train_sequence_length=sequence_length,
                        qat_mode=selected_qat,
                    )
                    selected_sparsity = str(sparsity_pattern.value or "none")
                    selected_backward = str(backward_precision.value or "bf16")
                    if (
                        selected_qat == "nvfp4"
                        and str(parameter_dtype.value or "") != "bf16"
                    ):
                        raise ValueError("native NVFP4 QAT requires BF16 parameter dtype")
                    if selected_qat == "nvfp4" and not native_nvfp4:
                        raise ValueError(f"native NVFP4 backend unavailable: {native_nvfp4_note}")
                    if (
                        selected_sparsity != "none"
                        and not native_sparse_nvfp4
                    ):
                        raise ValueError(
                            f"compressed sparse NVFP4 unavailable: {native_sparse_nvfp4_note}"
                        )
                    if selected_backward == "mxfp8" and selected_qat != "nvfp4":
                        raise ValueError("MXFP8 backward requires NVFP4 QAT")
                    if (
                        selected_backward == "mxfp8"
                        and not native_mxfp8
                    ):
                        raise ValueError(f"native MXFP8 unavailable: {native_mxfp8_note}")
                    if (
                        selected_backward == "mxfp8"
                        and parameter_dtype.value != "bf16"
                    ):
                        raise ValueError("native MXFP8 backward requires BF16 parameters")
                    if (
                        selected_sparsity != "none"
                        and selected_backward == "mxfp8"
                        and not native_sparse_mxfp8
                    ):
                        raise ValueError(
                            "sparse NVFP4 + MXFP8 combined path unavailable: "
                            f"{native_sparse_mxfp8_note}"
                        )
                except ValueError as exc:
                    ui.notify(str(exc), type="negative", multi_line=True)
                    return
                architecture = spec.to_dict()
                resolved_method = "full"
                resolved_quant = str(parameter_dtype.value or "")
                initialization = _scratch_initialization(
                    TOKENIZER_SOURCE_COMPACT
                    if compact_tokenizer_selected
                    else TOKENIZER_SOURCE_EXISTING,
                    tokenizer_ref=tokenizer_ref,
                    compact_tokenizer=compact_spec,
                )
                stats_key = tokenizer_stats_key(spec.family)
                base_name = (output_name.value or "").strip() or (
                    f"{spec.family}-{_parameter_target_slug(requested_parameter_target())}-scratch"
                )
            else:
                try:
                    resolved_method, resolved_quant, auto_plan = resolve_pretrained_method()
                except ValueError as exc:
                    ui.notify(str(exc), type="negative", multi_line=True)
                    return
                inherited_qat_format = _checkpoint_qat_format(checkpoint)
                if inherited_qat_format and resolved_method == "qlora":
                    ui.notify(
                        f"This base restores {inherited_qat_format} QAT numerics; QLoRA would "
                        "apply an incompatible second quantizer.",
                        type="negative",
                        multi_line=True,
                    )
                    return
                if inherited_qat_format and resolved_quant != "bf16":
                    ui.notify(
                        f"Restoring {inherited_qat_format} QAT requires BF16 shadow loading; "
                        f"{resolved_quant.upper()} was requested.",
                        type="negative",
                        multi_line=True,
                    )
                    return
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

            try:
                require_model_precision(resolved_quant, model_device)
            except (RuntimeError, ValueError) as exc:
                ui.notify(str(exc), type="negative", multi_line=True)
                return

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
                splits=source_splits,
            )
            try:
                budget = resolve_training_token_budget(
                    requested_tokens=int(requested_training_tokens.value or 0),
                    seq_len=sequence_length,
                    micro_batch=batch,
                    grad_accum=accumulation,
                    max_steps=int(max_steps_override.value or 0),
                    estimated_steps=estimated_steps,
                )
            except ValueError as exc:
                ui.notify(str(exc), type="negative", multi_line=True)
                return
            max_steps = int(budget["max_steps"])
            parameter_type = str(parameter_dtype.value or "")
            if not scratch and resolved_method == "lora":
                parameter_type = resolved_quant
            elif not scratch and resolved_method == "qlora":
                parameter_type = bitsandbytes_compute_precision(resolved_quant)
            precision = {
                "parameter_dtype": parameter_type,
                "qat": str(qat_mode.value or "none") if scratch else "none",
                "inherit_checkpoint_low_precision": bool(inherited_qat_format),
                "qat_group_size": 16,
                "activation_qat": (
                    "nvfp4" if scratch and qat_mode.value == "nvfp4" else "none"
                ),
                "qat_backend": (
                    "native" if scratch and qat_mode.value == "nvfp4" else "auto"
                ),
                "sparsity": str(sparsity_pattern.value or "none") if scratch else "none",
                "sparsity_backend": (
                    "compressed"
                    if scratch and sparsity_pattern.value != "none"
                    else "auto"
                ),
                "backward_precision": (
                    str(backward_precision.value or "bf16")
                    if scratch and qat_mode.value == "nvfp4"
                    else parameter_type
                ),
                "backward_backend": (
                    "native"
                    if scratch and backward_precision.value == "mxfp8"
                    else "auto"
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
                objective=str(training_objective.value or OBJECTIVE_CAUSAL_LM),
                requested_training_tokens=int(budget["requested_training_tokens"]),
                step_cap_source=str(budget["step_cap_source"]),
                splits=source_splits,
                target_parameters=(requested_parameter_target() if scratch else None),
                corpus_mix=str(corpus_mix_strategy.value or CORPUS_MIX_TOKEN_BALANCED),
                cross_corpus_dedup=bool(cross_corpus_dedup.value),
                source_max_ratio=float(source_ratio_cap.value or 0) or None,
            )
            data_check = corpus_training_preflight(
                config["corpora"],
                config.get("domains"),
                corpus_mix=config["corpus_mix"],
                splits=config.get("splits"),
                cross_corpus_dedup=config["cross_corpus_dedup"],
                corpus_weights=config.get("corpus_weights") or None,
                source_max_ratio=config.get("source_max_ratio"),
            )
            if data_check["errors"]:
                ui.notify(
                    "; ".join(data_check["errors"]),
                    type="negative",
                    multi_line=True,
                )
                return
            config["data_preflight_warnings"] = list(data_check["warnings"])

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
                f"Base training started — {base_name} ({max_steps:,} steps, "
                f"{int(config['scheduled_training_tokens']):,} scheduled tokens, pid {pid}).",
                type="positive",
            )

        init_mode.on_value_change(lambda _: update_mode())
        tokenizer_source.on_value_change(lambda _: update_tokenizer_source())
        training_objective.on_value_change(lambda _: update_preflight())
        preset.on_value_change(lambda _: apply_preset())
        training_recipe.on_value_change(lambda _: apply_training_recipe())
        fit_target.on_click(fit_parameter_target)
        qat_mode.on_value_change(lambda _: update_qat_controls())
        sparsity_pattern.on_value_change(lambda _: update_qat_controls())
        backward_precision.on_value_change(lambda _: update_qat_controls())
        optimizer_impl.on_value_change(lambda _: update_optimizer_controls())
        family.on_value_change(lambda _: update_qat_controls())
        corpora_select.on_value_change(lambda _: update_mix_controls())
        domain_filter.on_value_change(lambda _: update_data_summary())
        source_split_filter.on_value_change(lambda _: update_data_summary())
        corpus_mix_strategy.on_value_change(lambda _: update_mix_controls())
        source_ratio_cap.on_value_change(lambda _: update_data_summary())
        cross_corpus_dedup.on_value_change(lambda _: update_data_summary())
        start_button.on_click(on_start)
        for element in (
            checkpoint_select,
            registered_tokenizer_select,
            tokenizer_select,
            tokenizer_override,
            compact_vocab_size,
            compact_min_frequency,
            compact_max_documents,
            compact_lowercase,
            target_params_m,
            exact_target_params,
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
            sparsity_pattern,
            backward_precision,
            optimizer_state,
            optimizer_impl,
        ):
            element.on_value_change(lambda _: update_preflight())
        for element in (
            epochs,
            requested_training_tokens,
            max_steps_override,
            train_seq_len,
            micro_batch,
            grad_accum,
            method,
            checkpoint_select,
            family,
        ):
            element.on_value_change(lambda _: update_training_budget_summary())
        update_mode()
        update_qat_controls()
        update_optimizer_controls()
        update_mix_controls()
        update_training_budget_summary()
        update_preflight()
