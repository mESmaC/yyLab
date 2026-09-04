from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import orjson
from nicegui import ui

from ... import db
from ...hw import best_gpu, has_triton, ram_total_gb
from ...ingest.domains import DOMAINS
from ...models.bitsandbytes_capabilities import (
    BITSANDBYTES_QUANTIZATIONS,
    available_bitsandbytes_quantizations,
    bitsandbytes_compute_precision,
)
from ...models.families import checkpoint_family, hf_compatible_checkpoints
from ...models.precision_capabilities import (
    DENSE_MODEL_PRECISIONS,
    available_dense_precisions,
    model_execution_device,
    require_model_precision,
)
from ...mole.adaptive import AdaptiveOccupancyConfig, solve_expert_budget
from ...mole.inject import DEFAULT_TARGET_MODULES
from ...train.data import DEFAULT_TEXT_CHUNK_CHARS, estimate_steps, tokenizer_stats_key
from ...train.planner import PlanInput, plan_training
from ...workers.launch import launch_worker
from ..components import field_grid, form_section, hint
from ..shell import ACCENT, panel, shell

FIXED_ROUTING_MODE = "fixed_top_k"
ADAPTIVE_ROUTING_MODE = "adaptive_surprise"
SOFT_LORA_ROUTING_MODE = "soft_lora_identity"


def _adaptive_ui_config(
    *,
    target_total_multiplier: float = 1.25,
    expert_footprint_percent: float = 0.5,
    min_active_percent: float = 1.0,
    max_active_percent: float = 10.0,
    surprise_ema_decay: float = 0.95,
    surprise_ema_warmup: int = 32,
    expand_margin_nats: float = 0.0,
    contract_margin_nats: float = 0.05,
    expand_patience: int = 1,
    contract_patience: int = 8,
    min_marginal_nll_gain_nats: float = 0.01,
    max_recomputations: int = 4,
    vram_reserve_gb: float = 1.0,
    pin_host_memory: bool = True,
) -> dict:
    """Translate human-scale UI values into the versioned runtime contract."""

    config = AdaptiveOccupancyConfig(
        enabled=True,
        target_total_multiplier=float(target_total_multiplier),
        expert_footprint_fraction=float(expert_footprint_percent) / 100.0,
        min_active_fraction=float(min_active_percent) / 100.0,
        max_active_fraction=float(max_active_percent) / 100.0,
        surprise_ema_decay=float(surprise_ema_decay),
        surprise_ema_warmup=int(surprise_ema_warmup),
        expand_margin_nats=float(expand_margin_nats),
        contract_margin_nats=float(contract_margin_nats),
        expand_patience=int(expand_patience),
        contract_patience=int(contract_patience),
        min_marginal_nll_gain_nats=float(min_marginal_nll_gain_nats),
        max_recomputations=int(max_recomputations),
        vram_reserve_bytes=round(float(vram_reserve_gb) * 2**30),
        pin_host_memory=bool(pin_host_memory),
    )
    return config.to_dict()


def _routing_run_fields(mode: str, adaptive: dict | None = None) -> dict:
    """Return only routing fields, preserving the legacy run shape exactly."""

    if mode == FIXED_ROUTING_MODE:
        return {"training_strategy": "joint_per_token_v1"}
    if mode == SOFT_LORA_ROUTING_MODE:
        return {
            "training_strategy": "joint_soft_lora_identity_v1",
            "identity_expert": True,
        }
    if mode != ADAPTIVE_ROUTING_MODE:
        raise ValueError(f"unknown MoLE routing mode: {mode}")
    resolved = AdaptiveOccupancyConfig.from_dict(adaptive)
    if not resolved.enabled:
        raise ValueError("adaptive surprise routing requires an enabled adaptive config")
    return {
        "training_strategy": "joint_adaptive_surprise_v1",
        "adaptive": resolved.to_dict(),
    }


def _estimate_adaptive_architecture(
    shape: dict,
    target_modules: list[str] | tuple[str, ...],
    adaptive: dict,
) -> dict | None:
    """Estimate E/r/K from checkpoint metadata; the worker resolves exact tensors."""

    base_parameters = round(float(shape.get("params_b") or 0) * 1e9)
    hidden = int(shape.get("hidden_size") or 0)
    layers = int(shape.get("num_layers") or 0)
    if base_parameters < 1 or hidden < 1 or layers < 1:
        return None
    intermediate = int(shape.get("intermediate_size") or hidden * 4)
    heads = int(shape.get("num_attention_heads") or 0)
    kv_heads = int(shape.get("num_kv_heads") or 0)
    head_dim = int(shape.get("head_dim") or (hidden // heads if heads else 0))
    query_hidden = head_dim * heads if head_dim and heads else hidden
    kv_hidden = head_dim * kv_heads if head_dim and kv_heads else query_hidden
    widths = {
        "q_proj": hidden + query_hidden,
        "k_proj": hidden + kv_hidden,
        "v_proj": hidden + kv_hidden,
        "o_proj": query_hidden + hidden,
        "gate_proj": hidden + intermediate,
        "up_proj": hidden + intermediate,
        "down_proj": intermediate + hidden,
    }
    selected_width = sum(widths.get(name, 0) for name in target_modules)
    if selected_width < 1:
        return None
    config = AdaptiveOccupancyConfig.from_dict(adaptive)
    plan = solve_expert_budget(
        base_parameters=base_parameters,
        expert_rank_width=layers * selected_width,
        router_parameters_per_expert=layers * (hidden + 1),
        target_total_multiplier=config.target_total_multiplier,
        expert_footprint_fraction=config.expert_footprint_fraction,
        minimum_experts=config.minimum_experts,
    )
    config.resolved_expert_footprint_fraction = plan.expert_footprint_fraction
    return {
        "num_experts": plan.num_experts,
        "rank": plan.rank,
        "top_k": min(plan.num_experts, config.minimum_experts),
        "maximum_active_experts": config.maximum_experts(plan.num_experts),
        "requested_total_parameters": plan.requested_total_parameters,
        "resolved_total_parameters": plan.resolved_total_parameters,
        "expert_parameter_count": plan.parameters_per_expert,
        "expert_footprint_fraction": plan.expert_footprint_fraction,
        "budget_error_fraction": plan.budget_error_fraction,
    }


def _checkpoint_shape(row: dict | None) -> dict:
    if not row:
        return {}
    shape = {
        "params_b": float(row.get("params_b") or 0),
        "hidden_size": int(row.get("hidden_size") or 0),
        "num_layers": int(row.get("num_layers") or 0),
        "model_type": "",
        "fused_linear_ce": False,
        "full_liger_kernels": False,
        "prefer_cudnn_sdpa": False,
    }
    ref = Path(row.get("ref") or "")
    config_path = ref / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text("utf-8"))
            model_type = str(cfg.get("model_type") or "").lower()
            shape.update(
                {
                    "hidden_size": int(cfg.get("hidden_size") or shape["hidden_size"]),
                    "intermediate_size": int(cfg.get("intermediate_size") or 0),
                    "num_layers": int(cfg.get("num_hidden_layers") or shape["num_layers"]),
                    "num_attention_heads": int(cfg.get("num_attention_heads") or 0),
                    "num_kv_heads": int(cfg.get("num_key_value_heads") or 0),
                    "head_dim": int(cfg.get("head_dim") or 0),
                    "vocab_size": int(cfg.get("vocab_size") or 0),
                    "model_type": model_type,
                    "fused_linear_ce": model_type == "qwen3",
                    "full_liger_kernels": model_type == "qwen3",
                    "prefer_cudnn_sdpa": model_type == "qwen3",
                }
            )
        except (OSError, ValueError, TypeError):
            pass
    return shape


def _make_plan(
    checkpoint: dict | None,
    seq_len: int,
    method: str,
    quant: str,
    rank: int,
    num_experts: int,
    *,
    adapter_only: bool,
    isolated_expert_training: bool,
    train_token_routers: bool = False,
    optimizer_state: str = "gpu",
    dense_precision: str | None = None,
):
    gpu = best_gpu()
    shape = _checkpoint_shape(checkpoint)
    if gpu is None or not shape.get("params_b"):
        return None
    if method == "auto":
        method, quant = _auto_method_quant(checkpoint, quant)
    plan_input = PlanInput(
            **shape,
            vram_gb=gpu.vram_free_gb,
            ram_gb=ram_total_gb(),
            seq_len=seq_len,
            method=method,
            quant=quant,
            lora_rank=rank,
            num_experts=num_experts,
            adapter_only=adapter_only,
            isolated_expert_training=isolated_expert_training,
            train_token_routers=train_token_routers,
            optimizer_state=optimizer_state,
            sm_version=gpu.compute_capability,
            has_triton=has_triton(),
        )
    plan = plan_training(plan_input)
    if plan.method == "qlora" and quant in DENSE_MODEL_PRECISIONS:
        # The shared planner predates exact-device precision filtering and uses
        # its ``quant`` field only for QLoRA candidates. A dense value therefore
        # means no verified QLoRA lane exists: retain the best LoRA plan instead
        # of inventing a quantized execution mode.
        plan = plan_training(replace(plan_input, method="lora"))
    if plan.method in {"full", "lora"}:
        resolved_dense = (
            quant if method in {"full", "lora"} else str(dense_precision or "")
        )
        if resolved_dense not in DENSE_MODEL_PRECISIONS:
            raise ValueError("choose an available dense precision for Full/LoRA training")
        plan = replace(plan, quant=resolved_dense)
    return plan


def _auto_method_quant(checkpoint: dict | None, requested_quant: str) -> tuple[str, str]:
    """Respect an established checkpoint precision without widening it."""

    qat_format = _checkpoint_qat_format(checkpoint)
    locked = str((checkpoint or {}).get("quant_locked") or "").strip().casefold()
    if qat_format:
        # QAT checkpoints carry high-precision shadow weights and restore their
        # native low-precision forward path themselves. Stacking bitsandbytes
        # NF4/INT8 on top would be a second, incompatible quantization transform.
        if locked and locked != "bf16":
            raise ValueError(
                f"checkpoint has {qat_format} QAT but is quant-locked to {locked}; "
                "saved native QAT requires BF16 shadow loading"
            )
        return "lora", "bf16"
    if locked in BITSANDBYTES_QUANTIZATIONS:
        return "qlora", locked
    if locked in DENSE_MODEL_PRECISIONS:
        return "lora", locked
    if locked:
        raise ValueError(f"checkpoint has unsupported precision lock {locked!r}")
    requested = str(requested_quant or "").strip().casefold()
    if requested not in {*DENSE_MODEL_PRECISIONS, *BITSANDBYTES_QUANTIZATIONS}:
        raise ValueError(f"unsupported requested model precision {requested or '<empty>'!r}")
    return "auto", requested


def _checkpoint_qat_format(checkpoint: dict | None) -> str:
    if not checkpoint:
        return ""
    try:
        metadata = json.loads(checkpoint.get("meta_json") or "{}")
        qat = str((metadata.get("precision") or {}).get("qat") or "")
        if qat and qat != "none":
            return qat
    except (TypeError, ValueError):
        pass
    config_path = Path(checkpoint.get("ref") or "") / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text("utf-8"))
            return str((config.get("yylab_low_precision") or {}).get("format") or "")
        except (OSError, TypeError, ValueError):
            pass
    return ""


def render_plan(
    container,
    checkpoint: dict | None,
    seq_len: int,
    method: str,
    quant: str,
    rank: int,
    num_experts: int,
    *,
    dense_precision: str,
    adapter_only: bool,
    isolated_expert_training: bool,
    train_token_routers: bool = False,
    optimizer_state: str = "gpu",
) -> None:
    container.clear()
    with container:
        gpu = best_gpu()
        if gpu is None:
            ui.label(
                "No GPU detected — planner needs CUDA. CPU fallback training is not recommended."
            ).classes("text-amber-400 text-sm")
            return
        shape = _checkpoint_shape(checkpoint)
        if not shape.get("params_b"):
            ui.label(
                f"GPU: {gpu.name} ({gpu.vram_free_gb:.1f} GB free). Model size unknown — plan appears once checkpoint params are known."
            ).classes("text-sm text-gray-400")
            return
        try:
            plan = _make_plan(
                checkpoint,
                seq_len,
                method,
                quant,
                rank,
                num_experts,
                dense_precision=dense_precision,
                adapter_only=adapter_only,
                isolated_expert_training=isolated_expert_training,
                train_token_routers=train_token_routers,
                optimizer_state=optimizer_state,
            )
        except ValueError as exc:
            ui.label(str(exc)).classes("text-red-400 text-sm")
            return
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            ui.badge(f"{plan.method} · {plan.quant}").props(
                "color=primary" if plan.feasible else "color=negative"
            )
            ui.badge(f"est peak {plan.est_peak_gb} GB / {gpu.vram_free_gb:.0f} GB").props(
                "outline" + (" color=green" if plan.feasible else " color=negative")
            )
            ui.badge(f"bs {plan.micro_batch} × accum {plan.grad_accum}").props("outline color=grey")
            if plan.grad_ckpt:
                ui.badge("grad ckpt").props("outline color=grey")
            if plan.offload:
                ui.badge("CPU offload").props("outline color=orange")
            if plan.use_compile:
                ui.badge("torch.compile").props("outline color=teal")
        for n in plan.notes:
            ui.label(f"· {n}").classes("text-xs text-gray-400")


def _corpus_option_label(corpus: dict) -> str:
    stats = orjson.loads(corpus.get("stats_json") or b"{}")
    doms = stats.get("domains", {})
    top = ""
    if doms:
        top = max(doms.items(), key=lambda item: item[1].get("chars", 0))[0]
    cdas = stats.get("cdas") or {}
    profile = str((stats.get("ingest_config") or {}).get("selection_profile") or "")
    kind = str(corpus.get("kind") or "ingested")
    verdict = str(corpus.get("verdict") or "")
    if cdas:
        lineage = f"CDAS {profile or 'catalogued'}"
    elif kind == "approved":
        lineage = "approved store; no CDAS audit"
    elif verdict == "approved":
        lineage = "legacy reviewed; no CDAS audit"
    else:
        lineage = "legacy unreviewed; no CDAS audit"
    details = "; ".join(value for value in (top, lineage) if value)
    return f"{corpus['name']} [{details}]" if details else str(corpus["name"])


def corpus_options() -> dict[str, str]:
    out = {}
    for c in db.q(
        "SELECT id, name, kind, verdict, stats_json FROM corpora WHERE status='ready'"
        " AND COALESCE(kind,'ingested') != 'rejected'"
        " AND COALESCE(verdict,'') != 'rejected' ORDER BY name"
    ):
        out[c["id"]] = _corpus_option_label(c)
    return out


def corpus_domains(cid: str) -> set[str]:
    row = db.q1("SELECT stats_json FROM corpora WHERE id=?", (cid,))
    if not row:
        return set()
    return set(orjson.loads(row["stats_json"] or b"{}").get("domains", {}).keys())


@ui.page("/train/mole")
def train_mole_page():
    model_device = model_execution_device()
    dense_precisions = available_dense_precisions(model_device)
    bnb_quantizations = available_bitsandbytes_quantizations(model_device)
    dense_precision_options = {
        precision: precision.upper() for precision in dense_precisions
    }
    bnb_quantization_options = {
        quantization: (
            f"{quantization.upper()} ({bitsandbytes_compute_precision(quantization).upper()} compute)"
        )
        for quantization in bnb_quantizations
    }
    default_dense_precision = next(iter(dense_precisions), None)
    method_options = {"auto": "Auto (recommended)"}
    if default_dense_precision is not None:
        method_options["lora"] = f"LoRA ({default_dense_precision.upper()} base default)"
    if bnb_quantizations:
        method_options["qlora"] = "QLoRA (quantized base)"
    default_bnb_quantization = bnb_quantizations[0] if bnb_quantizations else None
    with shell("Train · Mixture of LoRA Experts", "/train/mole"):
        ui.label("Adapter training (frozen base, MoLE)").classes("text-xl font-semibold")
        ui.label("Jointly learn sparse per-token expert routes over one frozen checkpoint.").classes(
            "text-sm text-gray-400 -mt-2"
        )

        ckpts = {
            c["id"]: f"{c['name']} ({c['params_b']:.2f} B)" if c["params_b"] else c["name"]
            for c in hf_compatible_checkpoints(
                db.q("SELECT * FROM checkpoints WHERE kind!='pending' ORDER BY name")
            )
        }
        corpora = corpus_options()

        if not ckpts or not corpora:
            missing = []
            if not ckpts:
                missing.append(("a checkpoint", "/checkpoints"))
            if not corpora:
                missing.append(("a processed corpus", "/corpora"))
            with panel():
                with ui.row().classes("items-center gap-3"):
                    ui.icon("info", size="20px").classes("text-amber-400")
                    ui.label("Before training you need " + " and ".join(m[0] for m in missing) + ".").classes(
                        "text-sm text-gray-300"
                    )
                    for label, path in missing:
                        ui.button(
                            "Add " + label.split()[-1], on_click=lambda _, p=path: ui.navigate.to(p)
                        ).props("dense no-caps outline")

        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
            with panel().classes("w-[560px] shrink-0"):
                form_section("1", "Model")
                name_in = (
                    ui.input("Adapter name", placeholder="auto").props("dense outlined").classes("w-full")
                )
                ckpt_sel = (
                    ui.select(ckpts, label="Base checkpoint (frozen)")
                    .props("dense outlined")
                    .classes("w-full")
                )
                with field_grid(3):
                    method = (
                        ui.select(
                            method_options,
                            value="auto",
                            label="Training method",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    lora_precision = (
                        ui.select(
                            dense_precision_options,
                            value=default_dense_precision,
                            label="LoRA base precision",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    qlora_quant = (
                        ui.select(
                            bnb_quantization_options,
                            value=default_bnb_quantization,
                            label="QLoRA base quant",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                hint(
                    "The selected dense precision or quantized format is locked to this "
                    "checkpoint+adapter pair for the adapter's lifetime — no drift."
                )
                if not bnb_quantizations:
                    hint(
                        "QLoRA is omitted because no bitsandbytes NF4/INT8 CUDA "
                        "forward/backward path passed its capability probe."
                    )

                ui.separator().classes("my-3")
                form_section("2", "Training data")
                corp_sel = (
                    ui.select(corpora, multiple=True, label="Corpora")
                    .props("dense outlined use-chips")
                    .classes("w-full")
                )
                with ui.row().classes("items-center gap-2 -mt-1"):
                    ui.label("Quick pick:").classes("text-xs text-gray-500")
                    for dom in ("code", "math", "language", "all"):

                        def pick(dom=dom):
                            if dom == "all":
                                corp_sel.value = list(corpora.keys())
                            else:
                                corp_sel.value = [cid for cid in corpora if dom in corpus_domains(cid)]
                            corp_sel.update()
                            update_data_summary()

                        ui.button(dom, on_click=pick).props("dense flat no-caps size=sm")
                dom_filter = (
                    ui.select(
                        DOMAINS,
                        multiple=True,
                        value=[],
                        label="Domain filter (optional)",
                    )
                    .props("dense outlined use-chips")
                    .classes("w-full")
                )
                hint(
                    "Optional global filter only. Data is never split between experts; every token reaches the learned router."
                )
                data_lbl = (
                    ui.label("Select corpora to see totals.")
                    .classes("text-xs text-gray-400 px-3 py-2 rounded-md w-full")
                    .style(f"background: {ACCENT}10")
                )

                ui.separator().classes("my-3")
                form_section("3", "Expert architecture", "y = W_frozen(x) + Σ gₖ(x)·(BₖAₖx)")
                routing_mode = (
                    ui.select(
                        {
                            FIXED_ROUTING_MODE: "Legacy fixed top-K",
                            ADAPTIVE_ROUTING_MODE: "Adaptive surprise occupancy (recommended)",
                            SOFT_LORA_ROUTING_MODE: "Soft LoRA + identity bypass",
                        },
                        value=ADAPTIVE_ROUTING_MODE,
                        label="Routing policy",
                    )
                    .props("dense outlined")
                    .classes("w-full")
                )
                hint(
                    "Soft LoRA adds one parameter-free identity route to the learned token gate. "
                    "Selecting it applies no adapter delta; set K to all routes for a dense soft mixture."
                )
                with field_grid(3):
                    n_exp = (
                        ui.number("Experts", value=8, min=1, max=196)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    top_k = (
                        ui.number("Top-K active/token", value=2, min=1, max=196)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    rank = (
                        ui.number("Rank/expert", value=8, min=1, max=256)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                with ui.row().classes("items-center gap-2 -mt-1 flex-wrap") as capacity_presets:
                    ui.label("Capacity presets:").classes("text-xs text-gray-500")
                    for label, experts, expert_rank in (
                        ("balanced E8/r8", 8, 8),
                        ("deeper E4/r16", 4, 16),
                        ("wider E16/r4", 16, 4),
                    ):

                        def set_capacity(experts=experts, expert_rank=expert_rank):
                            n_exp.value = experts
                            rank.value = expert_rank
                            update_capacity_summary()
                            update_plan()

                        ui.button(label, on_click=set_capacity).props("dense flat no-caps size=sm")
                with ui.column().classes("w-full gap-2") as adaptive_box:
                    hint(
                        "The worker solves the exact rack from the frozen model. Each expert targets a 0.5% "
                        "weight quantum; execution starts at 1% and may round up to 10%."
                    )
                    with field_grid(3):
                        target_total_multiplier = (
                            ui.number("Target total × base", value=1.25, min=1.01, max=4, step=0.05)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        expert_footprint_percent = (
                            ui.number("Expert footprint (%)", value=0.5, min=0.5, max=0.5, step=0.5)
                            .props("dense outlined readonly")
                            .classes("w-full")
                        )
                        vram_reserve_gb = (
                            ui.number("Free VRAM reserve (GiB)", value=1.0, min=0, max=64, step=0.25)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with field_grid(2):
                        min_active_percent = (
                            ui.number("Minimum occupancy (%)", value=1.0, min=0.5, max=10, step=0.5)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        max_active_percent = (
                            ui.number("Maximum occupancy (%)", value=10.0, min=0.5, max=10, step=0.5)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                    with ui.expansion("Surprise threshold and hysteresis", icon="tune").classes("w-full"):
                        with field_grid(2):
                            surprise_ema_decay = (
                                ui.number("Surprise EMA decay", value=0.95, min=0, max=0.999, step=0.01)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            surprise_ema_warmup = (
                                ui.number("EMA warmup tokens", value=32, min=1, max=8192)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            expand_margin_nats = (
                                ui.number("Expand margin (nats)", value=0.0, min=0, max=10, step=0.01)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            contract_margin_nats = (
                                ui.number("Contract margin (nats)", value=0.05, min=0, max=10, step=0.01)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            expand_patience = (
                                ui.number("Expand patience", value=1, min=1, max=1024)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            contract_patience = (
                                ui.number("Contract patience", value=8, min=1, max=8192)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            min_marginal_nll_gain_nats = (
                                ui.number("Minimum marginal NLL gain", value=0.01, min=0, max=10, step=0.005)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            max_recomputations = (
                                ui.number("Recomputation cap/token", value=4, min=0, max=64)
                                .props("dense outlined")
                                .classes("w-full")
                            )
                        pin_host_memory = ui.switch("Pin CPU expert memory", value=True).tooltip(
                            "Speeds expert transfers when the resident pool changes; consumes pinned host RAM."
                        )
                adaptive_box.visible = False
                targets = (
                    ui.select(
                        list(DEFAULT_TARGET_MODULES),
                        multiple=True,
                        value=list(DEFAULT_TARGET_MODULES),
                        label="Target modules",
                    )
                    .props("dense outlined use-chips")
                    .classes("w-full")
                )
                with field_grid(2):
                    aux_w = (
                        ui.number("Router balance weight", value=0.01, min=0, max=1, step=0.005)
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    optimizer_state = (
                        ui.select(
                            {
                                "auto": "Auto optimizer memory",
                                "gpu": "Keep optimizer on GPU",
                                "cpu_paged": "Paged 8-bit state (CPU-backed)",
                                "cpu_offload": "Full CPU offload (more RAM, slower)",
                            },
                            value="auto",
                            label="Optimizer state",
                        )
                        .props("dense outlined")
                        .classes("w-full")
                    )
                capacity_lbl = (
                    ui.label("")
                    .classes("text-xs text-gray-400 px-3 py-2 rounded-md w-full")
                    .style(f"background: {ACCENT}10")
                )

                ui.separator().classes("my-3")
                form_section("4", "Training run")
                with field_grid(3):
                    seq_sel = (
                        ui.select([512, 1024, 2048, 4096, 8192], value=2048, label="Seq length")
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    epochs = (
                        ui.number("Epochs", value=1, min=1, max=20).props("dense outlined").classes("w-full")
                    )
                    lr = (
                        ui.select([1e-4, 3e-4, 5e-5, 1e-5], value=3e-4, label="Learning rate")
                        .props("dense outlined")
                        .classes("w-full")
                    )
                with ui.expansion("Advanced", icon="tune").classes("w-full"):
                    with field_grid(2):
                        micro_bs = (
                            ui.number("Micro batch", value=1, min=1, max=16)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        grad_accum = (
                            ui.number("Grad accum", value=16, min=1, max=256)
                            .props("dense outlined")
                            .classes("w-full")
                        )

                start_btn = (
                    ui.button("Start training", icon="play_arrow")
                    .props("no-caps unelevated")
                    .classes("w-full mt-4")
                )
                hint(
                    "All experts and one router per transformer block learn together. Auto pages optimizer state to CPU only for larger racks."
                )

            with ui.column().classes("flex-1 gap-4 min-w-[360px]"):
                with panel():
                    ui.label("Auto training plan").classes("text-base font-semibold mb-2")
                    plan_box = ui.column().classes("w-full gap-1")
                with panel():
                    with ui.expansion("What is MoLE?", icon="help_outline").classes("w-full"):
                        ui.label(
                            "Each transformer block learns a top-K decision for every token and reuses it across all "
                            "adapted attention and MLP projections in that block. The frozen base stays shared while "
                            "all low-rank experts and routers train jointly on the same mixed corpus. Reducing rank while "
                            "adding experts keeps total adapter width bounded. Weights remain plain safetensors + JSON."
                        ).classes("text-sm text-gray-300")

        def checkpoint_row() -> dict | None:
            cid = ckpt_sel.value
            if not cid:
                return None
            return db.q1("SELECT * FROM checkpoints WHERE id=?", (cid,))

        def adaptive_payload() -> dict:
            return _adaptive_ui_config(
                target_total_multiplier=float(target_total_multiplier.value or 1.25),
                expert_footprint_percent=float(expert_footprint_percent.value or 0.5),
                min_active_percent=float(min_active_percent.value or 1.0),
                max_active_percent=float(max_active_percent.value or 10.0),
                surprise_ema_decay=float(
                    0.95 if surprise_ema_decay.value is None else surprise_ema_decay.value
                ),
                surprise_ema_warmup=int(surprise_ema_warmup.value or 32),
                expand_margin_nats=float(
                    0.0 if expand_margin_nats.value is None else expand_margin_nats.value
                ),
                contract_margin_nats=float(
                    0.05 if contract_margin_nats.value is None else contract_margin_nats.value
                ),
                expand_patience=int(expand_patience.value or 1),
                contract_patience=int(contract_patience.value or 8),
                min_marginal_nll_gain_nats=float(
                    0.01 if min_marginal_nll_gain_nats.value is None else min_marginal_nll_gain_nats.value
                ),
                max_recomputations=int(4 if max_recomputations.value is None else max_recomputations.value),
                vram_reserve_gb=float(1.0 if vram_reserve_gb.value is None else vram_reserve_gb.value),
                pin_host_memory=bool(pin_host_memory.value),
            )

        legacy_capacity = {"experts": 8, "rank": 8, "top_k": 2}
        routing_state = {"adaptive": False}

        def update_routing_mode() -> None:
            adaptive = routing_mode.value == ADAPTIVE_ROUTING_MODE
            if adaptive and not routing_state["adaptive"]:
                legacy_capacity.update(
                    experts=int(n_exp.value or 8),
                    rank=int(rank.value or 8),
                    top_k=int(top_k.value or 2),
                )
            elif not adaptive and routing_state["adaptive"]:
                n_exp.value = legacy_capacity["experts"]
                rank.value = legacy_capacity["rank"]
                top_k.value = legacy_capacity["top_k"]
            routing_state["adaptive"] = adaptive
            adaptive_box.visible = adaptive
            capacity_presets.visible = not adaptive
            for control in (n_exp, rank, top_k):
                if adaptive:
                    control.props("readonly")
                else:
                    control.props(remove="readonly")
            update_capacity_summary()
            update_plan()

        def update_plan() -> None:
            requested_quant = (
                str(qlora_quant.value or "")
                if method.value in {"auto", "qlora"} and qlora_quant.value is not None
                else str(lora_precision.value or "")
            )
            render_plan(
                plan_box,
                checkpoint_row(),
                int(seq_sel.value or 2048),
                "auto" if method.value == "auto" else method.value,
                requested_quant,
                int(rank.value or 8),
                int(n_exp.value or 8),
                dense_precision=str(lora_precision.value or ""),
                adapter_only=True,
                isolated_expert_training=False,
                train_token_routers=True,
                optimizer_state=str(optimizer_state.value or "auto"),
            )

        def update_data_summary() -> None:
            sel = corp_sel.value or []
            if not sel:
                data_lbl.text = "Select corpora to see totals."
                return
            flt = set(dom_filter.value or [])
            docs = chars = 0
            doms: dict[str, int] = {}
            for cid in sel:
                row = db.q1("SELECT stats_json FROM corpora WHERE id=?", (cid,))
                if not row:
                    continue
                stats = orjson.loads(row["stats_json"] or b"{}")
                for d, v in stats.get("domains", {}).items():
                    if flt and d not in flt:
                        continue
                    doms[d] = doms.get(d, 0) + v.get("chars", 0)
                    docs += v.get("docs", 0)
                    chars += v.get("chars", 0)
            if flt and chars == 0:
                data_lbl.text = f"{len(sel)} corpora · no docs in domains: {', '.join(sorted(flt))}"
                return
            top = sorted(doms.items(), key=lambda kv: -kv[1])
            mix = ", ".join(f"{k} {v / max(chars, 1) * 100:.0f}%" for k, v in top)
            tag = f" · filter: {', '.join(sorted(flt))}" if flt else ""
            data_lbl.text = f"{len(sel)} corpora{tag} · {docs:,} docs · {chars / 1e9:.2f}B chars · mix: {mix}"

        def update_capacity_summary() -> None:
            if routing_mode.value == ADAPTIVE_ROUTING_MODE:
                try:
                    config = adaptive_payload()
                    estimate = _estimate_adaptive_architecture(
                        _checkpoint_shape(checkpoint_row()),
                        list(targets.value or []),
                        config,
                    )
                except (TypeError, ValueError) as exc:
                    capacity_lbl.text = f"Adaptive configuration: {exc}"
                    return
                adaptive_cfg = AdaptiveOccupancyConfig.from_dict(config)
                if estimate is None:
                    n_exp.value = adaptive_cfg.minimum_experts
                    rank.value = 1
                    top_k.value = adaptive_cfg.minimum_experts
                    capacity_lbl.text = (
                        "Adaptive estimate unavailable from checkpoint metadata; the worker will solve exact E/r/K. "
                        f"Occupancy remains {min_active_percent.value:g}–{max_active_percent.value:g}% in "
                        f"{expert_footprint_percent.value:g}% expert quanta."
                    )
                    return
                n_exp.value = estimate["num_experts"]
                rank.value = estimate["rank"]
                top_k.value = estimate["top_k"]
                capacity_lbl.text = (
                    f"Estimated E{estimate['num_experts']}/r{estimate['rank']}/K{estimate['top_k']} minimum; "
                    f"adaptive K={estimate['top_k']}–{estimate['maximum_active_experts']} "
                    f"({min_active_percent.value:g}–{max_active_percent.value:g}%). "
                    f"One expert ≈ {estimate['expert_footprint_fraction'] * 100:.3f}% of resolved weights. "
                    "The worker performs the exact tensor-shape solve."
                )
                return
            experts = int(n_exp.value or 1)
            expert_rank = int(rank.value or 1)
            soft_identity = routing_mode.value == SOFT_LORA_ROUTING_MODE
            active = min(int(top_k.value or 1), experts + int(soft_identity))
            identity_note = (
                " The additional identity route contributes zero delta, so tokens can keep the frozen base output."
                if soft_identity
                else ""
            )
            capacity_lbl.text = (
                f"Total adapter width E×r = {experts * expert_rank}; active routes/token K={active}. "
                f"E8/r8 and E4/r16 have equal adapter width.{identity_note}"
            )

        def update_precision_visibility() -> None:
            lora_precision.enabled = bool(dense_precisions) and method.value in {"auto", "lora"}
            qlora_quant.enabled = bool(bnb_quantizations) and method.value in {"auto", "qlora"}

        def resolve_method() -> tuple[str, str, object | None]:
            m = str(method.value or "auto")
            if m != "auto":
                requested = (
                    str(qlora_quant.value or "")
                    if m == "qlora"
                    else str(lora_precision.value or "")
                )
                return m, requested, None
            requested = (
                str(qlora_quant.value)
                if qlora_quant.value is not None
                else str(lora_precision.value or "")
            )
            plan_method, plan_quant = _auto_method_quant(
                checkpoint_row(), requested
            )
            plan = _make_plan(
                checkpoint_row(),
                int(seq_sel.value or 2048),
                plan_method,
                plan_quant,
                int(rank.value or 8),
                int(n_exp.value or 8),
                dense_precision=str(lora_precision.value or ""),
                adapter_only=True,
                isolated_expert_training=False,
                train_token_routers=True,
                optimizer_state=str(optimizer_state.value or "auto"),
            )
            if plan is None:
                if plan_method != "auto":
                    return plan_method, plan_quant, None
                fallback_method = (
                    "qlora" if plan_quant in BITSANDBYTES_QUANTIZATIONS else "lora"
                )
                return fallback_method, plan_quant, None
            return plan.method, plan.quant, plan

        def on_start() -> None:
            problems = []
            routing_fields: dict = {}
            if not ckpt_sel.value:
                problems.append("choose a checkpoint")
            if not corp_sel.value:
                problems.append("choose corpora")
            route_count = int(n_exp.value or 1) + int(
                routing_mode.value == SOFT_LORA_ROUTING_MODE
            )
            if routing_mode.value != ADAPTIVE_ROUTING_MODE and int(top_k.value or 1) > route_count:
                problems.append("top-K must be ≤ number of adapter and identity routes")
            if not targets.value:
                problems.append("choose target modules")
            try:
                routing_fields = _routing_run_fields(
                    str(routing_mode.value or FIXED_ROUTING_MODE),
                    adaptive_payload() if routing_mode.value == ADAPTIVE_ROUTING_MODE else None,
                )
            except (TypeError, ValueError) as exc:
                problems.append(str(exc))
            if problems:
                ui.notify("Fix: " + ", ".join(problems), type="warning")
                return

            ckpt = db.q1("SELECT * FROM checkpoints WHERE id=?", (ckpt_sel.value,))
            if not ckpt or checkpoint_family(ckpt) == "spalmer":
                ui.notify(
                    "MoLE training requires a Hugging Face-compatible base checkpoint.",
                    type="negative",
                )
                return
            try:
                m_resolved, q_resolved, auto_plan = resolve_method()
            except ValueError as exc:
                ui.notify(str(exc), type="negative", multi_line=True)
                return
            qat_format = _checkpoint_qat_format(ckpt)
            if qat_format and m_resolved == "qlora":
                ui.notify(
                    f"This base already restores {qat_format} QAT numerics. Use LoRA/BF16; "
                    "QLoRA would apply an incompatible second quantizer.",
                    type="negative",
                    multi_line=True,
                )
                return
            if auto_plan is not None and not auto_plan.feasible:
                ui.notify(
                    "The selected model/sequence does not pass the VRAM preflight. Reduce sequence length or rank.",
                    type="negative",
                    multi_line=True,
                )
                return
            if ckpt["quant_locked"] and ckpt["quant_locked"] != q_resolved:
                ui.notify(
                    f"Checkpoint is quant-locked to {ckpt['quant_locked']} — adapters on it must use the same quant.",
                    type="negative",
                    multi_line=True,
                )
                return
            if qat_format and q_resolved != "bf16":
                ui.notify(
                    f"This base restores {qat_format} QAT numerics and requires BF16 shadow loading; "
                    f"{q_resolved.upper()} was requested.",
                    type="negative",
                    multi_line=True,
                )
                return
            try:
                require_model_precision(q_resolved, model_device)
            except (RuntimeError, ValueError) as exc:
                ui.notify(str(exc), type="negative", multi_line=True)
                return

            seq = int(seq_sel.value or 2048)
            mb = auto_plan.micro_batch if auto_plan is not None else int(micro_bs.value or 1)
            ga = auto_plan.grad_accum if auto_plan is not None else int(grad_accum.value or 16)
            n_epochs = int(epochs.value or 1)
            domains_sel = list(dom_filter.value) if dom_filter.value else None
            stats_key = tokenizer_stats_key(_checkpoint_shape(ckpt).get("model_type"))
            max_steps = estimate_steps(
                list(corp_sel.value),
                domains_sel,
                n_epochs,
                seq,
                mb,
                ga,
                stats_key,
            )

            aid = db.new_id("adp")
            name = (
                name_in.value or ""
            ).strip() or f"{ckpt['name']}-mole-E{int(n_exp.value)}k{int(top_k.value)}r{int(rank.value)}"
            db.execute(
                "INSERT INTO adapters(id, name, created_at, checkpoint_id, arch, quant, num_experts, top_k, rank,"
                " target_modules_json, domains_json, corpora_json, status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    aid,
                    name,
                    db.now_iso(),
                    ckpt["id"],
                    "mole",
                    q_resolved,
                    int(n_exp.value or 8),
                    int(top_k.value or 2),
                    int(rank.value or 8),
                    json.dumps(list(targets.value)),
                    json.dumps(sorted(domains_sel or [])),
                    json.dumps(list(corp_sel.value)),
                    "training",
                ),
            )
            rid = db.new_id("run")
            config = {
                "checkpoint_id": ckpt["id"],
                "checkpoint_ref": ckpt["ref"],
                "adapter_id": aid,
                "method": m_resolved,
                "quant": q_resolved,
                "router_type": "per_token",
                "identity_expert": routing_mode.value == SOFT_LORA_ROUTING_MODE,
                "num_experts": int(n_exp.value or 8),
                "rank": int(rank.value or 8),
                "alpha": None,
                "top_k": int(top_k.value or 2),
                "target_modules": list(targets.value),
                "corpora": list(corp_sel.value),
                "domains": domains_sel,
                "seq_len": seq,
                "epochs": n_epochs,
                "micro_batch": mb,
                "grad_accum": ga,
                "lr": float(lr.value or 3e-4),
                "aux_weight": float(0.01 if aux_w.value is None else aux_w.value),
                "optimizer_state": str(optimizer_state.value or "auto"),
                "grad_ckpt": True,
                "liger_fused_ce": True,
                "liger_full_kernels": True,
                "cudnn_sdpa": True,
                "token_stats_key": stats_key,
                "seed": 42,
                "shuffle_buffer": 128,
                "shuffle_buffer_bytes": 64 * 2**20,
                "tokenize_chunk": 8,
                "max_text_chars": DEFAULT_TEXT_CHUNK_CHARS,
                "max_steps": max_steps,
                "log_every": 5,
                "save_every": None,
            }
            config.update(routing_fields)
            db.execute(
                "INSERT INTO runs(id, kind, status, created_at, config_json) VALUES(?,?,?,?,?)",
                (rid, "mole_stage1", "queued", db.now_iso(), json.dumps(config)),
            )
            pid = launch_worker(rid)
            db.execute("UPDATE runs SET metrics_json=? WHERE id=?", (json.dumps({"pid": pid}), rid))
            ui.notify(
                f"{('Adaptive surprise' if routing_mode.value == ADAPTIVE_ROUTING_MODE else ('Soft LoRA identity' if routing_mode.value == SOFT_LORA_ROUTING_MODE else 'Joint per-token'))} "
                f"MoLE training started — {name} (~{max_steps} steps, pid {pid}). See Runs.",
                type="positive",
            )

        for el in (
            ckpt_sel,
            seq_sel,
            method,
            lora_precision,
            qlora_quant,
            rank,
            n_exp,
            top_k,
            optimizer_state,
        ):
            el.on_value_change(lambda _: update_plan())
        ckpt_sel.on_value_change(lambda _: update_capacity_summary())
        method.on_value_change(lambda _: update_precision_visibility())
        corp_sel.on_value_change(lambda _: update_data_summary())
        dom_filter.on_value_change(lambda _: update_data_summary())
        for el in (rank, n_exp, top_k):
            el.on_value_change(lambda _: update_capacity_summary())
        routing_mode.on_value_change(lambda _: update_routing_mode())
        targets.on_value_change(lambda _: (update_capacity_summary(), update_plan()))
        for el in (
            target_total_multiplier,
            expert_footprint_percent,
            min_active_percent,
            max_active_percent,
            surprise_ema_decay,
            surprise_ema_warmup,
            expand_margin_nats,
            contract_margin_nats,
            expand_patience,
            contract_patience,
            min_marginal_nll_gain_nats,
            max_recomputations,
            vram_reserve_gb,
            pin_host_memory,
        ):
            el.on_value_change(lambda _: (update_capacity_summary(), update_plan()))
        start_btn.on_click(on_start)

        update_routing_mode()
        update_plan()
        update_precision_visibility()
        update_capacity_summary()
