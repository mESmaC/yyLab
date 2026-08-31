from __future__ import annotations

from nicegui import ui

from ... import db
from ...config import get_settings, update_settings
from ...hw import best_gpu, system_report
from ...infer.context_math import capacity_rows, kv_bytes_per_token, max_context_for_budget, model_weight_gb
from ..shell import panel, shell

CTX_PRESETS = [8_192, 32_768, 131_072, 524_288, 1_048_576, 2_500_000]


def _load_model_config(ref: str):
    from transformers import AutoConfig

    token = get_settings().get("hf_token") or None
    return AutoConfig.from_pretrained(ref, token=token)


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


@ui.page("/context")
def context_page():
    with shell("Context & KV-Cache", "/context"):
        ui.label("Context window & KV-cache settings").classes("text-xl font-semibold")
        ui.label("Applied at chat/inference load time. Use the capacity lab to plan long-context runs.").classes(
            "text-sm text-gray-400"
        )

        engine = dict(get_settings()["engine"])
        sysinfo = system_report()
        gpu = best_gpu()
        ram_gb = sysinfo["ram_gb"]

        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
            with panel().classes("w-[460px] shrink-0"):
                ui.label("Engine settings").classes("text-base font-semibold mb-2")
                attn = ui.select(
                    {"sdpa": "SDPA (recommended on Windows)", "eager": "Eager"},
                    value=engine["attn_implementation"],
                    label="Attention implementation",
                ).props("dense outlined").classes("w-full")
                cache = ui.select(
                    {
                        "dynamic": "Dynamic (default)",
                        "offloaded": "Offloaded to CPU RAM (dynamic)",
                        "offloaded_static": "Offloaded to CPU RAM (static, chunked prefill)",
                        "quantized": "Quantized KV cache (in VRAM)",
                        "quantized_offloaded": "Quantized KV offload (packed CPU int4/int8)",
                        "static": "Static (pre-allocated, in VRAM)",
                        "sliding_window": "Sliding window (deprecated upstream)",
                    },
                    value=engine["cache_implementation"],
                    label="KV cache implementation",
                ).props("dense outlined").classes("w-full")
                kvq = ui.select(
                    {
                        "none": "None",
                        "int8": "INT8 (~half; quantized offload)",
                        "int4": "INT4 (~quarter; resident or offloaded)",
                    },
                    value=(
                        "int4"
                        if engine.get("cache_implementation") == "quantized"
                        else engine["kv_quant"]
                        if engine["kv_quant"] in {"none", "int8", "int4"}
                        else "none"
                    ),
                    label="KV storage precision",
                ).props("dense outlined").classes("w-full")
                with ui.row().classes("w-full gap-3"):
                    kv_offload_group = ui.number(
                        "Quant group size",
                        value=engine.get("kv_offload_group_size", 64),
                        min=8,
                        max=512,
                        precision=0,
                    ).props("dense outlined").classes("flex-1")
                    kv_offload_residual = ui.number(
                        "Full-precision recent KV",
                        value=engine.get("kv_offload_residual_length", 32),
                        min=0,
                        max=4096,
                        precision=0,
                    ).props("dense outlined").classes("flex-1")
                kv_offload_pin = ui.switch(
                    "Pin quantized CPU KV buffers",
                    value=bool(engine.get("kv_offload_pin_memory", True)),
                ).tooltip("Pinned buffers improve CPU-to-GPU transfer overlap but consume pinned host memory.")

                def align_cache_precision() -> None:
                    if cache.value == "quantized" and kvq.value != "int4":
                        kvq.set_value("int4")
                        ui.notify("The resident Quanto cache uses INT4; selected INT4.", type="info")
                    elif cache.value == "quantized_offloaded" and kvq.value not in {"int4", "int8"}:
                        kvq.set_value("int4")
                        ui.notify("Quantized offload requires INT4 or INT8; selected INT4.", type="info")

                cache.on_value_change(lambda _: align_cache_precision())
                with ui.row().classes("w-full gap-3"):
                    sw = ui.number("Sliding window", value=engine["sliding_window"], min=256, max=65536).props(
                        "dense outlined"
                    ).classes("flex-1")
                    chunk = ui.number(
                        "Prefill chunk", value=engine.get("prefill_chunk_size", 8192), min=512, max=65536
                    ).props("dense outlined").classes("flex-1")
                maxctx = ui.number(
                    "Max context tokens", value=engine["max_context_tokens"], min=512, max=2_500_000, step=1024
                ).props("dense outlined").classes("w-full")
                with ui.row().classes("gap-2 -mt-1"):
                    for p in CTX_PRESETS:
                        def set_preset(p=p):
                            maxctx.set_value(p)
                        ui.button(_fmt_tokens(p), on_click=set_preset).props("dense flat no-caps size=sm")
                rope = ui.select(
                    {"none": "None", "linear": "Linear (position interpolation)", "yarn": "YaRN"},
                    value=engine["rope_scaling"] if engine["rope_scaling"] in {"none", "linear", "yarn"} else "none",
                    label="RoPE scaling (extend context beyond pretrain)",
                ).props("dense outlined").classes("w-full")
                factor = ui.number("RoPE factor", value=engine["rope_factor"], min=1.0, max=64.0, step=0.5).props(
                    "dense outlined"
                ).classes("w-full")
                with ui.expansion("Adaptive context selection", icon="manage_search").classes("w-full"):
                    compaction_strategy = ui.select(
                        {
                            "nll_atxy": "Adaptive query-NLL + ATXY",
                            "recency": "Recent turns only",
                        },
                        value=engine.get("context_compaction_strategy", "nll_atxy"),
                        label="Compaction policy",
                    ).props("dense outlined").classes("w-full")
                    with ui.row().classes("w-full gap-3"):
                        recent_turns = ui.number(
                            "Protected recent turns",
                            value=engine.get("context_recent_turns", 1),
                            min=0,
                            max=32,
                            precision=0,
                        ).props("dense outlined").classes("flex-1")
                        nll_probes = ui.number(
                            "Maximum NLL probes",
                            value=engine.get("context_nll_max_probes", 4),
                            min=0,
                            max=32,
                            precision=0,
                        ).props("dense outlined").classes("flex-1")
                    with ui.row().classes("w-full gap-3"):
                        nll_probe_tokens = ui.number(
                            "Tokens per query probe",
                            value=engine.get("context_nll_probe_tokens", 128),
                            min=16,
                            max=2048,
                            precision=0,
                        ).props("dense outlined").classes("flex-1")
                        nll_trigger = ui.number(
                            "Selection trigger (context fraction)",
                            value=engine.get("context_nll_trigger_ratio", 0.9),
                            min=0.1,
                            max=1.0,
                            step=0.05,
                        ).props("dense outlined").classes("flex-1")
                    with ui.row().classes("w-full gap-3"):
                        nll_gain = ui.number(
                            "Minimum net ΔNLL (nats)",
                            value=engine.get("context_min_nll_gain_nats", 0.01),
                            min=0,
                            max=10,
                            step=0.005,
                        ).props("dense outlined").classes("flex-1")
                        nll_token_cost = ui.number(
                            "Token cost (nats/token)",
                            value=engine.get("context_nll_token_cost_nats", 0.00002),
                            min=0,
                            max=0.1,
                            step=0.00001,
                            precision=5,
                        ).props("dense outlined").classes("flex-1")
                    ui.label(
                        "ATXY means factorized address A=(T,X,Y). Address metadata and lexical overlap only "
                        "propose candidates; positive net query-NLL gain accepts them. True answer-side R_y is "
                        "an offline training signal because the answer is unknown during live retrieval."
                    ).classes("text-xs text-gray-500")

                def save() -> None:
                    update_settings(
                        engine={
                            "attn_implementation": attn.value,
                            "cache_implementation": cache.value,
                            "kv_quant": kvq.value,
                            "kv_offload_group_size": int(kv_offload_group.value or 64),
                            "kv_offload_residual_length": int(kv_offload_residual.value or 0),
                            "kv_offload_pin_memory": bool(kv_offload_pin.value),
                            "kv_offload_growth_chunk_size": int(
                                engine.get("kv_offload_growth_chunk_size", 256)
                            ),
                            "sliding_window": int(sw.value or 4096),
                            "prefill_chunk_size": int(chunk.value or 8192),
                            "max_context_tokens": int(maxctx.value or 8192),
                            "rope_scaling": rope.value,
                            "rope_factor": float(factor.value or 2.0),
                            "context_compaction_strategy": compaction_strategy.value,
                            "context_recent_turns": int(recent_turns.value or 0),
                            "context_nll_max_probes": int(nll_probes.value or 0),
                            "context_nll_probe_tokens": int(nll_probe_tokens.value or 128),
                            "context_min_nll_gain_nats": float(nll_gain.value or 0.0),
                            "context_nll_token_cost_nats": float(nll_token_cost.value or 0.0),
                            "context_nll_trigger_ratio": float(nll_trigger.value or 0.9),
                        }
                    )
                    ui.notify("Engine settings saved", type="positive")

                ui.button("Save", icon="save", on_click=save).props("no-caps").classes("mt-3")

            with ui.column().classes("flex-1 gap-4 min-w-[420px]"):
                with panel():
                    ui.label("Context capacity lab").classes("text-base font-semibold")
                    ui.label(
                        "Estimates KV-cache memory for a checkpoint at a target context length. Offloading moves KV to "
                        "system RAM; quantized caches shrink KV in VRAM. Physics decides the rest."
                    ).classes("text-xs text-gray-500 mb-2")

                    ckpts = {
                        c["id"]: c["name"]
                        for c in db.q("SELECT * FROM checkpoints WHERE kind!='pending' ORDER BY name")
                    }
                    with ui.row().classes("w-full gap-3 items-end flex-wrap"):
                        cap_ckpt = ui.select(ckpts, label="Checkpoint").props("dense outlined").classes("w-64")
                        cap_len = ui.number("Context tokens", value=131_072, min=512, max=2_500_000, step=1024).props(
                            "dense outlined"
                        ).classes("w-40")
                        with ui.row().classes("gap-2"):
                            for p in CTX_PRESETS:
                                def set_len(p=p):
                                    cap_len.set_value(p)
                                    render_capacity()
                                ui.button(_fmt_tokens(p), on_click=set_len).props("dense flat no-caps size=sm")
                    cap_info = ui.label("").classes("text-xs text-gray-400 mt-2")
                    cap_rows = ui.column().classes("w-full gap-1 mt-2")

                with panel():
                    ui.label("Notes").classes("text-base font-semibold mb-2")
                    for line in (
                        "⚠ Upstream full-precision offloaded caches were unstable in earlier Windows testing. The "
                        "packed int4/int8 offload path is yyLab-owned and separate; validate quality/throughput on each model.",
                        "The stock resident cache uses Optimum Quanto INT4 in VRAM. yyLab's CPU-offloaded cache "
                        "supports both packed INT4 and INT8.",
                        "Quantized KV offload stores packed history in CPU RAM and materializes only the active layer "
                        "on the accelerator. It minimizes VRAM and host bytes at the cost of PCIe/dequantization latency.",
                        "Static variants pre-allocate the full max context; prefill chunking bounds prompt memory.",
                        "RoPE scaling stretches position encoding to extrapolate past the trained context length.",
                        "Adaptive selection runs before prefill, then rebuilds one chronological, contiguous prompt so "
                        "RoPE/YaRN positions and the KV cache stay coherent.",
                        "Rule of thumb: KV bytes/token = 2 · layers · kv_heads · head_dim · bytes_per_value.",
                    ):
                        ui.label(f"· {line}").classes("text-sm text-gray-300")

        def render_capacity() -> None:
            cap_rows.clear()
            if not cap_ckpt.value or not cap_len.value:
                cap_info.text = "Pick a checkpoint and context length."
                return
            c = db.q1("SELECT * FROM checkpoints WHERE id=?", (cap_ckpt.value,))
            if not c:
                cap_info.text = "Checkpoint not found."
                return
            try:
                cfg = _load_model_config(c["ref"])
            except Exception as e:  # noqa: BLE001
                cap_info.text = f"Could not load model config: {str(e).splitlines()[0][:140]}"
                return
            ctx = int(cap_len.value)
            quant_locked = c["quant_locked"] or "bf16"
            m_gb = model_weight_gb(c["params_b"], quant_locked) if c["params_b"] else 0.0
            vram_free = gpu.vram_free_gb if gpu else 0.0
            rows = capacity_rows(cfg, ctx, vram_free, ram_gb, m_gb)
            per_tok = kv_bytes_per_token(cfg, "bf16")
            max_ram = max_context_for_budget(cfg, max(ram_gb - 4.0, 0.0), "bf16")
            cap_info.text = (
                f'{c["name"]} · weights ≈ {m_gb:.2f} GB ({quant_locked}) · KV ≈ {per_tok / 1024:.1f} KB/token (bf16) · '
                f'VRAM {vram_free:.1f} GB free · RAM {ram_gb:.0f} GB · offload ceiling ≈ {_fmt_tokens(max_ram)} tokens'
            )
            with cap_rows:
                for r in rows:
                    with ui.row().classes("w-full items-center gap-3"):
                        ui.label(r["label"]).classes("w-52 text-sm text-gray-300")
                        ui.label(f'{r["kv_gb"]:,} GB').classes("w-24 text-sm font-mono")
                        ui.badge(r["location"]).props("outline color=grey")
                        if r["fits"]:
                            ui.badge("fits").props("color=green outline")
                        else:
                            ui.badge(f'too big (budget {r["budget_gb"]:,} GB)').props("color=red outline")

        cap_ckpt.on_value_change(lambda _: render_capacity())
        cap_len.on_value_change(lambda _: render_capacity())
