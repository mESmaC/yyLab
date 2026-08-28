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
                        "static": "Static (pre-allocated, in VRAM)",
                        "sliding_window": "Sliding window (deprecated upstream)",
                    },
                    value=engine["cache_implementation"],
                    label="KV cache implementation",
                ).props("dense outlined").classes("w-full")
                kvq = ui.select(
                    {"none": "None", "int8": "INT8 (~half cache memory)", "int4": "INT4 (~quarter cache memory)"},
                    value=engine["kv_quant"] if engine["kv_quant"] in {"none", "int8", "int4"} else "none",
                    label="KV quantization (quantized cache)",
                ).props("dense outlined").classes("w-full")
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

                def save() -> None:
                    update_settings(
                        engine={
                            "attn_implementation": attn.value,
                            "cache_implementation": cache.value,
                            "kv_quant": kvq.value,
                            "sliding_window": int(sw.value or 4096),
                            "prefill_chunk_size": int(chunk.value or 8192),
                            "max_context_tokens": int(maxctx.value or 8192),
                            "rope_scaling": rope.value,
                            "rope_factor": float(factor.value or 2.0),
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
                        "⚠ Offloaded caches are UNSTABLE on this Windows + torch + transformers build: NF4 produced "
                        "garbage output and BF16 hard-crashed in testing. Prefer quantized / static / dynamic for now.",
                        "Quantized KV (int8/int4) stays in VRAM: 2×/4× longer context for the same VRAM.",
                        "Static variants pre-allocate the full max context; prefill chunking bounds prompt memory.",
                        "RoPE scaling stretches position encoding to extrapolate past the trained context length.",
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
