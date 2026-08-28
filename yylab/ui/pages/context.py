from __future__ import annotations

from nicegui import ui

from ...config import get_settings, update_settings
from ..shell import panel, shell


@ui.page("/context")
def context_page():
    with shell("Context & KV-Cache", "/context"):
        ui.label("Context window & KV-cache settings").classes("text-xl font-semibold")
        ui.label("Applied at chat/inference load time and used as defaults for training context.").classes(
            "text-sm text-gray-400"
        )

        engine = dict(get_settings()["engine"])

        with panel().classes("max-w-[560px]"):
            attn = ui.select(
                {"sdpa": "SDPA (recommended on Windows)", "eager": "Eager"},
                value=engine["attn_implementation"],
                label="Attention implementation",
            ).props("dense outlined").classes("w-full")
            cache = ui.select(
                {
                    "dynamic": "Dynamic (default)",
                    "quantized": "Quantized KV cache (bitsandbytes)",
                    "sliding_window": "Sliding window",
                    "static": "Static (pre-allocated)",
                },
                value=engine["cache_implementation"],
                label="KV cache implementation",
            ).props("dense outlined").classes("w-full")
            kvq = ui.select(
                {"none": "None", "int8": "INT8 (~half cache memory)", "fp8": "FP8 (Ada/Blackwell)"},
                value=engine["kv_quant"],
                label="KV quantization",
            ).props("dense outlined").classes("w-full")
            sw = ui.number("Sliding window size", value=engine["sliding_window"], min=256, max=65536).props(
                "dense outlined"
            ).classes("w-full")
            maxctx = ui.number("Max context tokens", value=engine["max_context_tokens"], min=512, max=262144).props(
                "dense outlined"
            ).classes("w-full")
            rope = ui.select(
                {"none": "None", "linear": "Linear", "dynamic": "Dynamic NTK"},
                value=engine["rope_scaling"],
                label="RoPE scaling (extend context beyond pretrain)",
            ).props("dense outlined").classes("w-full")
            factor = ui.number("RoPE factor", value=engine["rope_factor"], min=1.0, max=32.0, step=0.5).props(
                "dense outlined"
            ).classes("w-full")

            def save() -> None:
                update_settings(
                    engine={
                        "attn_implementation": attn.value,
                        "cache_implementation": cache.value,
                        "kv_quant": kvq.value,
                        "sliding_window": int(sw.value or 4096),
                        "max_context_tokens": int(maxctx.value or 8192),
                        "rope_scaling": rope.value,
                        "rope_factor": float(factor.value or 2.0),
                    }
                )
                ui.notify("Engine settings saved", type="positive")

            ui.button("Save", icon="save", on_click=save).props("no-caps").classes("mt-3")

        with panel().classes("max-w-[560px]"):
            ui.label("Notes").classes("text-base font-semibold mb-2")
            for line in (
                "Quantized KV cache trades a little quality for ~2× longer context in the same VRAM.",
                "Sliding window keeps attention local beyond the window — best for long code/chat sessions.",
                "RoPE scaling stretches position encoding to extrapolate past the trained context length.",
                "FP8 KV needs an Ada/Blackwell GPU (RTX 40/50 series).",
            ):
                ui.label(f"· {line}").classes("text-sm text-gray-300")
