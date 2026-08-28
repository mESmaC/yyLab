from __future__ import annotations

from nicegui import ui

from ...config import get_settings, update_settings
from ...ingest.tokenize_stats import DEFAULT_SPECS
from ..shell import panel, shell


@ui.page("/settings")
def settings_page():
    with shell("Settings", "/settings"):
        ui.label("Settings").classes("text-xl font-semibold")
        s = get_settings()
        enabled = dict(s.get("tokenizers_enabled", {}))

        with panel().classes("max-w-[560px]"):
            ui.label("Hugging Face").classes("text-base font-semibold mb-2")
            token_in = ui.input("HF token (for gated tokenizers/models)", password=True).props(
                "dense outlined"
            ).classes("w-full")
            token_in.value = s.get("hf_token", "")
            ui.label("Used to download gated tokenizer repos (e.g. Gemma) and hub checkpoints.").classes(
                "text-xs text-gray-500"
            )

        with panel().classes("max-w-[560px]"):
            ui.label("Corpus tokenizers").classes("text-base font-semibold mb-2")
            checks: dict[str, object] = {}
            for spec in DEFAULT_SPECS:
                checks[spec.key] = ui.checkbox(f"{spec.label} ({spec.key})", value=enabled.get(spec.key, True))

        with panel().classes("max-w-[560px]"):
            ui.label("Defaults").classes("text-base font-semibold mb-2")
            quant_def = ui.select(
                {"nf4": "NF4", "int8": "INT8", "fp8": "FP8"},
                value=s.get("default_quant", "nf4"),
                label="Default quant for QLoRA",
            ).props("dense outlined").classes("w-56")

            def save() -> None:
                update_settings(
                    hf_token=(token_in.value or "").strip(),
                    default_quant=quant_def.value,
                    tokenizers_enabled={k: bool(c.value) for k, c in checks.items()},
                )
                ui.notify("Settings saved", type="positive")

            ui.button("Save", icon="save", on_click=save).props("no-caps").classes("mt-3")
