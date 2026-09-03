from __future__ import annotations

from nicegui import ui

from ...config import get_settings, update_settings
from ...ingest.parallel import auto_worker_count
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
            saved_quant = s.get("default_quant", "nf4")
            quant_def = ui.select(
                {"nf4": "NF4", "int8": "INT8"},
                value=saved_quant if saved_quant in {"nf4", "int8"} else "nf4",
                label="Default quant for QLoRA",
            ).props("dense outlined").classes("w-56")
            try:
                saved_workers = int(s.get("ingest_workers") or 0)
            except (TypeError, ValueError):
                saved_workers = 0
            workers_in = ui.number(
                "Ingest worker processes (0 = auto)",
                value=saved_workers,
                min=0,
                max=64,
                step=1,
            ).props("dense outlined").classes("w-56")
            ui.label(
                f"Auto currently means {auto_worker_count()} workers on this machine. "
                "Each worker extracts and analyzes documents in its own process; the "
                "ingest dialog can override this per run."
            ).classes("text-xs text-gray-500")

            def save() -> None:
                update_settings(
                    hf_token=(token_in.value or "").strip(),
                    default_quant=quant_def.value,
                    tokenizers_enabled={k: bool(c.value) for k, c in checks.items()},
                    ingest_workers=max(0, int(workers_in.value or 0)),
                )
                ui.notify("Settings saved", type="positive")

            ui.button("Save", icon="save", on_click=save).props("no-caps").classes("mt-3")
