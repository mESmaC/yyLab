from __future__ import annotations

from nicegui import ui

from ...config import get_settings, update_settings
from ...ingest.parallel import auto_worker_count
from ...ingest.tokenize_stats import DEFAULT_SPECS
from ...models.bitsandbytes_capabilities import (
    available_bitsandbytes_quantizations,
    bitsandbytes_compute_precision,
)
from ...models.precision_capabilities import model_execution_device
from ..shell import panel, shell


@ui.page("/settings")
def settings_page():
    with shell("Settings", "/settings"):
        ui.label("Settings").classes("text-xl font-semibold")
        s = get_settings()
        enabled = dict(s.get("tokenizers_enabled", {}))
        bnb_quantizations = available_bitsandbytes_quantizations(model_execution_device())
        bnb_quantization_options = {
            quantization: (
                f"{quantization.upper()} · "
                f"{bitsandbytes_compute_precision(quantization).upper()} compute"
            )
            for quantization in bnb_quantizations
        }

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
            selected_quant = (
                saved_quant
                if saved_quant in bnb_quantizations
                else (bnb_quantizations[0] if bnb_quantizations else None)
            )
            quant_def = ui.select(
                bnb_quantization_options,
                value=selected_quant,
                label="Default quant for QLoRA",
            ).props("dense outlined").classes("w-56")
            quant_def.enabled = bool(bnb_quantizations)
            if not bnb_quantizations:
                ui.label(
                    "QLoRA has no verified bitsandbytes CUDA precision on this GPU."
                ).classes("text-xs text-amber-400")
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
                    default_quant=str(quant_def.value or ""),
                    tokenizers_enabled={k: bool(c.value) for k, c in checks.items()},
                    ingest_workers=max(0, int(workers_in.value or 0)),
                )
                ui.notify("Settings saved", type="positive")

            ui.button("Save", icon="save", on_click=save).props("no-caps").classes("mt-3")
