from __future__ import annotations

from nicegui import ui

from ... import db
from ..shell import panel, shell


@ui.page("/train/router")
def train_router_page():
    with shell("Train · Router", "/train/router"):
        ui.label("Router training (experts frozen)").classes("text-xl font-semibold")

        adapters = {a["id"]: f'{a["name"]} (E={a["num_experts"]}, r={a["rank"]}, {a["quant"]})' for a in db.q("SELECT * FROM adapters ORDER BY name")}

        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
            with panel().classes("w-[440px]"):
                ui.label("Setup").classes("text-base font-semibold mb-2")
                if not adapters:
                    ui.label("No adapters yet — train a MoLE adapter first.").classes("text-sm text-gray-400")
                adp_sel = ui.select(adapters, label="Adapter (experts frozen)").props("dense outlined").classes("w-full")
                ui.number("Active experts per token (top-K)", value=2, min=1, max=16).props(
                    "dense outlined"
                ).classes("w-full")
                ui.label("Top-1, Top-2, … Top-K routing. Higher K = denser, more capacity per token.").classes(
                    "text-xs text-gray-500"
                )
                with ui.row().classes("w-full gap-3"):
                    ui.number("Epochs", value=1, min=1, max=20).props("dense outlined").classes("w-24")
                    ui.select([1e-4, 3e-4, 5e-5, 1e-5], value=1e-4, label="LR").props("dense outlined").classes("w-28")
                ui.number("Load-balance loss weight", value=0.01, min=0, max=1, step=0.005).props(
                    "dense outlined"
                ).classes("w-full")
                start_btn = ui.button("Start router training", icon="play_arrow").props(
                    "no-caps unelevated"
                ).classes("w-full mt-3")
                ui.label("Training runtime lands in phase 3.").classes("text-xs text-gray-500")

            with panel().classes("flex-1 min-w-[380px]"):
                ui.label("Why a separate stage?").classes("text-base font-semibold mb-2")
                ui.label(
                    "Stage 1 (adapter training) co-trains experts with a zero-init router so routing starts uniform. "
                    "Stage 2 freezes the experts and trains only the router gates on the same data — cheap, and lets you "
                    "sweep top-K or re-balance experts without re-training LoRA weights."
                ).classes("text-sm text-gray-300")

        def on_start() -> None:
            if not adp_sel.value:
                ui.notify("Choose an adapter", type="warning")
                return
            ui.notify("Config validated. Trainer runtime arrives in phase 3.", type="info")

        start_btn.on_click(on_start)
