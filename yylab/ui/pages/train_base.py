from __future__ import annotations

from nicegui import ui

from ... import db
from ..shell import panel, shell
from .train_mole import render_plan


@ui.page("/train/base")
def train_base_page():
    with shell("Train · Base", "/train/base"):
        ui.label("Trainable base (continued pretraining / finetune)").classes("text-xl font-semibold")

        ckpts = {c["id"]: f'{c["name"]} ({c["params_b"]:.2f} B)' if c["params_b"] else c["name"] for c in db.q("SELECT * FROM checkpoints ORDER BY name")}

        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
            with panel().classes("w-[440px]"):
                ui.label("Setup").classes("text-base font-semibold mb-2")
                ckpt_sel = ui.select(ckpts, label="Checkpoint to train").props("dense outlined").classes("w-full")
                method = ui.select(
                    {"auto": "Auto", "full": "Full fine-tune", "lora": "LoRA", "qlora": "QLoRA"},
                    value="auto",
                    label="Method",
                ).props("dense outlined").classes("w-full")
                with ui.row().classes("w-full gap-3"):
                    seq_sel = ui.select([512, 1024, 2048, 4096, 8192, 16384], value=2048, label="Context window").props(
                        "dense outlined"
                    ).classes("w-32")
                    ui.number("Epochs", value=1, min=1, max=20).props("dense outlined").classes("w-24")
                quant = ui.select({"nf4": "NF4", "int8": "INT8", "fp8": "FP8"}, value="nf4", label="Quant (QLoRA)").props(
                    "dense outlined"
                ).classes("w-full")
                ui.select(
                    {"none": "None", "linear": "Linear (YaRN-lite)", "dynamic": "Dynamic NTK"},
                    value="none",
                    label="RoPE scaling for longer context",
                ).props("dense outlined").classes("w-full")
                start_btn = ui.button("Start training", icon="play_arrow").props("no-caps unelevated").classes("w-full mt-3")
                ui.label(
                    "Outputs become new checkpoints that expert adapters can be trained on top of. "
                    "Training runtime lands in phase 3."
                ).classes("text-xs text-gray-500")

            with ui.column().classes("flex-1 gap-4 min-w-[380px]"), panel():
                ui.label("Auto training plan").classes("text-base font-semibold mb-2")
                plan_box = ui.column().classes("w-full gap-1")

        def params_b_of() -> float:
            if not ckpt_sel.value:
                return 0.0
            row = db.q1("SELECT params_b FROM checkpoints WHERE id=?", (ckpt_sel.value,))
            return row["params_b"] if row else 0.0

        def update_plan() -> None:
            render_plan(
                plan_box,
                params_b_of(),
                int(seq_sel.value or 2048),
                method.value,
                quant.value,
                rank=32,
                num_experts=1,
            )

        def on_start() -> None:
            if not ckpt_sel.value:
                ui.notify("Choose a checkpoint", type="warning")
                return
            ui.notify("Config validated. Trainer runtime arrives in phase 3.", type="info")

        for el in (ckpt_sel, method, seq_sel, quant):
            el.on_value_change(lambda _: update_plan())
        start_btn.on_click(on_start)
        update_plan()
