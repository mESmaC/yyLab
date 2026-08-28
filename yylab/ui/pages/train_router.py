from __future__ import annotations

import json

from nicegui import ui

from ... import db
from ...train.data import estimate_steps
from ...workers.launch import launch_worker
from ..components import field_grid, form_section, hint
from ..shell import ACCENT, panel, shell


@ui.page("/train/router")
def train_router_page():
    with shell("Train · Router", "/train/router"):
        ui.label("Router training (experts frozen)").classes("text-xl font-semibold")
        ui.label("Stage 2: freeze the experts and train only the top-K router gates.").classes(
            "text-sm text-gray-400 -mt-2"
        )

        adapters = {
            a["id"]: f'{a["name"]} (E={a["num_experts"]}, r={a["rank"]}, {a["quant"]}, {a["status"]})'
            for a in db.q("SELECT * FROM adapters WHERE arch='mole' ORDER BY created_at DESC")
        }

        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
            with panel().classes("w-[560px] shrink-0"):
                form_section("1", "Adapter", "Experts stay frozen; only router gates update")
                if not adapters:
                    ui.label("No MoLE adapters yet — train one on the MoLE page first.").classes(
                        "text-sm text-amber-400 px-3 py-2 rounded-md"
                    ).style("background: #f59e0b14")
                adp_sel = ui.select(adapters, label="Adapter (experts frozen)").props("dense outlined").classes("w-full")
                top_k = ui.number("Active experts per token (top-K)", value=2, min=1, max=16).props(
                    "dense outlined"
                ).classes("w-full")
                hint("Top-1, Top-2, … Top-K routing. Higher K = denser, more capacity per token.")

                ui.separator().classes("my-3")
                form_section("2", "Training run")
                with field_grid(3):
                    seq_sel = ui.select([512, 1024, 2048, 4096], value=2048, label="Seq length").props(
                        "dense outlined"
                    ).classes("w-full")
                    epochs = ui.number("Epochs", value=1, min=1, max=20).props("dense outlined").classes("w-full")
                    lr = ui.select([1e-4, 3e-4, 5e-5, 1e-5], value=1e-4, label="Learning rate").props(
                        "dense outlined"
                    ).classes("w-full")
                aux_w = ui.number("Load-balance loss weight", value=0.01, min=0, max=1, step=0.005).props(
                    "dense outlined"
                ).classes("w-full")
                with ui.expansion("Advanced", icon="tune").classes("w-full"):
                    with field_grid(2):
                        micro_bs = ui.number("Micro batch", value=1, min=1, max=16).props("dense outlined").classes("w-full")
                        grad_accum = ui.number("Grad accum", value=16, min=1, max=256).props("dense outlined").classes("w-full")

                start_btn = ui.button("Start router training", icon="play_arrow").props(
                    "no-caps unelevated"
                ).classes("w-full mt-4")
                hint("Runs in a background worker — follow live progress on the Runs page.")

            with ui.column().classes("flex-1 gap-4 min-w-[360px]"):
                with panel():
                    ui.label("Stage summary").classes("text-base font-semibold mb-2")
                    info_lbl = ui.label("Select an adapter to estimate the run.").classes(
                        "text-xs text-gray-400 px-3 py-2 rounded-md w-full"
                    ).style(f"background: {ACCENT}10")
                with panel():
                    with ui.expansion("Why a separate stage?", icon="help_outline", value=True).classes("w-full"):
                        ui.label(
                            "Stage 1 co-trains experts with a zero-init router so routing starts uniform. Stage 2 "
                            "freezes the experts and trains only the router gates on the same corpora — cheap, and it "
                            "lets you sweep top-K or re-balance experts without re-training LoRA weights. The snapshot "
                            "is saved as a full MoLE weight set (experts + new router)."
                        ).classes("text-sm text-gray-300")

        def update_summary() -> None:
            if not adp_sel.value:
                info_lbl.text = "Select an adapter to estimate the run."
                return
            a = db.q1("SELECT * FROM adapters WHERE id=?", (adp_sel.value,))
            if not a:
                info_lbl.text = "Select an adapter to estimate the run."
                return
            corpora = json.loads(a["corpora_json"] or "[]")
            seq = int(seq_sel.value or 2048)
            mb = int(micro_bs.value or 1)
            ga = int(grad_accum.value or 16)
            n_epochs = int(epochs.value or 1)
            if not corpora:
                info_lbl.text = f'{a["name"]} · no corpora recorded on this adapter.'
                return
            steps = estimate_steps(corpora, None, n_epochs, seq, mb, ga)
            info_lbl.text = (
                f'{a["name"]} · {len(corpora)} corpora · E={a["num_experts"]} → top-K={int(top_k.value or 2)} · '
                f'~{steps} steps'
            )

        def on_start() -> None:
            if not adp_sel.value:
                ui.notify("Choose an adapter", type="warning")
                return
            a = db.q1("SELECT * FROM adapters WHERE id=?", (adp_sel.value,))
            if not a or a["status"] != "trained":
                ui.notify("Adapter must be trained first", type="warning")
                return
            ckpt = db.q1("SELECT * FROM checkpoints WHERE id=?", (a["checkpoint_id"],))
            corpora = json.loads(a["corpora_json"] or "[]")
            if not corpora:
                ui.notify("Adapter has no corpora recorded", type="warning")
                return

            seq = int(seq_sel.value or 2048)
            mb = int(micro_bs.value or 1)
            ga = int(grad_accum.value or 16)
            n_epochs = int(epochs.value or 1)
            max_steps = estimate_steps(corpora, None, n_epochs, seq, mb, ga)

            router_id = db.new_id("rtr")
            db.execute(
                "INSERT INTO routers(id, name, created_at, adapter_id, top_k, status) VALUES(?,?,?,?,?,?)",
                (router_id, f'{a["name"]}-router-k{int(top_k.value or 2)}', db.now_iso(), a["id"], int(top_k.value or 2), "training"),
            )
            rid = db.new_id("run")
            config = {
                "checkpoint_id": ckpt["id"],
                "checkpoint_ref": ckpt["ref"],
                "adapter_id": a["id"],
                "router_id": router_id,
                "top_k": int(top_k.value or 2),
                "corpora": corpora,
                "domains": None,
                "seq_len": seq,
                "epochs": n_epochs,
                "micro_batch": mb,
                "grad_accum": ga,
                "lr": float(lr.value or 1e-4),
                "aux_weight": float(aux_w.value or 0.01),
                "grad_ckpt": True,
                "max_steps": max_steps,
                "log_every": 5,
                "save_every": None,
            }
            db.execute(
                "INSERT INTO runs(id, kind, status, created_at, config_json) VALUES(?,?,?,?,?)",
                (rid, "router_stage2", "queued", db.now_iso(), json.dumps(config)),
            )
            pid = launch_worker(rid)
            db.execute("UPDATE runs SET metrics_json=? WHERE id=?", (json.dumps({"pid": pid}), rid))
            ui.notify(f"Router training started (~{max_steps} steps, pid {pid}). See Runs.", type="positive")

        for el in (adp_sel, top_k, seq_sel, epochs, micro_bs, grad_accum):
            el.on_value_change(lambda _: update_summary())
        start_btn.on_click(on_start)
        update_summary()
