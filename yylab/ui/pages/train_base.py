from __future__ import annotations

import json

from nicegui import ui

from ... import db
from ...hw import best_gpu
from ...train.data import estimate_steps
from ...workers.launch import launch_worker
from ..shell import panel, shell
from .train_mole import corpus_options, render_plan


@ui.page("/train/base")
def train_base_page():
    with shell("Train · Base", "/train/base"):
        ui.label("Trainable base (continued pretraining / finetune)").classes("text-xl font-semibold")

        ckpts = {c["id"]: f'{c["name"]} ({c["params_b"]:.2f} B)' if c["params_b"] else c["name"] for c in db.q("SELECT * FROM checkpoints ORDER BY name")}
        corpora = corpus_options()

        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
            with panel().classes("w-[440px]"):
                ui.label("Setup").classes("text-base font-semibold mb-2")
                name_in = ui.input("Output name", placeholder="auto").props("dense outlined").classes("w-full")
                ckpt_sel = ui.select(ckpts, label="Checkpoint to train").props("dense outlined").classes("w-full")
                corp_sel = ui.select(corpora, multiple=True, label="Corpora").props(
                    "dense outlined use-chips"
                ).classes("w-full")
                method = ui.select(
                    {
                        "auto": "Auto (recommended)",
                        "full": "Full fine-tune",
                        "lora": "LoRA",
                        "qlora": "QLoRA",
                    },
                    value="auto",
                    label="Method",
                ).props("dense outlined").classes("w-full")
                quant = ui.select({"nf4": "NF4", "int8": "INT8", "fp8": "FP8"}, value="nf4", label="Quant (QLoRA)").props(
                    "dense outlined"
                ).classes("w-full")
                with ui.row().classes("w-full gap-3"):
                    seq_sel = ui.select([512, 1024, 2048, 4096, 8192, 16384], value=2048, label="Context window").props(
                        "dense outlined"
                    ).classes("w-32")
                    epochs = ui.number("Epochs", value=1, min=1, max=20).props("dense outlined").classes("w-24")
                    lr = ui.select([1e-4, 3e-4, 5e-5, 1e-5, 5e-6], value=5e-5, label="LR").props("dense outlined").classes("w-28")
                with ui.row().classes("w-full gap-3"):
                    rank = ui.number("LoRA rank", value=32, min=1, max=256).props("dense outlined").classes("w-28")
                    micro_bs = ui.number("Micro batch", value=1, min=1, max=16).props("dense outlined").classes("w-28")
                    grad_accum = ui.number("Grad accum", value=16, min=1, max=256).props("dense outlined").classes("w-28")
                start_btn = ui.button("Start training", icon="play_arrow").props("no-caps unelevated").classes("w-full mt-3")
                ui.label(
                    "Full fine-tunes become new checkpoints to train expert adapters on. LoRA/QLoRA become adapters. "
                    "Runs in a background worker."
                ).classes("text-xs text-gray-500")

            with ui.column().classes("flex-1 gap-4 min-w-[380px]"):
                with panel():
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
                rank=int(rank.value or 32),
                num_experts=1,
            )

        def resolve_method() -> tuple[str, str]:
            m = method.value
            if m == "auto":
                gpu = best_gpu()
                p = params_b_of()
                if gpu and p and p * 2.3 < gpu.vram_free_gb:
                    m = "lora"
                else:
                    m = "qlora"
            q = quant.value if m == "qlora" else "bf16"
            return m, q

        def on_start() -> None:
            problems = []
            if not ckpt_sel.value:
                problems.append("choose a checkpoint")
            if not corp_sel.value:
                problems.append("choose corpora")
            if problems:
                ui.notify("Fix: " + ", ".join(problems), type="warning")
                return

            ckpt = db.q1("SELECT * FROM checkpoints WHERE id=?", (ckpt_sel.value,))
            m_resolved, q_resolved = resolve_method()
            if m_resolved != "full" and ckpt["quant_locked"] and ckpt["quant_locked"] != q_resolved:
                ui.notify(
                    f'Checkpoint is quant-locked to {ckpt["quant_locked"]}.', type="negative"
                )
                return

            seq = int(seq_sel.value or 2048)
            mb = int(micro_bs.value or 1)
            ga = int(grad_accum.value or 16)
            n_epochs = int(epochs.value or 1)
            max_steps = estimate_steps(list(corp_sel.value), None, n_epochs, seq, mb, ga)
            base_name = (name_in.value or "").strip() or f'{ckpt["name"]}-{m_resolved}'

            config = {
                "checkpoint_id": ckpt["id"],
                "checkpoint_ref": ckpt["ref"],
                "method": m_resolved,
                "quant": q_resolved,
                "rank": int(rank.value or 32),
                "alpha": int(rank.value or 32) * 2,
                "corpora": list(corp_sel.value),
                "domains": None,
                "seq_len": seq,
                "epochs": n_epochs,
                "micro_batch": mb,
                "grad_accum": ga,
                "lr": float(lr.value or 5e-5),
                "grad_ckpt": True,
                "max_steps": max_steps,
                "log_every": 5,
                "save_every": None,
            }

            if m_resolved == "full":
                out_id = db.new_id("ckpt")
                db.execute(
                    "INSERT INTO checkpoints(id, name, created_at, kind, source, ref) VALUES(?,?,?,?,?,?)",
                    (out_id, base_name, db.now_iso(), "pending", "pending", ""),
                )
                config["out_checkpoint_id"] = out_id
            else:
                aid = db.new_id("adp")
                db.execute(
                    "INSERT INTO adapters(id, name, created_at, checkpoint_id, arch, quant, rank, corpora_json, status)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (aid, base_name, db.now_iso(), ckpt["id"], "lora", q_resolved, int(rank.value or 32),
                     json.dumps(list(corp_sel.value)), "training"),
                )
                config["adapter_id"] = aid

            rid = db.new_id("run")
            db.execute(
                "INSERT INTO runs(id, kind, status, created_at, config_json) VALUES(?,?,?,?,?)",
                (rid, "base", "queued", db.now_iso(), json.dumps(config)),
            )
            pid = launch_worker(rid)
            db.execute("UPDATE runs SET metrics_json=? WHERE id=?", (json.dumps({"pid": pid}), rid))
            ui.notify(f"Base training started — {base_name} (~{max_steps} steps, pid {pid}).", type="positive")

        for el in (ckpt_sel, method, seq_sel, quant):
            el.on_value_change(lambda _: update_plan())
        start_btn.on_click(on_start)
        update_plan()
