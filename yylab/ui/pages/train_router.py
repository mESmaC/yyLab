from __future__ import annotations

import json

from nicegui import ui

from ... import db
from ...train.data import DEFAULT_TEXT_CHUNK_CHARS, estimate_steps, tokenizer_stats_key
from ...workers.launch import launch_worker
from ..components import field_grid, form_section, hint
from ..shell import ACCENT, panel, shell
from .train_mole import _checkpoint_shape, _make_plan


@ui.page("/train/router")
def train_router_page():
    with shell("Train · Router", "/train/router"):
        ui.label("Router training (experts frozen)").classes("text-xl font-semibold")
        ui.label("Stage 2: freeze the experts and train one shared prompt-level top-K router.").classes(
            "text-sm text-gray-400 -mt-2"
        )

        all_adapters = db.q("SELECT * FROM adapters WHERE arch='mole' ORDER BY created_at DESC")

        def is_joint_token_adapter(adapter: dict) -> bool:
            try:
                metadata = json.loads(adapter.get("meta_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            return metadata.get("training_strategy") == "joint_per_token_v1"

        adapters = {
            a["id"]: f"{a['name']} (E={a['num_experts']}, r={a['rank']}, {a['quant']}, {a['status']})"
            for a in all_adapters
            if not is_joint_token_adapter(a)
        }
        joint_count = sum(is_joint_token_adapter(adapter) for adapter in all_adapters)

        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
            with panel().classes("w-[560px] shrink-0"):
                form_section("1", "Adapter", "Experts stay frozen; only the shared router updates")
                if joint_count:
                    ui.label(
                        f"{joint_count} learned per-token adapter(s) already include their routers and need no stage 2."
                    ).classes("text-sm text-green-400 px-3 py-2 rounded-md").style("background: #22c55e14")
                if not adapters:
                    ui.label("No legacy isolated MoLE adapters need prompt-router training.").classes(
                        "text-sm text-amber-400 px-3 py-2 rounded-md"
                    ).style("background: #f59e0b14")
                adp_sel = (
                    ui.select(adapters, label="Adapter (experts frozen)")
                    .props("dense outlined")
                    .classes("w-full")
                )
                top_k = (
                    ui.number("Active experts per prompt (top-K)", value=2, min=1, max=16)
                    .props("dense outlined")
                    .classes("w-full")
                )
                hint(
                    "One decision is reused across every adapted projection and generated token in the prompt."
                )

                ui.separator().classes("my-3")
                form_section("2", "Training run")
                with field_grid(3):
                    seq_sel = (
                        ui.select([512, 1024, 2048, 4096], value=2048, label="Seq length")
                        .props("dense outlined")
                        .classes("w-full")
                    )
                    epochs = (
                        ui.number("Epochs", value=1, min=1, max=20).props("dense outlined").classes("w-full")
                    )
                    lr = (
                        ui.select([1e-4, 3e-4, 5e-5, 1e-5], value=1e-4, label="Learning rate")
                        .props("dense outlined")
                        .classes("w-full")
                    )
                aux_w = (
                    ui.number("Load-balance loss weight", value=0.01, min=0, max=1, step=0.005)
                    .props("dense outlined")
                    .classes("w-full")
                )
                with ui.expansion("Advanced", icon="tune").classes("w-full"):
                    auto_batch = ui.checkbox("Auto-select fastest safe batch profile", value=True)
                    with field_grid(2):
                        micro_bs = (
                            ui.number("Micro batch", value=4, min=1, max=16)
                            .props("dense outlined")
                            .classes("w-full")
                        )
                        grad_accum = (
                            ui.number("Grad accum", value=4, min=1, max=256)
                            .props("dense outlined")
                            .classes("w-full")
                        )

                start_btn = (
                    ui.button("Start router training", icon="play_arrow")
                    .props("no-caps unelevated")
                    .classes("w-full mt-4")
                )
                hint("Runs in a background worker — follow live progress on the Runs page.")

            with ui.column().classes("flex-1 gap-4 min-w-[360px]"):
                with panel():
                    ui.label("Stage summary").classes("text-base font-semibold mb-2")
                    info_lbl = (
                        ui.label("Select an adapter to estimate the run.")
                        .classes("text-xs text-gray-400 px-3 py-2 rounded-md w-full")
                        .style(f"background: {ACCENT}10")
                    )
                with panel():
                    with ui.expansion("Why a separate stage?", icon="help_outline", value=True).classes(
                        "w-full"
                    ):
                        ui.label(
                            "Stage 1 trains each specialist independently with one forced whole-adapter route and keeps "
                            "all router gates frozen. Stage 2 loads the assembled bank, freezes every expert, and trains "
                            "one small router from pooled prompt embeddings. Its top-K mixture is shared across all "
                            "adapted projections. This preserves specialist identity and lets you "
                            "sweep top-K or re-balance experts without re-training LoRA weights. The snapshot "
                            "is saved as a full MoLE weight set (experts + new router)."
                        ).classes("text-sm text-gray-300")

        def selected_context():
            if not adp_sel.value:
                return None, None, None
            adapter = db.q1("SELECT * FROM adapters WHERE id=?", (adp_sel.value,))
            if not adapter:
                return None, None, None
            checkpoint = db.q1("SELECT * FROM checkpoints WHERE id=?", (adapter["checkpoint_id"],))
            if not checkpoint:
                return adapter, None, None
            quant = str(adapter["quant"] or "bf16")
            method = "qlora" if quant in {"nf4", "int8"} else "lora"
            plan = _make_plan(
                checkpoint,
                int(seq_sel.value or 2048),
                method,
                quant,
                int(adapter["rank"] or 16),
                int(adapter["num_experts"] or 1),
                adapter_only=True,
                isolated_expert_training=False,
            )
            return adapter, checkpoint, plan

        def update_batch_controls() -> None:
            automatic = bool(auto_batch.value)
            micro_bs.enabled = not automatic
            grad_accum.enabled = not automatic
            if not automatic:
                return
            _, _, plan = selected_context()
            if plan is not None:
                micro_bs.value = plan.micro_batch
                grad_accum.value = plan.grad_accum
            else:
                micro_bs.value = 1
                grad_accum.value = 16

        def update_summary() -> None:
            if not adp_sel.value:
                info_lbl.text = "Select an adapter to estimate the run."
                return
            a, checkpoint, plan = selected_context()
            if not a:
                info_lbl.text = "Select an adapter to estimate the run."
                return
            corpora = json.loads(a["corpora_json"] or "[]")
            seq = int(seq_sel.value or 2048)
            mb = int(micro_bs.value or 1)
            ga = int(grad_accum.value or 16)
            n_epochs = int(epochs.value or 1)
            if not corpora:
                info_lbl.text = f"{a['name']} · no corpora recorded on this adapter."
                return
            stats_key = tokenizer_stats_key(_checkpoint_shape(checkpoint).get("model_type"))
            steps = estimate_steps(corpora, None, n_epochs, seq, mb, ga, stats_key)
            profile = "auto" if auto_batch.value and plan is not None else "manual"
            peak = f" · est peak {plan.est_peak_gb} GB" if plan is not None else ""
            info_lbl.text = (
                f"{a['name']} · {len(corpora)} corpora · E={a['num_experts']} → top-K={int(top_k.value or 2)} · "
                f"{profile} batch {mb} × accum {ga}{peak} · ~{steps} steps"
            )

        def on_start() -> None:
            if not adp_sel.value:
                ui.notify("Choose an adapter", type="warning")
                return
            a, ckpt, plan = selected_context()
            if not a or a["status"] != "trained":
                ui.notify("Adapter must be trained first", type="warning")
                return
            if not ckpt:
                ui.notify("The adapter's base checkpoint is missing", type="negative")
                return
            if auto_batch.value and plan is not None and not plan.feasible:
                ui.notify(
                    "The router plan does not pass the VRAM preflight at this sequence length.",
                    type="negative",
                )
                return
            if int(top_k.value or 1) > int(a["num_experts"] or 1):
                ui.notify("Top-K cannot exceed the adapter's expert count", type="warning")
                return
            corpora = json.loads(a["corpora_json"] or "[]")
            if not corpora:
                ui.notify("Adapter has no corpora recorded", type="warning")
                return

            seq = int(seq_sel.value or 2048)
            mb = int(micro_bs.value or 1)
            ga = int(grad_accum.value or 16)
            n_epochs = int(epochs.value or 1)
            stats_key = tokenizer_stats_key(_checkpoint_shape(ckpt).get("model_type"))
            max_steps = estimate_steps(corpora, None, n_epochs, seq, mb, ga, stats_key)

            router_id = db.new_id("rtr")
            db.execute(
                "INSERT INTO routers(id, name, created_at, adapter_id, top_k, status) VALUES(?,?,?,?,?,?)",
                (
                    router_id,
                    f"{a['name']}-router-k{int(top_k.value or 2)}",
                    db.now_iso(),
                    a["id"],
                    int(top_k.value or 2),
                    "training",
                ),
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
                "aux_weight": float(0.01 if aux_w.value is None else aux_w.value),
                "grad_ckpt": True,
                "liger_fused_ce": True,
                "liger_full_kernels": True,
                "cudnn_sdpa": True,
                "token_stats_key": stats_key,
                "router_hidden_size": 256,
                "router_max_tokens": 256,
                "seed": 42,
                "shuffle_buffer": 128,
                "shuffle_buffer_bytes": 64 * 2**20,
                "tokenize_chunk": 8,
                "max_text_chars": DEFAULT_TEXT_CHUNK_CHARS,
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

        def refresh_batch_plan() -> None:
            update_batch_controls()
            update_summary()

        for el in (adp_sel, seq_sel):
            el.on_value_change(lambda _: refresh_batch_plan())
        auto_batch.on_value_change(lambda _: refresh_batch_plan())
        for el in (top_k, epochs, micro_bs, grad_accum):
            el.on_value_change(lambda _: update_summary())
        start_btn.on_click(on_start)
        update_batch_controls()
        update_summary()
