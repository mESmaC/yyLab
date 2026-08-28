from __future__ import annotations

import orjson
from nicegui import ui

from ... import db
from ...hw import best_gpu, has_triton
from ...mole.inject import DEFAULT_TARGET_MODULES
from ...train.planner import PlanInput, plan_training
from ..shell import panel, shell


def render_plan(container, params_b: float, seq_len: int, method: str, quant: str, rank: int, num_experts: int) -> None:
    container.clear()
    with container:
        gpu = best_gpu()
        if gpu is None:
            ui.label("No GPU detected — planner needs CUDA. CPU fallback training is not recommended.").classes(
                "text-amber-400 text-sm"
            )
            return
        if not params_b:
            ui.label(
                f"GPU: {gpu.name} ({gpu.vram_free_gb:.1f} GB free). Model size unknown — plan appears once checkpoint params are known."
            ).classes("text-sm text-gray-400")
            return
        plan = plan_training(
            PlanInput(
                params_b=params_b,
                vram_gb=gpu.vram_free_gb,
                seq_len=seq_len,
                method=method,
                quant=quant,
                lora_rank=rank,
                num_experts=num_experts,
                sm_version=gpu.compute_capability,
                has_triton=has_triton(),
            )
        )
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            ui.badge(f'{plan.method} · {plan.quant}').props("color=primary" if plan.feasible else "color=negative")
            ui.badge(f'est peak {plan.est_peak_gb} GB / {gpu.vram_free_gb:.0f} GB').props(
                "outline" + (" color=green" if plan.feasible else " color=negative")
            )
            ui.badge(f'bs {plan.micro_batch} × accum {plan.grad_accum}').props("outline color=grey")
            if plan.grad_ckpt:
                ui.badge("grad ckpt").props("outline color=grey")
            if plan.offload:
                ui.badge("CPU offload").props("outline color=orange")
            if plan.use_compile:
                ui.badge("torch.compile").props("outline color=teal")
        for n in plan.notes:
            ui.label(f"· {n}").classes("text-xs text-gray-400")


def corpus_options() -> dict[str, str]:
    out = {}
    for c in db.q("SELECT id, name, stats_json FROM corpora WHERE status='ready' ORDER BY name"):
        stats = orjson.loads(c["stats_json"] or b"{}")
        doms = stats.get("domains", {})
        top = ""
        if doms:
            top = max(doms.items(), key=lambda kv: kv[1].get("chars", 0))[0]
        out[c["id"]] = f'{c["name"]}' + (f" [{top}]" if top else "")
    return out


def corpus_domains(cid: str) -> set[str]:
    row = db.q1("SELECT stats_json FROM corpora WHERE id=?", (cid,))
    if not row:
        return set()
    return set(orjson.loads(row["stats_json"] or b"{}").get("domains", {}).keys())


@ui.page("/train/mole")
def train_mole_page():

    with shell("Train · Mixture of LoRA Experts", "/train/mole"):
        ui.label("Adapter training (frozen base, MoLE)").classes("text-xl font-semibold")

        ckpts = {c["id"]: f'{c["name"]} ({c["params_b"]:.2f} B)' if c["params_b"] else c["name"] for c in db.q("SELECT * FROM checkpoints ORDER BY name")}
        corpora = corpus_options()

        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
            with panel().classes("w-[440px]"):
                ui.label("Setup").classes("text-base font-semibold mb-2")
                ckpt_sel = ui.select(ckpts, label="Base checkpoint (frozen)").props("dense outlined").classes("w-full")
                corp_sel = ui.select(corpora, multiple=True, label="Corpora").props(
                    "dense outlined use-chips"
                ).classes("w-full")
                with ui.row().classes("gap-2"):
                    for dom in ("code", "math", "language", "all"):
                        def pick(dom=dom):
                            if dom == "all":
                                corp_sel.value = list(corpora.keys())
                            else:
                                corp_sel.value = [cid for cid in corpora if dom in corpus_domains(cid)]
                            corp_sel.update()
                        ui.button(dom, on_click=pick).props("dense flat no-caps")
                ui.label("Domain quick-picks select all corpora containing that domain.").classes(
                    "text-xs text-gray-500"
                )

                ui.separator().classes("my-3")
                with ui.row().classes("w-full gap-3"):
                    n_exp = ui.number("Experts", value=4, min=1, max=16).props("dense outlined").classes("w-24")
                    top_k = ui.number("Top-K active", value=2, min=1, max=16).props("dense outlined").classes("w-28")
                    rank = ui.number("Rank", value=16, min=1, max=256).props("dense outlined").classes("w-24")
                with ui.row().classes("w-full gap-3"):
                    seq_sel = ui.select([512, 1024, 2048, 4096, 8192], value=2048, label="Seq length").props(
                        "dense outlined"
                    ).classes("w-28")
                    ui.number("Epochs", value=1, min=1, max=20).props("dense outlined").classes("w-24")
                    ui.select([1e-4, 3e-4, 5e-5, 1e-5], value=3e-4, label="LR").props("dense outlined").classes("w-28")
                ui.select(
                    list(DEFAULT_TARGET_MODULES),
                    multiple=True,
                    value=list(DEFAULT_TARGET_MODULES),
                    label="Target modules",
                ).props("dense outlined use-chips").classes("w-full")
                ui.separator().classes("my-3")
                method = ui.select(
                    {"auto": "Auto (recommended)", "qlora": "QLoRA (4-bit base)", "lora": "LoRA (bf16 base)", "full": "Full fine-tune"},
                    value="auto",
                    label="Training method",
                ).props("dense outlined").classes("w-full")
                quant = ui.select(
                    {"nf4": "NF4", "int8": "INT8", "fp8": "FP8"},
                    value="nf4",
                    label="Base quantization (locked for this adapter)",
                ).props("dense outlined").classes("w-full")
                ui.label(
                    "The quant chosen here is locked to the checkpoint+adapter pair for the adapter's lifetime — no quant drift between adapters."
                ).classes("text-xs text-gray-500")

                start_btn = ui.button("Start training", icon="play_arrow").props("no-caps unelevated").classes("w-full mt-3")
                ui.label("Training runtime lands in phase 3 — config and planner are live now.").classes(
                    "text-xs text-gray-500"
                )

            with ui.column().classes("flex-1 gap-4 min-w-[380px]"):
                with panel():
                    ui.label("Auto training plan").classes("text-base font-semibold mb-2")
                    plan_box = ui.column().classes("w-full gap-1")

                with panel():
                    ui.label("MoLE architecture").classes("text-base font-semibold mb-2")
                    ui.label(
                        "y = W_frozen(x) + Σ_k g_k(x) · (B_k A_k x). Each expert is a rank-r LoRA branch; a per-layer router "
                        "selects top-K experts per token. Stage 1 trains experts (+router), stage 2 refines the router only. "
                        "Weights are stored as plain safetensors + JSON config, fully RL-friendly."
                    ).classes("text-sm text-gray-300")

                with panel():
                    ui.label("Training data summary").classes("text-base font-semibold mb-2")
                    data_lbl = ui.label("Select corpora to see totals.").classes("text-sm text-gray-400")

        def params_b_of() -> float:
            cid = ckpt_sel.value
            if not cid:
                return 0.0
            row = db.q1("SELECT params_b FROM checkpoints WHERE id=?", (cid,))
            return row["params_b"] if row else 0.0

        def update_plan() -> None:
            render_plan(
                plan_box,
                params_b_of(),
                int(seq_sel.value or 2048),
                "auto" if method.value == "auto" else method.value,
                quant.value,
                int(rank.value or 16),
                int(n_exp.value or 4),
            )

        def update_data_summary() -> None:
            sel = corp_sel.value or []
            if not sel:
                data_lbl.text = "Select corpora to see totals."
                return
            docs = chars = 0
            doms: dict[str, int] = {}
            for cid in sel:
                row = db.q1("SELECT stats_json FROM corpora WHERE id=?", (cid,))
                if not row:
                    continue
                stats = orjson.loads(row["stats_json"] or b"{}")
                docs += stats.get("docs_written", 0)
                chars += stats.get("chars_written", 0)
                for d, v in stats.get("domains", {}).items():
                    doms[d] = doms.get(d, 0) + v.get("chars", 0)
            top = sorted(doms.items(), key=lambda kv: -kv[1])
            mix = ", ".join(f"{k} {v / max(chars, 1) * 100:.0f}%" for k, v in top)
            data_lbl.text = f"{len(sel)} corpora · {docs:,} docs · {chars / 1e9:.2f}B chars · mix: {mix}"

        def update_quant_visibility() -> None:
            quant.enabled = method.value in {"auto", "qlora"}

        def on_start() -> None:
            problems = []
            if not ckpt_sel.value:
                problems.append("choose a checkpoint")
            if not corp_sel.value:
                problems.append("choose corpora")
            if int(top_k.value or 1) > int(n_exp.value or 1):
                problems.append("top-K must be ≤ number of experts")
            if problems:
                ui.notify("Fix: " + ", ".join(problems), type="warning")
                return
            ui.notify("Config validated. Trainer runtime arrives in phase 3.", type="info")

        for el in (ckpt_sel, seq_sel, method, quant, rank, n_exp):
            el.on_value_change(lambda _: update_plan())
        method.on_value_change(lambda _: update_quant_visibility())
        corp_sel.on_value_change(lambda _: update_data_summary())
        start_btn.on_click(on_start)

        update_plan()
        update_quant_visibility()
