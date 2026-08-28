from __future__ import annotations

from nicegui import ui

from ... import db
from ...hw import system_report
from ..shell import panel, shell, stat_card


@ui.page("/")
def dashboard_page():
    with shell("Dashboard", "/"):
        report = system_report()

        with ui.row().classes("w-full gap-4 flex-wrap"):
            stat_card("Corpora (ready)", str(db.q1(
                "SELECT COUNT(*) n FROM corpora WHERE status='ready'"
            )["n"]))
            stat_card("Checkpoints", str(db.q1("SELECT COUNT(*) n FROM checkpoints")["n"]))
            stat_card("Adapters", str(db.q1("SELECT COUNT(*) n FROM adapters")["n"]))
            stat_card("Routers", str(db.q1("SELECT COUNT(*) n FROM routers")["n"]))
            stat_card("Runs", str(db.q1("SELECT COUNT(*) n FROM runs")["n"]))

        if report["gpus"]:
            with panel():
                ui.label("GPUs").classes("text-base font-semibold mb-2")
                for g in report["gpus"]:
                    with ui.row().classes("w-full items-center gap-4"):
                        ui.icon("developer_board").classes("text-gray-400")
                        ui.label(g["name"]).classes("w-64")
                        bar = ui.linear_progress(
                            value=1 - g["vram_free_gb"] / max(g["vram_total_gb"], 0.001),
                            show_value=False,
                        ).classes("flex-1")
                        bar.props("color=primary track-color=grey-8")
                        ui.label(
                            f'{g["vram_free_gb"]:.1f} / {g["vram_total_gb"]:.1f} GB free'
                        ).classes("text-xs text-gray-400 w-36")
                        ui.label(f'cc {g["compute_capability"]}').classes("text-xs text-gray-500 w-16")
        else:
            with panel():
                ui.label("No CUDA GPU detected — training will fall back to CPU (not recommended).").classes(
                    "text-amber-400"
                )

        with panel():
            ui.label("System").classes("text-base font-semibold mb-2")
            ui.label(f'{report["os"]} · {report["cpu"]} · {report["cpu_cores"]} threads · {report["ram_gb"]} GB RAM')
            ui.label(
                "triton available (torch.compile enabled where useful)"
                if report["triton"]
                else "triton not available"
            ).classes("text-xs text-gray-400")

        with panel():
            ui.label("Workflow").classes("text-base font-semibold mb-2")
            steps = [
                "1. Ingest a corpora folder (dedup + token counts + mixture).",
                "2. Register a pretrained checkpoint.",
                "3. Train a Mixture-of-LoRA-Experts adapter over the frozen base (quant locked).",
                "4. Train the router (top-K experts per token).",
                "5. Chat with base + adapter + router.",
            ]
            for s in steps:
                ui.label(s).classes("text-sm text-gray-300")
