from __future__ import annotations

from nicegui import ui

from ... import db
from ...hw import system_report
from ..components import status_badge
from ..shell import ACCENT, PANEL, PANEL_EDGE, panel, shell, stat_card

QUICK_ACTIONS = [
    ("Ingest corpora", "folder_open", "/corpora", "Dedup, tokenize, mix"),
    ("Add checkpoint", "hub", "/checkpoints", "Register a pretrained base"),
    ("Train raw base", "memory", "/train/base", "Scratch architecture or continued train"),
    ("Train MoLE adapter", "psychology", "/train/mole", "Experts over a frozen base"),
    ("Train router", "alt_route", "/train/router", "Top-K routing stage"),
    ("Benchmarks", "science", "/benchmarks", "Quality + training speed"),
    ("Chat", "chat", "/chat", "Base + adapter + router"),
    ("Context & KV", "speed", "/context", "Cache & window settings"),
]

WORKFLOW = [
    ("1", "Ingest a corpora folder", "dedup + token counts + mixture", "/corpora"),
    ("2", "Register or initialize a base", "pretrained import or raw scratch model", "/train/base"),
    ("3", "Train a MoLE adapter", "joint learned per-token experts", "/train/mole"),
    ("4", "Benchmark the rack", "quality, routing regret, and throughput", "/benchmarks"),
    ("5", "Chat with base + adapter + router", "streaming inference", "/chat"),
]


def _recent_runs(limit: int = 6) -> list[dict]:
    return db.q("SELECT kind, status, created_at, error FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))


@ui.page("/")
def dashboard_page():
    with shell("Dashboard", "/"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Overview").classes("text-xl font-semibold")
            report = system_report()
            if report["gpus"]:
                g = report["gpus"][0]
                ui.badge(f"{g['name']} · {g['vram_free_gb']:.1f}/{g['vram_total_gb']:.1f} GB free").props(
                    "outline color=primary"
                )
            else:
                ui.badge("No CUDA GPU").props("outline color=orange")

        with ui.row().classes("w-full gap-4 flex-wrap"):
            stat_card(
                "Corpora (ready)",
                str(
                    db.q1(
                        "SELECT COUNT(*) n FROM corpora WHERE status='ready'"
                        " AND COALESCE(kind,'ingested') != 'rejected'"
                    )["n"]
                ),
            )
            stat_card(
                "Checkpoints", str(db.q1("SELECT COUNT(*) n FROM checkpoints WHERE kind!='pending'")["n"])
            )
            stat_card("Adapters", str(db.q1("SELECT COUNT(*) n FROM adapters")["n"]))
            stat_card("Routers", str(db.q1("SELECT COUNT(*) n FROM routers")["n"]))
            stat_card("Runs", str(db.q1("SELECT COUNT(*) n FROM runs")["n"]))

        with panel():
            ui.label("Quick actions").classes("text-base font-semibold mb-3")
            with ui.grid(columns="2 sm:3 lg:6").classes("w-full gap-3"):
                for label, icon, path, sub in QUICK_ACTIONS:
                    with ui.link(target=path).classes("no-underline"):
                        with (
                            ui.card()
                            .classes("w-full p-3 items-start gap-1 cursor-pointer")
                            .style(f"background: {PANEL}; border: 1px solid {PANEL_EDGE}")
                            .props("flat bordered")
                        ):
                            ui.icon(icon, size="22px").style(f"color: {ACCENT}")
                            ui.label(label).classes("text-sm font-medium text-gray-200 mt-1")
                            ui.label(sub).classes("text-xs text-gray-500 leading-tight")

        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
            with ui.column().classes("flex-1 min-w-[320px] gap-4"):
                if report["gpus"]:
                    with panel():
                        ui.label("GPUs").classes("text-base font-semibold mb-2")
                        for g in report["gpus"]:
                            with ui.row().classes("w-full items-center gap-4"):
                                ui.icon("developer_board").classes("text-gray-400")
                                ui.label(g["name"]).classes("w-56")
                                bar = ui.linear_progress(
                                    value=1 - g["vram_free_gb"] / max(g["vram_total_gb"], 0.001),
                                    show_value=False,
                                ).classes("flex-1")
                                bar.props("color=primary track-color=grey-8")
                                ui.label(f"{g['vram_free_gb']:.1f} / {g['vram_total_gb']:.1f} GB").classes(
                                    "text-xs text-gray-400 w-28"
                                )
                                ui.label(f"cc {g['compute_capability']}").classes(
                                    "text-xs text-gray-500 w-14"
                                )
                else:
                    with panel():
                        ui.label("No CUDA GPU detected — training will fall back to CPU.").classes(
                            "text-amber-400"
                        )

                with panel():
                    ui.label("System").classes("text-base font-semibold mb-2")
                    ui.label(f"{report['os']} · {report['cpu']}").classes("text-sm text-gray-300")
                    ui.label(
                        f"{report['cpu_cores']} threads · {report['ram_gb']} GB RAM · "
                        + ("triton available" if report["triton"] else "triton unavailable")
                    ).classes("text-xs text-gray-500")

            with panel().classes("flex-1 min-w-[320px]"):
                ui.label("Recent runs").classes("text-base font-semibold mb-2")
                runs = _recent_runs()
                if not runs:
                    ui.label("No runs yet — start a training or export job.").classes("text-sm text-gray-500")
                    ui.link("Go to training", "/train/mole").classes("text-sm").style(f"color: {ACCENT}")
                else:
                    for r in runs:
                        with (
                            ui.row()
                            .classes("w-full items-center justify-between gap-2 py-1")
                            .style(f"border-bottom: 1px solid {PANEL_EDGE}")
                        ):
                            with ui.row().classes("items-center gap-2 min-w-0"):
                                status_badge(r["status"])
                                ui.label(r["kind"]).classes("text-sm text-gray-300")
                            ui.label(r["created_at"][5:16]).classes("text-xs text-gray-500 whitespace-nowrap")
                    ui.link("View all runs", "/runs").classes("text-sm mt-2").style(f"color: {ACCENT}")

        with panel():
            ui.label("Workflow").classes("text-base font-semibold mb-3")
            for num, title, sub, path in WORKFLOW:
                with ui.link(target=path).classes("no-underline"):
                    with (
                        ui.row()
                        .classes("w-full items-center gap-3 py-1.5 px-2 rounded-md")
                        .style("border: 1px solid transparent")
                    ):
                        ui.label(num).classes(
                            "w-6 h-6 items-center justify-center rounded-full text-xs font-bold"
                        ).style(f"background: {ACCENT}33; color: {ACCENT}; display:flex")
                        with ui.column().classes("gap-0"):
                            ui.label(title).classes("text-sm text-gray-200")
                            ui.label(sub).classes("text-xs text-gray-500")
