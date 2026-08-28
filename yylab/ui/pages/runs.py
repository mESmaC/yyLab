from __future__ import annotations

from pathlib import Path

import orjson
from nicegui import ui

from ... import db
from ..shell import panel, shell


@ui.page("/runs")
def runs_page():
    with shell("Runs", "/runs"):
        ui.label("Training runs").classes("text-xl font-semibold")
        detail = ui.column().classes("w-full")

        def rows() -> list[dict]:
            out = []
            for r in db.q("SELECT * FROM runs ORDER BY created_at DESC"):
                out.append(
                    {
                        "id": r["id"],
                        "kind": r["kind"],
                        "status": r["status"],
                        "created": r["created_at"],
                        "error": (r["error"] or "")[:80],
                    }
                )
            return out

        def render_detail(e) -> None:
            detail.clear()
            r = db.q1("SELECT * FROM runs WHERE id=?", (e.args[1]["id"],))
            if not r:
                return
            with detail, panel():
                ui.label(f'{r["kind"]} · {r["status"]} · {r["created_at"]}').classes("font-semibold")
                cfg = orjson.loads(r["config_json"] or b"{}")
                if cfg:
                    ui.label(orjson.dumps(cfg, option=orjson.OPT_INDENT_2).decode()).classes(
                        "text-xs font-mono whitespace-pre-wrap text-gray-300"
                    )
                if r["log_path"] and Path(r["log_path"]).exists():
                    tail = Path(r["log_path"]).read_text("utf-8", errors="ignore")[-4000:]
                    ui.label(tail).classes("text-xs font-mono whitespace-pre-wrap text-gray-400")
                if r["error"]:
                    ui.label(r["error"]).classes("text-xs text-red-400 whitespace-pre-wrap")

        table = ui.table(
            columns=[
                {"name": "kind", "label": "Kind", "field": "kind", "sortable": True},
                {"name": "status", "label": "Status", "field": "status", "sortable": True},
                {"name": "created", "label": "Created", "field": "created", "sortable": True},
                {"name": "error", "label": "Error", "field": "error", "align": "left"},
            ],
            rows=rows(),
            row_key="id",
            pagination=20,
        ).classes("w-full")
        table.on("row-click", render_detail)

        if not rows():
            ui.label("No runs yet.").classes("text-sm text-gray-500")
