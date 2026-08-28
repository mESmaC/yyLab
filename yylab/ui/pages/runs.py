from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import orjson
from nicegui import ui

from ... import db
from ...artifacts import run_dir
from ..shell import panel, shell


def _progress_of(r: dict) -> str:
    if r["status"] not in {"queued", "running"}:
        return ""
    pf = run_dir(r["id"]) / "progress.json"
    if not pf.exists():
        return "starting…"
    try:
        p = json.loads(pf.read_text("utf-8"))
        return f'{p.get("stage", "")} {p.get("step", 0)}/{p.get("max_steps", "?")} · loss {p.get("loss", "?")} · {p.get("msg", "")}'
    except Exception:  # noqa: BLE001
        return ""


def _stop(pid: int) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:  # noqa: BLE001
        pass


@ui.page("/runs")
def runs_page():
    with shell("Runs", "/runs"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Training runs").classes("text-xl font-semibold")
        table_container = ui.column().classes("w-full")
        detail = ui.column().classes("w-full")

        def rows() -> list[dict]:
            out = []
            for r in db.q("SELECT * FROM runs ORDER BY created_at DESC"):
                meta = {}
                try:
                    meta = json.loads(r["metrics_json"] or "{}")
                except Exception:  # noqa: BLE001
                    pass
                out.append(
                    {
                        "id": r["id"],
                        "kind": r["kind"],
                        "status": r["status"],
                        "created": r["created_at"],
                        "progress": _progress_of(r),
                        "pid": meta.get("pid", ""),
                        "error": (r["error"] or "")[:80],
                    }
                )
            return out

        def stop_run(row: dict) -> None:
            if row.get("pid"):
                _stop(int(row["pid"]))
                ui.notify(f"Sent stop to pid {row['pid']}", type="warning")
            else:
                db.execute("UPDATE runs SET status='cancelled' WHERE id=?", (row["id"],))

        def render_table() -> None:
            table_container.clear()
            data = rows()
            with table_container:
                if not data:
                    from ..components import empty_state

                    empty_state(
                        "history",
                        "No runs yet",
                        "Training, router, base, and export jobs appear here with live progress, logs, and metrics.",
                    )
                    return
                table = ui.table(
                    columns=[
                        {"name": "kind", "label": "Kind", "field": "kind", "sortable": True},
                        {"name": "status", "label": "Status", "field": "status", "sortable": True},
                        {"name": "progress", "label": "Progress", "field": "progress", "align": "left"},
                        {"name": "created", "label": "Created", "field": "created", "sortable": True},
                        {"name": "error", "label": "Error", "field": "error", "align": "left"},
                    ],
                    rows=data,
                    row_key="id",
                    pagination=20,
                ).classes("w-full")
                table.add_slot(
                    "body-cell-status",
                    """
                    <q-td :props="props">
                      <q-badge :color="props.value === 'done' ? 'green' : props.value === 'failed' ? 'red' : props.value === 'running' ? 'primary' : 'grey'">
                        {{ props.value }}
                      </q-badge>
                    </q-td>
                    """,
                )

                def on_row(e) -> None:
                    render_detail(e.args[1])

                table.on("row-click", on_row)

        def render_detail(row: dict | None = None) -> None:
            detail.clear()
            if not row:
                return
            r = db.q1("SELECT * FROM runs WHERE id=?", (row["id"],))
            if not r:
                return
            with detail, panel():
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(f'{r["kind"]} · {r["status"]} · {r["created_at"]}').classes("font-semibold")
                    if r["status"] in {"queued", "running"}:
                        ui.button("Stop", icon="stop", on_click=lambda: (stop_run(row), render_table())).props(
                            "dense flat no-caps color=negative"
                        )
                cfg = orjson.loads(r["config_json"] or b"{}")
                if cfg:
                    with ui.expansion("config").classes("w-full"):
                        ui.label(orjson.dumps(cfg, option=orjson.OPT_INDENT_2).decode()).classes(
                            "text-xs font-mono whitespace-pre-wrap text-gray-300"
                        )
                log_path = Path(r["log_path"]) if r["log_path"] else run_dir(r["id"]) / "train.log"
                if log_path.exists():
                    tail = log_path.read_text("utf-8", errors="ignore")[-4000:]
                    with ui.expansion("log (tail)", value=r["status"] in {"running", "failed"}).classes("w-full"):
                        ui.label(tail).classes("text-xs font-mono whitespace-pre-wrap text-gray-400")
                if r["error"]:
                    ui.label(r["error"][-2000:]).classes("text-xs text-red-400 whitespace-pre-wrap")

        render_table()

        def tick() -> None:
            if any(r["status"] in {"queued", "running"} for r in db.q("SELECT status FROM runs")):
                render_table()

        ui.timer(2.0, tick)
