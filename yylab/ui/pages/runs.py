from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import orjson
from nicegui import ui

from ... import db
from ...artifacts import run_dir
from ..components import empty_state
from ..shell import panel, shell

ACTIVE_RUN_STATUSES = frozenset({"queued", "running"})
FAST_POLL_SECONDS = 1.0
IDLE_POLL_SECONDS = 5.0
EMPTY_POLL_SECONDS = 10.0
LOG_TAIL_BYTES = 16 * 1024
MAX_LOG_DELTA_BYTES = 64 * 1024


def _poll_interval(*, has_active_runs: bool, has_selection: bool) -> float:
    if has_active_runs:
        return FAST_POLL_SECONDS
    return IDLE_POLL_SECONDS if has_selection else EMPTY_POLL_SECONDS


def _read_log_chunk(
    path: Path,
    *,
    offset: int | None,
    identity: tuple[int, int] | None,
    mtime_ns: int | None,
) -> tuple[str, int, tuple[int, int] | None, int | None, bool]:
    """Read only new log bytes, falling back to a bounded tail after replacement/truncation."""
    try:
        stat = path.stat()
    except OSError:
        return "", 0, None, None, offset is not None

    current_identity = (stat.st_dev, stat.st_ino)
    reset = (
        offset is None
        or identity != current_identity
        or stat.st_size < offset
        or (stat.st_size == offset and mtime_ns is not None and stat.st_mtime_ns != mtime_ns)
    )
    if not reset and stat.st_size == offset:
        return "", offset, current_identity, stat.st_mtime_ns, False

    start = max(0, stat.st_size - LOG_TAIL_BYTES) if reset else offset
    if not reset and stat.st_size - start > MAX_LOG_DELTA_BYTES:
        start = max(0, stat.st_size - LOG_TAIL_BYTES)
        reset = True

    try:
        with path.open("rb") as stream:
            stream.seek(start)
            raw = stream.read(MAX_LOG_DELTA_BYTES)
            new_offset = stream.tell()
    except OSError:
        return "", offset or 0, identity, mtime_ns, False

    if reset and start > 0:
        # A bounded tail may begin inside a UTF-8 sequence or a partially written line.
        newline = raw.find(b"\n")
        if newline >= 0:
            raw = raw[newline + 1 :]
    return raw.decode("utf-8", errors="ignore"), new_offset, current_identity, stat.st_mtime_ns, reset


def _progress_of(r: dict) -> str:
    if r["status"] not in {"queued", "running"}:
        return ""
    pf = run_dir(r["id"]) / "progress.json"
    if not pf.exists():
        return "starting…"
    try:
        p = json.loads(pf.read_text("utf-8"))
        metric_label = p.get("metric_label", "loss")
        metric_value = p.get("value", p.get("loss", "?"))
        return (
            f"{p.get('stage', '')} {p.get('step', 0)}/{p.get('max_steps', '?')} · "
            f"{metric_label} {metric_value} · {p.get('msg', '')}"
        )
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
    state: dict[str, Any] = {
        "selected": None,
        "table_signature": None,
        "detail_signature": None,
        "has_active": False,
        "timer": None,
        "log_path": None,
        "log_offset": None,
        "log_identity": None,
        "log_mtime_ns": None,
        "log_has_content": False,
        "token_view_mtime_ns": None,
    }
    detail_refs: dict[str, Any] = {}

    with shell("Runs", "/runs"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Runs").classes("text-xl font-semibold")
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
                diagnostic = meta.get("diagnostic_verdict") or {}
                out.append(
                    {
                        "id": r["id"],
                        "kind": r["kind"],
                        "status": r["status"],
                        "created": r["created_at"],
                        "progress": _progress_of(r),
                        "pid": meta.get("pid", ""),
                        "diagnostic": diagnostic.get("label", ""),
                        "error": (r["error"] or "")[:80],
                    }
                )
            return out

        def stop_run(row: dict) -> None:
            if row.get("pid"):
                _stop(int(row["pid"]))
                ui.notify(f"Sent stop to pid {row['pid']}", type="warning")
            run = db.q1("SELECT kind, config_json FROM runs WHERE id=?", (row["id"],))
            try:
                config = json.loads(run["config_json"] or "{}") if run else {}
            except (TypeError, json.JSONDecodeError):
                config = {}
            db.execute(
                "UPDATE runs SET status='cancelled', finished_at=?, error=? WHERE id=?",
                (db.now_iso(), "cancelled by user", row["id"]),
            )
            if (
                run
                and run["kind"]
                in {
                    "mole_stage1",
                    "router_stage2",
                    "base",
                    "preference_dpo",
                    "antidoom_ftpo",
                }
                and config.get("adapter_id")
            ):
                db.execute(
                    "UPDATE adapters SET status='failed' WHERE id=? AND status='training'",
                    (config["adapter_id"],),
                )
            if run and run["kind"] == "router_stage2" and config.get("router_id"):
                db.execute(
                    "UPDATE routers SET status='failed' WHERE id=? AND status='training'",
                    (config["router_id"],),
                )
            if run and run["kind"] == "tokenizer" and config.get("tokenizer_id"):
                db.execute(
                    "UPDATE tokenizers SET status='failed',error=? "
                    "WHERE id=? AND status IN ('queued','processing')",
                    ("cancelled by user", config["tokenizer_id"]),
                )
            if (
                run
                and run["kind"] in {"base", "spalmer_base", "sprint"}
                and config.get("out_checkpoint_id")
            ):
                db.execute(
                    "DELETE FROM checkpoints WHERE id=? AND kind='pending'",
                    (config["out_checkpoint_id"],),
                )

        with table_container:
            table = ui.table(
                columns=[
                    {"name": "kind", "label": "Kind", "field": "kind", "sortable": True},
                    {"name": "status", "label": "Status", "field": "status", "sortable": True},
                    {"name": "progress", "label": "Progress", "field": "progress", "align": "left"},
                    {
                        "name": "diagnostic",
                        "label": "Diagnostic",
                        "field": "diagnostic",
                        "align": "left",
                    },
                    {"name": "created", "label": "Created", "field": "created", "sortable": True},
                    {"name": "error", "label": "Error", "field": "error", "align": "left"},
                ],
                rows=[],
                row_key="id",
                pagination=20,
            ).classes("w-full")
            table.add_slot(
                "body-cell-status",
                """
                <q-td :props="props">
                  <q-badge :color="props.value === 'done' ? 'green' : props.value === 'failed' ? 'red' : props.value === 'running' ? 'primary' : props.value === 'queued' ? 'teal' : 'grey'">
                    {{ props.value }}
                  </q-badge>
                </q-td>
                """,
            )
            with ui.column().classes("w-full") as empty_holder:
                empty_state(
                    "history",
                    "No runs yet",
                    "Training, tokenizer, benchmark, router, base, and export jobs appear here with live progress, logs, and metrics.",
                )

        def refresh_table(*, force: bool = False) -> bool:
            data = rows()
            signature = tuple(
                (
                    r["id"],
                    r["kind"],
                    r["status"],
                    r["created"],
                    r["progress"],
                    r["pid"],
                    r["diagnostic"],
                    r["error"],
                )
                for r in data
            )
            if force or signature != state["table_signature"]:
                table.update_rows(data, clear_selection=False)
                table.visible = bool(data)
                empty_holder.visible = not data
                state["table_signature"] = signature
            state["has_active"] = any(r["status"] in ACTIVE_RUN_STATUSES for r in data)
            return state["has_active"]

        def update_timer_interval() -> None:
            timer = state["timer"]
            if timer is not None:
                timer.interval = _poll_interval(
                    has_active_runs=state["has_active"],
                    has_selection=state["selected"] is not None,
                )

        def stop_selected(run_id: str) -> None:
            r = db.q1("SELECT metrics_json FROM runs WHERE id=?", (run_id,))
            if not r:
                return
            try:
                meta = json.loads(r["metrics_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                meta = {}
            stop_run({"id": run_id, "pid": meta.get("pid", "")})
            refresh_table(force=True)
            refresh_detail(force=True)
            update_timer_interval()

        def render_detail(run_id: str | None = None) -> None:
            detail.clear()
            detail_refs.clear()
            state["selected"] = run_id
            state["detail_signature"] = None
            state["log_path"] = None
            state["log_offset"] = None
            state["log_identity"] = None
            state["log_mtime_ns"] = None
            state["log_has_content"] = False
            state["token_view_mtime_ns"] = None
            if not run_id:
                update_timer_interval()
                return
            r = db.q1("SELECT * FROM runs WHERE id=?", (run_id,))
            if not r:
                state["selected"] = None
                update_timer_interval()
                return
            with detail, panel():
                with ui.row().classes("w-full items-center justify-between"):
                    header = ui.label("").classes("font-semibold")
                    stop_button = ui.button(
                        "Stop", icon="stop", on_click=lambda: stop_selected(run_id)
                    ).props("dense flat no-caps color=negative")
                progress_label = ui.label("").classes("text-xs text-gray-400 whitespace-pre-wrap")
                try:
                    cfg = orjson.loads(r["config_json"] or b"{}")
                except orjson.JSONDecodeError:
                    cfg = {}
                if cfg:
                    with ui.expansion("config").classes("w-full"):
                        ui.label(orjson.dumps(cfg, option=orjson.OPT_INDENT_2).decode()).classes(
                            "text-xs font-mono whitespace-pre-wrap text-gray-300"
                        )
                token_view_expansion = ui.expansion("tokenized data view").classes("w-full")
                with token_view_expansion:
                    token_view_label = ui.label("").classes(
                        "text-xs font-mono whitespace-pre-wrap text-gray-300"
                    )
                token_view_expansion.visible = False
                with ui.expansion(
                    "log (live tail)", value=r["status"] in ACTIVE_RUN_STATUSES or r["status"] == "failed"
                ).classes("w-full"):
                    log_placeholder = ui.label("Waiting for log output…").classes("text-xs text-gray-500")
                    log_view = (
                        ui.log(max_lines=500)
                        .classes("w-full h-80 text-xs font-mono text-gray-400 rounded p-2")
                        .style("background: #0d0d12")
                    )
                    log_view.visible = False
                error_label = ui.label("").classes("text-xs text-red-400 whitespace-pre-wrap")
            detail_refs.update(
                header=header,
                progress=progress_label,
                stop=stop_button,
                log=log_view,
                log_placeholder=log_placeholder,
                error=error_label,
                token_view_expansion=token_view_expansion,
                token_view=token_view_label,
            )
            refresh_detail(force=True)
            update_timer_interval()

        def refresh_detail(*, force: bool = False) -> str | None:
            run_id = state["selected"]
            if not run_id or not detail_refs:
                return None
            r = db.q1("SELECT * FROM runs WHERE id=?", (run_id,))
            if not r:
                render_detail(None)
                return None

            progress = _progress_of(r)
            signature = (r["kind"], r["status"], r["created_at"], progress, r["error"] or "")
            if force or signature != state["detail_signature"]:
                detail_refs["header"].text = f"{r['kind']} · {r['status']} · {r['created_at']}"
                detail_refs["progress"].text = progress
                detail_refs["progress"].visible = bool(progress)
                detail_refs["stop"].visible = r["status"] in ACTIVE_RUN_STATUSES
                error = (r["error"] or "")[-2000:]
                detail_refs["error"].text = error
                detail_refs["error"].visible = bool(error)
                state["detail_signature"] = signature

            token_view_path = run_dir(run_id) / "tokenized_view_manifest.json"
            try:
                token_view_mtime = token_view_path.stat().st_mtime_ns
            except OSError:
                token_view_mtime = None
            if token_view_mtime != state["token_view_mtime_ns"]:
                state["token_view_mtime_ns"] = token_view_mtime
                if token_view_mtime is not None:
                    try:
                        token_view = orjson.loads(token_view_path.read_bytes())
                    except (OSError, orjson.JSONDecodeError):
                        token_view = None
                    if token_view:
                        detail_refs["token_view"].text = orjson.dumps(
                            token_view,
                            option=orjson.OPT_INDENT_2,
                        ).decode()
                        detail_refs["token_view_expansion"].visible = True
                    else:
                        detail_refs["token_view_expansion"].visible = False

            log_path = Path(r["log_path"]) if r["log_path"] else run_dir(run_id) / "train.log"
            path_key = str(log_path)
            if state["log_path"] != path_key:
                detail_refs["log"].clear()
                state["log_path"] = path_key
                state["log_offset"] = None
                state["log_identity"] = None
                state["log_mtime_ns"] = None
                state["log_has_content"] = False

            chunk, offset, identity, mtime_ns, reset = _read_log_chunk(
                log_path,
                offset=state["log_offset"],
                identity=state["log_identity"],
                mtime_ns=state["log_mtime_ns"],
            )
            if reset:
                detail_refs["log"].clear()
                state["log_has_content"] = False
            if chunk:
                detail_refs["log"].push(
                    chunk,
                    classes="text-xs font-mono whitespace-pre-wrap text-gray-400",
                )
                state["log_has_content"] = True
            state["log_offset"] = offset
            state["log_identity"] = identity
            state["log_mtime_ns"] = mtime_ns

            has_log = state["log_has_content"]
            detail_refs["log"].visible = has_log
            detail_refs["log_placeholder"].visible = not has_log
            detail_refs["log_placeholder"].text = (
                "Waiting for log output…"
                if r["status"] in ACTIVE_RUN_STATUSES
                else "No log output was written."
            )
            return r["status"]

        table.on("row-click", lambda e: render_detail(e.args[1]["id"]))

        def tick() -> None:
            refresh_table()
            refresh_detail()
            update_timer_interval()

        refresh_table(force=True)
        state["timer"] = ui.timer(
            _poll_interval(has_active_runs=state["has_active"], has_selection=False),
            tick,
            immediate=False,
        )
