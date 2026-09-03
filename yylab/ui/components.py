from __future__ import annotations

import string
from pathlib import Path

from nicegui import ui

from .shell import ACCENT, PANEL, PANEL_EDGE

STATUS_COLORS = {
    "ready": "green",
    "done": "green",
    "trained": "green",
    "running": "primary",
    "processing": "primary",
    "training": "primary",
    "loading": "primary",
    "queued": "teal",
    "pending": "grey",
    "untrained": "grey",
    "failed": "red",
    "error": "red",
    "cancelled": "grey",
}


def status_badge(status: str):
    color = STATUS_COLORS.get(status, "grey")
    b = ui.badge(status, color=color)
    if status in {"running", "processing", "training", "loading"}:
        b.props("outline")
    return b


def empty_state(icon: str, title: str, desc: str, action_label: str | None = None, on_action=None):
    with ui.column().classes("w-full items-center justify-center gap-2 py-14 px-6").style(
        f"border: 1px dashed {PANEL_EDGE}; border-radius: 12px; background: {PANEL}55"
    ):
        ui.icon(icon, size="44px").style(f"color: {ACCENT}66")
        ui.label(title).classes("text-base font-semibold text-gray-200")
        ui.label(desc).classes("text-sm text-gray-500 text-center max-w-lg")
        if action_label and on_action is not None:
            ui.button(action_label, on_click=on_action).props("no-caps unelevated").classes("mt-2")


def confirm(title: str, message: str, on_confirm, confirm_label: str = "Delete", danger: bool = True):
    with ui.dialog() as dlg, ui.card().classes("w-[420px] p-4").style(
        f"background: {PANEL}; border: 1px solid {PANEL_EDGE}"
    ):
        ui.label(title).classes("text-base font-semibold")
        ui.label(message).classes("text-sm text-gray-400 mt-1")
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dlg.close).props("flat dense no-caps")

            def do() -> None:
                dlg.close()
                on_confirm()

            ui.button(confirm_label, on_click=do).props(
                f"dense no-caps {'color=negative' if danger else 'color=primary'}"
            )
    return dlg


def form_section(step: str, title: str, sub: str = ""):
    with ui.row().classes("w-full items-center gap-2 mt-1"):
        ui.label(step).classes(
            "w-5 h-5 items-center justify-center rounded-full text-[11px] font-bold shrink-0"
        ).style(f"background: {ACCENT}2e; color: {ACCENT}; display: flex")
        ui.label(title).classes("text-sm font-semibold text-gray-200")
    if sub:
        ui.label(sub).classes("text-xs text-gray-500 ml-7 -mt-0.5")


def hint(text: str):
    return ui.label(text).classes("text-xs text-gray-500")


def field_grid(cols: int = 3):
    return ui.grid(columns=cols).classes("w-full gap-3")


def drive_roots() -> list[str]:
    return [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]


def dir_browser(title: str, on_select, initial: str | None = None):
    state = {"path": initial or ""}

    with ui.dialog() as dlg, ui.card().classes("w-[560px]").style(
        f"background: {PANEL}; border: 1px solid {PANEL_EDGE}"
    ):
        ui.label(title).classes("text-base font-semibold")
        path_input = ui.input(value=state["path"]).props("dense outlined dark").classes("w-full")
        path_input.on_value_change(lambda e: goto(e.value, from_input=True))
        listing = ui.column().classes("w-full h-72 overflow-auto gap-0.5")

        def goto(p: str, from_input: bool = False) -> None:
            if not p:
                state["path"] = ""
            else:
                candidate = Path(p)
                if not candidate.is_dir():
                    return
                state["path"] = str(candidate)
            if not from_input:
                path_input.set_value(state["path"])
            render()

        def render() -> None:
            listing.clear()
            with listing:
                if not state["path"]:
                    for r in drive_roots():
                        ui.button(r, on_click=lambda _, r=r: goto(r)).props(
                            "flat dense no-caps align=left icon=storage"
                        ).classes("w-full")
                    return
                cur = Path(state["path"])
                if cur.parent != cur:
                    ui.button("..", on_click=lambda _: goto(str(cur.parent))).props(
                        "flat dense no-caps align=left icon=arrow_upward"
                    ).classes("w-full")
                try:
                    subdirs = sorted(
                        (d for d in cur.iterdir() if d.is_dir() and not d.name.startswith(".")),
                        key=lambda d: d.name.lower(),
                    )
                except (PermissionError, OSError):
                    subdirs = []
                if not subdirs:
                    ui.label("(no subfolders)").classes("text-xs text-gray-500 px-2 py-1")
                for d in subdirs:
                    ui.button(d.name, on_click=lambda _, d=d: goto(str(d))).props(
                        "flat dense no-caps align=left icon=folder"
                    ).classes("w-full")

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=dlg.close).props("flat dense no-caps")
            ui.button("Select folder", on_click=lambda: (on_select(state["path"]), dlg.close())).props(
                "dense no-caps color=primary"
            )
        render()
    return dlg
