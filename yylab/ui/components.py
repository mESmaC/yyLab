from __future__ import annotations

import string
from pathlib import Path

from nicegui import ui

from .shell import PANEL, PANEL_EDGE


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
