from __future__ import annotations

from contextlib import contextmanager

from nicegui import ui

BG = "#0e0e13"
PANEL = "#16161d"
PANEL_EDGE = "#262633"
ACCENT = "#8b6bff"

NAV = [
    ("Dashboard", "/", "dashboard"),
    ("Corpora", "/corpora", "folder_open"),
    ("Review", "/review", "rate_review"),
    ("Checkpoints", "/checkpoints", "hub"),
    ("Train · MoLE", "/train/mole", "psychology"),
    ("Train · Router", "/train/router", "alt_route"),
    ("Train · Base", "/train/base", "memory"),
    ("Context & KV", "/context", "speed"),
    ("Benchmarks", "/benchmarks", "science"),
    ("Chat", "/chat", "chat"),
    ("Runs", "/runs", "history"),
    ("Settings", "/settings", "settings"),
]


def panel():
    return ui.card().classes("w-full p-4").style(f"background: {PANEL}; border: 1px solid {PANEL_EDGE}")


def stat_card(label: str, value: str):
    with ui.card().classes("p-3 min-w-36").style(f"background: {PANEL}; border: 1px solid {PANEL_EDGE}"):
        ui.label(value).classes("text-xl font-semibold")
        ui.label(label).classes("text-xs text-gray-400")


@contextmanager
def shell(title: str, active: str):
    ui.add_head_html(f"<style>body {{ background: {BG}; }}</style>")
    with ui.header().classes("items-center px-4 gap-3").style(
        f"background: {PANEL}; border-bottom: 1px solid {PANEL_EDGE}"
    ):
        ui.label("yyLab").classes("text-lg font-bold").style(f"color: {ACCENT}")
        ui.label(title).classes("text-sm text-gray-400")
    with ui.left_drawer(value=True, fixed=True).props("width=180 bordered").style(f"background: {PANEL}"):
        for label, path, icon in NAV:
            color = ACCENT if path == active else "#b8b8c4"
            bg = f"background: {ACCENT}22;" if path == active else ""
            with ui.link(target=path) as link:
                with ui.row().classes("items-center gap-2 w-full px-3 py-2 rounded-md text-sm no-wrap"):
                    ui.icon(icon, size="18px")
                    ui.label(label)
            link.classes("w-full no-underline")
            link.style(f"color: {color}; {bg}")
    with ui.column().classes("w-full p-4 gap-4") as content:
        yield content
