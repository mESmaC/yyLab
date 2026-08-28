from __future__ import annotations

from nicegui import ui

from ... import db
from ...config import get_settings
from ..shell import panel, shell


@ui.page("/chat")
def chat_page():
    with shell("Chat", "/chat"):
        ui.label("Chat").classes("text-xl font-semibold")

        ckpts = {c["id"]: c["name"] for c in db.q("SELECT * FROM checkpoints ORDER BY name")}
        engine = get_settings()["engine"]

        with panel().classes("w-full"), ui.row().classes("w-full gap-3 items-end flex-wrap"):
            ckpt_sel = ui.select(ckpts, label="Base checkpoint").props("dense outlined").classes("w-64")
            adp_sel = ui.select({}, label="Adapter (+router)").props("dense outlined").classes("w-64")
            ui.badge(
                f'{engine["attn_implementation"]} · cache {engine["cache_implementation"]}'
                + (f' · kv {engine["kv_quant"]}' if engine["kv_quant"] != "none" else "")
                + f' · ctx {engine["max_context_tokens"]}'
            ).props("outline color=grey")

        def refresh_adapters(e=None) -> None:
            opts = {}
            if ckpt_sel.value:
                for a in db.q("SELECT * FROM adapters WHERE checkpoint_id=? AND status='trained'", (ckpt_sel.value,)):
                    opts[a["id"]] = f'{a["name"]} (E={a["num_experts"]}, k={a["top_k"]})'
            adp_sel.options = opts
            adp_sel.value = None
            adp_sel.update()

        ckpt_sel.on_value_change(refresh_adapters)

        messages = ui.column().classes("w-full gap-2 min-h-64")
        with panel().classes("w-full"), ui.row().classes("w-full items-end gap-2"):
            inp = ui.input(placeholder="Message…").props("dense outlined autogrow").classes("flex-1")
            send = ui.button(icon="send").props("dense")

        def on_send() -> None:
            text = (inp.value or "").strip()
            if not text:
                return
            with messages:
                ui.label(text).classes("self-end bg-[#2a2a3a] px-3 py-2 rounded-lg text-sm")
            inp.value = ""
            if not ckpt_sel.value:
                with messages:
                    ui.label("(select a checkpoint first)").classes("text-xs text-gray-500")
                return
            with messages:
                ui.label(
                    "(inference engine lands in phase 4 — checkpoint loading + MoLE routing + streaming)"
                ).classes("text-xs text-gray-500")

        send.on_click(on_send)
        inp.on("keydown.enter", on_send)
