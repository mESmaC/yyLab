from __future__ import annotations

import queue as queue_mod
import threading

from nicegui import ui

from ... import db
from ...config import get_settings
from ...hw import best_gpu
from ...infer.engine import ENGINE, LoadSpec
from ...jobs import JOBS
from ..shell import PANEL_EDGE, panel, shell


@ui.page("/chat")
def chat_page():
    with shell("Chat", "/chat"):
        ui.label("Chat").classes("text-xl font-semibold")

        ckpts = {c["id"]: c["name"] for c in db.q("SELECT * FROM checkpoints WHERE kind!='pending' ORDER BY name")}
        engine_settings = dict(get_settings()["engine"])
        messages: list[dict] = []
        stream_state: dict = {"q": None, "label": None, "active": False, "buf": ""}
        load_job: dict = {"id": None}

        with panel().classes("w-full"):
            with ui.row().classes("w-full gap-3 items-end flex-wrap"):
                ckpt_sel = ui.select(ckpts, label="Base checkpoint").props("dense outlined").classes("w-64")
                adp_sel = ui.select({}, label="Adapter").props("dense outlined").classes("w-64")
                rtr_sel = ui.select({}, label="Router snapshot").props("dense outlined").classes("w-56")
                quant_sel = ui.select(
                    {"auto": "Auto quant", "bf16": "BF16", "nf4": "NF4", "int8": "INT8"},
                    value="auto",
                    label="Base quant",
                ).props("dense outlined").classes("w-36")
                load_btn = ui.button("Load", icon="download").props("dense no-caps")
                unload_btn = ui.button("Unload", icon="eject").props("dense flat no-caps")
            with ui.row().classes("w-full items-center gap-3 mt-2"):
                status_lbl = ui.label(ENGINE.status()).classes("text-xs text-gray-400")
                ui.badge(
                    f'{engine_settings["attn_implementation"]} · cache {engine_settings["cache_implementation"]}'
                    + (f' · kv {engine_settings["kv_quant"]}' if engine_settings["kv_quant"] != "none" else "")
                    + f' · ctx {engine_settings["max_context_tokens"]}'
                ).props("outline color=grey")
            with ui.row().classes("w-full gap-3 mt-2 items-end"):
                temp = ui.number("Temperature", value=0.7, min=0, max=2, step=0.1).props("dense outlined").classes("w-32")
                top_p = ui.number("Top-p", value=0.9, min=0.1, max=1, step=0.05).props("dense outlined").classes("w-28")
                max_new = ui.number("Max new tokens", value=512, min=16, max=16384).props("dense outlined").classes("w-36")
                clear_btn = ui.button("Clear chat", icon="delete_sweep").props("dense flat no-caps")

        chat_box = ui.column().classes(
            "w-full gap-2 min-h-[300px] max-h-[520px] overflow-auto p-2 rounded-lg"
        ).style(f"border: 1px solid {PANEL_EDGE}")

        def bubble(role: str, text: str = "") -> ui.label:
            with chat_box:
                if role == "user":
                    lbl = ui.label(text).classes(
                        "self-end bg-[#2a2a3a] px-3 py-2 rounded-lg text-sm whitespace-pre-wrap max-w-[80%]"
                    )
                else:
                    lbl = ui.label(text or "…").classes(
                        "self-start bg-[#1d1d28] px-3 py-2 rounded-lg text-sm whitespace-pre-wrap max-w-[80%]"
                    ).style(f"border: 1px solid {PANEL_EDGE}")
            return lbl

        def refresh_adapters(_e=None) -> None:
            opts = {"": "(none)"}
            if ckpt_sel.value:
                for a in db.q(
                    "SELECT * FROM adapters WHERE checkpoint_id=? AND status='trained' ORDER BY created_at DESC",
                    (ckpt_sel.value,),
                ):
                    opts[a["id"]] = f'{a["name"]} ({a["arch"]}, {a["quant"]})'
            adp_sel.options = opts
            adp_sel.value = ""
            adp_sel.update()
            refresh_routers()

        def refresh_routers(_e=None) -> None:
            opts = {"": "(adapter router)"}
            if adp_sel.value:
                for r in db.q(
                    "SELECT * FROM routers WHERE adapter_id=? AND status='trained' ORDER BY created_at DESC",
                    (adp_sel.value,),
                ):
                    opts[r["id"]] = f'{r["name"]} (k={r["top_k"]})'
            rtr_sel.options = opts
            rtr_sel.value = ""
            rtr_sel.update()

        ckpt_sel.on_value_change(refresh_adapters)
        adp_sel.on_value_change(refresh_routers)

        def resolve_quant(ckpt_row, adapter_row) -> str:
            if adapter_row:
                return adapter_row["quant"]
            if ckpt_row["quant_locked"]:
                return ckpt_row["quant_locked"]
            if quant_sel.value and quant_sel.value != "auto":
                return quant_sel.value
            gpu = best_gpu()
            if gpu and ckpt_row["params_b"] and ckpt_row["params_b"] * 2.3 > gpu.vram_free_gb:
                return "nf4"
            return "bf16"

        def on_load() -> None:
            if not ckpt_sel.value:
                ui.notify("Choose a checkpoint", type="warning")
                return
            if load_job["id"] and JOBS.get(load_job["id"]) and JOBS.get(load_job["id"]).status == "running":
                ui.notify("Already loading", type="warning")
                return
            ckpt_row = db.q1("SELECT * FROM checkpoints WHERE id=?", (ckpt_sel.value,))
            adapter_row = db.q1("SELECT * FROM adapters WHERE id=?", (adp_sel.value,)) if adp_sel.value else None
            router_row = db.q1("SELECT * FROM routers WHERE id=?", (rtr_sel.value,)) if rtr_sel.value else None
            if router_row and not adapter_row:
                ui.notify("Router requires its adapter selected", type="warning")
                return

            spec = LoadSpec(
                ref=ckpt_row["ref"],
                quant=resolve_quant(ckpt_row, adapter_row),
                adapter_path=adapter_row["path"] if adapter_row else None,
                adapter_arch=adapter_row["arch"] if adapter_row else "",
                router_path=router_row["path"] if router_row else None,
                settings=engine_settings,
            )

            def do_load(progress) -> list[str]:
                progress("load", 0.1, spec.ref)
                notes = ENGINE.load(spec)
                progress("load", 1.0, "loaded")
                return notes

            load_job["id"] = JOBS.submit("load_model", do_load)
            status_lbl.text = "loading…"
            load_btn.disable()

        def poll_load() -> None:
            if not load_job["id"]:
                return
            job = JOBS.get(load_job["id"])
            if job is None or job.status == "running":
                return
            load_job["id"] = None
            load_btn.enable()
            if job.status == "done":
                ui.notify(f"Loaded: {ENGINE.status()}", type="positive")
                for n in ENGINE.notes[-4:]:
                    ui.notify(n, type="info")
            else:
                ui.notify(f"Load failed: {job.error.splitlines()[0]}", type="negative", multi_line=True)
            status_lbl.text = ENGINE.status()

        def on_unload() -> None:
            ENGINE.unload()
            status_lbl.text = ENGINE.status()
            ui.notify("Model unloaded", type="info")

        def drain() -> None:
            q = stream_state["q"]
            if q is None or not stream_state["active"]:
                return
            try:
                while True:
                    item = q.get_nowait()
                    if item is None:
                        stream_state["active"] = False
                        if not stream_state["buf"]:
                            stream_state["label"].text = "(empty response)"
                        messages.append({"role": "assistant", "content": stream_state["buf"]})
                        send_btn.enable()
                        return
                    if isinstance(item, tuple) and item[0] == "__error__":
                        stream_state["active"] = False
                        stream_state["label"].text = stream_state["buf"] or ""
                        ui.notify(f"Generation error: {item[1]}", type="negative", multi_line=True)
                        send_btn.enable()
                        return
                    stream_state["buf"] += item
                    stream_state["label"].text = stream_state["buf"]
            except queue_mod.Empty:
                pass

        def on_send() -> None:
            text = (inp.value or "").strip()
            if not text:
                return
            if not ENGINE.is_loaded():
                ui.notify("Load a model first", type="warning")
                return
            if stream_state["active"]:
                return
            messages.append({"role": "user", "content": text})
            bubble("user", text)
            inp.value = ""
            stream_state["label"] = bubble("assistant", "")
            stream_state["buf"] = ""
            stream_state["q"] = queue_mod.Queue()
            stream_state["active"] = True
            send_btn.disable()

            q = stream_state["q"]
            snapshot = list(messages)

            def gen_worker() -> None:
                try:
                    for chunk in ENGINE.generate_stream(
                        snapshot,
                        temperature=float(temp.value or 0.0),
                        top_p=float(top_p.value or 0.9),
                        max_new_tokens=int(max_new.value or 512),
                    ):
                        q.put(chunk)
                    q.put(None)
                except Exception as e:  # noqa: BLE001
                    q.put(("__error__", str(e)))

            threading.Thread(target=gen_worker, daemon=True).start()

        def on_stop() -> None:
            if stream_state["active"]:
                stream_state["active"] = False
                if stream_state["buf"]:
                    messages.append({"role": "assistant", "content": stream_state["buf"]})
                send_btn.enable()

        def on_clear() -> None:
            messages.clear()
            chat_box.clear()

        with ui.row().classes("w-full items-end gap-2"):
            inp = ui.input(placeholder="Message…").props("outlined autogrow").classes("flex-1")
            send_btn = ui.button(icon="send", on_click=on_send).props("dense")
            ui.button(icon="stop", on_click=on_stop).props("dense flat")

        load_btn.on_click(on_load)
        unload_btn.on_click(on_unload)
        clear_btn.on_click(on_clear)
        inp.on("keydown.enter", lambda e: on_send() if not e.args.get("shiftKey") else None)
        ui.timer(0.05, drain)
        ui.timer(1.0, poll_load)
