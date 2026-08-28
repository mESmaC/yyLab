from __future__ import annotations

import queue as queue_mod
import threading

from nicegui import ui

from ... import db
from ...config import get_settings
from ...hw import best_gpu
from ...infer.engine import ENGINE, LoadSpec
from ...jobs import JOBS
from ..components import empty_state
from ..shell import ACCENT, PANEL, PANEL_EDGE, panel, shell

MD_CSS = """
<style>
.assistant-md { font-size: 0.875rem; line-height: 1.5; }
.assistant-md p { margin: 0.25rem 0; }
.assistant-md pre { background: #0d0d12; border: 1px solid #262633; padding: 0.6rem;
  border-radius: 8px; overflow-x: auto; }
.assistant-md code { background: #0d0d12; padding: 0.1rem 0.3rem; border-radius: 4px;
  font-size: 0.8rem; }
.assistant-md pre code { background: transparent; padding: 0; }
</style>
"""


@ui.page("/chat")
def chat_page():
    with shell("Chat", "/chat"):
        ui.add_head_html(MD_CSS)
        ui.label("Chat").classes("text-xl font-semibold")

        ckpts = {c["id"]: c["name"] for c in db.q("SELECT * FROM checkpoints WHERE kind!='pending' ORDER BY name")}
        engine_settings = dict(get_settings()["engine"])
        messages: list[dict] = []
        stream_state: dict = {"q": None, "col": None, "label": None, "active": False, "buf": "", "spin": None}
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
            "w-full gap-3 min-h-[320px] max-h-[540px] overflow-auto p-4 rounded-xl"
        ).style(f"border: 1px solid {PANEL_EDGE}; background: {PANEL}66")

        def scroll_down() -> None:
            ui.run_javascript(
                f"const el=document.getElementById('{chat_box.id}'); if(el) el.scrollTop=el.scrollHeight;"
            )

        def add_user(text: str) -> None:
            with chat_box:
                with ui.row().classes("w-full justify-end"):
                    with ui.row().classes("items-start gap-2 max-w-[80%]").style("flex-direction: row-reverse"):
                        ui.icon("person", size="20px").classes("mt-1 text-gray-400")
                        ui.label(text).classes(
                            "bg-[#2a2a3a] px-3 py-2 rounded-xl rounded-tr-sm text-sm whitespace-pre-wrap"
                        )
            scroll_down()

        def add_assistant_start() -> None:
            with chat_box:
                with ui.row().classes("w-full items-start gap-2"):
                    ui.icon("smart_toy", size="20px").classes("mt-1").style(f"color: {ACCENT}")
                    with ui.column().classes("gap-1 flex-1 min-w-0 max-w-[85%]") as col:
                        spin = ui.spinner("dots", size="md")
                        lbl = ui.label("").classes(
                            "px-3 py-2 rounded-xl rounded-tl-sm text-sm whitespace-pre-wrap w-full"
                        ).style(f"background: {PANEL}; border: 1px solid {PANEL_EDGE}")
            stream_state["col"] = col
            stream_state["label"] = lbl
            stream_state["spin"] = spin
            scroll_down()

        def finish_assistant(buf: str, error: bool = False) -> None:
            col = stream_state["col"]
            if col is None:
                return
            col.clear()
            with col:
                if error:
                    ui.label(buf or "(generation error)").classes("text-sm text-red-400 whitespace-pre-wrap")
                elif not buf:
                    ui.label("(empty response)").classes("text-sm text-gray-500 italic")
                else:
                    ui.markdown(buf).classes("assistant-md w-full")
            stream_state["col"] = None
            stream_state["label"] = None
            stream_state["spin"] = None
            scroll_down()

        def show_empty_hint() -> None:
            chat_box.clear()
            with chat_box:
                empty_state(
                    "chat",
                    "Start a conversation",
                    "Load a checkpoint (optionally with an adapter and router), then send a message below.",
                )

        show_empty_hint()

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
            load_btn.props("loading").disable()

        def poll_load() -> None:
            if not load_job["id"]:
                return
            job = JOBS.get(load_job["id"])
            if job is None or job.status == "running":
                return
            load_job["id"] = None
            load_btn.props(remove="loading").enable()
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
            got_new = False
            try:
                while True:
                    item = q.get_nowait()
                    if item is None:
                        stream_state["active"] = False
                        if not stream_state["buf"]:
                            pass
                        messages.append({"role": "assistant", "content": stream_state["buf"]})
                        finish_assistant(stream_state["buf"])
                        send_btn.enable()
                        return
                    if isinstance(item, tuple) and item[0] == "__error__":
                        stream_state["active"] = False
                        ui.notify(f"Generation error: {item[1]}", type="negative", multi_line=True)
                        finish_assistant(stream_state["buf"], error=True)
                        send_btn.enable()
                        return
                    stream_state["buf"] += item
                    got_new = True
            except queue_mod.Empty:
                pass
            if got_new and stream_state["label"] is not None:
                if stream_state["spin"] is not None:
                    stream_state["spin"].delete()
                    stream_state["spin"] = None
                stream_state["label"].text = stream_state["buf"]
                scroll_down()

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
            add_user(text)
            inp.value = ""
            add_assistant_start()
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
                buf = stream_state["buf"]
                messages.append({"role": "assistant", "content": buf})
                finish_assistant(buf)
                send_btn.enable()

        def on_clear() -> None:
            messages.clear()
            show_empty_hint()

        with ui.row().classes("w-full items-end gap-2"):
            inp = ui.input(placeholder="Message…").props("outlined autogrow").classes("flex-1")
            send_btn = ui.button(icon="send", on_click=on_send).props("dense")
            ui.button(icon="stop", on_click=on_stop).props("dense flat").tooltip("Stop generating")

        load_btn.on_click(on_load)
        unload_btn.on_click(on_unload)
        clear_btn.on_click(on_clear)
        inp.on("keydown.enter", lambda e: on_send() if not e.args.get("shiftKey") else None)
        ui.timer(0.05, drain)
        ui.timer(1.0, poll_load)
