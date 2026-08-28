from __future__ import annotations

import json
import math
from pathlib import Path

from nicegui import ui
from safetensors import safe_open

from ... import db
from ...artifacts import checkpoint_dir
from ..components import dir_browser
from ..shell import panel, shell


def scan_local_checkpoint(path: Path) -> dict:
    cfg_file = path / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError("config.json not found — not a HuggingFace checkpoint folder")
    cfg = json.loads(cfg_file.read_text("utf-8"))
    arch = (cfg.get("architectures") or ["unknown"])[0]
    params = 0
    files = list(path.glob("*.safetensors"))
    for f in files:
        with safe_open(str(f), framework="numpy") as sf:
            for key in sf.keys():
                shape = sf.get_slice(key).get_shape()
                params += math.prod(shape) if shape else 0
    return {
        "arch": arch,
        "params_b": round(params / 1e9, 3),
        "hidden_size": cfg.get("hidden_size", 0) or cfg.get("n_embd", 0),
        "num_layers": cfg.get("num_hidden_layers", 0) or cfg.get("n_layer", 0),
        "base_cfg": cfg,
    }


@ui.page("/checkpoints")
def checkpoints_page():
    state: dict = {"selected": None}

    with shell("Checkpoints", "/checkpoints"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Checkpoints").classes("text-xl font-semibold")
            add_btn = ui.button("Add checkpoint", icon="add").props("no-caps")

        table_container = ui.column().classes("w-full")
        detail_container = ui.column().classes("w-full")

        def quant_of(cid: str) -> str:
            row = db.q1("SELECT quant FROM adapters WHERE checkpoint_id=? AND status!='deleted' ORDER BY created_at LIMIT 1", (cid,))
            return row["quant"] if row else ""

        def rows() -> list[dict]:
            out = []
            for c in db.q("SELECT * FROM checkpoints ORDER BY created_at DESC"):
                out.append(
                    {
                        "id": c["id"],
                        "name": c["name"],
                        "arch": c["arch"],
                        "params": f'{c["params_b"]:.2f} B' if c["params_b"] else "?",
                        "source": f'{c["source"]}: {c["ref"]}',
                        "quant_locked": c["quant_locked"] or "-",
                    }
                )
            return out

        def refresh() -> None:
            table_container.clear()
            with table_container:
                table = ui.table(
                    columns=[
                        {"name": "name", "label": "Name", "field": "name", "sortable": True, "align": "left"},
                        {"name": "arch", "label": "Architecture", "field": "arch", "align": "left"},
                        {"name": "params", "label": "Params", "field": "params", "sortable": True},
                        {"name": "source", "label": "Source", "field": "source", "align": "left"},
                        {"name": "quant_locked", "label": "Quant lock", "field": "quant_locked"},
                    ],
                    rows=rows(),
                    row_key="id",
                    pagination=15,
                ).classes("w-full")
                table.on("row-click", lambda e: select(e.args[1]["id"]))

        def select(cid: str) -> None:
            state["selected"] = cid
            render_detail()

        def render_detail() -> None:
            detail_container.clear()
            cid = state["selected"]
            if not cid:
                return
            c = db.q1("SELECT * FROM checkpoints WHERE id=?", (cid,))
            if not c:
                state["selected"] = None
                return
            with detail_container, panel():
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(c["name"]).classes("text-lg font-semibold")
                    ui.button("Remove", icon="delete", on_click=lambda: remove(cid)).props(
                        "flat dense no-caps color=negative"
                    )
                ui.label(f'arch: {c["arch"]} · hidden {c["hidden_size"]} · {c["num_layers"]} layers · {c["params_b"]:.2f} B params')
                ui.label(f'source: {c["source"]} → {c["ref"]}')
                if c["quant_locked"]:
                    ui.label(
                        f'Quantization locked to {c["quant_locked"]} for all adapters on this checkpoint.'
                    ).classes("text-amber-400 text-sm")
                else:
                    ui.label("Quantization not locked yet — set when the first adapter is trained.").classes(
                        "text-xs text-gray-500"
                    )
                adapters = db.q("SELECT * FROM adapters WHERE checkpoint_id=?", (cid,))
                if adapters:
                    ui.label("Adapters").classes("text-sm font-semibold mt-2")
                    for a in adapters:
                        ui.label(
                            f'· {a["name"]} — {a["arch"]} E={a["num_experts"]} k={a["top_k"]} r={a["rank"]} quant={a["quant"]} [{a["status"]}]'
                        ).classes("text-xs text-gray-400")

        def remove(cid: str) -> None:
            if db.q1("SELECT id FROM adapters WHERE checkpoint_id=?", (cid,)):
                ui.notify("Delete adapters on this checkpoint first", type="warning")
                return
            db.execute("DELETE FROM checkpoints WHERE id=?", (cid,))
            state["selected"] = None
            refresh()
            render_detail()
            ui.notify("Checkpoint removed from registry", type="info")

        with ui.dialog() as add_dlg, ui.card().classes("w-[560px]").style("background: #16161d; border: 1px solid #262633"):
            ui.label("Add checkpoint").classes("text-base font-semibold")
            name_in = ui.input("Name").props("dense outlined").classes("w-full")
            mode = ui.radio(["Local folder", "HuggingFace Hub"], value="Local folder").props("dense inline")
            with ui.row().classes("w-full items-end gap-2") as local_row:
                ref_in = ui.input("Folder path").props("dense outlined").classes("flex-1")
                browser = dir_browser("Select checkpoint folder", lambda p: ref_in.set_value(p))
                ui.button(icon="folder", on_click=browser.open).props("dense outline")
            hub_hint = ui.label("Hub id, e.g. Qwen/Qwen2.5-1.5B-Instruct (weights downloaded on first use)").classes(
                "text-xs text-gray-500"
            )
            hub_hint.visible = False

            def mode_changed(e) -> None:
                local = e.value == "Local folder"
                local_row.visible = local
                hub_hint.visible = not local
                ref_in.props(f'label={"Folder path" if local else "Hub model id"}')

            mode.on_value_change(mode_changed)
            status_lbl = ui.label("").classes("text-xs text-gray-400")
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=add_dlg.close).props("flat dense no-caps")

                def add() -> None:
                    ref = (ref_in.value or "").strip()
                    if not ref:
                        ui.notify("Provide a path or hub id", type="warning")
                        return
                    name = (name_in.value or "").strip() or ref.split("/")[-1].split("\\")[-1]
                    meta = {"arch": "", "params_b": 0, "hidden_size": 0, "num_layers": 0}
                    if mode.value == "Local folder":
                        try:
                            status_lbl.text = "scanning safetensors..."
                            info = scan_local_checkpoint(Path(ref))
                            meta.update(
                                {
                                    "arch": info["arch"],
                                    "params_b": info["params_b"],
                                    "hidden_size": info["hidden_size"],
                                    "num_layers": info["num_layers"],
                                }
                            )
                        except Exception as e:  # noqa: BLE001
                            ui.notify(f"Scan failed: {e}", type="negative")
                            return
                    cid = db.new_id("ckpt")
                    db.execute(
                        "INSERT INTO checkpoints(id, name, created_at, kind, source, ref, arch, params_b, hidden_size, num_layers, path, meta_json)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            cid,
                            name,
                            db.now_iso(),
                            "base",
                            "local" if mode.value == "Local folder" else "hub",
                            ref,
                            meta["arch"],
                            meta["params_b"],
                            meta["hidden_size"],
                            meta["num_layers"],
                            ref if mode.value == "Local folder" else str(checkpoint_dir(cid)),
                            json.dumps(meta),
                        ),
                    )
                    add_dlg.close()
                    refresh()
                    ui.notify("Checkpoint registered", type="positive")

                ui.button("Add", on_click=add).props("dense no-caps")

        add_btn.on_click(add_dlg.open)
        refresh()
