from __future__ import annotations

import json
import shutil
from pathlib import Path

from nicegui import ui

from ... import db
from ...artifacts import checkpoint_dir, read_json
from ...models.families import checkpoint_family
from ...models.loading import scan_checkpoint as scan_local_checkpoint
from ...models.spalmer import require_spalmer_precision_capability
from ...workers.launch import launch_worker
from ..components import dir_browser, empty_state
from ..shell import panel, shell

JOINT_PER_TOKEN_STRATEGY = "joint_per_token_v1"
PER_TOKEN_ROUTER = "per_token"
STATIC_TOKEN_EXPORT_MODE = "calibrated_static_mole_v1"


def _parameter_label(params_b: float) -> str:
    count = float(params_b or 0.0) * 1e9
    if count >= 1e9:
        return f"{count / 1e9:.2f} B"
    if count >= 1e6:
        return f"{count / 1e6:.1f} M"
    if count > 0:
        return f"{count / 1e3:.1f} K"
    return "?"


def _precision_label(value: object, fallback: str) -> str:
    normalized = str(value or fallback).strip().casefold()
    aliases = {
        "bfloat16": "BF16",
        "float16": "FP16",
        "float32": "FP32",
    }
    return aliases.get(normalized, normalized.upper())


def _spalmer_recurrence(metadata: dict) -> dict | None:
    """Locate a recurrent-core block in a registry receipt or a scanned bundle."""

    nested = metadata.get("metadata")
    nested = nested if isinstance(nested, dict) else {}
    for block in (
        metadata.get("recurrence"),
        metadata.get("recurrence_config"),
        nested.get("recurrence"),
        nested.get("recurrence_config"),
    ):
        if isinstance(block, dict) and block and block.get("enabled") is not False:
            return block
    return None


def _spalmer_effective_depth(metadata: dict) -> int | None:
    """Physical layers stay in the registry column; this is the walked depth."""

    nested = metadata.get("metadata")
    nested = nested if isinstance(nested, dict) else {}
    recurrence = _spalmer_recurrence(metadata) or {}
    for value in (
        metadata.get("effective_depth"),
        nested.get("effective_depth"),
        recurrence.get("effective_depth"),
    ):
        try:
            depth = int(value)
        except (TypeError, ValueError):
            continue
        if depth > 0:
            return depth
    return None


def _spalmer_base_recipe(metadata: dict) -> str:
    """Describe either a yyLab registry receipt or a directly scanned bundle."""

    precision = metadata.get("expert_precision") or metadata.get("precision")
    precision = precision if isinstance(precision, dict) else {}
    native_metadata = metadata.get("metadata")
    native_metadata = native_metadata if isinstance(native_metadata, dict) else {}
    native_precision = native_metadata.get("precision")
    native_precision = native_precision if isinstance(native_precision, dict) else {}
    experts = metadata.get("experts_config")
    experts = experts if isinstance(experts, dict) else {}

    weight_format = (
        precision.get("expert_weights")
        or precision.get("expert_weight_format")
        or precision.get("weight_format")
        or native_precision.get("expert_weights")
        or native_precision.get("expert_weight_format")
        or experts.get("expert_weight_format")
        or metadata.get("expert_weight_format")
    )
    activation_format = (
        precision.get("expert_activations")
        or precision.get("expert_activation_format")
        or precision.get("activation_format")
        or native_precision.get("expert_activations")
        or native_precision.get("expert_activation_format")
        or experts.get("expert_activation_format")
    )
    master_dtype = (
        precision.get("master_weights")
        or precision.get("expert_master_dtype")
        or precision.get("master_dtype")
        or native_precision.get("master_weights")
        or native_precision.get("expert_master_dtype")
        or experts.get("expert_master_dtype")
    )
    recipe = (
        f"SPALMER · {_precision_label(weight_format, 'unknown')}/"
        f"{_precision_label(activation_format, 'unknown')} · "
        f"{_precision_label(master_dtype, 'unknown')} master"
    )
    executable = metadata.get("precision_executable", precision.get("executable"))
    if executable is False:
        recipe += " · metadata-only (blocked legacy precision)"
    recurrence = _spalmer_recurrence(metadata)
    if recurrence is not None:
        recipe += (
            f" · recurrent {int(recurrence.get('prelude_layers') or 0)}/"
            f"{int(recurrence.get('core_layers') or 0)}/"
            f"{int(recurrence.get('coda_layers') or 0)} × "
            f"{int(recurrence.get('default_steps') or 0)}"
        )
    return recipe


def _adapter_metadata(adapter: dict) -> dict:
    try:
        metadata = json.loads(adapter.get("meta_json") or "{}")
    except (TypeError, ValueError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _adapter_router_type(adapter: dict) -> str:
    """Resolve routing from registry metadata, then the artifact itself."""

    metadata = _adapter_metadata(adapter)
    if metadata.get("training_strategy") == JOINT_PER_TOKEN_STRATEGY:
        return PER_TOKEN_ROUTER
    if metadata.get("router_arch") == "shared_per_block_per_token_v1":
        return PER_TOKEN_ROUTER

    artifact_path = adapter.get("path")
    if artifact_path:
        config_path = Path(artifact_path) / "mole_config.json"
        try:
            payload = read_json(config_path)
            router_type = payload.get("config", {}).get("router_type")
            if isinstance(router_type, str):
                return router_type
        except (OSError, TypeError, ValueError):
            pass
    return "per_layer"


def _export_job_config(
    checkpoint: dict,
    adapter: dict,
    router: dict | None,
    out_checkpoint_id: str,
) -> dict:
    router_type = _adapter_router_type(adapter)
    joint_per_token = router_type == PER_TOKEN_ROUTER
    try:
        corpora = json.loads(adapter.get("corpora_json") or "[]")
    except (TypeError, ValueError):
        corpora = []
    return {
        "checkpoint_ref": checkpoint["ref"],
        "adapter_id": adapter["id"],
        "adapter_path": adapter["path"],
        "adapter_arch": adapter["arch"],
        # A stage-2 shared-prompt router is a different architecture and must
        # never override the block token routers stored in a joint artifact.
        "router_path": None if joint_per_token else router["path"] if router else None,
        "source_router_type": router_type,
        "export_mode": STATIC_TOKEN_EXPORT_MODE if joint_per_token else "merged_weights_v1",
        "dynamic_token_routing_preserved": False if joint_per_token else None,
        "corpora": corpora,
        "out_checkpoint_id": out_checkpoint_id,
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

        def base_recipe(c: dict) -> str:
            try:
                metadata = json.loads(c.get("meta_json") or "{}")
            except (TypeError, ValueError):
                metadata = {}
            precision = metadata.get("precision") or {}
            if checkpoint_family(c) == "spalmer":
                return _spalmer_base_recipe(metadata)
            qat = str(precision.get("qat") or "none")
            dtype = str(precision.get("parameter_dtype") or "")
            initialization = str(metadata.get("initialization") or "")
            if qat != "none":
                return f"{initialization or 'base'} · {qat} QAT / {dtype or 'shadow'}"
            if initialization or dtype:
                return " · ".join(value for value in (initialization, dtype) if value)
            return "imported"

        def rows() -> list[dict]:
            out = []
            for c in db.q("SELECT * FROM checkpoints ORDER BY created_at DESC"):
                out.append(
                    {
                        "id": c["id"],
                        "name": c["name"],
                        "arch": c["arch"],
                        "params": _parameter_label(c["params_b"]),
                        "source": f'{c["source"]}: {c["ref"]}',
                        "recipe": base_recipe(c),
                        "quant_locked": c["quant_locked"] or "-",
                    }
                )
            return out

        def refresh() -> None:
            table_container.clear()
            data = rows()
            with table_container:
                if not data:
                    empty_state(
                        "hub",
                        "No checkpoints registered",
                        "Add a pretrained model — a local HuggingFace folder (params scanned instantly) or a hub id "
                        "like Qwen/Qwen2.5-1.5B-Instruct.",
                        "Add checkpoint",
                        add_dlg.open,
                    )
                    return
                table = ui.table(
                    columns=[
                        {"name": "name", "label": "Name", "field": "name", "sortable": True, "align": "left"},
                        {"name": "arch", "label": "Architecture", "field": "arch", "align": "left"},
                        {"name": "params", "label": "Params", "field": "params", "sortable": True},
                        {"name": "source", "label": "Source", "field": "source", "align": "left"},
                        {"name": "recipe", "label": "Base recipe", "field": "recipe", "align": "left"},
                        {"name": "quant_locked", "label": "Quant lock", "field": "quant_locked"},
                    ],
                    rows=data,
                    row_key="id",
                    pagination=15,
                ).classes("w-full")
                table.add_slot(
                    "body-cell-quant_locked",
                    """
                    <q-td :props="props">
                      <q-badge v-if="props.value !== '-'" color="amber" outline>
                        <q-icon name="lock" size="12px" class="q-mr-xs"></q-icon>{{ props.value }}
                      </q-badge>
                      <span v-else class="text-grey-6">-</span>
                    </q-td>
                    """,
                )
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
                    ui.button(
                        "Remove",
                        icon="delete",
                        on_click=lambda cid=cid, name=c["name"]: ask_remove(cid, name),
                    ).props("flat dense no-caps color=negative")
                try:
                    detail_metadata = json.loads(c.get("meta_json") or "{}")
                except (TypeError, ValueError):
                    detail_metadata = {}
                effective_depth = (
                    _spalmer_effective_depth(detail_metadata)
                    if isinstance(detail_metadata, dict)
                    else None
                )
                layers_label = f'{c["num_layers"]} layers'
                if effective_depth and effective_depth != c["num_layers"]:
                    layers_label += f" (effective {effective_depth})"
                ui.label(
                    f'arch: {c["arch"]} · hidden {c["hidden_size"]} · '
                    f'{layers_label} · {_parameter_label(c["params_b"])} params'
                )
                ui.label(f'source: {c["source"]} → {c["ref"]}')
                ui.label(f"base recipe: {base_recipe(c)}").classes("text-sm text-gray-400")
                if checkpoint_family(c) == "spalmer":
                    precision_executable = detail_metadata.get("precision_executable")
                    if precision_executable is False:
                        reason = str(
                            detail_metadata.get("precision_rejection_reason")
                            or "the bundle requires a retired emulated precision path"
                        )
                        ui.label(
                            "Metadata-only SPALMER checkpoint: loading, inference, and continued "
                            f"training are blocked because {reason}."
                        ).classes("text-red-300 text-sm")
                    else:
                        try:
                            capability = require_spalmer_precision_capability(c["ref"])
                        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                            ui.label(
                                "SPALMER precision is unavailable on this runtime: "
                                f"{str(exc).splitlines()[0][:240]}"
                            ).classes("text-red-300 text-sm")
                        else:
                            ui.label(
                                "Native recurrent KDA/NoPE MLA with its checkpointed micro-expert "
                                "routing and exact-device verified "
                                f"{str(capability['weight_format']).upper()}/"
                                f"{str(capability['activation_format']).upper()} precision via "
                                f"{capability['provider_id']}. yyLab PEFT adapters do not apply."
                            ).classes("text-cyan-300 text-sm")
                elif c["quant_locked"]:
                    ui.label(
                        f'Quantization locked to {c["quant_locked"]} for all adapters on this checkpoint.'
                    ).classes("text-amber-400 text-sm")
                else:
                    ui.label("Quantization not locked yet — set when the first adapter is trained.").classes(
                        "text-xs text-gray-500"
                    )
                adapters = (
                    []
                    if checkpoint_family(c) == "spalmer"
                    else db.q("SELECT * FROM adapters WHERE checkpoint_id=?", (cid,))
                )
                if adapters:
                    ui.label("Adapters").classes("text-sm font-semibold mt-2")
                    for a in adapters:
                        joint_per_token = _adapter_router_type(a) == PER_TOKEN_ROUTER
                        adapter_metadata = _adapter_metadata(a)
                        merge_safe = adapter_metadata.get("merge_safe") is not False
                        diagnostic = adapter_metadata.get("diagnostic_verdict") or {}
                        with ui.row().classes("w-full items-center justify-between gap-2"):
                            with ui.column().classes("gap-0"):
                                ui.label(
                                    f'· {a["name"]} — {a["arch"]} E={a["num_experts"]} k={a["top_k"]} r={a["rank"]} quant={a["quant"]} [{a["status"]}]'
                                ).classes("text-xs text-gray-400")
                                if joint_per_token:
                                    ui.label(
                                        "Export calibrates a static expert blend; dynamic per-token routing is not preserved."
                                    ).classes("text-[11px] text-amber-400")
                                if diagnostic.get("label"):
                                    recommended = bool(
                                        diagnostic.get("recommended_for_inference")
                                    )
                                    ui.label(
                                        f"AntiDoom diagnostic: {diagnostic['label']} — "
                                        f"{diagnostic.get('explanation', '')}"
                                    ).classes(
                                        "text-[11px] text-green-400"
                                        if recommended
                                        else "text-[11px] text-red-400"
                                    )
                                if not merge_safe:
                                    ui.label(
                                        "Adapter-only artifact: merging is disabled for tied-output safety."
                                    ).classes("text-[11px] text-amber-400")
                            with ui.row().classes("items-center gap-1"):
                                if a["status"] == "trained" and merge_safe:
                                    export_button = ui.button(
                                        (
                                            "Export calibrated static approximation"
                                            if joint_per_token
                                            else "Export merged"
                                        ),
                                        icon="ios_share",
                                        on_click=lambda _, a=a: export_adapter(c, a),
                                    ).props("dense flat no-caps")
                                    if joint_per_token:
                                        export_button.tooltip(
                                            "Uses the adapter's corpora to calibrate static expert weights. "
                                            "The exported base model cannot route each token dynamically."
                                        )
                                ui.button(
                                    icon="delete",
                                    on_click=lambda _, a=a: ask_delete_adapter(a),
                                ).props("dense flat no-caps color=negative").tooltip("Delete adapter + its routers")

        def export_adapter(ckpt_row: dict, a: dict) -> None:
            if _adapter_metadata(a).get("merge_safe") is False:
                ui.notify(
                    "This output-head AntiDoom adapter is intentionally adapter-only and "
                    "cannot be merged safely into the tied-embedding base.",
                    type="warning",
                    multi_line=True,
                )
                return
            joint_per_token = _adapter_router_type(a) == PER_TOKEN_ROUTER
            router = None
            if not joint_per_token:
                router = db.q1(
                    "SELECT * FROM routers WHERE adapter_id=? AND status='trained' ORDER BY created_at DESC",
                    (a["id"],),
                )
            out_id = db.new_id("ckpt")
            suffix = "calibrated-static" if joint_per_token else "merged"
            out_name = f'{ckpt_row["name"]}+{a["name"]}-{suffix}'
            config = _export_job_config(ckpt_row, a, router, out_id)
            if joint_per_token and not config["corpora"]:
                ui.notify(
                    "A joint per-token adapter needs its training corpora for calibrated static export.",
                    type="warning",
                )
                return
            db.execute(
                "INSERT INTO checkpoints(id, name, created_at, kind, source, ref) VALUES(?,?,?,?,?,?)",
                (out_id, out_name, db.now_iso(), "pending", "pending", ""),
            )
            rid = db.new_id("run")
            db.execute(
                "INSERT INTO runs(id, kind, status, created_at, config_json) VALUES(?,?,?,?,?)",
                (rid, "export", "queued", db.now_iso(), json.dumps(config)),
            )
            pid = launch_worker(rid)
            db.execute("UPDATE runs SET metrics_json=? WHERE id=?", (json.dumps({"pid": pid}), rid))
            if joint_per_token:
                ui.notify(
                    f"Calibrated static approximation started — {out_name} (pid {pid}). "
                    "Dynamic token routing will not be preserved.",
                    type="warning",
                )
            else:
                ui.notify(f"Export started — {out_name} (pid {pid}). See Runs.", type="positive")

        def remove(cid: str) -> None:
            if db.q1("SELECT id FROM adapters WHERE checkpoint_id=?", (cid,)):
                ui.notify("Delete adapters on this checkpoint first", type="warning")
                return
            db.execute("DELETE FROM checkpoints WHERE id=?", (cid,))
            state["selected"] = None
            refresh()
            render_detail()
            ui.notify("Checkpoint removed from registry", type="info")

        rm_state: dict = {"cid": None}
        with ui.dialog() as rm_dlg, ui.card().classes("w-[440px] p-4").style(
            "background: #16161d; border: 1px solid #262633"
        ):
            ui.label("Remove checkpoint?").classes("text-base font-semibold")
            rm_msg = ui.label("").classes("text-sm text-gray-400 mt-1")
            ui.label(
                "This only removes the registry entry. Local folders stay on disk; hub models stay cached."
            ).classes("text-xs text-gray-500 mt-1")
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=rm_dlg.close).props("flat dense no-caps")

                def do_remove() -> None:
                    rm_dlg.close()
                    if rm_state["cid"]:
                        remove(rm_state["cid"])

                ui.button("Remove", on_click=do_remove).props("dense no-caps color=negative")

        def ask_remove(cid: str, name: str) -> None:
            rm_state["cid"] = cid
            rm_msg.text = f'Remove "{name}" from the registry?'
            rm_dlg.open()

        def delete_adapter(aid: str) -> None:
            for r in db.q("SELECT id, path FROM routers WHERE adapter_id=?", (aid,)):
                if r["path"]:
                    shutil.rmtree(r["path"], ignore_errors=True)
                db.execute("DELETE FROM routers WHERE id=?", (r["id"],))
            a = db.q1("SELECT path FROM adapters WHERE id=?", (aid,))
            if a and a["path"]:
                shutil.rmtree(a["path"], ignore_errors=True)
            db.execute("DELETE FROM adapters WHERE id=?", (aid,))
            ui.notify("Adapter and its routers deleted", type="info")
            render_detail()

        adp_state: dict = {"aid": None}
        with ui.dialog() as adp_dlg, ui.card().classes("w-[440px] p-4").style(
            "background: #16161d; border: 1px solid #262633"
        ):
            ui.label("Delete adapter?").classes("text-base font-semibold")
            adp_msg = ui.label("").classes("text-sm text-gray-400 mt-1")
            ui.label(
                "Removes the adapter and any router snapshots trained on it, including their files on disk."
            ).classes("text-xs text-gray-500 mt-1")
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=adp_dlg.close).props("flat dense no-caps")

                def do_delete_adapter() -> None:
                    adp_dlg.close()
                    if adp_state["aid"]:
                        delete_adapter(adp_state["aid"])

                ui.button("Delete", on_click=do_delete_adapter).props("dense no-caps color=negative")

        def ask_delete_adapter(a: dict) -> None:
            adp_state["aid"] = a["id"]
            n_routers = db.q1("SELECT COUNT(*) n FROM routers WHERE adapter_id=?", (a["id"],))["n"]
            extra = f" and {n_routers} router snapshot(s)" if n_routers else ""
            adp_msg.text = f'Delete "{a["name"]}"{extra}?'
            adp_dlg.open()

        with ui.dialog() as add_dlg, ui.card().classes("w-[560px]").style("background: #16161d; border: 1px solid #262633"):
            ui.label("Add checkpoint").classes("text-base font-semibold")
            name_in = ui.input("Name").props("dense outlined").classes("w-full")
            mode = ui.radio(["Local checkpoint", "HuggingFace Hub"], value="Local checkpoint").props("dense inline")
            with ui.row().classes("w-full items-end gap-2") as local_row:
                ref_in = ui.input("HF folder or SPALMER .pt path").props("dense outlined").classes("flex-1")
                browser = dir_browser("Select checkpoint folder", lambda p: ref_in.set_value(p))
                ui.button(icon="folder", on_click=browser.open).props("dense outline")
            hub_hint = ui.label("Hub id, e.g. Qwen/Qwen2.5-1.5B-Instruct (weights downloaded on first use)").classes(
                "text-xs text-gray-500"
            )
            hub_hint.visible = False

            def mode_changed(e) -> None:
                local = e.value == "Local checkpoint"
                local_row.visible = local
                hub_hint.visible = not local
                ref_in.props(
                    f'label={"HF folder or SPALMER .pt path" if local else "Hub model id"}'
                )

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
                    if mode.value == "Local checkpoint":
                        try:
                            status_lbl.text = "scanning checkpoint metadata..."
                            info = scan_local_checkpoint(Path(ref))
                            meta.update(info)
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
                            "local" if mode.value == "Local checkpoint" else "hub",
                            str(meta.get("checkpoint_ref") or ref),
                            meta["arch"],
                            meta["params_b"],
                            meta["hidden_size"],
                            meta["num_layers"],
                            (
                                str(
                                    meta.get("path")
                                    or (
                                        Path(str(meta.get("checkpoint_ref") or ref)).parent
                                        if Path(str(meta.get("checkpoint_ref") or ref)).is_file()
                                        else Path(str(meta.get("checkpoint_ref") or ref))
                                    )
                                )
                                if mode.value == "Local checkpoint"
                                else str(checkpoint_dir(cid))
                            ),
                            json.dumps(meta),
                        ),
                    )
                    add_dlg.close()
                    refresh()
                    ui.notify("Checkpoint registered", type="positive")

                ui.button("Add", on_click=add).props("dense no-caps")

        add_btn.on_click(add_dlg.open)
        refresh()
