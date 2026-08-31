from __future__ import annotations

import shutil

import orjson
from nicegui import ui

from ... import db
from ...artifacts import corpus_dir, fmt_bytes
from ...config import get_settings, update_settings
from ...ingest.domains import DOMAINS
from ...ingest.library import ingested_by_source, scan_dataset_folders
from ...ingest.pipeline import IngestConfig, _iter_shards, run_ingest
from ...ingest.tokenize_stats import SPEC_KEYS, SPEC_LABELS
from ...jobs import JOBS
from ..components import dir_browser, empty_state, hint, status_badge
from ..shell import panel, shell, stat_card


def _stats(c: dict) -> dict:
    return orjson.loads(c["stats_json"] or b"{}")


def _domain_tokens(stats: dict, tok: str) -> dict[str, int]:
    out = {}
    for dom, d in stats.get("domains", {}).items():
        out[dom] = d.get("tokens", {}).get(tok, 0) or d.get("chars", 0)
    return out


@ui.page("/corpora")
def corpora_page():
    state: dict = {"job": None, "selected": None, "tok": "gpt2"}

    with shell("Corpora", "/corpora"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Corpora").classes("text-xl font-semibold")
            ingest_btn = ui.button("Ingest folder", icon="add").props("no-caps")

        with ui.row().classes("w-full items-center gap-3"):
            tok_select = ui.select(SPEC_LABELS, value=state["tok"], label="Token count basis").props(
                "dense outlined"
            ).classes("w-64")
            progress_row = ui.row().classes("flex-1 items-center gap-3")
            bar = ui.linear_progress(value=0, show_value=False).classes("flex-1")
            bar.props("color=primary track-color=grey-8")
            plabel = ui.label("").classes("text-xs text-gray-400 whitespace-nowrap")
            progress_row.visible = False

        library_container = ui.column().classes("w-full")
        table_container = ui.column().classes("w-full")
        detail_container = ui.column().classes("w-full gap-4")

        def start_library_ingest(path: str, name: str) -> None:
            if state["job"]:
                ui.notify("An ingest job is already running", type="warning")
                return
            cfg = IngestConfig(name=name, source_dir=path, tokenizer_keys=[])
            state["job"] = JOBS.submit("ingest", run_ingest, cfg)
            progress_row.visible = True
            plabel.text = f"queued · {name}"
            render_library()

        def render_library() -> None:
            library_container.clear()
            root = (get_settings().get("datasets_root") or "").strip()
            with library_container, panel():
                with ui.row().classes("w-full items-end gap-2"):
                    root_in = ui.input(
                        "Datasets folder (one subfolder per corpus)", value=root,
                        placeholder="D:\\datasets",
                    ).props("dense outlined").classes("flex-1")
                    browser = dir_browser("Select datasets folder", lambda p: root_in.set_value(p))
                    ui.button(icon="folder", on_click=browser.open).props("dense outline")

                    def save_root() -> None:
                        update_settings(datasets_root=(root_in.value or "").strip())
                        render_library()

                    ui.button("Scan", icon="refresh", on_click=save_root).props("dense outline no-caps")
                root = (get_settings().get("datasets_root") or "").strip()
                if not root:
                    hint("Point this at the parent folder containing your corpus folders, then Scan.")
                    return
                folders = scan_dataset_folders(root)
                if not folders:
                    hint("No subfolders found in that directory.")
                    return
                known = ingested_by_source()
                ui.label(f"{len(folders)} dataset folders").classes("text-xs text-gray-500 mt-1")
                for f in folders:
                    ing = known.get(f.path)
                    busy = state["job"] is not None
                    with ui.row().classes("w-full items-center gap-3 py-1 border-b border-gray-800"):
                        ui.label(f.name).classes("text-sm text-gray-200 flex-1 truncate")
                        ui.label(f"{f.files} files · {fmt_bytes(f.size_bytes)}").classes(
                            "text-xs text-gray-500 whitespace-nowrap"
                        )
                        if ing:
                            status_badge(ing["status"])
                        elif busy:
                            ui.button("Ingest", on_click=lambda p=f.path, n=f.name: start_library_ingest(p, n)).props(
                                "dense no-caps outline"
                            ).classes("w-24").set_enabled(False)
                        else:
                            ui.button("Ingest", on_click=lambda p=f.path, n=f.name: start_library_ingest(p, n)).props(
                                "dense no-caps outline"
                            ).classes("w-24")

        def corpus_rows() -> list[dict]:
            rows = []
            for c in db.q(
                "SELECT * FROM corpora WHERE COALESCE(kind,'ingested') != 'rejected' ORDER BY created_at DESC"
            ):
                stats = _stats(c)
                dom_tokens = sorted(_domain_tokens(stats, state["tok"]).items(), key=lambda kv: -kv[1])
                dom_str = ", ".join(f"{k}" for k, _ in dom_tokens[:3]) or "-"
                rows.append(
                    {
                        "id": c["id"],
                        "name": c["name"],
                        "status": c["status"],
                        "verdict": c.get("verdict") or "",
                        "domains": dom_str,
                        "docs": stats.get("docs_written", 0),
                        "size": fmt_bytes(stats.get("bytes_on_disk", 0)),
                        "tokens": stats.get("totals", {}).get("tokens", {}).get(state["tok"], 0),
                    }
                )
            return rows

        table_holder: dict = {}

        def refresh_table() -> None:
            table_container.clear()
            rows = corpus_rows()
            with table_container:
                if not rows:
                    empty_state(
                        "folder_open",
                        "No corpora yet",
                        "Point yyLab at a folder of text, code, or JSONL data. It will deduplicate, tag domains, "
                        "and count tokens across several tokenizer families.",
                        "Ingest folder",
                        ingest_dlg.open,
                    )
                    table_holder["t"] = None
                    return
                table = ui.table(
                    columns=[
                        {"name": "name", "label": "Name", "field": "name", "sortable": True, "align": "left"},
                        {"name": "status", "label": "Status", "field": "status", "sortable": True},
                        {"name": "verdict", "label": "Verdict", "field": "verdict"},
                        {"name": "domains", "label": "Domains", "field": "domains", "align": "left"},
                        {"name": "docs", "label": "Docs", "field": "docs", "sortable": True},
                        {"name": "size", "label": "On disk", "field": "size"},
                        {"name": "tokens", "label": f"Tokens ({SPEC_LABELS[state['tok']]})", "field": "tokens", "sortable": True},
                    ],
                    rows=rows,
                    row_key="id",
                    pagination=15,
                ).classes("w-full")
                table.add_slot(
                    "body-cell-status",
                    """
                    <q-td :props="props">
                      <q-badge :color="props.value === 'ready' ? 'green' : props.value === 'failed' ? 'red' : 'primary'"
                               :outline="props.value === 'processing'">
                        {{ props.value }}
                      </q-badge>
                    </q-td>
                    """,
                )
                table.add_slot(
                    "body-cell-verdict",
                    """
                    <q-td :props="props">
                      <q-badge v-if="props.value"
                               :color="props.value === 'approved' ? 'teal' : 'red'" outline>
                        {{ props.value }}
                      </q-badge>
                    </q-td>
                    """,
                )
                table.on("row-click", lambda e: select_corpus(e.args[1]["id"]))
                table_holder["t"] = table

        def select_corpus(cid: str) -> None:
            state["selected"] = cid
            render_detail()

        def render_detail() -> None:
            detail_container.clear()
            cid = state["selected"]
            if not cid:
                return
            c = db.q1("SELECT * FROM corpora WHERE id=?", (cid,))
            if not c:
                state["selected"] = None
                return
            stats = _stats(c)
            with detail_container, panel():
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-3"):
                        ui.label(c["name"]).classes("text-lg font-semibold")
                        status_badge(c["status"])
                        if c.get("verdict"):
                            v = ui.badge(c["verdict"], color="teal" if c["verdict"] == "approved" else "red")
                            v.props("outline")
                    with ui.row().classes("items-center gap-2"):
                        ui.link("Review", f"/review?cid={cid}").props("no-caps dense outline").classes(
                            "no-underline text-gray-300"
                        )
                        ui.button(
                            "Delete",
                            icon="delete",
                            on_click=lambda cid=cid, name=c["name"]: ask_delete(cid, name),
                        ).props("flat dense no-caps color=negative")
                ui.label(f'source: {c["source_path"]}').classes("text-xs text-gray-500")
                if c["error"]:
                    ui.label(c["error"]).classes("text-xs text-red-400")

                with ui.row().classes("w-full gap-3 flex-wrap mt-2"):
                    stat_card("Documents", f'{stats.get("docs_written", 0):,}')
                    stat_card("Characters", f'{stats.get("chars_written", 0):,}')
                    stat_card("On disk", fmt_bytes(stats.get("bytes_on_disk", 0)))
                    stat_card("Exact dupes", f'{stats.get("exact_dupes_local", 0) + stats.get("exact_dupes_global", 0):,}')
                    stat_card("Near dupes", f'{stats.get("near_dupes", 0):,}')
                    stat_card("Filtered", f'{stats.get("filtered", 0):,}')
                    if stats.get("non_english_docs"):
                        stat_card("Non-English", f'{stats["non_english_docs"]:,}')
                    if stats.get("quarantined_docs"):
                        stat_card("Quarantined", f'{stats["quarantined_docs"]:,}')
                    if stats.get("rejected_known_bad"):
                        stat_card("Known-bad blocked", f'{stats["rejected_known_bad"]:,}')
                langs = stats.get("langs") or {}
                if langs:
                    top = sorted(langs.items(), key=lambda kv: -kv[1])[:4]
                    ui.label("Languages: " + ", ".join(f"{k} ({v:,})" for k, v in top)).classes(
                        "text-xs text-gray-500 mt-1"
                    )
                qreasons = stats.get("quarantine_reasons") or {}
                if qreasons:
                    top = sorted(qreasons.items(), key=lambda kv: -kv[1])[:5]
                    ui.label("Quarantine reasons: " + ", ".join(f"{k} ({v:,})" for k, v in top)).classes(
                        "text-xs text-amber-400 mt-1"
                    )

                dom_tokens = _domain_tokens(stats, state["tok"])
                totals = stats.get("totals", {}).get("tokens", {})
                if dom_tokens:
                    with ui.row().classes("w-full gap-4 mt-2 flex-wrap"):
                        ui.echart(
                            {
                                "backgroundColor": "transparent",
                                "title": {"text": f"Mixture by domain ({SPEC_LABELS[state['tok']]})", "textStyle": {"color": "#ccc", "fontSize": 12}},
                                "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                                "series": [
                                    {
                                        "type": "pie",
                                        "radius": ["35%", "65%"],
                                        "data": [{"name": k, "value": v} for k, v in sorted(dom_tokens.items(), key=lambda kv: -kv[1])],
                                        "label": {"color": "#bbb"},
                                    }
                                ],
                            }
                        ).classes("w-[420px] h-64")
                        if totals:
                            ui.echart(
                                {
                                    "backgroundColor": "transparent",
                                    "title": {"text": "Tokens by tokenizer", "textStyle": {"color": "#ccc", "fontSize": 12}},
                                    "tooltip": {"trigger": "axis"},
                                    "xAxis": {"type": "category", "data": [SPEC_LABELS.get(k, k) for k in totals], "axisLabel": {"color": "#999", "rotate": 30}},
                                    "yAxis": {"type": "value", "axisLabel": {"color": "#999"}},
                                    "series": [{"type": "bar", "data": [totals[k] for k in totals], "itemStyle": {"color": "#8b6bff"}}],
                                }
                            ).classes("w-[420px] h-64")

                failures = stats.get("tokenizer_failures", {})
                if failures:
                    ui.label("Tokenizers unavailable: " + ", ".join(f"{k} ({v[:80]})" for k, v in failures.items())).classes(
                        "text-xs text-amber-400 mt-2"
                    )

                sample = []
                shards = stats.get("shards", [])
                if shards:
                    for line in _iter_shards(corpus_dir(cid) / "data", shards[:1]):
                        sample.append(line)
                        if len(sample) >= 3:
                            break
                if sample:
                    ui.label("Preview").classes("text-sm font-semibold mt-3")
                    for i, line in enumerate(sample):
                        with ui.expansion(f'doc {i + 1} · {line["domain"]} · {len(line["text"])} chars').classes(
                            "w-full text-sm"
                        ):
                            ui.label(line["text"][:800]).classes("whitespace-pre-wrap text-gray-300 font-mono text-xs")

        def delete_corpus(cid: str) -> None:
            db.execute("DELETE FROM doc_hashes WHERE corpus_id=?", (cid,))
            db.execute("DELETE FROM corpora WHERE id=?", (cid,))
            shutil.rmtree(corpus_dir(cid), ignore_errors=True)
            if state["selected"] == cid:
                state["selected"] = None
            refresh_table()
            render_detail()
            ui.notify("Corpus deleted", type="info")

        def poll_job() -> None:
            if not state["job"]:
                return
            job = JOBS.get(state["job"])
            if job is None:
                state["job"] = None
                return
            p = job.progress
            bar.value = p.get("frac", 0.0)
            plabel.text = f'{p.get("stage", "")} · {p.get("message", "")}'
            if job.status == "done":
                state["job"] = None
                progress_row.visible = False
                ui.notify("Corpus processed", type="positive")
                refresh_table()
                render_library()
                if isinstance(job.result, str):
                    select_corpus(job.result)
            elif job.status == "error":
                state["job"] = None
                progress_row.visible = False
                ui.notify(f"Ingest failed: {job.error.splitlines()[0]}", type="negative", multi_line=True)
                refresh_table()
                render_library()

        tok_select.on_value_change(lambda e: (state.update(tok=e.value), refresh_table(), render_detail()))
        ui.timer(1.0, poll_job)

        del_state: dict = {"cid": None}
        with ui.dialog() as del_dlg, ui.card().classes("w-[420px] p-4").style(
            "background: #16161d; border: 1px solid #262633"
        ):
            ui.label("Delete corpus?").classes("text-base font-semibold")
            del_msg = ui.label("").classes("text-sm text-gray-400 mt-1")
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=del_dlg.close).props("flat dense no-caps")

                def do_delete() -> None:
                    del_dlg.close()
                    if del_state["cid"]:
                        delete_corpus(del_state["cid"])

                ui.button("Delete", on_click=do_delete).props("dense no-caps color=negative")

        def ask_delete(cid: str, name: str) -> None:
            del_state["cid"] = cid
            del_msg.text = f'"{name}" and its shards + dedup hashes will be removed permanently.'
            del_dlg.open()

        with ui.dialog() as ingest_dlg, ui.card().classes("w-[560px]").style("background: #16161d; border: 1px solid #262633"):
            ui.label("Ingest corpora folder").classes("text-base font-semibold")
            name_in = ui.input("Corpus name").props("dense outlined").classes("w-full")
            with ui.row().classes("w-full items-end gap-2"):
                src_in = ui.input("Source folder", placeholder="D:\\data\\corpora").props("dense outlined").classes("flex-1")
                browser = dir_browser("Select corpora folder", lambda p: src_in.set_value(p))
                ui.button(icon="folder", on_click=browser.open).props("dense outline")
            dom_sel = ui.select(
                ["auto"] + DOMAINS, value="auto", label="Domain override"
            ).props("dense outlined").classes("w-full")
            with ui.row().classes("w-full gap-6"):
                nd_chk = ui.checkbox("Near-duplicate removal (MinHash)", value=True)
                gd_chk = ui.checkbox("Global exact dedup", value=True)
            with ui.row().classes("w-full gap-6"):
                lang_chk = ui.checkbox("English-only (non-English kept in side bucket)", value=True)
                qual_chk = ui.checkbox("Quality quarantine (score < 0.35)", value=True)
            tok_multi = ui.select(
                SPEC_LABELS, multiple=True, value=list(SPEC_KEYS), label="Tokenizers for stats"
            ).props("dense outlined use-chips").classes("w-full")
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=ingest_dlg.close).props("flat dense no-caps")

                def begin() -> None:
                    src = (src_in.value or "").strip()
                    if not src:
                        ui.notify("Choose a source folder", type="warning")
                        return
                    if state["job"]:
                        ui.notify("An ingest job is already running", type="warning")
                        return
                    cfg = IngestConfig(
                        name=(name_in.value or "").strip() or src.rstrip("\\/").split("\\")[-1].split("/")[-1],
                        source_dir=src,
                        domain_override=None if dom_sel.value == "auto" else dom_sel.value,
                        near_dedup=nd_chk.value,
                        global_dedup=gd_chk.value,
                        language_filter=bool(lang_chk.value),
                        quality_filter=bool(qual_chk.value),
                        tokenizer_keys=list(tok_multi.value or []),
                    )
                    state["job"] = JOBS.submit("ingest", run_ingest, cfg)
                    progress_row.visible = True
                    ingest_dlg.close()

                ui.button("Process", on_click=begin).props("dense no-caps")

        ingest_btn.on_click(ingest_dlg.open)

        render_library()
        refresh_table()
