from __future__ import annotations

import shutil

import orjson
from nicegui import ui

from ... import db
from ...artifacts import corpus_dir, fmt_bytes
from ...ingest.domains import DOMAINS
from ...ingest.pipeline import IngestConfig, _iter_shards, run_ingest
from ...ingest.tokenize_stats import SPEC_KEYS, SPEC_LABELS
from ...jobs import JOBS
from ..components import dir_browser
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

        table_container = ui.column().classes("w-full")
        detail_container = ui.column().classes("w-full gap-4")

        def corpus_rows() -> list[dict]:
            rows = []
            for c in db.q("SELECT * FROM corpora ORDER BY created_at DESC"):
                stats = _stats(c)
                dom_tokens = sorted(_domain_tokens(stats, state["tok"]).items(), key=lambda kv: -kv[1])
                dom_str = ", ".join(f"{k}" for k, _ in dom_tokens[:3]) or "-"
                rows.append(
                    {
                        "id": c["id"],
                        "name": c["name"],
                        "status": c["status"],
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
            with table_container:
                table = ui.table(
                    columns=[
                        {"name": "name", "label": "Name", "field": "name", "sortable": True, "align": "left"},
                        {"name": "status", "label": "Status", "field": "status", "sortable": True},
                        {"name": "domains", "label": "Domains", "field": "domains", "align": "left"},
                        {"name": "docs", "label": "Docs", "field": "docs", "sortable": True},
                        {"name": "size", "label": "On disk", "field": "size"},
                        {"name": "tokens", "label": f"Tokens ({SPEC_LABELS[state['tok']]})", "field": "tokens", "sortable": True},
                    ],
                    rows=corpus_rows(),
                    row_key="id",
                    pagination=15,
                ).classes("w-full")
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
                        ui.badge(c["status"]).props(
                            "color=green" if c["status"] == "ready" else "color=orange"
                        )
                    ui.button("Delete", icon="delete", on_click=lambda: delete_corpus(cid)).props(
                        "flat dense no-caps color=negative"
                    )
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
                if isinstance(job.result, str):
                    select_corpus(job.result)
            elif job.status == "error":
                state["job"] = None
                progress_row.visible = False
                ui.notify(f"Ingest failed: {job.error.splitlines()[0]}", type="negative", multi_line=True)
                refresh_table()

        tok_select.on_value_change(lambda e: (state.update(tok=e.value), refresh_table(), render_detail()))
        ui.timer(1.0, poll_job)

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
                        tokenizer_keys=list(tok_multi.value or []),
                    )
                    state["job"] = JOBS.submit("ingest", run_ingest, cfg)
                    progress_row.visible = True
                    ingest_dlg.close()

                ui.button("Process", on_click=begin).props("dense no-caps")

        ingest_btn.on_click(ingest_dlg.open)

        refresh_table()
