from __future__ import annotations

import os
import shutil

import orjson
from nicegui import ui

from ... import db
from ...artifacts import corpus_bucket_dir, corpus_dir, fmt_bytes, read_corpus_layout
from ...config import get_settings, update_settings
from ...ingest.cache import cache_size_bytes, clear_cache
from ...ingest.domains import DOMAINS
from ...ingest.library import ingested_by_source, scan_dataset_folders
from ...ingest.parallel import auto_worker_count, resolve_worker_count
from ...ingest.pipeline import IngestConfig, _iter_shards, ingest_in_progress, reimport_corpus, run_ingest
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


def _same_source(left: str, right: str) -> bool:
    try:
        return os.path.normcase(os.path.normpath(left)) == os.path.normcase(os.path.normpath(right))
    except (TypeError, ValueError):
        return False


def ingested_corpus_for_source(source: str) -> dict | None:
    """Return the ingested corpus that already covers ``source``, if any."""

    for row in db.q(
        "SELECT id, name, status, source_path, stats_json FROM corpora "
        "WHERE COALESCE(kind,'ingested')='ingested' AND source_path != ''"
    ):
        if _same_source(str(row["source_path"]), source):
            return row
    return None


def curated_library_ingest_config(*, path: str, name: str) -> IngestConfig:
    """Fast path for datasets that are already published as curated corpora.

    The library browser is intentionally preservation-oriented. Expensive
    heuristic cleaning remains available from the explicit ingest dialog.
    """

    return IngestConfig(
        name=name,
        source_dir=path,
        near_dedup=False,
        global_dedup=True,
        language_filter=False,
        quality_filter=False,
        tokenizer_keys=[],
        selection_profile="dataset",
        max_file_mb=0,
    )


@ui.page("/corpora")
def corpora_page():
    state: dict = {"job": None, "selected": None, "tok": "gpt2"}

    with shell("Corpora", "/corpora"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Corpora").classes("text-xl font-semibold")
            gccf_btn = ui.button("Import GCCF receipt", icon="fact_check").props("no-caps")
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

        def busy() -> bool:
            return bool(state["job"]) or ingest_in_progress()

        def start_library_ingest(path: str, name: str) -> None:
            if busy():
                ui.notify("An ingest job is already running", type="warning")
                return
            cfg = curated_library_ingest_config(path=path, name=name)
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
                hint(
                    "Fast ingest preserves dataset records, inventories provenance, and excludes obvious "
                    "dataset sidecars (README/license/config/CSS/SVG/XML). Use Re-import on an ingested "
                    "folder to change the asset policy or enable language, quality, and MinHash cleaning "
                    "without extracting unchanged files again."
                )
                for f in folders:
                    ing = known.get(f.path)
                    is_busy = busy()
                    with ui.row().classes("w-full items-center gap-3 py-1 border-b border-gray-800"):
                        ui.label(f.name).classes("text-sm text-gray-200 flex-1 truncate")
                        ui.label(f"{f.files} files · {fmt_bytes(f.size_bytes)}").classes(
                            "text-xs text-gray-500 whitespace-nowrap"
                        )
                        if ing:
                            status_badge(ing["status"])
                            ui.button(
                                "Re-import",
                                on_click=lambda cid=ing["id"]: open_reimport(cid),
                            ).props("dense no-caps outline").classes("w-28").set_enabled(
                                not is_busy and ing["status"] in {"ready", "failed"}
                            )
                        else:
                            ui.button("Fast ingest", on_click=lambda p=f.path, n=f.name: start_library_ingest(p, n)).props(
                                "dense no-caps outline"
                            ).classes("w-28").set_enabled(not is_busy)

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
                        lambda: open_ingest(),
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
                        if (c.get("kind") or "ingested") == "ingested" and c["source_path"]:
                            ui.button(
                                "Re-import",
                                icon="replay",
                                on_click=lambda cid=cid: open_reimport(cid),
                            ).props("dense no-caps outline").set_enabled(
                                c["status"] in {"ready", "failed"} and not busy()
                            )
                        ui.button(
                            "Delete",
                            icon="delete",
                            on_click=lambda cid=cid, name=c["name"]: ask_delete(cid, name),
                        ).props("flat dense no-caps color=negative")
                ui.label(f'source: {c["source_path"]}').classes("text-xs text-gray-500")
                gccf = stats.get("gccf") or {}
                if gccf:
                    coverage = "complete export" if gccf.get("complete_source_scan") else "partial sample export"
                    ui.label(
                        f"GCCF approved · {coverage} · source and receipt provenance retained"
                    ).classes("text-xs text-teal-300")
                    changed = int(stats.get("integrity_rejections", 0))
                    missing = int(stats.get("unavailable_documents", 0))
                    if changed or missing:
                        ui.label(
                            f"Export integrity: {changed:,} changed records excluded; "
                            f"{missing:,} records unavailable"
                        ).classes("text-xs text-amber-300")
                if c["error"]:
                    ui.label(c["error"]).classes("text-xs text-red-400")
                reimport_marker = stats.get("reimport") if isinstance(stats.get("reimport"), dict) else None
                if reimport_marker:
                    ui.label(
                        f"Re-import running since {reimport_marker.get('started_at', '?')} · the current "
                        "shards stay usable until the new generation is published."
                    ).classes("text-xs text-amber-400")
                if stats.get("pending_hash_commit"):
                    ui.label(
                        "The last re-import was published but its global-dedup hashes were not "
                        "committed; the next re-import or app start completes it."
                    ).classes("text-xs text-amber-400")
                timing = stats.get("timing") or {}
                layout = read_corpus_layout(corpus_dir(cid))
                cache = stats.get("prepared_cache") or {}
                cache_bytes = cache_size_bytes(corpus_dir(cid))
                details = []
                if timing.get("seconds"):
                    workers = int(timing.get("workers") or 1)
                    details.append(
                        f"last run {timing['seconds']:,.0f}s · {timing.get('docs_per_second', 0):,.0f} docs/s "
                        f"· {workers} worker{'s' if workers != 1 else ''}"
                    )
                if layout["generation"]:
                    details.append(f"generation {layout['generation']}")
                    if layout["retired"]:
                        details.append(f"{len(layout['retired'])} retired output(s) pending cleanup")
                if cache.get("enabled"):
                    reused = int(cache.get("assets_reused") or 0)
                    stored = int(cache.get("assets_stored") or 0)
                    details.append(
                        f"prepared cache {fmt_bytes(cache_bytes)} · {int(cache.get('assets') or 0):,} assets"
                        + (f" · reused {reused:,}, extracted {stored:,}" if reused or stored else "")
                    )
                elif cache_bytes:
                    details.append(f"prepared cache {fmt_bytes(cache_bytes)} (disabled for the last run)")
                if details:
                    with ui.row().classes("w-full items-center gap-3"):
                        ui.label(" · ".join(details)).classes("text-xs text-gray-500")
                        if cache_bytes:

                            def do_clear_cache(cid=cid) -> None:
                                if busy():
                                    ui.notify("Wait for the running ingest to finish", type="warning")
                                    return
                                clear_cache(corpus_dir(cid))
                                ui.notify("Prepared-document cache cleared", type="info")
                                render_detail()

                            ui.button("Clear cache", icon="cleaning_services", on_click=do_clear_cache).props(
                                "flat dense no-caps"
                            )

                with ui.row().classes("w-full gap-3 flex-wrap mt-2"):
                    stat_card("Documents", f'{stats.get("docs_written", 0):,}')
                    stat_card("Characters", f'{stats.get("chars_written", 0):,}')
                    stat_card("On disk", fmt_bytes(stats.get("bytes_on_disk", 0)))
                    stat_card("Exact dupes", f'{stats.get("exact_dupes_local", 0) + stats.get("exact_dupes_global", 0):,}')
                    stat_card("Near dupes", f'{stats.get("near_dupes", 0):,}')
                    if stats.get("near_duplicate_candidates"):
                        stat_card(
                            "Near-match evidence",
                            f'{stats["near_duplicate_candidates"]:,}',
                        )
                    stat_card("Filtered", f'{stats.get("filtered", 0):,}')
                    if stats.get("non_english_docs"):
                        stat_card("Non-English", f'{stats["non_english_docs"]:,}')
                    if stats.get("quarantined_docs"):
                        stat_card("Quarantined", f'{stats["quarantined_docs"]:,}')
                    if stats.get("rejected_known_bad"):
                        stat_card("Known-bad blocked", f'{stats["rejected_known_bad"]:,}')
                    if stats.get("assets_policy_skipped"):
                        stat_card("Assets policy-skipped", f'{stats["assets_policy_skipped"]:,}')
                    if stats.get("docs_policy_skipped"):
                        stat_card("Records policy-skipped", f'{stats["docs_policy_skipped"]:,}')
                    if stats.get("split_leakage_dupes"):
                        stat_card("Cross-split dupes", f'{stats["split_leakage_dupes"]:,}')
                    if stats.get("duplicate_sample_ids"):
                        stat_card("Duplicate sample IDs", f'{stats["duplicate_sample_ids"]:,}')
                    if stats.get("records_failed"):
                        stat_card("Malformed records", f'{stats["records_failed"]:,}')
                cdas = stats.get("cdas") or {}
                if cdas:
                    compatibility_changes = int(cdas.get("compatibility_fold_would_change_docs") or 0)
                    ui.label(
                        f"CDAS inventory: {int(cdas.get('assets') or 0):,} assets · "
                        f"{int(cdas.get('documents') or 0):,} document decisions · "
                        f"{int(cdas.get('issues') or 0):,} issues · semantic text preserved"
                    ).classes("text-xs text-teal-300 mt-1")
                    if compatibility_changes:
                        ui.label(
                            f"NFKC would have changed {compatibility_changes:,} documents; yyLab kept NFC text."
                        ).classes("text-xs text-amber-400")
                    kept_roles = cdas.get("kept_role_counts") or {}
                    if kept_roles:
                        top_roles = sorted(kept_roles.items(), key=lambda item: -item[1])[:6]
                        ui.label(
                            "Kept roles: " + ", ".join(f"{name} ({count:,})" for name, count in top_roles)
                        ).classes("text-xs text-gray-500")
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
                    for line in _iter_shards(corpus_bucket_dir(cid, "data"), shards[:1]):
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
            if busy():
                ui.notify("Wait for the running ingest to finish before deleting", type="warning")
                return
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
                message = "Corpus re-imported" if job.kind == "reimport" else "Corpus processed"
                if job.kind == "gccf-ingest":
                    message = f"Imported {len(job.result['corpora'])} classified corpora"
                ui.notify(message, type="positive")
                refresh_table()
                render_library()
                if isinstance(job.result, str):
                    select_corpus(job.result)
            elif job.status == "error":
                state["job"] = None
                progress_row.visible = False
                verb = "Re-import" if job.kind == "reimport" else "Ingest"
                ui.notify(f"{verb} failed: {job.error.splitlines()[0]}", type="negative", multi_line=True)
                refresh_table()
                render_library()
                if job.kind == "reimport":
                    render_detail()

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

        dialog_state: dict = {"mode": "ingest", "cid": None}
        with ui.dialog() as ingest_dlg, ui.card().classes("w-[560px]").style("background: #16161d; border: 1px solid #262633"):
            dlg_title = ui.label("Ingest corpora folder").classes("text-base font-semibold")
            reimport_hint = ui.label("").classes("text-xs text-teal-300")
            reimport_hint.visible = False
            name_in = ui.input("Corpus name").props("dense outlined").classes("w-full")
            with ui.row().classes("w-full items-end gap-2"):
                src_in = ui.input("Source folder", placeholder="D:\\data\\corpora").props("dense outlined").classes("flex-1")
                browser = dir_browser("Select corpora folder", lambda p: src_in.set_value(p))
                src_btn = ui.button(icon="folder", on_click=browser.open).props("dense outline")
            dom_sel = ui.select(
                ["auto"] + DOMAINS, value="auto", label="Domain override"
            ).props("dense outlined").classes("w-full")
            with ui.row().classes("w-full gap-6"):
                nd_chk = ui.checkbox("Near-duplicate removal (MinHash)", value=True)
                gd_chk = ui.checkbox("Global exact dedup", value=True)
            hint(
                "MinHash is a full-corpus CPU and memory pass. Leave it off for published datasets "
                "that already document near-duplicate removal."
            )
            with ui.row().classes("w-full gap-6"):
                lang_chk = ui.checkbox("English-only (non-English kept in side bucket)", value=True)
                qual_chk = ui.checkbox("Quality quarantine (score < 0.35)", value=True)
            profile_select = ui.select(
                {
                    "dataset": "Dataset (exclude metadata + web/config sidecars)",
                    "repository": "Repository (keep docs/config/web source)",
                    "strict": "Strict dataset (also reject repetition/boilerplate)",
                    "preserve": "Preserve all supported assets",
                },
                value="dataset",
                label="CDAS asset selection profile",
            ).props("dense outlined").classes("w-full")
            hint(
                "The asset profile runs before extraction, so large SVG/CSS/XML trees never reach "
                "tokenization. Every observed asset and decision is retained in inventory.sqlite3."
            )
            with ui.expansion("Advanced path policy", icon="filter_alt").classes("w-full"):
                include_path_in = ui.input(
                    "Include path regex (optional)", placeholder=r"(?:train|papers)/"
                ).props("dense outlined").classes("w-full")
                exclude_path_in = ui.input(
                    "Exclude path regex (optional)", placeholder=r"(?:generated|assets)/"
                ).props("dense outlined").classes("w-full")
                max_file_mb_in = ui.number(
                    "Maximum asset MiB (0 = unlimited)",
                    value=300,
                    min=0,
                    step=64,
                ).props("dense outlined").classes("w-full")
                hint(
                    "Oversized assets remain visible in the CDAS inventory with an explicit "
                    "excluded-by-limit decision. Curated-library fast ingest is unlimited."
                )
            tok_multi = ui.select(
                SPEC_LABELS, multiple=True, value=list(SPEC_KEYS), label="Tokenizers for stats"
            ).props("dense outlined use-chips").classes("w-full")
            with ui.row().classes("w-full items-center gap-4"):
                workers_in = ui.number(
                    "Parallel workers (0 = auto)",
                    value=0,
                    min=0,
                    max=64,
                    step=1,
                ).props("dense outlined").classes("w-56")
                cache_chk = ui.checkbox("Keep prepared-document cache for quick re-import", value=True)
            hint(
                f"Auto uses {resolve_worker_count(None, get_settings().get('ingest_workers'))} worker "
                f"processes (machine default {auto_worker_count()}; change it under Settings). Large JSONL "
                "and Parquet files are split across workers; results are merged in file order, so output "
                "is identical to a single-worker run. The cache stores every extracted document so a "
                "later re-import with different policies skips unchanged files."
            )
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=ingest_dlg.close).props("flat dense no-caps")

                def dialog_config(name: str, src: str) -> IngestConfig:
                    workers = int(workers_in.value or 0)
                    return IngestConfig(
                        name=name,
                        source_dir=src,
                        domain_override=None if dom_sel.value == "auto" else dom_sel.value,
                        near_dedup=nd_chk.value,
                        global_dedup=gd_chk.value,
                        language_filter=bool(lang_chk.value),
                        quality_filter=bool(qual_chk.value),
                        tokenizer_keys=list(tok_multi.value or []),
                        selection_profile=str(profile_select.value or "dataset"),
                        include_path_regex=(include_path_in.value or "").strip(),
                        exclude_path_regex=(exclude_path_in.value or "").strip(),
                        max_file_mb=float(max_file_mb_in.value or 0),
                        workers=workers or None,
                        prepared_cache=bool(cache_chk.value),
                    )

                def begin() -> None:
                    src = (src_in.value or "").strip()
                    if not src:
                        ui.notify("Choose a source folder", type="warning")
                        return
                    if busy():
                        ui.notify("An ingest job is already running", type="warning")
                        return
                    if dialog_state["mode"] == "reimport":
                        cid = dialog_state["cid"]
                        cfg = dialog_config(name=(name_in.value or "").strip() or "corpus", src=src)
                        state["job"] = JOBS.submit("reimport", reimport_corpus, cid, cfg)
                        progress_row.visible = True
                        plabel.text = "queued · re-import"
                        ingest_dlg.close()
                        render_detail()
                        return
                    existing = ingested_corpus_for_source(src)
                    if existing is not None and existing["status"] not in {"ready", "failed"}:
                        ui.notify(
                            f'"{existing["name"]}" already covers this folder but is still marked '
                            f'{existing["status"]}; delete it or restart the app to clear the stale state.',
                            type="warning",
                            multi_line=True,
                        )
                        return
                    if existing is not None:
                        ui.notify(
                            f'"{existing["name"]}" already covers this folder; a second corpus would only '
                            "contain global duplicates. Re-importing it instead.",
                            type="warning",
                            multi_line=True,
                        )
                        open_reimport(existing["id"])
                        return
                    cfg = dialog_config(
                        name=(name_in.value or "").strip() or src.rstrip("\\/").split("\\")[-1].split("/")[-1],
                        src=src,
                    )
                    state["job"] = JOBS.submit("ingest", run_ingest, cfg)
                    progress_row.visible = True
                    ingest_dlg.close()

                process_btn = ui.button("Process", on_click=begin).props("dense no-caps")

        def set_dialog_mode(mode: str) -> None:
            dialog_state["mode"] = mode
            reimporting = mode == "reimport"
            dlg_title.text = "Re-import corpus" if reimporting else "Ingest corpora folder"
            process_btn.text = "Re-import" if reimporting else "Process"
            reimport_hint.visible = reimporting
            name_in.set_enabled(not reimporting)
            src_in.set_enabled(not reimporting)
            src_btn.set_enabled(not reimporting)

        def open_ingest() -> None:
            set_dialog_mode("ingest")
            dialog_state["cid"] = None
            ingest_dlg.open()

        def open_reimport(cid: str) -> None:
            c = db.q1("SELECT * FROM corpora WHERE id=?", (cid,))
            if not c:
                ui.notify("Corpus not found", type="warning")
                return
            if busy():
                ui.notify("An ingest job is already running", type="warning")
                return
            stats = _stats(c)
            stored = stats.get("ingest_config") or {}
            cfg = IngestConfig.from_ingest_config(
                name=c["name"], source_dir=c["source_path"], ingest_config=stored
            )
            dialog_state["cid"] = cid
            set_dialog_mode("reimport")
            name_in.value = cfg.name
            src_in.value = cfg.source_dir
            dom_sel.value = cfg.domain_override or "auto"
            nd_chk.value = bool(cfg.near_dedup)
            gd_chk.value = bool(cfg.global_dedup)
            lang_chk.value = bool(cfg.language_filter)
            qual_chk.value = bool(cfg.quality_filter)
            profile_select.value = cfg.selection_profile
            include_path_in.value = cfg.include_path_regex
            exclude_path_in.value = cfg.exclude_path_regex
            max_file_mb_in.value = cfg.max_file_mb
            tok_multi.value = list(cfg.tokenizer_keys)
            cache_chk.value = bool(cfg.prepared_cache)
            workers_in.value = 0
            cache = stats.get("prepared_cache") or {}
            cache_bytes = cache_size_bytes(corpus_dir(cid))
            if cache.get("enabled") and cache_bytes:
                cache_text = (
                    f"{int(cache.get('assets') or 0):,} extracted files are cached ({fmt_bytes(cache_bytes)}); "
                    "unchanged files are not extracted again."
                )
            else:
                cache_text = "No prepared-document cache exists yet; every file is extracted once more."
            warning = ""
            if "domain_override" not in stored:
                warning = (
                    " The domain override was not recorded when this corpus was created; "
                    f"domains seen: {', '.join(sorted(stats.get('domains') or {})) or 'none'}. Verify the override."
                )
            reimport_hint.text = (
                "Settings are prefilled from the original ingest. The corpus keeps its id, reviews, and "
                "verdict; the current shards stay usable until the new output is published. " + cache_text + warning
            )
            ingest_dlg.open()

        ingest_btn.on_click(open_ingest)

        with ui.dialog() as gccf_dialog, ui.card().classes("w-[640px]"):
            ui.label("Import classified corpora").classes("text-lg font-semibold")
            hint(
                "Reads the files approved by GCCF, removes listed bad records, and preserves "
                "held-out splits. Each source becomes a separate corpus for training mixtures."
            )
            gccf_path = ui.input("GCCF receipt JSON path").props("outlined dense").classes("w-full")

            def begin_gccf() -> None:
                if busy():
                    ui.notify("An ingest job is already running", type="warning")
                    return
                path = str(gccf_path.value or "").strip().strip('"')
                if not path:
                    ui.notify("Enter a GCCF receipt path", type="warning")
                    return
                from ...ingest.gccf import run_gccf_ingest

                state["job"] = JOBS.submit("gccf-ingest", run_gccf_ingest, path)
                progress_row.visible = True
                plabel.text = "queued · GCCF import"
                gccf_dialog.close()

            with ui.row():
                ui.button("Import", on_click=begin_gccf).props("no-caps")
                ui.button("Cancel", on_click=gccf_dialog.close).props("flat no-caps")
        gccf_btn.on_click(gccf_dialog.open)

        render_library()
        refresh_table()
