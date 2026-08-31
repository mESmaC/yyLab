from __future__ import annotations

from nicegui import ui

from ... import db
from ...config import PATHS
from ...ingest.ranker import (
    Stage1Ranker,
    grouped_cross_validation,
    load_ranker_if_available,
    train_ranker,
)
from ...review import (
    finish_session,
    load_review_labels,
    record_verdict,
    sample_review_docs,
    start_session,
)
from ..components import empty_state, status_badge
from ..shell import panel, shell

MAX_PREVIEW = 4_000
RANKER_NAME = "corpus-stage1"


def _ranker_dir():
    return PATHS.rankers / RANKER_NAME


def _metric(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _verdict_counts(verdicts: dict[str, dict]) -> tuple[int, int]:
    good = sum(1 for value in verdicts.values() if value.get("verdict") == "good")
    bad = sum(1 for value in verdicts.values() if value.get("verdict") == "bad")
    return good, bad


@ui.page("/review")
def review_page(cid: str | None = None):
    ranker_error = ""
    try:
        ranker = load_ranker_if_available(_ranker_dir())
    except Exception as exc:  # a corrupt optional artifact must not strand manual review
        ranker = None
        ranker_error = f"{type(exc).__name__}: {exc}"
    state: dict = {
        "docs": [],
        "idx": 0,
        "verdicts": {},
        "cid": cid,
        "review_id": None,
        "ranker": ranker,
        "ranker_error": ranker_error,
    }

    with shell("Review", "/review"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Swipe review").classes("text-xl font-semibold")
            ui.label("← bad · → good · esc finish").classes("text-xs text-gray-500")

        setup_holder = ui.column().classes("w-full")
        swipe_holder = ui.column().classes("w-full gap-4")
        summary_holder = ui.column().classes("w-full")

        def ready_corpora() -> dict[str, str]:
            rows = db.q(
                "SELECT id, name FROM corpora WHERE status='ready' AND COALESCE(kind,'ingested')='ingested'"
                " ORDER BY created_at DESC"
            )
            return {r["id"]: r["name"] for r in rows}

        def render_setup() -> None:
            setup_holder.clear()
            corpora = ready_corpora()
            with setup_holder, panel():
                if not corpora:
                    empty_state("rate_review", "No corpora to review", "Ingest a folder first, then review its documents here.")
                    return
                ui.label("Review session").classes("text-base font-semibold")
                ui.label(
                    "Swipe through sampled documents. Left (bad) sends docs to the rejected store and blocks them "
                    "from future ingests; right (good) assembles an approved corpus for training."
                ).classes("text-xs text-gray-500")
                with ui.row().classes("w-full items-end gap-3"):
                    corpus_sel = ui.select(corpora, value=state["cid"] if state["cid"] in corpora else None, label="Corpus").props(
                        "dense outlined"
                    ).classes("flex-1")
                    n_in = ui.number("Sample size", value=50, min=5, max=500).props("dense outlined").classes("w-36")
                    q_chk = ui.checkbox("Include quarantined docs", value=False)
                    reviewed_chk = ui.checkbox("Include reviewed docs", value=False)
                    strategy_sel = (
                        ui.select(
                            {"auto": "Auto", "uniform": "Uniform", "active": "Active blend"},
                            value="auto",
                            label="Queue",
                        )
                        .props("dense outlined")
                        .classes("w-36")
                    )
                    start_btn = ui.button("Start session", icon="play_arrow").props("no-caps")

                ui.separator().classes("my-2")
                annotation_stats = db.q1(
                    "SELECT COUNT(*) AS total, "
                    "SUM(CASE WHEN verdict='good' THEN 1 ELSE 0 END) AS good, "
                    "SUM(CASE WHEN verdict='bad' THEN 1 ELSE 0 END) AS bad FROM review_annotations"
                ) or {"total": 0, "good": 0, "bad": 0}
                trained: Stage1Ranker | None = state.get("ranker")
                with ui.row().classes("w-full items-center justify-between gap-3"):
                    with ui.column().classes("gap-1"):
                        ui.label("Learned review ranker").classes("text-sm font-semibold")
                        ui.label(
                            f'{int(annotation_stats["total"] or 0)} labels · '
                            f'{int(annotation_stats["good"] or 0)} good · '
                            f'{int(annotation_stats["bad"] or 0)} bad'
                        ).classes("text-xs text-gray-400")
                        if trained is not None:
                            validation = trained.metrics.get("cross_validation", {}).get("overall") or trained.metrics.get(
                                "holdout", {}
                            ).get("validation", {})
                            trained_count = int(trained.training_config.get("training_examples", 0) or 0)
                            new_labels = max(0, int(annotation_stats["total"] or 0) - trained_count)
                            ui.label(
                                f"{trained.model_version} · corpus-CV AUC {_metric(validation.get('roc_auc'))} · "
                                f"balanced {_metric(validation.get('balanced_accuracy'))} · "
                                f"bad PR-AUC {_metric(validation.get('bad_pr_auc'))}"
                            ).classes("text-xs text-gray-400")
                            if new_labels:
                                ui.label(f"{new_labels} new labels available for retraining").classes(
                                    "text-xs text-amber-400"
                                )
                        elif state.get("ranker_error"):
                            ui.label(f'Ranker unavailable: {state["ranker_error"]}').classes(
                                "text-xs text-red-400"
                            )
                        else:
                            ui.label("No trained ranker yet; Auto uses uniform sampling.").classes(
                                "text-xs text-gray-500"
                            )
                        ui.label(
                            "Corpus-held-out validation · queue priority only · never gates ingest"
                        ).classes("text-[11px] text-gray-500")

                    train_btn = ui.button(
                        "Retrain ranker" if trained is not None else "Train ranker",
                        icon="model_training",
                    ).props("no-caps")

                def train_classifier() -> None:
                    try:
                        labels = load_review_labels()
                        fitted = train_ranker(labels, seed=0)
                        fitted.metrics["cross_validation"] = grouped_cross_validation(labels, l2=0.1)
                        fitted.save(_ranker_dir())
                    except Exception as exc:
                        state["ranker_error"] = f"{type(exc).__name__}: {exc}"
                        ui.notify(f'Ranker training failed: {state["ranker_error"]}', type="negative")
                        render_setup()
                        return
                    state["ranker"] = fitted
                    state["ranker_error"] = ""
                    validation = fitted.metrics["cross_validation"]["overall"]
                    ui.notify(
                        f"Ranker trained · corpus-CV AUC {_metric(validation.get('roc_auc'))}",
                        type="positive",
                    )
                    render_setup()

                train_btn.on_click(train_classifier)

                def start() -> None:
                    if not corpus_sel.value:
                        ui.notify("Pick a corpus", type="warning")
                        return
                    docs = sample_review_docs(
                        corpus_sel.value,
                        n=int(n_in.value or 50),
                        include_quarantine=bool(q_chk.value),
                        strategy=strategy_sel.value or "auto",
                        ranker=state.get("ranker"),
                        skip_reviewed=not bool(reviewed_chk.value),
                    )
                    if not docs:
                        ui.notify("No unreviewed documents match this queue", type="warning")
                        return
                    state["cid"] = corpus_sel.value
                    state["docs"] = docs
                    state["idx"] = 0
                    state["verdicts"] = {}
                    state["review_id"] = start_session(
                        corpus_sel.value,
                        sampling_strategy=strategy_sel.value or "auto",
                        model_version=str(docs[0].get("model_version") or ""),
                    )
                    if docs[0].get("sampling_fallback_reason"):
                        ui.notify("No trained ranker found; using a uniform queue", type="info")
                    render_swipe()

                start_btn.on_click(start)

        def verdict_counts() -> tuple[int, int]:
            return _verdict_counts(state["verdicts"])

        def render_swipe() -> None:
            swipe_holder.clear()
            summary_holder.clear()
            docs = state["docs"]
            idx = state["idx"]
            if not docs or idx >= len(docs):
                render_finish_prompt()
                return
            doc = docs[idx]
            good, bad = verdict_counts()
            with swipe_holder, panel():
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-2"):
                        ui.badge(doc.get("domain", "other"), color="grey-8")
                        if doc.get("lang"):
                            ui.badge(doc["lang"], color="grey-8")
                        if doc.get("quality") is not None:
                            q = doc["quality"]
                            ui.badge(f"q={q:.2f}", color="green" if q >= 0.7 else "amber" if q >= 0.35 else "red")
                        if doc.get("bucket") == "quarantine":
                            ui.badge("quarantined", color="orange")
                        if doc.get("sampling_component"):
                            ui.badge(doc["sampling_component"].replace("_", " "), color="blue-grey")
                        if doc.get("rank_score") is not None:
                            ui.badge(f'rank={doc["rank_score"]:.2f}', color="blue-grey")
                    ui.label(f'{idx + 1} / {len(docs)}   ·   good {good} · bad {bad}').classes(
                        "text-xs text-gray-400"
                    )
                ui.label(doc.get("source", "")).classes("text-[11px] text-gray-600 truncate w-full")
                ui.label(doc["text"][:MAX_PREVIEW]).classes(
                    "whitespace-pre-wrap text-sm text-gray-200 font-mono mt-2"
                ).style("max-height: 46vh; overflow-y: auto")
                with ui.row().classes("w-full justify-center gap-6 mt-3"):
                    ui.button("Bad ←", icon="thumb_down", on_click=lambda: vote("bad")).props(
                        "no-caps color=negative outline"
                    )
                    ui.button("Skip", icon="redo", on_click=skip).props("flat no-caps dense")
                    ui.button("→ Good", icon="thumb_up", on_click=lambda: vote("good")).props(
                        "no-caps color=positive outline"
                    )
                with ui.row().classes("w-full justify-center"):
                    ui.button("Finish session", icon="flag", on_click=finish).props("flat dense no-caps")

        def render_finish_prompt() -> None:
            swipe_holder.clear()
            good, bad = verdict_counts()
            with swipe_holder, panel():
                ui.label("All documents reviewed").classes("text-base font-semibold")
                ui.label(f"good: {good} · bad: {bad}").classes("text-sm text-gray-400")
                ui.button("Finish session", icon="flag", on_click=finish).props("no-caps")

        def advance() -> None:
            state["idx"] += 1
            render_swipe()

        def vote(verdict: str) -> None:
            docs = state["docs"]
            idx = state["idx"]
            if not docs or idx >= len(docs):
                return
            document = docs[idx]
            try:
                record_verdict(state["review_id"], state["cid"], document, verdict)
            except Exception as exc:
                ui.notify(f"Could not save verdict: {exc}", type="negative")
                return
            state["verdicts"][document["hash"]] = {**document, "verdict": verdict}
            advance()

        def skip() -> None:
            advance()

        def finish() -> None:
            if not state["verdicts"]:
                ui.notify("No verdicts recorded", type="warning")
                return
            result = finish_session(state["cid"], review_id=state["review_id"])
            state["docs"] = []
            state["review_id"] = None
            render_summary(result)

        def render_summary(r: dict) -> None:
            swipe_holder.clear()
            summary_holder.clear()
            with summary_holder, panel():
                ui.label("Session saved").classes("text-base font-semibold")
                with ui.row().classes("gap-3 flex-wrap mt-2"):
                    ui.badge(f'reviewed {r["docs_reviewed"]}', color="grey-8")
                    ui.badge(f'good {r["docs_good"]}', color="green")
                    ui.badge(f'bad {r["docs_bad"]}', color="red")
                    if r.get("dupes"):
                        ui.badge(f'{r["dupes"]} already annotated', color="grey-7")
                    if r["verdict"]:
                        status_badge(r["verdict"])
                if r["approved_corpus_id"]:
                    ui.label(f'added to shared approved corpus: {r["approved_corpus_id"]}').classes(
                        "text-xs text-gray-400 mt-2"
                    )
                if r["rejected_corpus_id"]:
                    ui.label(f'added to hidden rejected corpus: {r["rejected_corpus_id"]}').classes(
                        "text-xs text-gray-400"
                    )
                if r.get("annotation_path"):
                    ui.label(f'annotations: {r["annotation_path"]}').classes("text-xs text-gray-500")
                ui.label(
                    "Approved docs accumulate in one shared corpus for training. Rejected docs go to a hidden "
                    "corpus and block future ingests by hash. Each human swipe is saved immediately with its "
                    "queue-selection metadata; model scores only prioritize review and never gate ingest."
                ).classes("text-xs text-gray-500 mt-2")
                ui.button("New session", icon="replay", on_click=lambda: (summary_holder.clear(), render_setup())).props(
                    "no-caps dense mt-2"
                )

        def on_key(e) -> None:
            if not state["docs"] or state["idx"] >= len(state["docs"]):
                return
            if e.action.keydown:
                if e.key == "ArrowLeft":
                    vote("bad")
                elif e.key == "ArrowRight":
                    vote("good")
                elif e.key == "Escape":
                    finish()

        keyboard = ui.keyboard(on_key=on_key)
        keyboard.active = True

        if cid:
            docs_ready = db.q1("SELECT id FROM corpora WHERE id=? AND status='ready'", (cid,))
            if docs_ready:
                state["cid"] = cid
        render_setup()
