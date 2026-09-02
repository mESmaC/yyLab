from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path

import orjson
import pytest

import yylab.review as review_module
from yylab import db
from yylab.artifacts import corpus_dir, read_json
from yylab.config import PATHS
from yylab.ingest.pipeline import IngestConfig, _iter_shards, run_ingest
from yylab.review import finish_session, load_review_labels, record_verdict, sample_review_docs, start_session
from yylab.ui.pages import review as review_ui
from yylab.ui.pages.review import _verdict_counts


@pytest.fixture(autouse=True)
def _clean_review_state():
    db.execute("DELETE FROM doc_hashes")
    db.execute("DELETE FROM rejected_hashes")
    db.execute("DELETE FROM review_swipes")
    db.execute("DELETE FROM review_annotations")
    db.execute("DELETE FROM reviews")
    for r in db.q("SELECT id FROM corpora WHERE kind IN ('approved','rejected')"):
        shutil.rmtree(corpus_dir(r["id"]), ignore_errors=True)
        db.execute("DELETE FROM corpora WHERE id=?", (r["id"],))
    for f in PATHS.annotations.glob("ann_*.jsonl.zst"):
        f.unlink()


GOOD_A = (
    "The operations team validated the nightly backup procedure and confirmed that every shard "
    "was written with checksums before the retention window closed on schedule."
)
GOOD_B = (
    "Field engineers replaced the worn bearing assembly on line three and documented the "
    "measured vibration levels before and after the repair for the maintenance archive."
)
BAD_DOC = (
    "casino bonus free spins casino bonus free spins casino bonus free spins casino bonus "
    "free spins casino bonus free spins casino bonus free spins casino bonus free spins now"
)
QUARANTINED_DOC = (
    "buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now "
    "buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now "
    "buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now"
)


def _make_corpus(tmp_path: Path, name: str, extra: dict[str, str] | None = None) -> str:
    src = tmp_path / name
    src.mkdir()
    (src / "a.txt").write_text(GOOD_A, "utf-8")
    (src / "b.txt").write_text(GOOD_B, "utf-8")
    (src / "c.txt").write_text(BAD_DOC, "utf-8")
    for fname, text in (extra or {}).items():
        (src / fname).write_text(text, "utf-8")
    cfg = IngestConfig(
        name=name,
        source_dir=str(src),
        tokenizer_keys=[],
        near_dedup=False,
        global_dedup=False,
        language_filter=False,
        quality_filter=False,
    )
    return run_ingest(lambda *a: None, cfg)


def _swipe_all(cid: str) -> dict:
    docs = sample_review_docs(cid, n=50)
    verdicts = [{**d, "verdict": "bad" if "casino" in d["text"] else "good"} for d in docs]
    return finish_session(cid, verdicts)


def test_single_accumulating_approved_corpus(tmp_path):
    cid1 = _make_corpus(tmp_path, "src1")
    r1 = _swipe_all(cid1)
    assert r1["docs_good"] == 2 and r1["docs_bad"] == 1
    approved_id = r1["approved_corpus_id"]
    assert approved_id

    approved = db.q1("SELECT * FROM corpora WHERE id=?", (approved_id,))
    assert approved["kind"] == "approved"
    assert approved["status"] == "ready"
    stats = read_json(corpus_dir(approved_id) / "stats.json")
    assert stats["docs_written"] == 2
    assert stats["review_store"] == "approved"
    docs = list(_iter_shards(corpus_dir(approved_id) / "data", stats["shards"]))
    assert {d["text"] for d in docs} == {GOOD_A, GOOD_B}

    cid2 = _make_corpus(
        tmp_path,
        "src2",
        extra={
            "d.txt": "The logistics coordinator rerouted the delayed shipment and updated every "
            "stakeholder before the end of the afternoon window."
        },
    )
    r2 = _swipe_all(cid2)
    assert r2["approved_corpus_id"] == approved_id, "second session must reuse the same approved corpus"
    assert r2["docs_good"] == 1, "reviewed docs must be skipped and not re-added"
    assert r2["dupes"] == 0
    src2_stats = read_json(corpus_dir(cid2) / "stats.json")
    assert src2_stats["rejected_known_bad"] == 1, "BAD_DOC must be blocked at ingest after rejection"
    stats2 = read_json(corpus_dir(approved_id) / "stats.json")
    assert stats2["docs_written"] == 3
    assert len(stats2["shards"]) >= 2


def test_hidden_rejected_corpus(tmp_path):
    cid = _make_corpus(tmp_path, "srcbad")
    r = _swipe_all(cid)
    rejected_id = r["rejected_corpus_id"]
    assert rejected_id

    rejected = db.q1("SELECT * FROM corpora WHERE id=?", (rejected_id,))
    assert rejected["kind"] == "rejected"
    stats = read_json(corpus_dir(rejected_id) / "stats.json")
    assert stats["docs_written"] == 1
    docs = list(_iter_shards(corpus_dir(rejected_id) / "data", stats["shards"]))
    assert docs[0]["text"] == BAD_DOC

    visible = db.q("SELECT id FROM corpora WHERE COALESCE(kind,'ingested') != 'rejected'")
    assert rejected_id not in {row["id"] for row in visible}
    assert cid in {row["id"] for row in visible}
    assert r["approved_corpus_id"] in {row["id"] for row in visible}

    from yylab.ingest.dedup import sha256_hex, training_hash_key

    assert (
        db.q1(
            "SELECT hash FROM rejected_hashes WHERE hash=?",
            (training_hash_key(sha256_hex(BAD_DOC)),),
        )
        is not None
    )


def _read_zst_jsonl(path: Path) -> list[dict]:
    import zstandard as zstd

    with path.open("rb") as fh:
        raw = zstd.ZstdDecompressor().stream_reader(fh).read()
    return [orjson.loads(line) for line in raw.splitlines() if line]


def test_annotations_recorded_for_scorer(tmp_path):
    cid = _make_corpus(tmp_path, "srconn")
    r = _swipe_all(cid)
    ann_path = Path(r["annotation_path"])
    assert ann_path.exists() and ann_path.suffix == ".zst"

    records = _read_zst_jsonl(ann_path)
    assert len(records) == 3
    for rec in records:
        assert {"hash", "verdict", "text", "corpus_id", "review_id", "domain", "lang", "source"} <= set(rec)
        assert rec["corpus_id"] == cid
    assert {rec["verdict"] for rec in records} == {"good", "bad"}

    rows = db.q("SELECT hash, verdict, text, sampling_strategy, sampling_propensity FROM review_annotations")
    assert len(rows) == 3
    assert all(row["text"] for row in rows)
    assert all(row["sampling_strategy"] == "uniform" for row in rows)
    assert all(row["sampling_propensity"] == 1.0 for row in rows)
    assert db.q1("SELECT verdict FROM review_annotations WHERE verdict='bad'") is not None


def test_reannotated_doc_not_duplicated_in_stores(tmp_path):
    cid = _make_corpus(tmp_path, "srcdup")
    r1 = _swipe_all(cid)
    approved_id = r1["approved_corpus_id"]

    docs = sample_review_docs(cid, n=50, skip_reviewed=False)
    verdicts = [{**d, "verdict": "good"} for d in docs]
    r2 = finish_session(cid, verdicts)
    assert r2["dupes"] == 3
    assert r2["docs_good"] == 0

    stats = read_json(corpus_dir(approved_id) / "stats.json")
    assert stats["docs_written"] == 2, "re-annotations must not duplicate store docs"
    ann_rows = db.q("SELECT * FROM review_annotations")
    assert len(ann_rows) == 3

    files = sorted(PATHS.annotations.glob("ann_*.jsonl.zst"))
    total_events = sum(len(_read_zst_jsonl(f)) for f in files)
    assert total_events == 6, "every annotation event stays in the scorer log"


def test_rejected_doc_blocks_future_ingest(tmp_path):
    cid = _make_corpus(tmp_path, "srcblock")
    _swipe_all(cid)

    src2 = tmp_path / "again"
    src2.mkdir()
    (src2 / "c2.txt").write_text(BAD_DOC, "utf-8")
    cfg = IngestConfig(
        name="rev-again",
        source_dir=str(src2),
        tokenizer_keys=[],
        near_dedup=False,
        language_filter=False,
        quality_filter=False,
    )
    cid2 = run_ingest(lambda *a: None, cfg)
    stats = read_json(corpus_dir(cid2) / "stats.json")
    assert stats["rejected_known_bad"] == 1
    assert stats["docs_written"] == 0


def test_swipe_is_durable_before_session_finishes(tmp_path):
    cid = _make_corpus(tmp_path, "durable")
    doc = sample_review_docs(cid, n=1, strategy="uniform")[0]
    review_id = start_session(cid, sampling_strategy="uniform")

    record_verdict(review_id, cid, doc, "good")

    session = db.q1("SELECT * FROM reviews WHERE id=?", (review_id,))
    assert session["status"] == "active"
    assert session["docs_reviewed"] == 1
    swipe = db.q1("SELECT * FROM review_swipes WHERE review_id=?", (review_id,))
    assert swipe["text"] == doc["text"]
    assert swipe["sampling_strategy"] == "uniform"
    assert swipe["sampling_component"] == "uniform"
    assert swipe["sampling_propensity"] == pytest.approx(1 / 3)
    annotation = db.q1("SELECT * FROM review_annotations WHERE hash=?", (doc["hash"],))
    assert annotation["text"] == doc["text"]
    assert annotation["verdict"] == "good"


def test_default_queue_skips_reviewed_documents(tmp_path):
    cid = _make_corpus(tmp_path, "skip-reviewed")
    _swipe_all(cid)

    assert sample_review_docs(cid, n=50) == []
    assert len(sample_review_docs(cid, n=50, skip_reviewed=False)) == 3


def test_queue_reads_quarantine_shards_from_ingest_directory(tmp_path):
    src = tmp_path / "review-quarantine"
    src.mkdir()
    (src / "good.txt").write_text(GOOD_A, "utf-8")
    (src / "junk.txt").write_text(QUARANTINED_DOC, "utf-8")
    cid = run_ingest(
        lambda *a: None,
        IngestConfig(
            name="review-quarantine",
            source_dir=str(src),
            tokenizer_keys=[],
            near_dedup=False,
            global_dedup=False,
            language_filter=False,
            quality_filter=True,
        ),
    )

    stats = read_json(corpus_dir(cid) / "stats.json")
    assert stats["quarantine_shards"]
    assert (corpus_dir(cid) / "data_quarantine" / stats["quarantine_shards"][0]).exists()
    assert not (corpus_dir(cid) / "quarantine").exists()

    docs = sample_review_docs(cid, n=50, include_quarantine=True, strategy="uniform")

    assert {doc["text"] for doc in docs} == {GOOD_A, QUARANTINED_DOC}
    assert {doc["bucket"] for doc in docs} == {"data", "quarantine"}


class _FakeRanker:
    model_version = "test-ranker-v1"

    def score_docs(self, docs: list[dict]) -> list[dict]:
        scored = []
        for doc in docs:
            value = int(doc["hash"][:8], 16) / 0xFFFFFFFF
            scored.append(
                {
                    "rank_score": value,
                    "uncertainty": 1.0 - abs(value - 0.5) * 2.0,
                    "model_version": self.model_version,
                }
            )
        return scored


def test_active_queue_blends_uncertainty_audit_and_random_with_metadata(tmp_path):
    extra = {
        f"extra-{idx}.txt": (
            f"Document {idx} contains a distinct operational account of maintenance planning, measured "
            "outcomes, responsible owners, and the follow-up actions recorded for the engineering archive."
        )
        for idx in range(20)
    }
    cid = _make_corpus(tmp_path, "active", extra=extra)

    docs = sample_review_docs(cid, n=10, strategy="active", ranker=_FakeRanker(), seed=11)

    assert len(docs) == 10
    components = {doc["sampling_component"] for doc in docs}
    assert components == {"uncertainty", "confidence_audit", "random_exploration"}
    assert all(doc["sampling_candidate_count"] == 23 for doc in docs)
    assert all(doc["sampling_scored_candidate_count"] == 23 for doc in docs)
    assert all(doc["model_version"] == "test-ranker-v1" for doc in docs)
    assert all(doc["rank_score"] is not None and doc["uncertainty"] is not None for doc in docs)
    deterministic = [doc for doc in docs if doc["sampling_component"] != "random_exploration"]
    exploration = [doc for doc in docs if doc["sampling_component"] == "random_exploration"]
    assert all(doc["sampling_propensity"] is None for doc in deterministic)
    assert all(0 < doc["sampling_propensity"] <= 1 for doc in exploration)

    review_id = start_session(cid, sampling_strategy="active", model_version=_FakeRanker.model_version)
    record_verdict(review_id, cid, docs[0], "bad")
    stored = db.q1("SELECT * FROM review_swipes WHERE review_id=?", (review_id,))
    assert stored["sampling_component"] == docs[0]["sampling_component"]
    assert stored["model_version"] == "test-ranker-v1"
    assert stored["rank_score"] == pytest.approx(docs[0]["rank_score"])
    assert stored["uncertainty"] == pytest.approx(docs[0]["uncertainty"])


def test_active_queue_bounds_expensive_ranker_scoring(tmp_path, monkeypatch):
    extra = {
        f"extra-{idx}.txt": (
            f"Document {idx} provides a distinct technical account with measured results, "
            "responsible owners, and enough prose to form a useful review candidate."
        )
        for idx in range(20)
    }
    cid = _make_corpus(tmp_path, "bounded-active", extra=extra)

    class CountingRanker(_FakeRanker):
        scored = 0

        def score_docs(self, docs: list[dict]) -> list[dict]:
            self.scored += len(docs)
            return super().score_docs(docs)

    monkeypatch.setattr(review_module, "_ACTIVE_MIN_CANDIDATES", 8)
    monkeypatch.setattr(review_module, "_ACTIVE_MAX_CANDIDATES", 8)
    ranker = CountingRanker()

    docs = sample_review_docs(cid, n=5, strategy="active", ranker=ranker, seed=17)

    assert len(docs) == 5
    assert ranker.scored == 8
    assert all(doc["sampling_candidate_count"] == 23 for doc in docs)
    assert all(doc["sampling_scored_candidate_count"] == 8 for doc in docs)


def test_auto_queue_without_ranker_falls_back_to_uniform(tmp_path):
    cid = _make_corpus(tmp_path, "fallback")
    docs = sample_review_docs(cid, n=2, strategy="auto", ranker=None)
    assert len(docs) == 2
    assert all(doc["sampling_strategy"] == "uniform" for doc in docs)
    assert all(doc["sampling_requested_strategy"] == "auto" for doc in docs)
    assert all(doc["sampling_propensity"] == pytest.approx(2 / 3) for doc in docs)
    assert all(doc["sampling_fallback_reason"] == "ranker unavailable" for doc in docs)


def test_label_loader_survives_derived_corpus_deletion(tmp_path):
    cid = _make_corpus(tmp_path, "loader")
    result = _swipe_all(cid)
    expected = {GOOD_A, GOOD_B, BAD_DOC}

    for derived_id in (result["approved_corpus_id"], result["rejected_corpus_id"]):
        shutil.rmtree(corpus_dir(derived_id), ignore_errors=True)
        db.execute("DELETE FROM corpora WHERE id=?", (derived_id,))

    labels = load_review_labels()
    assert {label["text"] for label in labels} == expected
    assert {label["verdict"] for label in labels} == {"good", "bad"}
    assert all(label["group"] == cid for label in labels)


def test_live_verdict_counts_read_stored_payloads():
    verdicts = {
        "a": {"text": GOOD_A, "verdict": "good"},
        "b": {"text": GOOD_B, "verdict": "good"},
        "c": {"text": BAD_DOC, "verdict": "bad"},
    }
    assert _verdict_counts(verdicts) == (2, 1)


def test_review_queue_sampling_runs_off_ui_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    call: dict = {}
    expected = [{"hash": "sampled"}]

    def fake_sample(cid, n=50, include_quarantine=False, **kwargs):
        call.update(
            {
                "thread": threading.get_ident(),
                "cid": cid,
                "n": n,
                "include_quarantine": include_quarantine,
                **kwargs,
            }
        )
        return expected

    monkeypatch.setattr(review_ui, "sample_review_docs", fake_sample)
    docs = asyncio.run(
        review_ui._sample_review_queue(
            "corpus-id",
            n=75,
            include_quarantine=True,
            strategy="active",
            ranker="ranker",
            skip_reviewed=False,
        )
    )

    assert docs is expected
    assert call == {
        "thread": call["thread"],
        "cid": "corpus-id",
        "n": 75,
        "include_quarantine": True,
        "strategy": "active",
        "ranker": "ranker",
        "skip_reviewed": False,
    }
    assert call["thread"] != loop_thread
