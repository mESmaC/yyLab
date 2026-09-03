from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from yylab import db
from yylab.artifacts import (
    corpus_bucket_dir,
    corpus_dir,
    read_corpus_layout,
    read_json,
    reclaim_retired_generations,
)
from yylab.ingest import pipeline
from yylab.ingest.adapters.base import Adapter
from yylab.ingest.adapters.loader import discover_adapters
from yylab.ingest.dedup import sha256_hex, training_hash_key
from yylab.ingest.pipeline import (
    IngestConfig,
    _iter_shards,
    reimport_corpus,
    run_ingest,
    startup_housekeeping,
)
from yylab.ingest.records import Doc


@pytest.fixture(autouse=True)
def _clean_dedup_state():
    db.execute("DELETE FROM doc_hashes")
    db.execute("DELETE FROM rejected_hashes")


def _sentence(index: int) -> str:
    return (
        f"Record {index} explains how the scheduling service balances maintenance windows "
        f"against demand forecasts and escalates conflicts to the duty engineer. Marker "
        f"{index * 104729 % 99991} keeps every record distinct."
    )


def _source(root: Path, count: int = 12) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (root / f"doc-{index:02d}.txt").write_text(_sentence(index), "utf-8")
    (root / "junk.txt").write_text("buy now " * 60, "utf-8")
    # Sorts first, so an adapter failure on it happens before any other asset
    # is re-extracted.
    (root / "00-notes.md").write_text("# Notes\n\n" + _sentence(900), "utf-8")
    return root


def _cfg(source: Path, **overrides) -> IngestConfig:
    values = dict(
        name="safety",
        source_dir=str(source),
        tokenizer_keys=[],
        near_dedup=False,
        global_dedup=True,
        language_filter=False,
        quality_filter=False,
        selection_profile="preserve",
        workers=1,
    )
    values.update(overrides)
    return IngestConfig(**values)


def _owned_hashes(cid: str) -> set[str]:
    return {row["hash"] for row in db.q("SELECT hash FROM doc_hashes WHERE corpus_id=?", (cid,))}


def _kept_hashes(cid: str) -> set[str]:
    stats = read_json(corpus_dir(cid) / "stats.json")
    return {
        training_hash_key(sha256_hex(line["text"]))
        for line in _iter_shards(corpus_bucket_dir(cid, "data"), stats["shards"])
    }


def _cache_asset_count(cid: str) -> int:
    with sqlite3.connect(corpus_dir(cid) / "prepared.sqlite3") as con:
        return int(con.execute("SELECT COUNT(*) FROM assets").fetchone()[0])


def test_hash_commit_failure_after_publish_is_reconciled_at_startup(tmp_path, monkeypatch):
    source = _source(tmp_path / "src")
    cid = run_ingest(lambda *args: None, _cfg(source))
    before = _owned_hashes(cid)
    assert before == _kept_hashes(cid)

    cfg = _cfg(source, quality_filter=True)
    real_replace = db.replace_corpus_hashes

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "replace_corpus_hashes", locked)
    with pytest.raises(sqlite3.OperationalError):
        reimport_corpus(lambda *args: None, cid, cfg)

    layout = read_corpus_layout(corpus_dir(cid))
    assert layout["generation"] == "gen-1", "the generation was published before the commit"
    row = db.q1("SELECT status, error, stats_json FROM corpora WHERE id=?", (cid,))
    assert row["status"] == "ready"
    assert "published but the hash commit failed" in row["error"]
    assert json.loads(row["stats_json"])["pending_hash_commit"] == "gen-1"
    assert (corpus_dir(cid) / "gen-1" / "hashes.sqlite3").exists()
    assert _owned_hashes(cid) == before, "the registry is untouched until the commit succeeds"
    new_kept = _kept_hashes(cid)
    assert new_kept < before, "quality quarantine removed the junk document"

    monkeypatch.setattr(db, "replace_corpus_hashes", real_replace)
    report = startup_housekeeping()
    assert cid in report["reconciled"]
    assert _owned_hashes(cid) == new_kept
    assert not (corpus_dir(cid) / "gen-1" / "hashes.sqlite3").exists()
    row = db.q1("SELECT status, error, stats_json FROM corpora WHERE id=?", (cid,))
    assert row["status"] == "ready" and row["error"] == ""
    stats = json.loads(row["stats_json"])
    assert "pending_hash_commit" not in stats and "reimport" not in stats
    assert stats["docs_written"] == read_json(corpus_dir(cid) / "stats.json")["docs_written"]
    assert startup_housekeeping()["reconciled"] == []


def test_next_reimport_reconciles_a_pending_hash_commit_first(tmp_path, monkeypatch):
    source = _source(tmp_path / "src")
    cid = run_ingest(lambda *args: None, _cfg(source))
    real_replace = db.replace_corpus_hashes
    monkeypatch.setattr(db, "replace_corpus_hashes", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        reimport_corpus(lambda *args: None, cid, _cfg(source, quality_filter=True))
    monkeypatch.setattr(db, "replace_corpus_hashes", real_replace)

    reimport_corpus(lambda *args: None, cid, _cfg(source, quality_filter=True, min_chars=10))
    assert read_corpus_layout(corpus_dir(cid))["generation"] == "gen-2"
    assert _owned_hashes(cid) == _kept_hashes(cid)
    row = db.q1("SELECT error, stats_json FROM corpora WHERE id=?", (cid,))
    assert row["error"] == "" and "pending_hash_commit" not in json.loads(row["stats_json"])


class _FailingMarkdown(Adapter):
    name = "markdown"
    version = 99
    extensions = {".md", ".markdown", ".rst"}

    def extract(self, path: Path) -> Iterator[Doc]:
        raise ValueError("synthetic markdown failure")
        yield  # pragma: no cover


def test_analysis_key_change_keeps_untouched_cache_entries(tmp_path, monkeypatch):
    source = _source(tmp_path / "src")
    cid = run_ingest(lambda *args: None, _cfg(source))
    first = read_json(corpus_dir(cid) / "stats.json")
    assert _cache_asset_count(cid) == first["files_processed"]

    real_adapters = [adapter for adapter in discover_adapters() if adapter.name != "markdown"]
    monkeypatch.setattr(pipeline, "discover_adapters", lambda: [_FailingMarkdown(), *real_adapters])
    with pytest.raises(RuntimeError, match="asset extraction failed"):
        reimport_corpus(lambda *args: None, cid, _cfg(source, domain_override="code"))
    assert _cache_asset_count(cid) >= first["files_processed"] - 1, (
        "a changed analysis key must not wipe assets the failed run never touched"
    )
    assert read_corpus_layout(corpus_dir(cid))["generation"] == ""

    monkeypatch.setattr(pipeline, "discover_adapters", discover_adapters)
    reimport_corpus(lambda *args: None, cid, _cfg(source))
    stats = read_json(corpus_dir(cid) / "stats.json")
    cache = stats["prepared_cache"]
    assert cache["assets_reused"] == first["files_processed"] - 1
    assert cache["assets_stored"] == 1
    assert _cache_asset_count(cid) == stats["files_processed"]


def test_cache_prunes_assets_removed_from_the_source(tmp_path):
    source = _source(tmp_path / "src")
    cid = run_ingest(lambda *args: None, _cfg(source))
    first = read_json(corpus_dir(cid) / "stats.json")
    (source / "doc-00.txt").unlink()
    (source / "doc-01.txt").unlink()

    reimport_corpus(lambda *args: None, cid, _cfg(source))
    stats = read_json(corpus_dir(cid) / "stats.json")
    assert stats["prepared_cache"]["assets_pruned"] == 2
    assert stats["prepared_cache"]["assets"] == first["files_processed"] - 2
    assert _cache_asset_count(cid) == first["files_processed"] - 2


def test_worker_import_does_not_touch_the_registry(tmp_path):
    home = tmp_path / "home"
    root = Path(pipeline.__file__).resolve().parents[2]
    env = {**os.environ, "YYLAB_HOME": str(home), "PYTHONPATH": str(root)}
    code = (
        "import sys, yylab.ingest.parallel as p; "
        "forbidden = [m for m in ('yylab.db', 'yylab.ingest.pipeline', 'nicegui', 'torch') if m in sys.modules]; "
        "assert not forbidden, forbidden; "
        "p._worker_init(); "
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
    assert not (home / "yylab.sqlite3").exists(), "spawned workers must not create or migrate the registry"


def test_docs_seen_counts_a_record_whose_preparation_fails(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    lines = [
        json.dumps({"text": _sentence(1)}),
        json.dumps({"text": _sentence(2)}),
        '{"text": "bad \\udc80 lone surrogate that orjson rejects and stdlib accepts"}',
        json.dumps({"text": _sentence(4)}),
    ]
    (source / "records.jsonl").write_text("\n".join(lines) + "\n", "utf-8")
    with pytest.raises(RuntimeError, match="asset extraction failed"):
        run_ingest(lambda *args: None, _cfg(source, name="surrogate"))
    row = db.q1("SELECT stats_json, status FROM corpora WHERE name='surrogate'")
    assert row["status"] == "failed"
    assert json.loads(row["stats_json"])["docs_seen"] == 3


def test_startup_marks_stale_processing_rows_failed():
    cid = db.new_id("cor")
    now = db.now_iso()
    db.execute(
        "INSERT INTO corpora(id,name,created_at,updated_at,status,source_path) VALUES(?,?,?,?,?,?)",
        (cid, "stale", now, now, "processing", ""),
    )
    try:
        report = startup_housekeeping()
        assert report["marked_failed"] >= 1
        row = db.q1("SELECT status, error FROM corpora WHERE id=?", (cid,))
        assert row["status"] == "failed"
        assert "interrupted" in row["error"]
    finally:
        db.execute("DELETE FROM corpora WHERE id=?", (cid,))


def test_reclaim_probe_reports_retired_outputs(tmp_path):
    source = _source(tmp_path / "src")
    cid = run_ingest(lambda *args: None, _cfg(source))
    reimport_corpus(lambda *args: None, cid, _cfg(source, quality_filter=True))
    reclaim_retired_generations(cid)
    layout = read_corpus_layout(corpus_dir(cid))
    assert layout["generation"] == "gen-1"
    assert layout["retired"] == []
    assert not (corpus_dir(cid) / "data").exists()
    assert not (corpus_dir(cid) / "inventory.sqlite3").exists()
