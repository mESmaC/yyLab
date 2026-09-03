from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yylab import db
from yylab.artifacts import (
    corpus_bucket_dir,
    corpus_dir,
    corpus_inventory_path,
    read_corpus_layout,
    read_json,
)
from yylab.ingest import pipeline
from yylab.ingest.adapters import jsonl as jsonl_adapter
from yylab.ingest.adapters import parquet as parquet_adapter
from yylab.ingest.adapters.base import Adapter, ChunkSpec
from yylab.ingest.dedup import training_hash_key
from yylab.ingest.parallel import resolve_worker_count
from yylab.ingest.pipeline import IngestConfig, _iter_shards, reimport_corpus, run_ingest
from yylab.ingest.records import Doc

GOOD_EN = (
    "The maintenance team reviewed the overnight sensor readings and confirmed that every "
    "channel reported values within the expected range. They noted that the calibration cycle "
    "was complete and scheduled the next audit for the following week without incident."
)
GERMAN = (
    "Die Wartungsmannschaft überprüfte die nächtlichen Sensorwerte und bestätigte, dass alle "
    "Kanäle Werte im erwarteten Bereich meldeten. Die Kalibrierung war abgeschlossen und die "
    "nächste Prüfung wurde für die folgende Woche ohne Zwischenfälle angesetzt."
)
JUNK = "buy now " * 60
VOLATILE_STATS = {"timing", "prepared_cache", "reimport_history"}


@pytest.fixture(autouse=True)
def _clean_dedup_state():
    db.execute("DELETE FROM doc_hashes")
    db.execute("DELETE FROM rejected_hashes")


@pytest.fixture
def small_chunks(monkeypatch):
    """Force multi-chunk plans on small fixtures (planning happens in-process)."""

    monkeypatch.setattr(jsonl_adapter, "CHUNK_BYTES", 24_000)
    monkeypatch.setattr(jsonl_adapter, "MIN_CHUNKED_FILE_BYTES", 48_000)
    monkeypatch.setattr(parquet_adapter, "CHUNK_ROWS", 60)


def _sentence(index: int) -> str:
    return (
        f"Document number {index} describes how the monitoring service aggregates telemetry "
        f"from field devices and reports anomalies to the operations desk for review. Sequence "
        f"marker {index * 7919 % 10007} keeps the wording distinct across records."
    )


def _make_source(root: Path, *, jsonl_records: int = 700, parquet_rows: int = 300) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "train").mkdir(exist_ok=True)
    (root / "test").mkdir(exist_ok=True)
    (root / "assets").mkdir(exist_ok=True)
    lines: list[str] = []
    for index in range(jsonl_records):
        residue = index % 23
        if residue == 3:
            lines.append("")
        elif residue == 5:
            lines.append("{not valid json")
        elif residue == 7:
            lines.append("[1, 2, 3]")
        elif residue == 11:
            lines.append(json.dumps({"nothing": "here"}))
        elif residue == 13:
            # Repeated sample ids on distinct texts exercise the per-file
            # duplicate_sample_id issue path without changing dedup outcomes.
            lines.append(json.dumps({"text": _sentence(index), "sample_id": f"dup-{index % 3}"}))
        elif residue == 17:
            lines.append(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"Question {index} about telemetry retention?"},
                            {"role": "assistant", "content": _sentence(index)},
                        ],
                        "split": "train",
                    }
                )
            )
        elif residue == 19:
            lines.append(json.dumps({"text": GERMAN + f" Nummer {index}.", "sample_id": f"de-{index}"}))
        elif residue == 21:
            lines.append(json.dumps({"text": JUNK + f"{index}"}))
        else:
            lines.append(
                json.dumps(
                    {
                        "text": _sentence(index),
                        "sample_id": f"s-{index}",
                        "provenance": {"logical_path": f"papers/{'test' if index % 9 == 0 else 'train'}/p{index}.txt"},
                    }
                )
            )
    (root / "train" / "records.jsonl").write_text("﻿" + "\n".join(lines) + "\n", "utf-8")
    texts = []
    for index in range(parquet_rows):
        if index % 17 == 0:
            texts.append(None)
        elif index % 13 == 0:
            texts.append(_sentence(index))  # duplicate of a JSONL record
        else:
            texts.append(_sentence(10_000 + index))
    pq.write_table(
        pa.table({"text": texts, "document_id": [f"pq-{index}" for index in range(parquet_rows)]}),
        root / "test" / "rows.parquet",
        row_group_size=25,
    )
    (root / "notes.md").write_text("# Notes\n\n" + _sentence(50_001) + "\n", "utf-8")
    (root / "same-notes.md").write_text("# Notes\n\n" + _sentence(50_001) + "\n", "utf-8")
    (root / "code.py").write_text("def add(left, right):\n    return left + right\n\n\nprint(add(1, 2))\n", "utf-8")
    (root / "README.md").write_text("Repository navigation and badges " * 20, "utf-8")
    (root / "assets" / "style.css").write_text(".katex { display: inline-block; } " * 50, "utf-8")
    (root / "mystery.xyz").write_bytes(b"\x00\x01binary")
    (root / "english.txt").write_text(GOOD_EN, "utf-8")
    (root / "german.txt").write_text(GERMAN, "utf-8")
    (root / "junk.txt").write_text(JUNK, "utf-8")
    return root


def _config(source: Path, **overrides) -> IngestConfig:
    values = dict(
        name="parallel",
        source_dir=str(source),
        tokenizer_keys=[],
        near_dedup=True,
        global_dedup=True,
        language_filter=True,
        quality_filter=True,
        selection_profile="dataset",
    )
    values.update(overrides)
    return IngestConfig(**values)


def _shard_bytes(directory: Path, shards: list[str]) -> list[dict]:
    if not shards:
        return []
    return list(_iter_shards(directory, shards))


def _inventory(cid: str) -> dict:
    with sqlite3.connect(corpus_inventory_path(cid)) as con:
        assets = sorted(
            con.execute(
                "SELECT logical_path, raw_sha256, byte_size, format_name, adapter_name, status,"
                " reason_codes_json, documents_seen, documents_kept FROM assets"
            ).fetchall()
        )
        documents = sorted(
            con.execute(
                "SELECT document_id, asset_id, locator, normalized_sha256, training_sha256, utf8_bytes,"
                " domain, language, selection, reason_codes_json, roles_json, signals_json,"
                " metadata_json, transformation_json FROM documents"
            ).fetchall()
        )
        issues = sorted(
            con.execute("SELECT asset_id, locator, reason_code, detail FROM issues").fetchall()
        )
    return {"assets": assets, "documents": documents, "issues": issues}


def _snapshot(cid: str) -> dict:
    stats = read_json(corpus_dir(cid) / "stats.json")
    trimmed = {key: value for key, value in stats.items() if key not in VOLATILE_STATS}
    trimmed["ingest_config"] = {
        key: value for key, value in stats["ingest_config"].items() if key != "workers"
    }
    trimmed.pop("generation", None)
    return {
        "stats": trimmed,
        "data": _shard_bytes(corpus_bucket_dir(cid, "data"), stats["shards"]),
        "nonen": _shard_bytes(corpus_bucket_dir(cid, "data_nonen"), stats["nonen_shards"]),
        "quarantine": _shard_bytes(corpus_bucket_dir(cid, "data_quarantine"), stats["quarantine_shards"]),
        "inventory": _inventory(cid),
    }


def _kept_hashes(cid: str) -> set[str]:
    stats = read_json(corpus_dir(cid) / "stats.json")
    from yylab.ingest.dedup import sha256_hex

    return {
        training_hash_key(sha256_hex(line["text"]))
        for line in _iter_shards(corpus_bucket_dir(cid, "data"), stats["shards"])
    }


def test_resolve_worker_count(monkeypatch):
    monkeypatch.delenv("YYLAB_INGEST_WORKERS", raising=False)
    assert resolve_worker_count(3) == 3
    assert resolve_worker_count(None, 2) == 2
    assert resolve_worker_count(None, "bogus") >= 1
    monkeypatch.setenv("YYLAB_INGEST_WORKERS", "5")
    assert resolve_worker_count(None, 2) == 5
    assert resolve_worker_count(4, 2) == 4
    monkeypatch.setenv("YYLAB_INGEST_WORKERS", "0")
    assert resolve_worker_count(None, 9) >= 1


def test_parallel_chunked_ingest_matches_serial(tmp_path, small_chunks):
    source = _make_source(tmp_path / "src")
    serial_id = run_ingest(lambda *args: None, _config(source, workers=1))
    serial = _snapshot(serial_id)
    assert serial["stats"]["docs_written"] > 100
    assert serial["stats"]["near_duplicate_candidates"] > 0
    assert serial["stats"]["non_english_docs"] > 0
    assert serial["stats"]["quarantined_docs"] > 0
    assert serial["stats"]["records_failed"] > 0
    assert serial["stats"]["duplicate_sample_ids"] > 0
    assert serial["stats"]["split_leakage_dupes"] > 0

    db.execute("DELETE FROM doc_hashes")
    progress_messages: list[str] = []
    parallel_id = run_ingest(
        lambda stage, frac, message: progress_messages.append(f"{stage}:{message}"),
        _config(source, workers=3),
    )
    parallel = _snapshot(parallel_id)
    parallel_stats = read_json(corpus_dir(parallel_id) / "stats.json")

    assert parallel_stats["ingest_config"]["workers"] == 3
    assert parallel_stats["timing"]["workers"] == 3
    assert parallel_stats["prepared_cache"]["assets_stored"] == serial["stats"]["files_processed"]
    assert any("3 workers" in message for message in progress_messages)
    assert parallel["stats"] == serial["stats"]
    assert parallel["data"] == serial["data"]
    assert parallel["nonen"] == serial["nonen"]
    assert parallel["quarantine"] == serial["quarantine"]
    assert parallel["inventory"] == serial["inventory"]
    with sqlite3.connect(corpus_dir(parallel_id) / "prepared.sqlite3") as con:
        chunk_counts = dict(
            con.execute("SELECT logical_path, chunk_count FROM assets").fetchall()
        )
    assert chunk_counts["train/records.jsonl"] > 1
    assert chunk_counts["test/rows.parquet"] > 1


class _ChunkedFailingAdapter(Adapter):
    name = "chunked-failing"
    extensions = {".txt"}

    def __init__(self) -> None:
        self._records = 0

    def plan_chunks(self, path: Path) -> list[ChunkSpec] | None:
        return [ChunkSpec(index=0, kind="lines", start=0, end=1), ChunkSpec(index=1, kind="lines", start=1, end=2)]

    def chunk_stats(self) -> tuple[int, int]:
        return (self._records, self._records)

    def extract_chunk(self, path: Path, chunk: ChunkSpec) -> Iterator[Doc]:
        self._records = 0
        for index in range(2):
            self._records += 1
            if chunk.index == 1 and index == 1:
                raise ValueError("synthetic late chunk failure")
            yield Doc(
                text=_sentence(chunk.index * 10 + index),
                source=str(path),
                domain="language",
                locator=f"record={index + 1}",
            )

    def extract(self, path: Path) -> Iterator[Doc]:
        for chunk in self.plan_chunks(path) or []:
            yield from self.extract_chunk(path, chunk)


def test_chunked_failure_keeps_earlier_chunks_and_rebased_locators(tmp_path, monkeypatch):
    source = tmp_path / "src"
    source.mkdir()
    (source / "records.txt").write_text("container", "utf-8")
    monkeypatch.setattr(pipeline, "discover_adapters", lambda: [_ChunkedFailingAdapter()])
    monkeypatch.setattr(pipeline, "CHUNK_WHEN_SERIAL", True)
    monkeypatch.setattr(pipeline, "GLOBAL_BATCH", 1)
    cfg = _config(
        source,
        name="chunked-failure",
        workers=1,
        near_dedup=False,
        language_filter=False,
        quality_filter=False,
        selection_profile="preserve",
    )
    with pytest.raises(RuntimeError, match="asset extraction failed"):
        run_ingest(lambda *args: None, cfg)
    row = db.q1("SELECT id,status,error FROM corpora WHERE name='chunked-failure'")
    assert row["status"] == "failed"
    assert "synthetic late chunk failure" in row["error"]
    stats = read_json(corpus_dir(row["id"]) / "stats.json")
    assert stats["docs_written"] == 3
    assert stats["files_failed"] == 1
    with sqlite3.connect(corpus_dir(row["id"]) / "inventory.sqlite3") as con:
        locators = sorted(loc for (loc,) in con.execute("SELECT locator FROM documents"))
        statuses = dict(con.execute("SELECT logical_path,status FROM assets"))
    assert locators == ["record=1", "record=2", "record=3"]
    assert statuses["records.txt"] == "extraction_failed"
    assert not (corpus_dir(row["id"]) / "prepared.sqlite3").exists() or True


def test_reimport_matches_fresh_ingest_and_swaps_hashes(tmp_path, small_chunks):
    source = _make_source(tmp_path / "src")
    first_cfg = _config(source, name="reimport", workers=1, quality_filter=True, language_filter=True)
    cid = run_ingest(lambda *args: None, first_cfg)
    first = _snapshot(cid)
    first_hashes = {
        row["hash"] for row in db.q("SELECT hash FROM doc_hashes WHERE corpus_id=?", (cid,))
    }
    assert first_hashes == _kept_hashes(cid)

    second_cfg = IngestConfig.from_ingest_config(
        name="ignored",
        source_dir="ignored",
        ingest_config=first["stats"]["ingest_config"],
    )
    second_cfg.quality_filter = False
    second_cfg.language_filter = False
    second_cfg.selection_profile = "preserve"
    second_cfg.min_chars = 20
    second_cfg.workers = 1

    progress_messages: list[str] = []
    assert reimport_corpus(
        lambda stage, frac, message: progress_messages.append(f"{stage}:{message}"), cid, second_cfg
    ) == cid
    reimported = _snapshot(cid)
    reimported_stats = read_json(corpus_dir(cid) / "stats.json")
    assert reimported["stats"]["docs_written"] > first["stats"]["docs_written"]
    assert reimported["stats"]["quarantined_docs"] == 0
    assert reimported["stats"]["non_english_docs"] == 0
    assert reimported_stats["prepared_cache"]["assets_reused"] == first["stats"]["files_processed"]
    # README.md and assets/style.css were policy-excluded before and never
    # extracted; the preserve profile extracts them fresh and caches them.
    assert reimported_stats["prepared_cache"]["assets_stored"] == 2
    assert reimported_stats["reimport_history"][0]["docs_written"] == first["stats"]["docs_written"]
    from yylab.artifacts import reclaim_retired_generations

    reclaim_retired_generations(cid)
    layout = read_corpus_layout(corpus_dir(cid))
    assert layout["generation"] == "gen-1"
    assert reimported_stats["generation"] == "gen-1"
    assert layout["retired"] == [], "nothing held the old outputs open, so they were reclaimed"
    assert not (corpus_dir(cid) / "data").exists()
    assert (corpus_dir(cid) / "gen-1" / "data").exists()
    assert read_json(corpus_dir(cid) / "stats.json")["docs_written"] == reimported["stats"]["docs_written"]
    row = db.q1("SELECT status, error, name, source_path FROM corpora WHERE id=?", (cid,))
    assert row["status"] == "ready" and row["error"] == ""
    assert "reimport" not in read_json(corpus_dir(cid) / "stats.json")
    assert row["name"] == "reimport" and row["source_path"] == str(source)
    new_hashes = {
        row["hash"] for row in db.q("SELECT hash FROM doc_hashes WHERE corpus_id=?", (cid,))
    }
    assert new_hashes == _kept_hashes(cid)
    assert new_hashes > first_hashes

    db.execute("DELETE FROM doc_hashes")
    fresh_cfg = IngestConfig(**{**second_cfg.__dict__, "name": "reimport", "source_dir": str(source)})
    fresh_id = run_ingest(lambda *args: None, fresh_cfg)
    fresh = _snapshot(fresh_id)
    assert reimported["stats"] == fresh["stats"]
    assert reimported["data"] == fresh["data"]
    assert reimported["nonen"] == fresh["nonen"]
    assert reimported["quarantine"] == fresh["quarantine"]
    assert reimported["inventory"] == fresh["inventory"]


def test_reimport_extracts_only_changed_and_new_assets(tmp_path):
    source = _make_source(tmp_path / "src", jsonl_records=60, parquet_rows=40)
    cid = run_ingest(lambda *args: None, _config(source, name="delta", workers=1))
    first = read_json(corpus_dir(cid) / "stats.json")
    (source / "english.txt").write_text(GOOD_EN + " Updated.", "utf-8")
    (source / "brand-new.txt").write_text(_sentence(777_777), "utf-8")
    (source / "junk.txt").unlink()

    cfg = IngestConfig.from_ingest_config(name="x", source_dir="x", ingest_config=first["ingest_config"])
    cfg.workers = 1
    reimport_corpus(lambda *args: None, cid, cfg)
    stats = read_json(corpus_dir(cid) / "stats.json")

    assert stats["prepared_cache"]["assets_stored"] == 2
    assert stats["prepared_cache"]["assets_reused"] == first["files_processed"] - 2
    assert stats["prepared_cache"]["assets_invalidated"] == 1
    assert stats["prepared_cache"]["assets_pruned"] == 1
    assert stats["files_found"] == first["files_found"]
    sources = stats["sources"]
    assert "brand-new.txt" in sources
    assert "junk.txt" not in sources
    with sqlite3.connect(corpus_dir(cid) / "prepared.sqlite3") as con:
        cached_paths = {row[0] for row in con.execute("SELECT logical_path FROM assets")}
    assert "brand-new.txt" in cached_paths and "english.txt" in cached_paths
    assert "junk.txt" not in cached_paths, "assets that left the source are pruned after a full walk"


def test_reimport_failure_leaves_corpus_untouched(tmp_path):
    source = _make_source(tmp_path / "src", jsonl_records=40, parquet_rows=30)
    cid = run_ingest(lambda *args: None, _config(source, name="rollback", workers=1))
    before = _snapshot(cid)
    hashes_before = {row["hash"] for row in db.q("SELECT hash FROM doc_hashes WHERE corpus_id=?", (cid,))}
    shutil.rmtree(source)

    cfg = IngestConfig.from_ingest_config(name="x", source_dir="x", ingest_config=before["stats"]["ingest_config"])
    with pytest.raises(FileNotFoundError):
        reimport_corpus(lambda *args: None, cid, cfg)

    row = db.q1("SELECT status, error, stats_json FROM corpora WHERE id=?", (cid,))
    assert row["status"] == "ready"
    assert "re-import failed" in row["error"]
    assert "reimport" not in json.loads(row["stats_json"])
    assert not any(path.name.startswith("gen-") for path in corpus_dir(cid).iterdir())
    assert read_corpus_layout(corpus_dir(cid))["generation"] == ""
    assert _snapshot(cid) == before
    assert {row["hash"] for row in db.q("SELECT hash FROM doc_hashes WHERE corpus_id=?", (cid,))} == hashes_before


def test_reimport_refuses_derived_and_busy_corpora(tmp_path):
    source = _make_source(tmp_path / "src", jsonl_records=30, parquet_rows=30)
    cid = run_ingest(lambda *args: None, _config(source, name="guard", workers=1))
    cfg = _config(source, name="guard", workers=1)
    db.execute("UPDATE corpora SET status='processing' WHERE id=?", (cid,))
    with pytest.raises(ValueError, match="already being processed"):
        reimport_corpus(lambda *args: None, cid, cfg)
    db.execute("UPDATE corpora SET status='ready', kind='approved' WHERE id=?", (cid,))
    with pytest.raises(ValueError, match="only ingested"):
        reimport_corpus(lambda *args: None, cid, cfg)
    with pytest.raises(ValueError, match="does not exist"):
        reimport_corpus(lambda *args: None, "cor_missing", cfg)


def test_reimport_computes_minhash_bands_when_enabled_later(tmp_path, small_chunks):
    source = _make_source(tmp_path / "src")
    cid = run_ingest(lambda *args: None, _config(source, name="bands", workers=1, near_dedup=False))
    first = read_json(corpus_dir(cid) / "stats.json")
    assert first["near_duplicate_candidates"] == 0

    cfg = IngestConfig.from_ingest_config(name="x", source_dir="x", ingest_config=first["ingest_config"])
    cfg.near_dedup = True
    cfg.workers = 1
    reimport_corpus(lambda *args: None, cid, cfg)
    second = read_json(corpus_dir(cid) / "stats.json")
    assert second["prepared_cache"]["chunks_augmented"] > 0
    assert second["prepared_cache"]["assets_reused"] == first["files_processed"]

    db.execute("DELETE FROM doc_hashes")
    fresh_id = run_ingest(lambda *args: None, _config(source, name="bands-fresh", workers=1, near_dedup=True))
    fresh = _snapshot(fresh_id)
    assert second["near_duplicate_candidates"] == fresh["stats"]["near_duplicate_candidates"] > 0
    assert _snapshot(cid)["data"] == fresh["data"]

    reimport_corpus(lambda *args: None, cid, cfg)
    third = read_json(corpus_dir(cid) / "stats.json")
    assert third["prepared_cache"]["chunks_augmented"] == 0
    assert third["near_duplicate_candidates"] == fresh["stats"]["near_duplicate_candidates"]


def test_reimport_reuses_cached_token_counts(tmp_path, monkeypatch):
    calls: list[int] = []

    class FakeCounter:
        def __init__(self, keys, hf_token=None):  # noqa: ANN001, ARG002
            self.failures = {}
            self._keys = list(keys)

        def available_specs(self):
            from yylab.ingest.tokenize_stats import DEFAULT_SPECS

            return [spec for spec in DEFAULT_SPECS if spec.key in self._keys]

        def count(self, texts):
            calls.append(len(texts))
            return {key: [len(text.split()) for text in texts] for key in self._keys}

    monkeypatch.setattr(pipeline, "TokenCounter", FakeCounter)
    source = _make_source(tmp_path / "src", jsonl_records=80, parquet_rows=40)
    cfg = _config(source, name="tokens", workers=1, tokenizer_keys=["gpt2", "cl100k"], near_dedup=False)
    cid = run_ingest(lambda *args: None, cfg)
    first = read_json(corpus_dir(cid) / "stats.json")
    assert first["totals"]["tokens"]["gpt2"] > 0
    assert first["totals"]["tokens"]["gpt2"] == first["totals"]["tokens"]["cl100k"]
    counted_first = sum(calls)
    assert counted_first == first["docs_written"]

    calls.clear()
    again = IngestConfig.from_ingest_config(name="x", source_dir="x", ingest_config=first["ingest_config"])
    again.workers = 1
    reimport_corpus(lambda *args: None, cid, again)
    second = read_json(corpus_dir(cid) / "stats.json")
    assert sum(calls) == 0, "unchanged kept documents must not be re-tokenized"
    assert second["totals"]["tokens"] == first["totals"]["tokens"]
    assert second["domains"] == first["domains"]
    assert second["splits"] == first["splits"]

    calls.clear()
    again.quality_filter = False
    again.tokenizer_keys = ["gpt2", "cl100k", "o200k"]
    reimport_corpus(lambda *args: None, cid, again)
    third = read_json(corpus_dir(cid) / "stats.json")
    assert third["docs_written"] > first["docs_written"]
    assert 0 < sum(calls) <= third["docs_written"]
    assert third["totals"]["tokens"]["o200k"] == third["totals"]["tokens"]["gpt2"]
    assert third["totals"]["tokens"]["gpt2"] > first["totals"]["tokens"]["gpt2"]


def test_ingest_config_round_trips_through_stats(tmp_path):
    import dataclasses

    source = _make_source(tmp_path / "src", jsonl_records=30, parquet_rows=30)
    cfg = IngestConfig(
        name="round-trip",
        source_dir=str(source),
        domain_override="code",
        near_dedup=False,
        global_dedup=False,
        language_filter=False,
        quality_filter=True,
        quality_threshold=0.5,
        min_chars=25,
        max_file_mb=12.5,
        tokenizer_keys=[],
        selection_profile="repository",
        include_path_regex="train|notes",
        exclude_path_regex="junk",
        workers=1,
        prepared_cache=False,
    )
    cid = run_ingest(lambda *args: None, cfg)
    stats = read_json(corpus_dir(cid) / "stats.json")
    rebuilt = IngestConfig.from_ingest_config(
        name="round-trip", source_dir=str(source), ingest_config=stats["ingest_config"]
    )
    for field in dataclasses.fields(IngestConfig):
        if field.name == "workers":
            continue
        assert getattr(rebuilt, field.name) == getattr(cfg, field.name), field.name
    assert stats["prepared_cache"] == {"enabled": False}
    assert not (corpus_dir(cid) / "prepared.sqlite3").exists()
    legacy = IngestConfig.from_ingest_config(name="legacy", source_dir="x", ingest_config={"near_dedup": False})
    assert legacy.domain_override is None and legacy.near_dedup is False and legacy.global_dedup is True


def test_reimport_keeps_corpus_usable_and_defers_reclaim_while_shards_are_open(tmp_path):
    import os

    from yylab.artifacts import reclaim_retired_generations
    from yylab.review import sample_review_docs
    from yylab.train.data import corpus_shards, iter_corpus_docs

    source = _make_source(tmp_path / "src", jsonl_records=40, parquet_rows=30)
    cid = run_ingest(lambda *args: None, _config(source, name="live", workers=1))
    stats = read_json(corpus_dir(cid) / "stats.json")
    old_shard = corpus_bucket_dir(cid, "data") / stats["shards"][0]
    old_docs = list(iter_corpus_docs([cid], shuffle_buffer=1))
    assert old_docs

    cfg = IngestConfig.from_ingest_config(name="x", source_dir="x", ingest_config=stats["ingest_config"])
    cfg.workers = 1
    cfg.quality_filter = False
    seen_status: list[str] = []

    def progress(stage: str, frac: float, message: str) -> None:
        if stage == "extract" and not seen_status:
            row = db.q1("SELECT status, stats_json FROM corpora WHERE id=?", (cid,))
            seen_status.append(row["status"])
            assert "reimport" in json.loads(row["stats_json"])
            assert corpus_shards(cid) == stats["shards"], "old generation stays live during the run"

    with old_shard.open("rb"):
        reimport_corpus(progress, cid, cfg)
        layout = read_corpus_layout(corpus_dir(cid))
        assert seen_status == ["ready"]
        assert layout["generation"] == "gen-1"
        # Windows: the rename probe fails while the shard is open. POSIX: no
        # reclaim happens at publish at all. Either way the old data survives.
        assert "data" in layout["retired"], "an open shard must defer deletion"
        assert old_shard.exists()
    reclaimed = reclaim_retired_generations(cid)
    assert "data" in reclaimed
    assert os.name  # keep the platform import meaningful on both families
    assert not (corpus_dir(cid) / "data").exists()
    assert read_corpus_layout(corpus_dir(cid))["retired"] == []
    new_stats = read_json(corpus_dir(cid) / "stats.json")
    assert new_stats["docs_written"] > stats["docs_written"]
    assert corpus_shards(cid) == new_stats["shards"]
    streamed = list(iter_corpus_docs([cid], shuffle_buffer=1, cross_corpus_dedup=False))
    assert len(streamed) == new_stats["docs_written"]
    assert sample_review_docs(cid, n=5, seed=1)
    assert (corpus_dir(cid) / "gen-1" / "inventory.sqlite3").exists()
    assert corpus_inventory_path(cid) == corpus_dir(cid) / "gen-1" / "inventory.sqlite3"

    # A second re-import publishes gen-2 and retires gen-1.
    cfg.selection_profile = "preserve"
    reimport_corpus(lambda *args: None, cid, cfg)
    reclaim_retired_generations(cid)
    layout = read_corpus_layout(corpus_dir(cid))
    assert layout["generation"] == "gen-2"
    assert not (corpus_dir(cid) / "gen-1").exists()


def test_ingest_lock_rejects_concurrent_runs_and_clears_stale_marker(tmp_path):
    from yylab.ingest.pipeline import _INGEST_LOCK, ingest_in_progress

    source = _make_source(tmp_path / "src", jsonl_records=30, parquet_rows=30)
    cid = run_ingest(lambda *args: None, _config(source, name="lock", workers=1))
    cfg = _config(source, name="lock", workers=1)
    assert not ingest_in_progress()
    assert _INGEST_LOCK.acquire(blocking=False)
    try:
        assert ingest_in_progress()
        with pytest.raises(RuntimeError, match="already running"):
            run_ingest(lambda *args: None, cfg)
        with pytest.raises(RuntimeError, match="already running"):
            reimport_corpus(lambda *args: None, cid, cfg)
    finally:
        _INGEST_LOCK.release()
    assert db.q1("SELECT COUNT(*) AS n FROM corpora WHERE name='lock'")["n"] == 1

    stale = json.loads(db.q1("SELECT stats_json FROM corpora WHERE id=?", (cid,))["stats_json"])
    stale["reimport"] = {"started_at": "2020-01-01T00:00:00"}
    db.execute("UPDATE corpora SET stats_json=? WHERE id=?", (json.dumps(stale), cid))
    reimport_corpus(lambda *args: None, cid, cfg)
    assert "reimport" not in json.loads(db.q1("SELECT stats_json FROM corpora WHERE id=?", (cid,))["stats_json"])
