from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from yylab import db
from yylab.artifacts import corpus_dir, read_json
from yylab.ingest import pipeline
from yylab.ingest.adapters.base import Adapter
from yylab.ingest.dedup import norm_sha256_hex, sha256_hex, training_hash_key
from yylab.ingest.pipeline import IngestConfig, _iter_shards, run_ingest
from yylab.ingest.records import Doc


def _safe_config(source: Path, **overrides) -> IngestConfig:
    return IngestConfig(
        name=str(overrides.pop("name", "ingest-safety")),
        source_dir=str(source),
        tokenizer_keys=[],
        near_dedup=False,
        global_dedup=bool(overrides.pop("global_dedup", False)),
        language_filter=False,
        quality_filter=False,
        min_chars=1,
        selection_profile=str(overrides.pop("selection_profile", "preserve")),
        **overrides,
    )


def test_legacy_normalized_global_hash_cannot_block_exact_text(tmp_path: Path):
    legacy = "US = 1 and casing is part of this model-visible training statement."
    distinct = "us = 1 and casing is part of this model-visible training statement."
    assert norm_sha256_hex(legacy) == norm_sha256_hex(distinct)
    db.execute(
        "INSERT OR IGNORE INTO doc_hashes(hash,corpus_id) VALUES(?,?)",
        (norm_sha256_hex(legacy), "legacy-corpus"),
    )
    (tmp_path / "record.txt").write_text(distinct, "utf-8")

    corpus_id = run_ingest(
        lambda *_args: None,
        _safe_config(tmp_path, global_dedup=True),
    )
    stats = read_json(corpus_dir(corpus_id) / "stats.json")

    assert stats["docs_written"] == 1
    assert stats["exact_dupes_global"] == 0
    assert stats["ingest_config"]["global_hash_storage"] == (
        "training_sha256_v3_namespaced"
    )


def test_oversized_asset_is_catalogued_with_explicit_decision(tmp_path: Path):
    (tmp_path / "large.txt").write_text("useful corpus row " * 200, "utf-8")

    corpus_id = run_ingest(
        lambda *_args: None,
        _safe_config(tmp_path, max_file_mb=0.0001),
    )
    stats = read_json(corpus_dir(corpus_id) / "stats.json")
    with sqlite3.connect(corpus_dir(corpus_id) / "inventory.sqlite3") as connection:
        asset = connection.execute(
            "SELECT status,raw_sha256,reason_codes_json FROM assets"
        ).fetchone()

    assert stats["files_found"] == 1
    assert stats["assets_size_skipped"] == 1
    assert stats["docs_written"] == 0
    assert asset[0] == "excluded_by_limit"
    assert asset[1] == ""
    assert "asset_too_large" in asset[2]


def test_historical_bad_review_is_safely_upgraded_to_namespaced_rejection(
    tmp_path: Path,
):
    text = "A human reviewer rejected this exact model-visible document in an older yyLab build."
    exact_hash = sha256_hex(text)
    now = db.now_iso()
    db.execute(
        "INSERT INTO review_annotations(hash,verdict,corpus_id,review_id,created_at,text) "
        "VALUES(?,?,?,?,?,?)",
        (exact_hash, "bad", "legacy-corpus", "legacy-review", now, text),
    )
    db.execute(
        "INSERT INTO rejected_hashes(hash,added_at,source) VALUES(?,?,?)",
        (exact_hash, now, "legacy-review"),
    )
    (tmp_path / "rejected.txt").write_text(text, "utf-8")

    corpus_id = run_ingest(lambda *_args: None, _safe_config(tmp_path))
    stats = read_json(corpus_dir(corpus_id) / "stats.json")

    assert stats["docs_written"] == 0
    assert stats["rejected_known_bad"] == 1
    assert db.q1(
        "SELECT hash FROM rejected_hashes WHERE hash=?",
        (training_hash_key(exact_hash),),
    ) is not None


def test_native_cdas_shard_uses_row_lineage_for_policy_domain_and_provenance(
    tmp_path: Path,
):
    readme_text = "Repository navigation badges and setup notes that are not corpus content."
    code_text = "def add(left, right):\n    return left + right\n"
    records = [
        {
            "schema_version": 1,
            "document_id": "doc-readme",
            "source_id": "source-cdas",
            "asset_id": "asset-readme",
            "logical_path": "README.md",
            "locator": "$",
            "format": "text",
            "content_sha256": sha256_hex(readme_text),
            "text": readme_text,
        },
        {
            "schema_version": 1,
            "document_id": "doc-code",
            "source_id": "source-cdas",
            "asset_id": "asset-code",
            "logical_path": "src/main.py",
            "locator": "$",
            "format": "text",
            "content_sha256": sha256_hex(code_text),
            "text": code_text,
        },
    ]
    (tmp_path / "shard-00000.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records),
        "utf-8",
    )

    corpus_id = run_ingest(
        lambda *_args: None,
        _safe_config(tmp_path, selection_profile="dataset"),
    )
    stats = read_json(corpus_dir(corpus_id) / "stats.json")
    docs = list(
        _iter_shards(corpus_dir(corpus_id) / "data", stats["shards"])
    )

    assert stats["docs_seen"] == 2
    assert stats["docs_policy_skipped"] == 1
    assert stats["docs_written"] == 1
    assert docs[0]["source"] == "src/main.py"
    assert docs[0]["domain"] == "code"
    assert docs[0]["provenance"]["document_id"] == "doc-code"
    assert docs[0]["provenance"]["content_sha256"] == sha256_hex(code_text)


def test_native_cdas_content_hash_mismatch_is_quarantined(tmp_path: Path):
    text = "A CDAS export row whose declared content hash does not match its text."
    record = {
        "schema_version": 1,
        "document_id": "doc-tampered",
        "logical_path": "train/article.txt",
        "content_sha256": "0" * 64,
        "text": text,
    }
    (tmp_path / "shard-00000.jsonl").write_text(json.dumps(record), "utf-8")

    corpus_id = run_ingest(lambda *_args: None, _safe_config(tmp_path))
    stats = read_json(corpus_dir(corpus_id) / "stats.json")

    assert stats["docs_written"] == 0
    assert stats["quarantined_docs"] == 1
    assert stats["quarantine_reasons"]["cdas:content_hash_mismatch"] == 1


def test_adapter_exception_marks_entire_corpus_failed_after_partial_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FailingAdapter(Adapter):
        name = "failing"
        extensions = {".txt"}

        def extract(self, path: Path) -> Iterator[Doc]:
            yield Doc(
                text="A valid first record is deliberately emitted before the failure.",
                source=str(path),
                domain="language",
            )
            raise ValueError("synthetic late row-group failure")

    (tmp_path / "records.txt").write_text("container", "utf-8")
    monkeypatch.setattr(pipeline, "discover_adapters", lambda: [FailingAdapter()])
    monkeypatch.setattr(pipeline, "GLOBAL_BATCH", 1)

    with pytest.raises(RuntimeError, match="asset extraction failed"):
        run_ingest(lambda *_args: None, _safe_config(tmp_path, name="partial-failure"))

    row = db.q1("SELECT id,status,error FROM corpora WHERE name='partial-failure'")
    assert row is not None
    assert row["status"] == "failed"
    assert "synthetic late row-group failure" in row["error"]
    stats = read_json(corpus_dir(row["id"]) / "stats.json")
    assert stats["docs_written"] == 1
    assert stats["files_failed"] == 1
