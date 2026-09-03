from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from yylab import db
from yylab.artifacts import corpus_dir, read_json
from yylab.ingest.dedup import dedup_fingerprints
from yylab.ingest.pipeline import IngestConfig, _iter_shards, run_ingest
from yylab.train import data
from yylab.workers.train import normalize_base_data_config


def _run(tmp_path: Path, files: dict[str, str], **overrides):
    source = tmp_path / "source"
    source.mkdir()
    for relative, text in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, "utf-8")
    config = IngestConfig(
        name=f"cdas-{tmp_path.name}",
        source_dir=str(source),
        tokenizer_keys=[],
        near_dedup=False,
        global_dedup=False,
        language_filter=False,
        quality_filter=False,
        **overrides,
    )
    corpus_id = run_ingest(lambda *args: None, config)
    stats = read_json(corpus_dir(corpus_id) / "stats.json")
    docs = list(_iter_shards(corpus_dir(corpus_id) / "data", stats["shards"]))
    return corpus_id, stats, docs


def test_training_normalization_preserves_math_compatibility_characters(tmp_path):
    text = (
        "A prime factorization preserves mathematical notation: 16 = 2⁴ and 24 = 2³ × 3. "
        "The subscripted sequence a₁, a₂, a₃ remains distinct from ordinary adjacent digits."
    )
    corpus_id, stats, docs = _run(
        tmp_path,
        {"train.jsonl": json.dumps({"text": text, "split": "train"})},
    )

    assert docs[0]["text"] == text
    assert "2⁴" in docs[0]["text"] and "2³ × 3" in docs[0]["text"]
    assert stats["ingest_config"]["training_normalization"] == "nfc_whitespace_v2"
    assert stats["cdas"]["compatibility_fold_would_change_docs"] == 1

    with sqlite3.connect(corpus_dir(corpus_id) / "inventory.sqlite3") as connection:
        transformation = json.loads(
            connection.execute("SELECT transformation_json FROM documents").fetchone()[0]
        )
    assert transformation["source_normalization"] == ["utf8_bom", "newline"]
    assert transformation["training_normalization"] == "nfc_whitespace_v2"


def test_dedup_evidence_keeps_math_distinct_and_finds_code_structure():
    math_notation = dedup_fingerprints(
        "16 = 2⁴",
        mode="natural",
        logical_path="math.txt",
    )
    folded_math = dedup_fingerprints(
        "16 = 24",
        mode="natural",
        logical_path="math.txt",
    )
    first_code = dedup_fingerprints(
        "def add(left, right):\n    return left + right + 7\n",
        mode="code",
        logical_path="first.py",
        language="python",
    )
    renamed_code = dedup_fingerprints(
        "def combine(alpha, beta):\n    return alpha + beta + 19\n",
        mode="code",
        logical_path="second.py",
        language="python",
    )

    assert math_notation["normalized_sha256"] != folded_math["normalized_sha256"]
    assert first_code["normalized_sha256"] != renamed_code["normalized_sha256"]
    assert first_code["structural_sha256"] == renamed_code["structural_sha256"]
    assert first_code["parse_status"] == renamed_code["parse_status"] == "valid"


def test_structured_dedup_evidence_canonicalizes_object_key_order():
    first = dedup_fingerprints(
        '{"beta":2,"alpha":1}',
        mode="structured",
        logical_path="records.jsonl",
    )
    reordered = dedup_fingerprints(
        '{ "alpha": 1, "beta": 2 }',
        mode="structured",
        logical_path="records.jsonl",
    )

    assert first["content_sha256"] != reordered["content_sha256"]
    assert first["normalized_sha256"] == reordered["normalized_sha256"]
    assert first["structural_sha256"] == reordered["structural_sha256"]


def test_dataset_policy_excludes_sidecars_before_extraction(tmp_path):
    real = (
        "This training record contains the actual article prose and enough unique language "
        "to serve as a useful causal language modeling document."
    )
    files = {
        "README.md": "Repository navigation and badges " * 20,
        "LICENSE": "Copyright and redistribution terms " * 20,
        "assets/style.css": ".katex { display: inline-block; } " * 100,
        "assets/equation.svg": "<svg><path d='M 0 0 L 100 100'/></svg>" * 100,
        "nbproject/configurations.xml": "<configuration><toolchain>generated</toolchain></configuration>" * 30,
        "train.jsonl": json.dumps({"text": real, "split": "train"}),
    }
    corpus_id, stats, docs = _run(tmp_path, files, selection_profile="dataset")

    assert [doc["text"] for doc in docs] == [real]
    assert stats["assets_policy_skipped"] == 5
    assert stats["docs_written"] == 1
    assert stats["cdas"]["assets"] == 6
    assert stats["cdas"]["selection_counts"] == {"kept": 1}

    with sqlite3.connect(corpus_dir(corpus_id) / "inventory.sqlite3") as connection:
        statuses = dict(connection.execute("SELECT logical_path,status FROM assets"))
    assert statuses["train.jsonl"] == "normalized"
    assert statuses["README.md"] == "excluded_by_policy"
    assert statuses["assets/equation.svg"] == "excluded_by_policy"


def test_compiled_cdas_jsonl_preserves_split_labels_and_provenance(tmp_path):
    text = (
        "Tensor training evidence is retained with a source identity, split, and stable document "
        "provenance instead of being flattened into anonymous text."
    )
    record = {
        "schema_version": 1,
        "sample_id": "sample-17",
        "split": "train",
        "text": text,
        "provenance": {
            "source_id": "src_v1_example",
            "document_id": "doc_v1_example",
            "logical_path": "papers/example.html",
        },
        "labels": {"content_type": "natural-text", "domains": ["topic/machine-learning"]},
    }
    corpus_id, _stats, docs = _run(tmp_path, {"compiled.jsonl": json.dumps(record)})

    assert docs[0]["split"] == "train"
    assert docs[0]["sample_id"] == "sample-17"
    assert docs[0]["provenance"]["document_id"] == "doc_v1_example"
    assert docs[0]["labels"]["domains"] == ["topic/machine-learning"]

    with sqlite3.connect(corpus_dir(corpus_id) / "inventory.sqlite3") as connection:
        locator, metadata_json = connection.execute(
            "SELECT locator,metadata_json FROM documents"
        ).fetchone()
    assert locator == "record=1;sample=sample-17"
    metadata = json.loads(metadata_json)
    assert metadata["provenance"]["logical_path"] == "papers/example.html"


def test_compiled_dataset_applies_policy_to_each_records_original_path(tmp_path):
    readme = {
        "sample_id": "readme-row",
        "text": "Repository badges installation instructions and navigation. " * 8,
        "provenance": {"logical_path": "projects/example/README.md"},
    }
    article = {
        "sample_id": "article-row",
        "text": "A retained scientific article with useful explanatory prose. " * 8,
        "provenance": {"logical_path": "papers/train/article.txt"},
    }
    _corpus_id, stats, docs = _run(
        tmp_path,
        {"compiled.jsonl": "\n".join((json.dumps(readme), json.dumps(article)))},
    )

    assert [doc["sample_id"] for doc in docs] == ["article-row"]
    assert docs[0]["source"] == "papers/train/article.txt"
    assert docs[0]["split"] == "train"
    assert stats["docs_policy_skipped"] == 1
    assert stats["quarantined_docs"] == 1


def test_duplicate_sample_ids_do_not_overwrite_catalog_decisions(tmp_path):
    records = [
        {"sample_id": "duplicate", "text": "First independently catalogued record. " * 4},
        {"sample_id": "duplicate", "text": "Second independently catalogued record. " * 4},
    ]
    corpus_id, stats, docs = _run(
        tmp_path,
        {"train.jsonl": "\n".join(json.dumps(record) for record in records)},
    )

    assert len(docs) == 2
    assert stats["duplicate_sample_ids"] == 1
    with sqlite3.connect(corpus_dir(corpus_id) / "inventory.sqlite3") as connection:
        locators = [
            row[0]
            for row in connection.execute("SELECT locator FROM documents ORDER BY locator")
        ]
        issues = connection.execute(
            "SELECT reason_code,detail FROM issues"
        ).fetchall()
    assert locators == ["record=1;sample=duplicate", "record=2;sample=duplicate"]
    assert issues == [("duplicate_sample_id", "duplicate")]


def test_quarantine_does_not_poison_global_dedup(tmp_path):
    source_text = "boiler boiler boiler boiler boiler boiler boiler boiler boiler boiler " * 8
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "train.txt").write_text(source_text, "utf-8")
    (second_root / "train.txt").write_text(source_text, "utf-8")

    first_id = run_ingest(
        lambda *args: None,
        IngestConfig(
            name="quarantined-first",
            source_dir=str(first_root),
            tokenizer_keys=[],
            near_dedup=False,
            global_dedup=True,
            language_filter=False,
            quality_filter=True,
            quality_threshold=0.99,
            selection_profile="preserve",
        ),
    )
    assert read_json(corpus_dir(first_id) / "stats.json")["quarantined_docs"] == 1

    second_id = run_ingest(
        lambda *args: None,
        IngestConfig(
            name="accepted-second",
            source_dir=str(second_root),
            tokenizer_keys=[],
            near_dedup=False,
            global_dedup=True,
            language_filter=False,
            quality_filter=False,
            selection_profile="preserve",
        ),
    )
    second_stats = read_json(corpus_dir(second_id) / "stats.json")
    assert second_stats["docs_written"] == 1
    assert second_stats["exact_dupes_global"] == 0


def test_cross_split_duplicates_are_reported_without_deleting_train(tmp_path):
    text = "A duplicated evaluation example remains visible for leakage review. " * 3
    _corpus_id, stats, docs = _run(
        tmp_path,
        {"test.txt": text, "train.txt": text},
    )

    assert len(docs) == 2
    assert {doc["split"] for doc in docs} == {"train", "test"}
    assert stats["split_leakage_dupes"] == 1
    assert stats["exact_dupes_local"] == 0


def test_training_stream_deduplicates_derived_and_source_corpora(monkeypatch):
    documents = {
        "source": [
            {"text": "The same reviewed training record.", "domain": "language", "origin": "source"},
            {"text": "A unique source record.", "domain": "language", "origin": "source-unique"},
        ],
        "approved": [
            {"text": "The same reviewed training record.", "domain": "language", "origin": "approved"},
            {"text": "A unique approved record.", "domain": "language", "origin": "approved-unique"},
        ],
    }
    monkeypatch.setattr(data, "corpus_shards", lambda corpus_id, **_: [f"{corpus_id}.jsonl.zst"])
    monkeypatch.setattr(data, "corpus_dir", lambda corpus_id: Path(corpus_id))
    monkeypatch.setattr(
        data,
        "_iter_shards",
        lambda directory, _shards: iter(documents[directory.parent.name]),
    )

    rows = list(
        data.iter_corpus_docs(
            ["source", "approved"],
            shuffle_buffer=1,
            corpus_mix="concatenate",
        )
    )
    assert [row["origin"] for row in rows] == ["source", "source-unique", "approved-unique"]


def test_training_stream_does_not_casefold_code_for_hard_dedup(monkeypatch):
    documents = {
        "first": [{"text": "US = 1", "domain": "code"}],
        "second": [{"text": "us = 1", "domain": "code"}],
    }
    monkeypatch.setattr(data, "corpus_shards", lambda corpus_id, **_: [f"{corpus_id}.jsonl.zst"])
    monkeypatch.setattr(data, "corpus_dir", lambda corpus_id: Path(corpus_id))
    monkeypatch.setattr(
        data,
        "_iter_shards",
        lambda directory, _shards: iter(documents[directory.parent.name]),
    )

    rows = list(
        data.iter_corpus_docs(
            ["first", "second"],
            shuffle_buffer=1,
            corpus_mix="concatenate",
        )
    )
    assert [row["text"] for row in rows] == ["US = 1", "us = 1"]


def test_token_balanced_mixing_accounts_for_document_size_and_honors_cap(monkeypatch):
    documents = {
        "large": [
            {"text": f"L{index}-" + "x" * 996, "domain": "language", "origin": f"large-{index}"}
            for index in range(10)
        ],
        "small": [
            {"text": f"S{index}-" + "y" * 96, "domain": "language", "origin": f"small-{index}"}
            for index in range(10)
        ],
    }
    monkeypatch.setattr(data, "corpus_shards", lambda corpus_id, **_: [f"{corpus_id}.jsonl.zst"])
    monkeypatch.setattr(data, "corpus_dir", lambda corpus_id: Path(corpus_id))
    monkeypatch.setattr(
        data,
        "_iter_shards",
        lambda directory, _shards: iter(documents[directory.parent.name]),
    )

    rows = list(
        data.iter_corpus_docs(
            ["large", "small"],
            shuffle_buffer=1,
            corpus_mix="token_balanced",
            source_max_ratio=0.6,
        )
    )
    origins = [row["origin"] for row in rows]
    assert origins[:7] == [f"small-{index}" for index in range(7)]
    assert origins[7] == "large-0"
    assert sum(origin.startswith("large-") for origin in origins) == 1
    assert sum(origin.startswith("small-") for origin in origins) == 10


def test_token_balanced_guard_does_not_admit_one_giant_first_document(monkeypatch):
    documents = {
        "giant": [{"text": "g" * 100_000, "domain": "language", "origin": "giant"}],
        "small": [
            {
                "text": f"small-{index}-" + "s" * 90,
                "domain": "language",
                "origin": f"small-{index}",
            }
            for index in range(10)
        ],
    }
    monkeypatch.setattr(data, "corpus_shards", lambda corpus_id, **_: [f"{corpus_id}.jsonl.zst"])
    monkeypatch.setattr(data, "corpus_dir", lambda corpus_id: Path(corpus_id))
    monkeypatch.setattr(
        data,
        "_iter_shards",
        lambda directory, _shards: iter(documents[directory.parent.name]),
    )

    rows = list(
        data.iter_corpus_docs(
            ["giant", "small"],
            shuffle_buffer=1,
            corpus_mix="token_balanced",
            source_max_ratio=0.5,
        )
    )
    assert [row["origin"] for row in rows] == [f"small-{index}" for index in range(10)]


def test_base_training_refuses_rejected_corpus():
    corpus_id = db.new_id("cor")
    now = db.now_iso()
    db.execute(
        """INSERT INTO corpora(
          id,name,created_at,updated_at,status,source_path,verdict
        ) VALUES(?,?,?,?,?,?,?)""",
        (corpus_id, "known contaminated", now, now, "ready", "", "rejected"),
    )
    try:
        try:
            normalize_base_data_config({"corpora": [corpus_id]})
        except ValueError as error:
            assert "refuses rejected corpus" in str(error)
        else:
            raise AssertionError("rejected corpus must not reach a base worker")
    finally:
        db.execute("DELETE FROM corpora WHERE id=?", (corpus_id,))


def test_base_training_refuses_rejected_kind_and_missing_corpus():
    corpus_id = db.new_id("cor")
    now = db.now_iso()
    db.execute(
        """INSERT INTO corpora(
          id,name,created_at,updated_at,status,source_path,kind,verdict
        ) VALUES(?,?,?,?,?,?,?,?)""",
        (corpus_id, "rejected store", now, now, "ready", "", "rejected", ""),
    )
    try:
        try:
            normalize_base_data_config({"corpora": [corpus_id]})
        except ValueError as error:
            assert "refuses rejected corpus" in str(error)
        else:
            raise AssertionError("rejected corpus kind must not reach a base worker")
        try:
            normalize_base_data_config({"corpora": ["missing-corpus"]})
        except ValueError as error:
            assert "does not exist" in str(error)
        else:
            raise AssertionError("missing corpus must fail preflight")
    finally:
        db.execute("DELETE FROM corpora WHERE id=?", (corpus_id,))


def test_base_data_config_validates_token_balanced_caps_and_weights(monkeypatch):
    monkeypatch.setattr(
        "yylab.workers.train.db.q1",
        lambda _query, params: {
            "name": params[0],
            "status": "ready",
            "kind": "ingested",
            "verdict": "",
        },
    )

    for config, message in (
        (
            {"corpora": ["a", "b"], "corpus_mix": "round_robin", "source_max_ratio": 0.5},
            "require token-balanced",
        ),
        (
            {"corpora": ["a", "b"], "corpus_mix": "token_balanced", "source_max_ratio": 0.4},
            "minimum feasible",
        ),
        (
            {"corpora": ["a"], "corpus_mix": "token_balanced", "corpus_weights": {"b": 1}},
            "unselected corpus",
        ),
    ):
        try:
            normalize_base_data_config(config)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"expected config rejection: {config}")

    single = normalize_base_data_config(
        {
            "corpora": ["a"],
            "corpus_mix": "token_balanced",
            "source_max_ratio": 0.5,
        }
    )
    assert single["source_max_ratio"] is None
