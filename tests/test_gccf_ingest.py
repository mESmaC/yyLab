from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from yylab import db
from yylab.artifacts import corpus_dir
from yylab.ingest.gccf import gccf_ingest_config, import_export, run_gccf_ingest
from yylab.train.data import corpus_data_provenance, iter_corpus_docs


@pytest.mark.parametrize("materialize", [False, True])
def test_receipt_to_yylab_training_preserves_selection_splits_and_provenance(tmp_path, materialize):
    source = tmp_path / "source"
    source.mkdir()
    data = source / "data.jsonl"
    kept = (
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Explain the sum of two integers."},
                    {"role": "assistant", "content": "Add the values together."},
                ],
                "split": "test",
            }
        ).encode()
        + b"\n"
    )
    bad = b'{"text":"This record was rejected by the classifier."}\n'
    data.write_bytes(kept + bad)
    r = tmp_path / "receipt.json"
    r.write_text(
        json.dumps(
            {
                "schema": "gccf.receipt/1",
                "roots": [{"id": 1, "path": str(source)}],
                "summary": {},
                "files": [
                    {
                        "root_id": 1,
                        "path": data.name,
                        "abs_path": str(data),
                        "format": "jsonl",
                        "kind": "rows",
                        "compression": "none",
                        "size_bytes": data.stat().st_size,
                        "mtime": int(data.stat().st_mtime),
                        "verdict": "include",
                        "good_ratio": 0.9,
                        "counts": {"human": 0, "classifier": 20},
                        "tags": {"english": 20},
                        "exclusions": [
                            {"type": "bytes", "byte_start": len(kept), "byte_end": len(kept) + len(bad)}
                        ],
                    }
                ],
            }
        )
    )
    output = tmp_path / "export"
    result = run_gccf_ingest(lambda *a: None, str(r), output=str(output), workers=1, materialize=materialize)
    cid = next(iter(result["corpora"].values()))
    assert db.q1("SELECT status FROM corpora WHERE id=?", (cid,))["status"] == "ready"
    assert list(iter_corpus_docs([cid], splits=["train"])) == []
    heldout = list(iter_corpus_docs([cid], splits=["test"]))
    assert len(heldout) == 1
    assert "rejected" not in heldout[0]["text"]
    assert heldout[0]["messages"][1]["role"] == "assistant"
    assert heldout[0]["provenance"]["receipt_sha256"] == result["receipt_sha256"]
    assert (corpus_dir(cid) / "gccf-provenance.json").exists()
    lineage = corpus_data_provenance([cid])["corpora"][0]
    assert lineage["catalog_status"] == "gccf_catalogued"
    assert lineage["gccf"]["receipt_sha256"] == result["receipt_sha256"]
    if materialize:
        assert import_export(lambda *a: None, output)["corpora"] == result["corpora"]
    else:
        assert result["storage"] == "original_sources"
        assert result["text_bytes_copied"] == 0
        assert not list(output.rglob("*.jsonl*"))
        assert not (corpus_dir(cid) / "data").exists()
        assert run_gccf_ingest(lambda *a: None, str(r), output=str(output))["corpora"] == result["corpora"]
        from yylab.train.data import corpus_training_preflight

        assert not corpus_training_preflight([cid], splits=["test"])["errors"]
        from yylab.ingest.pipeline import reimport_corpus

        with pytest.raises(ValueError, match="new GCCF receipt"):
            reimport_corpus(lambda *a: None, cid, gccf_ingest_config(name="direct", source_dir=str(source)))
        data.unlink()
        with pytest.raises(FileNotFoundError):
            list(iter_corpus_docs([cid], splits=["test"]))


def test_classified_import_does_not_get_shadowed_by_existing_raw_data():
    cfg = gccf_ingest_config(name="classified", source_dir="export")
    assert cfg.global_dedup is False
    assert cfg.language_filter is False
    assert cfg.quality_filter is False
    assert cfg.selection_profile == "preserve"


@pytest.mark.parametrize("expected_documents", [1, 2, 3])
def test_failed_import_retries_and_audits_changed_records(tmp_path, expected_documents):
    from spalmer.data.discovery import discover_jsonl, iter_approved_jsonl
    from spalmer.data.gccf import EXPORT_SCHEMA
    from spalmer.data.manifest import normalize_for_dedup

    root = tmp_path / "export"
    source = root / "data" / "approved-source"
    source.mkdir(parents=True)
    receipt_hash = "a" * 64
    records = []
    for text in ("def retained(): return 1", "def changed(): return 2"):
        digest = hashlib.sha256(("code\0" + normalize_for_dedup(text, kind="code")).encode()).hexdigest()
        records.append(
            {
                "text": text,
                "kind": "code",
                "domain": "code",
                "language": "python",
                "source": "approved-source",
                "split": "train",
                "document_id": "gccf-" + digest,
                "provenance": {
                    "gccf_schema": "gccf.receipt/1",
                    "receipt_sha256": receipt_hash,
                    "verdict": "include",
                    "path": "source.parquet",
                    "locator": {"type": "row", "index": len(records)},
                },
            }
        )
    records[1]["text"] = "modified after export"
    from yylab.ingest.adapters.jsonl import _record_text

    with pytest.raises(ValueError, match="GCCF text changed after export"):
        _record_text(records[1])
    (source / "approved.jsonl").write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
    (root / "gccf-export.json").write_text(
        json.dumps(
            {
                "schema": EXPORT_SCHEMA,
                "audit": {"receipt_sha256": receipt_hash},
                "complete_source_scan": True,
                "corpora": {
                    "approved-source": {
                        "documents": expected_documents,
                        "shards": ["data/approved-source/approved.jsonl"],
                    }
                },
            }
        )
    )
    cid = db.new_id("cor")
    now = db.now_iso()
    db.execute(
        "INSERT INTO corpora(id,name,created_at,updated_at,status,source_path) VALUES(?,?,?,?,?,?)",
        (cid, "GCCF failed import", now, now, "failed", str(source.resolve())),
    )
    failed_data = corpus_dir(cid) / "data"
    failed_data.mkdir(parents=True)
    (failed_data / "shard-00002.jsonl.zst").write_bytes(b"partial prior import")

    if expected_documents == 1:
        with pytest.raises(ValueError, match="more records than its manifest"):
            import_export(lambda *a: None, root)
        assert db.q1("SELECT status FROM corpora WHERE id=?", (cid,))["status"] == "failed"
        assert not (root / "verified").exists()
        return
    result = import_export(lambda *a: None, root)
    assert result["corpora"] == {"approved-source": cid}
    assert result["integrity_rejections"] == 1
    assert result["unavailable_documents"] == expected_documents - 2
    assert not (failed_data / "shard-00002.jsonl.zst").exists()
    native = list(iter_corpus_docs([cid], splits=["train"]))
    assert [r["text"] for r in native] == [records[0]["text"]]
    rejected = json.loads((corpus_dir(cid) / "gccf-integrity-rejections.jsonl").read_text())
    assert rejected["document_id"] == records[1]["document_id"]
    assert "text" not in rejected
    assert rejected["locator"] == {"type": "row", "index": 1}
    docs = list(iter_approved_jsonl(discover_jsonl([Path(result["verified_directory"]) / "data"])))
    assert [(d.text, d.kind, d.language, d.document_id) for d in docs] == [
        (records[0]["text"], "code", "python", records[0]["document_id"])
    ]
