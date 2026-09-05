from __future__ import annotations

import json

from yylab import db
from yylab.artifacts import corpus_dir
from yylab.ingest.gccf import gccf_ingest_config, import_export, run_gccf_ingest
from yylab.train.data import corpus_data_provenance, iter_corpus_docs


def test_receipt_to_yylab_training_preserves_selection_splits_and_provenance(tmp_path):
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
    result = run_gccf_ingest(lambda *a: None, str(r), output=str(output), workers=1)
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
    assert import_export(lambda *a: None, output)["corpora"] == result["corpora"]


def test_classified_import_does_not_get_shadowed_by_existing_raw_data():
    cfg = gccf_ingest_config(name="classified", source_dir="export")
    assert cfg.global_dedup is False
    assert cfg.language_filter is False
    assert cfg.quality_filter is False
    assert cfg.selection_profile == "preserve"
