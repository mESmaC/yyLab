from __future__ import annotations

import hashlib
import json

import pytest

from yylab import db
from yylab.artifacts import corpus_dir
from yylab.ingest import gccf
from yylab.ingest.gccf import gccf_ingest_config, import_export, run_gccf_ingest
from yylab.train.data import corpus_data_provenance, iter_corpus_docs


@pytest.mark.parametrize("options", [{}, {"materialize": False}])
def test_receipt_to_yylab_training_preserves_selection_splits_and_provenance(tmp_path, options):
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
    source_digest = hashlib.sha256(data.read_bytes()).hexdigest()
    output = tmp_path / "source-index"
    result = run_gccf_ingest(lambda *a: None, str(r), output=str(output), workers=1, **options)
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
    assert result["storage"] == "original_sources"
    assert result["text_bytes_copied"] == 0
    assert hashlib.sha256(data.read_bytes()).hexdigest() == source_digest
    assert not list(output.rglob("*.jsonl*"))
    assert not (corpus_dir(cid) / "data").exists()
    assert not (output / "data").exists()
    assert not (output / "verified").exists()
    assert not (output / "gccf-export.json").exists()
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


def _forbid_import_side_effects(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("retired GCCF import reached dependency loading, receipt inspection or registry I/O")

    monkeypatch.setattr(gccf, "_receipt_tools", unexpected)
    monkeypatch.setattr(gccf, "corpus_dir", unexpected)
    for name in ("init_db", "execute", "q1", "new_id"):
        monkeypatch.setattr(gccf.db, name, unexpected)
    return unexpected


def test_materialize_request_fails_before_receipt_or_output_access(tmp_path, monkeypatch):
    unexpected = _forbid_import_side_effects(monkeypatch)
    output = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="materialization is unsupported"):
        run_gccf_ingest(
            unexpected, str(tmp_path / "missing-receipt.json"), output=str(output), materialize=True
        )
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_retired_export_import_leaves_existing_files_and_corpus_unchanged(tmp_path, monkeypatch):
    import zstandard as zstd

    root = tmp_path / "old-export"
    source = root / "data" / "approved-source"
    source.mkdir(parents=True)
    (root / "gccf-export.json").write_text('{"schema":"old-export"}')
    (source / "approved.jsonl").write_text('{"text":"Existing exported text."}\n')
    cid = db.new_id("cor")
    now = db.now_iso()
    db.execute(
        "INSERT INTO corpora(id,name,created_at,updated_at,status,source_path) VALUES(?,?,?,?,?,?)",
        (cid, "Existing native GCCF corpus", now, now, "ready", str(source.resolve())),
    )
    native = corpus_dir(cid)
    (native / "data").mkdir(parents=True)
    row = {"text": "Existing native corpus remains readable.", "domain": "language", "split": "train"}
    shard = native / "data/shard-00000.jsonl.zst"
    shard.write_bytes(zstd.ZstdCompressor().compress((json.dumps(row) + "\n").encode()))
    (native / "stats.json").write_text(json.dumps({"shards": [shard.name], "docs_written": 1}))
    files = [p for directory in (root, native) for p in directory.rglob("*") if p.is_file()]
    before = {p: p.read_bytes() for p in files}
    registry_before = dict(db.q1("SELECT * FROM corpora WHERE id=?", (cid,)))
    with monkeypatch.context() as patch:
        unexpected = _forbid_import_side_effects(patch)
        with pytest.raises(ValueError, match="export copying is unsupported"):
            import_export(unexpected, root)
    assert {p: p.read_bytes() for p in files} == before
    assert set(p for directory in (root, native) for p in directory.rglob("*") if p.is_file()) == set(before)
    assert dict(db.q1("SELECT * FROM corpora WHERE id=?", (cid,))) == registry_before
    assert [r["text"] for r in iter_corpus_docs([cid], splits=["train"])] == [row["text"]]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--materialize"],
        ["missing-receipt.json", "--materialize"],
        ["missing-receipt.json", "--materialize=true"],
        ["--export-dir"],
        ["--export-dir", "existing-export"],
        ["--export-dir=existing-export"],
    ],
)
def test_retired_cli_options_fail_before_database_initialization(arguments, monkeypatch, capsys):
    _forbid_import_side_effects(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        gccf.main(arguments)
    assert exc.value.code == 2
    assert "is unsupported: GCCF receipts reference original source data without copying it" in capsys.readouterr().err


def test_gccf_cli_accepts_only_receipt_index_registration(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(db, "init_db", lambda: calls.append("init"))

    def register(progress, receipt_path, **options):
        calls.append((receipt_path, options))
        return {"storage": "original_sources", "text_bytes_copied": 0}

    monkeypatch.setattr(gccf, "run_gccf_ingest", register)
    receipt = tmp_path / "receipt.json"
    output = tmp_path / "index"
    gccf.main([str(receipt), "--output", str(output), "--limit-per-file", "2", "--workers", "1"])
    assert calls == ["init", (str(receipt), {"output": str(output), "limit_per_file": 2, "workers": 1})]
    assert json.loads(capsys.readouterr().out)["text_bytes_copied"] == 0
