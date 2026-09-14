from __future__ import annotations

import json

import pytest

from yylab import db
from yylab.artifacts import corpus_dir
from yylab.ingest.pipeline import IngestConfig, reimport_corpus
from yylab.ingest.text_source import run_text_source
from yylab.train.data import corpus_data_provenance, corpus_training_preflight, iter_corpus_docs


def test_original_prose_available_to_training_with_raw_provenance_and_holdouts(tmp_path):
    source = tmp_path / "books"
    source.mkdir()
    for split in ("train", "validation", "test"):
        (source / (split + ".txt")).write_text(
            f"The {split} book has a distinct story.\n" * 30, encoding="utf-8"
        )
    output = tmp_path / "index"
    selection = {"validation_files": ["validation.txt"], "test_files": ["test.txt"]}
    result = run_text_source(lambda *args: None, source, output=output, **selection)
    cid = result["corpus_id"]
    assert result["classified"] is False and result["text_bytes_copied"] == 0
    assert db.q1("SELECT status FROM corpora WHERE id=?", (cid,))["status"] == "ready"
    assert not (corpus_dir(cid) / "data").exists()
    assert not list(output.rglob("*.txt")) and not list(output.rglob("*.jsonl*"))
    assert json.loads((corpus_dir(cid) / "text-source.json").read_text())["index_directory"] == str(
        output.resolve()
    )
    for split in ("train", "validation", "test"):
        rows = list(iter_corpus_docs([cid], splits=[split]))
        assert len(rows) == 1 and rows[0]["split"] == split
        assert rows[0]["source"].endswith(split + ".txt")
        assert rows[0]["provenance"]["classified"] is False
        assert not corpus_training_preflight([cid], splits=[split])["errors"]
    provenance = corpus_data_provenance([cid])["corpora"][0]
    assert provenance["catalog_status"] == "raw_source_index"
    assert provenance["source_index"]["classified"] is False
    assert not provenance["gccf"]
    assert run_text_source(lambda *args: None, source, output=output, **selection)["corpus_id"] == cid
    with pytest.raises(ValueError, match="requested source selection"):
        run_text_source(lambda *args: None, source, output=output, test_files=["validation.txt"])
    with pytest.raises(ValueError, match="Re-index"):
        reimport_corpus(lambda *args: None, cid, IngestConfig(name="invalid", source_dir=str(source)))
    (source / "train.txt").unlink()
    with pytest.raises(FileNotFoundError):
        list(iter_corpus_docs([cid], splits=["train"]))


def test_cleaning_profile_cannot_reuse_raw_index_and_defaults_to_preprocessed_text(tmp_path):
    source = tmp_path / "books"
    source.mkdir()
    raw = b"A book paragraph has Windows line endings.\r\n\r\n\r\n\r\nSecond paragraph.\r\n"
    (source / "book.txt").write_bytes(raw)
    old = run_text_source(lambda *args: None, source, output=tmp_path / "raw", preprocessing=None)
    with pytest.raises(ValueError, match="requested source selection"):
        run_text_source(lambda *args: None, source, output=tmp_path / "raw")
    new = run_text_source(lambda *args: None, source, output=tmp_path / "clean")
    assert old["corpus_id"] != new["corpus_id"]
    old_text = next(iter_corpus_docs([old["corpus_id"]]))["text"]
    clean = next(iter_corpus_docs([new["corpus_id"]]))
    assert old_text == raw.decode()
    assert clean["text"] == "A book paragraph has Windows line endings.\n\nSecond paragraph.\n"
    assert clean["provenance"]["preprocessing"] == new["preprocessing"]
    assert (source / "book.txt").read_bytes() == raw
