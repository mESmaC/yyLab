"""A processed text view cannot reuse the raw view's cached token payload."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from spalmer.data.text_index import build_text_index, iter_indexed_text

from yylab.ingest import text_source
from yylab.train import data, prepared


class ByteTokenizer:
    identity = SimpleNamespace(fingerprint="text-profile-test-byte-codec-v1")

    def tokenize(self, texts):
        return [list(text.encode("utf-8")) for text in texts]


def test_profile_view_changes_token_cache_identity_and_preserves_split_metadata(tmp_path, monkeypatch):
    source = tmp_path / "books"
    source.mkdir()
    originals = {}
    for split in ("train", "validation", "test"):
        originals[split] = (
            f"The {split} reader followed the detailed account across the long summer\r\n"
            "evening, considering every sentence before turning to another page.\r\n\r\n"
        ).encode()
        (source / f"{split}.txt").write_bytes(originals[split])
    corpus_root = tmp_path / "corpora" / "books"
    corpus_root.mkdir(parents=True)
    for module in (text_source, data, prepared):
        monkeypatch.setattr(module, "corpus_dir", lambda cid: corpus_root)
    pointer = corpus_root / "text-source.json"
    tokenizer = ByteTokenizer()
    stores, snapshots, index_bytes = {}, {}, {}
    try:
        for name, profile in (("raw", None), ("processed", "prose_v1")):
            index = tmp_path / name
            build_text_index(
                source,
                index,
                validation_files=["validation.txt"],
                test_files=["test.txt"],
                preprocessing=profile,
            )
            index_bytes[name] = (index / "text-index.json").read_bytes()
            pointer.write_text(json.dumps({"index_directory": str(index)}), encoding="utf-8")
            snapshots[name] = prepared.source_snapshot("books")
            store = prepared.prepare_corpus(
                "books",
                tokenizer,
                tokenizer.tokenize,
                cache_dir=tmp_path / "cache",
                max_text_chars=19,
                shard_bytes=73,
                progress=lambda _: None,
            )
            stores[name] = store
            rows = {row["document_id"]: row for row in iter_indexed_text(index)}
            cached = list(store.records())
            assert len(cached) == len(rows) == 3
            assert {record["split"] for record in cached} == {"train", "validation", "test"}
            assert store.manifest["identity"]["source"] == snapshots[name]
            for record in cached:
                row = rows[record["sample_id"]]
                payload = row["text"].encode("utf-8")
                metadata = record[data.PREPARED_FIELD]
                assert metadata["text_sha256"] == hashlib.sha256(payload).hexdigest()
                assert bytes(token for unit in metadata["units"] for token in unit.read_ids()) == payload
                assert record["provenance"] == row["provenance"]
                assert record["split"] == row["split"]

        assert snapshots["raw"] != snapshots["processed"]
        assert stores["raw"].root != stores["processed"].root
        assert stores["raw"].manifest["tokens"] > stores["processed"].manifest["tokens"]
        for name, payload in index_bytes.items():
            assert (tmp_path / name / "text-index.json").read_bytes() == payload
        assert {path.stem: path.read_bytes() for path in source.glob("*.txt")} == originals
    finally:
        for store in stores.values():
            store.close()
