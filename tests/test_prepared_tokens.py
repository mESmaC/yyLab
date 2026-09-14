from __future__ import annotations

import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest
import zstandard as zstd
from test_train import NonPrefixStableQwenLikeTokenizer, TemplateTokenizer

from yylab.train import data, prepared
from yylab.train.data import ChatSFTBatches, PackedBatches, iter_corpus_docs
from yylab.train.prefetch import BufferedBatches, batch_stream


class ByteTokenizer:
    identity = SimpleNamespace(fingerprint="test-byte-codec-v1")
    eos_token_id = 1
    pad_token_id = 0

    def __init__(self):
        self.calls = 0

    def tokenize(self, texts):
        self.calls += 1
        return [[70000 + value for value in text.encode()] for text in texts]

    def __call__(self, texts, **kwargs):
        return {"input_ids": self.tokenize(texts)}


@pytest.fixture
def corpus_factory(tmp_path, monkeypatch):
    def root(cid):
        return tmp_path / "corpora" / cid

    monkeypatch.setattr(data, "corpus_dir", root)
    monkeypatch.setattr(prepared, "corpus_dir", root)

    def create(cid, records):
        directory = root(cid) / "data"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "shard-00000.jsonl.zst"
        path.write_bytes(zstd.ZstdCompressor().compress(b"\n".join(json.dumps(row).encode() for row in records)))
        return path

    return create


def make_store(cid, tokenizer, tmp_path, **kwargs):
    return prepared.prepare_corpus(
        cid, tokenizer, getattr(tokenizer, "tokenize", lambda _: pytest.fail("unexpected causal tokenizer")),
        cache_dir=tmp_path / "cache", max_text_chars=11, tokenize_chunk=2, shard_bytes=73,
        progress=lambda _: None, **kwargs,
    )


@pytest.mark.parametrize("mix", ["concatenate", "round_robin", "token_balanced"])
@pytest.mark.parametrize("byte_budget", [1, 1200, 64 * 2**20])
@pytest.mark.parametrize("dedup", [True, False])
def test_cached_causal_stream_preserves_mixing_packing_and_accounting(corpus_factory, tmp_path, mix, byte_budget, dedup):
    rows = [
        {"text": "α beta " * 9, "domain": "language", "split": "train"},
        {"text": "same bytes", "domain": "code", "split": "train"},
        {"text": "short", "domain": "language", "split": "validation"},
        {"text": "", "domain": "language", "split": "train"},
        {"text": "last document " * 5, "domain": "language", "split": "train"},
    ]
    corpus_factory("a", rows)
    corpus_factory("b", list(reversed(rows)))
    tokenizer = ByteTokenizer()
    stores = {cid: make_store(cid, tokenizer, tmp_path) for cid in ("a", "b")}
    reader = prepared.PreparedCorpora(stores)
    for epoch in range(2):
        options = dict(
            seed=17 + epoch, shuffle_buffer=6, shuffle_buffer_bytes=byte_budget,
            corpus_mix=mix, cross_corpus_dedup=dedup, annotate_source=True,
            shard_index=1, num_shards=2, splits=["train"],
        )
        if mix == "token_balanced":
            options.update(corpus_weights={"a": 2, "b": 1}, source_max_ratio=0.7)

        def pack(corpus_reader=None):
            return PackedBatches(
                iter_corpus_docs(["a", "b"], corpus_reader=corpus_reader, **options),
                tokenizer.tokenize, seq_len=13, micro_batch=3, eos_token_id=1,
                max_text_chars=11, tokenize_chunk=2,
            )

        live, cached = pack(), pack(reader)
        assert list(live) == list(cached)
        assert live.data_stats() == cached.data_stats()
    for store in stores.values():
        assert max(row["bytes"] for row in store.manifest["files"] if row["name"].endswith(".bin")) <= 73
        with sqlite3.connect(store.root / "records.sqlite3") as connection:
            for (record,) in connection.execute("SELECT metadata FROM records"):
                assert "text" not in json.loads(record)
                assert "messages" not in json.loads(record)
    reader.close()


@pytest.mark.parametrize("tokenizer_class", [TemplateTokenizer, NonPrefixStableQwenLikeTokenizer])
@pytest.mark.parametrize("seq_len", [4, 9, 24])
def test_cached_chat_masks_and_render_failures_match_live(corpus_factory, tmp_path, tokenizer_class, seq_len):
    corpus_factory("chat", [
        {"text": "user: ask\nassistant: " + "a long answer " * 12},
        {"text": "flat metadata", "messages": [
            {"role": "user", "content": "first"}, {"role": "assistant", "content": "one two three"},
            {"role": "user", "content": "again"}, {"role": "assistant", "content": "last reply"},
        ]},
        {"text": "malformed", "messages": []},
        {"text": "ordinary prose"},
    ])
    tokenizer = tokenizer_class()
    tokenizer.identity = SimpleNamespace(fingerprint=tokenizer_class.__name__)
    store = make_store("chat", tokenizer, tmp_path, objective="chat_sft")
    reader = prepared.PreparedCorpora({"chat": store})

    def pack(reader=None):
        docs = iter_corpus_docs(["chat"], seed=6, annotate_source=True, corpus_reader=reader)
        return ChatSFTBatches(docs, tokenizer, seq_len=seq_len, micro_batch=2)

    live, cached = pack(), pack(reader)
    assert list(live) == list(cached)
    assert live.data_stats() == cached.data_stats()
    payload = sum(row["bytes"] for row in store.manifest["files"] if row["name"].endswith(".bin"))
    assert payload == store.manifest["tokens"] * 5
    reader.close()


def test_reuse_and_source_tokenizer_invalidation(corpus_factory, tmp_path):
    corpus_factory("a", [{"text": "initial content"}])
    tokenizer = ByteTokenizer()
    first = make_store("a", tokenizer, tmp_path)
    calls = tokenizer.calls
    second = make_store("a", tokenizer, tmp_path)
    assert second.root == first.root
    assert tokenizer.calls == calls
    corpus_factory("a", [{"text": "changed content with a different size"}])
    third = make_store("a", tokenizer, tmp_path)
    assert third.root != first.root
    tokenizer.identity = SimpleNamespace(fingerprint="test-byte-codec-v2")
    fourth = make_store("a", tokenizer, tmp_path)
    assert fourth.root != third.root
    for store in (first, second, third, fourth):
        store.close()


def test_hf_identity_covers_normalizer_template_and_native_wrapper():
    from tokenizers import Tokenizer, models, normalizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    initial = prepared.tokenizer_fingerprint(tokenizer)
    assert prepared.tokenizer_fingerprint(SimpleNamespace(tokenizer=tokenizer)) == initial
    tokenizer.backend_tokenizer.normalizer = normalizers.Lowercase()
    normalized = prepared.tokenizer_fingerprint(tokenizer)
    assert normalized != initial
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    assert prepared.tokenizer_fingerprint(tokenizer) != normalized


@pytest.mark.parametrize("file_kind", ["tokens", "records"])
def test_corrupted_cache_fails_closed(corpus_factory, tmp_path, file_kind):
    corpus_factory("a", [{"text": "hello"}])
    store = make_store("a", ByteTokenizer(), tmp_path)
    target = next(store.root.glob(f"{file_kind}*"))
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 1
    target.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum"):
        if file_kind == "records":
            prepared.TokenStore(store.root)
        else:
            next(store.records())[data.PREPARED_FIELD]["units"][0].read_ids()
    store.close()


def test_failed_build_does_not_publish_partial_cache(corpus_factory, tmp_path):
    corpus_factory("a", [{"text": "hello"}])
    tokenizer = ByteTokenizer()
    tokenizer.tokenize = lambda _: (_ for _ in ()).throw(RuntimeError("broken tokenizer"))
    with pytest.raises(RuntimeError, match="broken tokenizer"):
        make_store("a", tokenizer, tmp_path)
    assert not list((tmp_path / "cache").iterdir())


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_prefetch_consumption_stats_early_close_and_errors(depth):
    class Batches:
        def __init__(self):
            self.docs_seen = 0
            self.closed = False

        def data_stats(self):
            return {"docs_seen": self.docs_seen}

        def __iter__(self):
            try:
                for index in range(10):
                    self.docs_seen += 1
                    yield {"input_ids": [[index, index + 1]]}
                raise ValueError("producer failed")
            finally:
                self.closed = True

    raw = Batches()
    batches = BufferedBatches(raw, depth)
    with batch_stream(batches) as iterator:
        assert next(iterator)["input_ids"] == [[0, 1]]
        assert batches.docs_seen == 1
        assert batches.data_stats()["docs_seen"] == 1
    assert raw.closed
    assert raw.docs_seen <= depth + 2
    assert not any(thread.name == "yylab-data-prefetch" for thread in threading.enumerate())
    with pytest.raises(ValueError, match="producer failed"):
        list(BufferedBatches(Batches(), depth))


def test_worker_defaults_to_prepared_reuses_across_packing_configs(corpus_factory, tmp_path):
    from yylab.workers.train import build_batches

    corpus_factory("a", [{"text": "example training stream " * 30}])
    tokenizer = ByteTokenizer()
    cfg = {"corpora": ["a"], "seq_len": 16, "micro_batch": 1, "grad_accum": 1, "token_cache_dir": str(tmp_path / "cache")}
    first = build_batches(cfg, tokenizer, 3)
    assert len(list(first)) == 3
    calls = tokenizer.calls
    cached = build_batches({**cfg, "seq_len": 9, "seed": 12, "epochs": 2}, tokenizer, 7)
    live = build_batches({**cfg, "token_cache": False, "seq_len": 9, "seed": 12, "epochs": 2}, tokenizer, 7)
    actual = list(cached)
    assert tokenizer.calls == calls
    assert actual == list(live)
    assert cached.data_stats()["prepared_tokens"]["a"]["tokens"] > 0


def test_text_pointer_snapshot_tracks_original_files(corpus_factory, tmp_path):
    root = prepared.corpus_dir("text")
    root.mkdir(parents=True)
    source = tmp_path / "book.txt"
    source.write_text("original prose", encoding="utf-8")
    index = tmp_path / "text-index"
    index.mkdir()
    (index / "text-index.json").write_text(json.dumps({"files": [{"path": str(source)}]}))
    (root / "text-source.json").write_text(json.dumps({"index_directory": str(index)}))
    first = prepared.source_snapshot("text", verify="sha256")
    source.write_text("modified prose", encoding="utf-8")
    assert prepared.source_snapshot("text", verify="sha256") != first
    assert first["sources"][0]["path"] == str(source)


def test_gccf_snapshot_uses_absolute_sources_and_selection(corpus_factory, tmp_path):
    root = prepared.corpus_dir("gccf")
    root.mkdir(parents=True)
    index = tmp_path / "gccf-index"
    index.mkdir()
    source = tmp_path / "dataset.jsonl"
    source.write_text('{"text":"approved"}', encoding="utf-8")
    with sqlite3.connect(index / "index.sqlite3") as connection:
        connection.execute("CREATE TABLE files(id INTEGER, source TEXT, entry TEXT)")
        connection.execute("INSERT INTO files VALUES(1, 'selected', ?)", (json.dumps({"abs_path": str(source), "path": "relative/dataset.jsonl"}),))
        connection.execute("INSERT INTO files VALUES(2, 'different', ?)", (json.dumps({"abs_path": "not-present"}),))
    (index / "gccf-index.json").write_text("{}")
    (root / "gccf-source.json").write_text(json.dumps({"index_directory": str(index), "source": "selected"}))
    snapshot = prepared.source_snapshot("gccf")
    assert [row["path"] for row in snapshot["sources"]] == [str(source)]


def test_native_builder_preserves_n_plus_one_width_and_resume(corpus_factory, tmp_path):
    from spalmer.training import CausalBatch
    from test_spalmer_training import _config

    from yylab.train.spalmer import build_spalmer_batch_source

    corpus_factory("a", [{"text": "repeatable training data " * 80}])
    tokenizer = ByteTokenizer()
    tokenizer.special_tokens = SimpleNamespace(eos_id=1)
    tokenizer.tokenizer = tokenizer.__call__
    cfg = _config(corpora=["a"], max_steps=8, grad_accum=2, token_cache_dir=str(tmp_path / "cache"))
    first = build_spalmer_batch_source(cfg, tokenizer, causal_batch_type=CausalBatch)
    batch = first.next_batch(batch_size=1, sequence_length=8)
    assert batch.input_ids.shape == (1, 9)
    state = first.state_dict()
    expected = [first.next_batch(batch_size=1, sequence_length=8).input_ids.tolist() for _ in range(5)]
    first.close()
    calls = tokenizer.calls
    resumed = build_spalmer_batch_source(cfg, tokenizer, causal_batch_type=CausalBatch)
    resumed.load_state_dict(state)
    assert [resumed.next_batch(batch_size=1, sequence_length=8).input_ids.tolist() for _ in range(5)] == expected
    assert resumed.consumed_batches == 6
    assert tokenizer.calls == calls
    resumed.close()
