from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from yylab.train import prepared
from yylab.train.cached_spalmer import CachedSPALMERBatchSource
from yylab.train.data import PREPARED_FIELD


class ToyTokenizer:
    identity = SimpleNamespace(fingerprint="cached-spalmer-test-v1")
    vocab_size = 64
    eos_token_id = 2
    pad_token_id = 0


@pytest.fixture
def cache_factory(tmp_path):
    def make(cid, records, *, objective="causal_lm"):
        root = tmp_path / cid
        root.mkdir()
        writer = prepared._ShardWriter(root, 20, False)
        connection = sqlite3.connect(root / "records.sqlite3")
        connection.execute("CREATE TABLE records(ordinal INTEGER PRIMARY KEY, metadata TEXT NOT NULL)")
        for ordinal, (split, units) in enumerate(records):
            record = {"split": split, PREPARED_FIELD: {"units": [writer.write(ids) for ids in units]}}
            connection.execute("INSERT INTO records VALUES (?, ?)", (ordinal, json.dumps(record)))
        connection.commit()
        connection.close()
        writer.close()
        index = root / "records.sqlite3"
        manifest = {
            "schema": prepared.SCHEMA,
            "identity": {
                "source": {"corpus_id": cid},
                "tokenizer": prepared.tokenizer_fingerprint(ToyTokenizer()),
                "objective": objective,
            },
            "documents": len(records),
            "tokens": writer.tokens,
            "files": [*writer.files, {
                "name": index.name, "bytes": index.stat().st_size,
                "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            }],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        return root
    return make


def make_source(paths, weights=None, **kwargs):
    return CachedSPALMERBatchSource(
        paths, weights=weights or dict.fromkeys(paths, 1), tokenizer=ToyTokenizer(),
        shuffle_buffer=3, read_chunk_tokens=2, **kwargs,
    )


def test_train_only_eos_per_document_overlap_and_bridge_stats(cache_factory):
    path = cache_factory("prose", [
        ("train", [[10, 11], [12]]), ("validation", [[40]]), ("test", [[41]]), (None, [[42]]),
    ])
    source = make_source({"prose": path})
    try:
        batch = source.next_batch(batch_size=3, sequence_length=3)
        assert batch.input_ids.tolist() == [[10, 11, 12, 2], [2, 10, 11, 12], [12, 2, 10, 11]]
        assert batch.labels is None and batch.state_reset_mask is None
        assert batch.strata == ("prose",) * 3
        stats = source.data_stats()
        assert stats["sources"]["prose"]["excluded_nontrain_documents"] == 3
        assert stats["tokens"]["rendered_token_positions"] == 10
        assert stats["tokens"]["packing_replay_inputs"] == 2
        assert stats["tokens"]["optimizer_supervised_targets"] == 9
        assert source.tokens_seen == 12 and source.discarded_incomplete_batches == 0
        assert source.docs_seen == 3
        from yylab.train.spalmer import _tokenized_view_manifest

        view = _tokenized_view_manifest(source, sequence_length=3)
        assert all(view["reconciliation"].values())
        assert view["rendered_tokens"] == 10
        assert view["prepared_tokens"]["prose"]["train_documents"] == 1
    finally:
        source.close()
    with pytest.raises(RuntimeError, match="closed"):
        source.next_batch(batch_size=3, sequence_length=3)


@pytest.mark.parametrize("weights,counts", [({"p": 70, "k": 15, "r": 15}, {"p": 14, "k": 3, "r": 3}),
                                          ({"p": .7, "k": .3}, {"p": 7, "k": 3})])
def test_exact_mixture_survives_unequal_source_exhaustion(cache_factory, weights, counts):
    paths = {cid: cache_factory(cid, [("train", [[10 + index] * (index * 20 + 1)])])
             for index, cid in enumerate(weights)}
    source = make_source(paths, weights)
    rng_before = random.getstate()
    try:
        for _ in range(3):
            batch = source.next_batch(batch_size=sum(counts.values()), sequence_length=7)
            assert Counter(batch.strata) == counts
        assert random.getstate() == rng_before
        stats = source.data_stats()
        assert stats["cycles_completed"] == 3
        for cid, count in counts.items():
            assert stats["sources"][cid]["served_target_tokens"] == 3 * count * 7
        assert stats["sources"]["p"]["epochs_completed"] > 1
    finally:
        source.close()


def test_shuffle_and_midcycle_checkpoint_replay_are_exact(cache_factory):
    paths = {cid: cache_factory(cid, [("train", [[10 + i, 20 + i]]) for i in range(7)])
             for cid in ("p", "k", "r")}
    weights = {"p": .7, "k": .15, "r": .15}
    first, restored = make_source(paths, weights), make_source(paths, weights)
    try:
        for _ in range(2):
            first.next_batch(batch_size=7, sequence_length=9)
        state = first.state_dict()
        expected = [first.next_batch(batch_size=7, sequence_length=9) for _ in range(3)]
        restored.load_state_dict(json.loads(json.dumps(state)))
        actual = [restored.next_batch(batch_size=7, sequence_length=9) for _ in range(3)]
        for left, right in zip(actual, expected, strict=True):
            assert torch.equal(left.input_ids, right.input_ids) and left.strata == right.strata
        assert first.state_dict() == restored.state_dict()
        assert first.data_stats() == restored.data_stats()
        with pytest.raises(ValueError, match="shape"):
            restored.next_batch(batch_size=8, sequence_length=9)
    finally:
        first.close()
        restored.close()


def test_rejects_noncausal_empty_train_identity_and_tokenizer(cache_factory):
    plain = cache_factory("p", [("train", [[10]])])
    chat = cache_factory("chat", [("train", [[10]])], objective="chat_sft")
    validation = cache_factory("validation", [("validation", [[10]])])
    with pytest.raises(ValueError, match="causal supervision"):
        make_source({"chat": chat})
    with pytest.raises(ValueError, match="explicit train"):
        make_source({"validation": validation})
    with pytest.raises(ValueError, match="corpus identity"):
        make_source({"wrong": plain})
    wrong = ToyTokenizer()
    wrong.identity = SimpleNamespace(fingerprint="different")
    with pytest.raises(ValueError, match="actual tokenizer"):
        CachedSPALMERBatchSource({"p": plain}, weights={"p": 1}, tokenizer=wrong)


def test_out_of_range_token_aborts_batch_and_checkpoint(cache_factory):
    path = cache_factory("bad", [("train", [[10, 999]])])
    source = make_source({"bad": path})
    try:
        with pytest.raises(ValueError, match="out-of-vocabulary"):
            source.next_batch(batch_size=1, sequence_length=4)
        with pytest.raises(RuntimeError, match="partially failed"):
            source.state_dict()
        with pytest.raises(RuntimeError, match="failed"):
            source.next_batch(batch_size=1, sequence_length=4)
    finally:
        source.close()
