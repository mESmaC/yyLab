from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np
import pytest

from yylab.train.data import CORPUS_ID_FIELD, PREPARED_FIELD, PackedBatches
from yylab.train.sprint_packing import SprintCausalBatches, make_sprint_tokenize


def tokenize(texts):
    # Tokens depend on both the text and its chunk-local position, so changing
    # the existing long-document chunk boundaries fails the equivalence check.
    return [[ord(char) + 7 * index for index, char in enumerate(text)] for text in texts]


@dataclass
class PreparedUnit:
    ids: list[int]

    def read_ids(self):
        return list(self.ids)


def compare_packers(docs, *, seq_len=7, eos_token_id=2, max_rows=None,
                    rows_per_chunk=3, tokenize_chunk=2, max_text_chars=11,
                    tokenization_batch_size=128):
    options = dict(seq_len=seq_len, eos_token_id=eos_token_id,
                   tokenize_chunk=tokenize_chunk, max_text_chars=max_text_chars)
    reference = PackedBatches(docs, tokenize, micro_batch=1, max_batches=max_rows, **options)
    fast = SprintCausalBatches(docs, tokenize, max_rows=max_rows,
                               rows_per_chunk=rows_per_chunk,
                               tokenization_batch_size=tokenization_batch_size, **options)
    expected = [row for batch in reference for row in batch["input_ids"]]
    chunks = list(fast)
    actual = [row.tolist() for batch in chunks for row in batch["input_ids"]]
    assert actual == expected
    assert fast.data_stats() == reference.data_stats()
    assert fast.sequences_made == reference.sequences_made
    for batch in chunks:
        array = batch["input_ids"]
        assert array.dtype == np.dtype(np.int64)
        assert array.flags.c_contiguous and array.flags.owndata
        assert array.shape[1] == seq_len
        assert 0 < array.shape[0] <= rows_per_chunk
    return actual, fast


@pytest.mark.parametrize("seq_len", [1, 2, 7, 32])
@pytest.mark.parametrize("rows_per_chunk", [1, 3, 64])
@pytest.mark.parametrize("eos_token_id", [None, 2])
@pytest.mark.parametrize("max_rows", [None, 1, 4, 17])
def test_causal_rows_and_source_accounting_match(seq_len, rows_per_chunk, eos_token_id, max_rows):
    docs = [
        {"text": "", CORPUS_ID_FIELD: "empty"},
        {"text": "one", CORPUS_ID_FIELD: "prose"},
        {"text": "a longer document α and β " * 17, CORPUS_ID_FIELD: "math"},
        {"text": "x", CORPUS_ID_FIELD: "prose"},
        {"text": "last short record"},
    ]
    compare_packers(docs, seq_len=seq_len, rows_per_chunk=rows_per_chunk,
                    eos_token_id=eos_token_id, max_rows=max_rows)


@pytest.mark.parametrize("max_rows", [None, 1, 3, 17])
def test_prepared_units_match_raw_chunk_boundaries(max_rows):
    raw = [
        {"text": "", CORPUS_ID_FIELD: "a"},
        {"text": "prepared text " * 8, CORPUS_ID_FIELD: "b"},
        {"text": "tail", CORPUS_ID_FIELD: "a"},
    ]
    prepared = []
    for doc in raw:
        text = doc["text"]
        parts = [text[start : start + 11] for start in range(0, max(len(text), 1), 11)]
        prepared.append({
            CORPUS_ID_FIELD: doc[CORPUS_ID_FIELD],
            PREPARED_FIELD: {"units": [PreparedUnit(ids) for ids in tokenize(parts)]},
        })
    raw_rows, raw_packer = compare_packers(raw, max_rows=max_rows)
    prepared_rows, prepared_packer = compare_packers(prepared, max_rows=max_rows)
    assert prepared_rows == raw_rows
    assert prepared_packer.data_stats() == raw_packer.data_stats()


def test_budget_keeps_exact_reference_read_ahead_statistics():
    rows, packer = compare_packers(
        [{"text": "abcdefghijklmnop", CORPUS_ID_FIELD: "one"},
         {"text": "second", CORPUS_ID_FIELD: "two"}],
        max_rows=1, seq_len=4, max_text_chars=100, tokenize_chunk=2,
    )
    assert len(rows) == 1
    assert packer.docs_seen == 2  # tokenizer batch read-ahead
    assert packer.tokens_seen == 16  # only the first encoding was consumed
    assert packer.boundary_tokens == 1
    assert packer.data_stats()["tokens"]["unforwarded_rendered_positions"] == 13


def test_tokenization_and_output_buffers_stay_bounded():
    calls = []

    def checked_tokenize(texts):
        calls.append([len(text) for text in texts])
        return tokenize(texts)

    packer = SprintCausalBatches(
        [{"text": "x" * 100_000}], checked_tokenize,
        seq_len=32, eos_token_id=2, max_rows=17,
        rows_per_chunk=5, tokenize_chunk=3, max_text_chars=100,
    )
    chunks = list(packer)
    assert [batch["input_ids"].shape[0] for batch in chunks] == [5, 5, 5, 2]
    assert calls and max(map(len, calls)) <= 128
    assert all(len(call) % 3 == 0 for call in calls)
    assert max(size for call in calls for size in call) <= 100
    assert all(batch["input_ids"].nbytes <= 5 * 32 * 8 for batch in chunks)


def test_eos_once_per_original_document_and_incomplete_tail_dropped():
    rows, packer = compare_packers(
        [{"text": "abcdefg"}, {"text": "hijkl"}],
        seq_len=1, max_text_chars=3, eos_token_id=1,
    )
    assert [row[0] for row in rows].count(1) == 2
    assert packer.boundary_tokens == 2
    rows, packer = compare_packers([{"text": "abcde"}], seq_len=4, eos_token_id=2)
    assert len(rows) == 1
    assert packer.data_stats()["tokens"]["unforwarded_rendered_positions"] == 2


def test_prepared_documents_with_no_units_and_no_rows():
    compare_packers([{PREPARED_FIELD: {"units": []}, CORPUS_ID_FIELD: "empty"}])
    compare_packers([])
    compare_packers([{"text": "x"}], seq_len=32)


@pytest.mark.parametrize("kwargs", [{"rows_per_chunk": 0}, {"max_rows": 0}, {"max_rows": -1},
                                    {"tokenization_batch_size": 0}])
def test_invalid_chunk_or_budget_rejected(kwargs):
    with pytest.raises(ValueError, match="positive"):
        SprintCausalBatches([], tokenize, seq_len=4, eos_token_id=2, **kwargs)


@pytest.mark.parametrize("logical_chunk", [1, 3, 8])
@pytest.mark.parametrize("physical_batch", [1, 8, 32, 128])
@pytest.mark.parametrize("prepared", [False, True])
def test_larger_physical_batches_keep_logical_read_ahead(logical_chunk, physical_batch, prepared):
    rng = Random(17)
    docs = []
    for index in range(80):
        row = {CORPUS_ID_FIELD: str(index % 3)}
        text = "abc " * rng.randrange(40)
        if prepared:
            parts = [text[start : start + 11] for start in range(0, len(text), 11)]
            row[PREPARED_FIELD] = {"units": [PreparedUnit(ids) for ids in tokenize(parts)]}
        else:
            row["text"] = text
        docs.append(row)
    for budget in [1, 2, 7, 13, 80, 1000, None]:
        compare_packers(docs, max_rows=budget, tokenize_chunk=logical_chunk,
                        tokenization_batch_size=physical_batch)


@pytest.mark.parametrize("empty_position", [0, 2, 3, 8, 10])
@pytest.mark.parametrize("budget", [1, 3, 8, None])
def test_empty_prepared_documents_at_logical_boundaries(empty_position, budget):
    docs = [{PREPARED_FIELD: {"units": [PreparedUnit([42] * 7)]}} for _ in range(10)]
    docs[empty_position:empty_position] = [
        {PREPARED_FIELD: {"units": []}, CORPUS_ID_FIELD: "empty"} for _ in range(25)
    ]
    compare_packers(docs, seq_len=8, tokenize_chunk=3, max_rows=budget,
                    tokenization_batch_size=128)


def test_text_batch_bound_uses_configured_chunk_size():
    packer = SprintCausalBatches([], tokenize, seq_len=4, eos_token_id=2,
                                 tokenization_batch_size=4096)
    assert packer.tokenization_batch_size == 128
    packer = SprintCausalBatches([], tokenize, seq_len=4, eos_token_id=2,
                                 max_text_chars=2**20, tokenization_batch_size=4096)
    assert packer.tokenization_batch_size == 8


@pytest.mark.parametrize("split_special_tokens", [False, True])
def test_ids_only_fast_backend_preserves_bpe_unicode_added_tokens_and_identity(split_special_tokens):
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    from yylab.train.prepared import tokenizer_fingerprint

    texts = [
        "ASCII prose and arithmetic: 12 + 30 = 42.",
        "Unicode café naïve Ελληνικά 汉字 🙂\n\t spaces  ",
        "<|user|>quoted text<eos><|assistant|>An answer.<eos>",
        " new-token new-tokenish <eos> literal EOS inside prose",
        "",
    ]
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=350, special_tokens=["<unk>", "<eos>"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>")
    tokenizer.add_tokens([AddedToken("new-token", single_word=True, lstrip=True)])
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>"]})
    tokenizer.split_special_tokens = split_special_tokens
    expected = tokenizer(texts, add_special_tokens=False, return_attention_mask=False, verbose=False)["input_ids"]
    # The encoder clears these flags on a private clone, just as the standard
    # HF call does by default, without modifying the caller's artifact identity.
    tokenizer.backend_tokenizer.enable_padding(length=7)
    tokenizer.backend_tokenizer.enable_truncation(max_length=7)
    before = tokenizer_fingerprint(tokenizer)
    encoder = make_sprint_tokenize(tokenizer)
    assert encoder.__name__ == "ids_only"
    assert encoder.supports_batched_tokenization is True
    assert encoder(texts) == expected
    assert tokenizer_fingerprint(tokenizer) == before
    assert tokenizer.backend_tokenizer.padding is not None
    assert tokenizer.backend_tokenizer.truncation is not None


def test_ids_only_factory_preserves_custom_and_slow_tokenizer_calls():
    class CustomTokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs == dict(add_special_tokens=False, return_attention_mask=False, verbose=False)
            return {"input_ids": tokenize(texts)}

    tokenizer = CustomTokenizer()
    encoder = make_sprint_tokenize(tokenizer)
    assert not getattr(encoder, "supports_batched_tokenization", False)
    assert encoder(["abc", "def"]) == tokenize(["abc", "def"])

    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    class CustomFastTokenizer(PreTrainedTokenizerFast):
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[99] for _ in texts]}

    custom_fast = CustomFastTokenizer(tokenizer_object=Tokenizer(models.WordLevel({"x": 0})))
    encoder = make_sprint_tokenize(custom_fast)
    assert not getattr(encoder, "supports_batched_tokenization", False)
    assert encoder(["x"]) == [[99]]


def collect_with_error(packer):
    rows = []
    error = None
    try:
        for batch in packer:
            rows.extend(np.asarray(batch["input_ids"]).tolist())
    except Exception as exc:
        error = (type(exc), str(exc))
    return rows, error, packer.data_stats()


def compare_failure_streams(docs_factory, *, max_rows, encode=tokenize, rows_per_chunk=3):
    options = dict(seq_len=4, eos_token_id=2, tokenize_chunk=8, max_text_chars=100)
    reference = PackedBatches(docs_factory(), encode, micro_batch=1, max_batches=max_rows, **options)
    fast = SprintCausalBatches(docs_factory(), encode, max_rows=max_rows,
                               rows_per_chunk=rows_per_chunk, tokenization_batch_size=128, **options)
    actual, expected = collect_with_error(fast), collect_with_error(reference)
    assert actual == expected
    return actual


@pytest.mark.parametrize("valid_docs", [2, 8, 9, 16, 20])
@pytest.mark.parametrize("budget", [1, 4, 10, 11, 50])
def test_source_lookahead_errors_are_deferred_until_needed(valid_docs, budget):
    def docs():
        for _ in range(valid_docs):
            yield {"text": "abcd", CORPUS_ID_FIELD: "good"}
        raise OSError("source read failed")

    rows, error, _ = compare_failure_streams(docs, max_rows=budget)
    if valid_docs < 8:
        assert not rows and error is not None  # no complete logical group to encode
    if valid_docs >= 8 and budget == 1:
        assert len(rows) == 1 and error is None


@pytest.mark.parametrize("valid_docs", [2, 8, 9, 16])
@pytest.mark.parametrize("budget", [1, 10, 11, 50])
def test_malformed_lookahead_document_obeys_original_logical_boundary(valid_docs, budget):
    def docs():
        yield from ({"text": "abcd"} for _ in range(valid_docs))
        yield {"text": None}

    compare_failure_streams(docs, max_rows=budget)


@pytest.mark.parametrize("bad_index", [0, 2, 8, 10, 16])
@pytest.mark.parametrize("budget", [1, 10, 11, 50])
@pytest.mark.parametrize("prepared", [False, True])
def test_encoding_lookahead_failure_retries_only_needed_logical_groups(bad_index, budget, prepared):
    class BadUnit:
        def read_ids(self):
            raise ValueError("bad tokenization record")

    def docs():
        for index in range(24):
            if prepared:
                unit = BadUnit() if index == bad_index else PreparedUnit(tokenize(["abcd"])[0])
                yield {PREPARED_FIELD: {"units": [unit]}}
            else:
                yield {"text": "bad" if index == bad_index else "abcd"}

    def encode(texts):
        if "bad" in texts:
            raise ValueError("bad tokenization record")
        return tokenize(texts)

    compare_failure_streams(docs, max_rows=budget, encode=encode)
