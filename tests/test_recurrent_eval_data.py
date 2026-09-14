from types import SimpleNamespace

import numpy as np
import pytest

from yylab.train import recurrent_eval_data as data


class Tokenizer:
    identity = SimpleNamespace(to_dict=lambda: {"fingerprint": "test"})

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return list(text.encode())


def test_frozen_depth_view_uses_validation_and_one_window_per_document(monkeypatch, tmp_path):
    calls = []

    def records(corpus_ids, **kwargs):
        calls.append((corpus_ids, kwargs))
        yield from [
            {"text": "abcdefgh", "source": "one"},
            {"text": "abcdefgh", "source": None},
            {"text": "ijklmnop", "source": "two"},
            {"text": "x", "source": "short"},
        ]

    monkeypatch.setattr(data, "iter_corpus_docs", records)
    monkeypatch.setattr(data, "corpus_data_dir", lambda _: tmp_path)
    sources = [{"name": "code", "corpus_id": "sample", "domains": ["code"]}]
    first, manifest = data.prepare_view(Tokenizer(), sources, documents=2, targets=4, pool=4)
    second, repeated = data.prepare_view(Tokenizer(), sources, documents=2, targets=4, pool=4)
    assert first.shape == (2, 5) and first.dtype == np.dtype("<u4")
    np.testing.assert_array_equal(first, second)
    assert manifest == repeated
    assert len({r["text_sha256"] for r in manifest["references"]}) == 2
    assert all(r["split"] == "validation" for r in manifest["references"])
    assert all(kwargs["splits"] == ["validation"] for _, kwargs in calls)
    assert all(kwargs["domains"] == ["code"] for _, kwargs in calls)


def test_frozen_depth_view_refuses_insufficient_validation(monkeypatch, tmp_path):
    def records(*args, **kwargs):
        yield {"text": "x"}

    monkeypatch.setattr(data, "iter_corpus_docs", records)
    monkeypatch.setattr(data, "corpus_data_dir", lambda _: tmp_path)
    with pytest.raises(ValueError, match="only 0/2"):
        data.prepare_view(
            Tokenizer(), [{"name": "code", "corpus_id": "sample"}], documents=2, targets=4, pool=4
        )
