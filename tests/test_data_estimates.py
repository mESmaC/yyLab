from __future__ import annotations

import json

from yylab.train import data


def test_tokenizer_stats_key_matches_supported_model_families() -> None:
    assert data.tokenizer_stats_key("qwen3") == "qwen25"
    assert data.tokenizer_stats_key("Qwen2_5") == "qwen25"
    assert data.tokenizer_stats_key("llama") == "llama3"
    assert data.tokenizer_stats_key("phi-3") == "phi3"
    assert data.tokenizer_stats_key("gemma2") == "gemma3"
    assert data.tokenizer_stats_key("mistral") == "mistral"
    assert data.tokenizer_stats_key("unknown") is None


def test_estimate_steps_prefers_recorded_tokens_and_falls_back_per_domain(monkeypatch) -> None:
    stats = {
        "chars_written": 1_050,
        "totals": {"tokens": {"qwen25": 360}},
        "domains": {
            "code": {"chars": 350, "tokens": {"qwen25": 140}},
            "language": {"chars": 700, "tokens": {}},
        },
    }

    monkeypatch.setattr(
        data.db,
        "q1",
        lambda _query, _params: {"stats_json": json.dumps(stats)},
    )

    # 140 recorded code tokens + the 700-char language fallback at 3.5 chars/token.
    assert data.estimate_steps(["corpus"], None, 1, 10, 1, 1, "qwen25") == 34
    assert data.estimate_steps(["corpus"], ["code"], 1, 10, 1, 1, "qwen25") == 14
    # Without a tokenizer match, both domains retain the historical character estimate.
    assert data.estimate_steps(["corpus"], None, 1, 10, 1, 1) == 30
