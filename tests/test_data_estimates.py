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


def test_estimate_steps_uses_split_totals_and_legacy_unspecified_fallback(monkeypatch) -> None:
    modern = {
        "chars_written": 1_400,
        "totals": {"tokens": {"qwen25": 500}},
        "domains": {
            "code": {"chars": 700, "tokens": {"qwen25": 280}},
            "language": {"chars": 700, "tokens": {"qwen25": 220}},
        },
        "splits": {
            "train": {
                "chars": 1_050,
                "tokens": {"qwen25": 360},
                "domains": {
                    "code": {"chars": 350, "tokens": {"qwen25": 140}},
                    "language": {"chars": 700, "tokens": {"qwen25": 220}},
                },
            },
            "test": {
                "chars": 350,
                "tokens": {"qwen25": 140},
                "domains": {
                    "code": {"chars": 350, "tokens": {"qwen25": 140}},
                },
            },
        },
    }
    legacy = {
        "chars_written": 700,
        "totals": {"tokens": {"qwen25": 200}},
        "domains": {"code": {"chars": 700, "tokens": {"qwen25": 200}}},
    }
    rows = {
        "modern": {"stats_json": json.dumps(modern)},
        "legacy": {"stats_json": json.dumps(legacy)},
    }
    monkeypatch.setattr(data.db, "q1", lambda _query, params: rows[params[0]])

    assert data.estimate_steps(
        ["modern"], None, 1, 10, 1, 1, "qwen25", splits=["train"]
    ) == 36
    assert data.estimate_steps(
        ["modern"], ["code"], 1, 10, 1, 1, "qwen25", splits=["test"]
    ) == 14
    assert data.estimate_steps(
        ["modern", "legacy"], None, 1, 10, 1, 1, "qwen25", splits=["train"]
    ) == 36
    assert data.estimate_steps(
        ["modern", "legacy"], None, 1, 10, 1, 1, "qwen25", splits=["unspecified"]
    ) == 20
