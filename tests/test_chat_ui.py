from __future__ import annotations

import pytest
from nicegui.elements.markdown import prepare_content

from yylab.ui.pages.chat import (
    ASSISTANT_MARKDOWN_EXTRAS,
    _assistant_markdown_content,
    _chat_load_settings,
    _context_telemetry_text,
    _generation_snapshot,
    _user_message,
)


def test_final_assistant_markdown_preserves_streamed_text() -> None:
    streamed = " First line\n<S>tep 1\n<T>ake the numbers\nThis stays intact"

    content = _assistant_markdown_content(streamed)
    rendered = prepare_content(content, extras=" ".join(ASSISTANT_MARKDOWN_EXTRAS))

    assert content == (
        "First line\n&lt;S&gt;tep 1\n&lt;T&gt;ake the numbers\nThis stays intact"
    )
    assert "&lt;S&gt;tep 1" in rendered
    assert "&lt;T&gt;ake the numbers" in rendered
    assert "\ntep 1" not in rendered
    assert "\nake the numbers" not in rendered
    assert rendered.count("<br />") == 3


@pytest.mark.parametrize(
    ("mode", "rope", "factor"),
    [
        ("native", "none", 1.0),
        ("yarn_2x", "yarn", 2.0),
        ("yarn_4x", "yarn", 4.0),
    ],
)
def test_chat_context_mode_builds_isolated_load_settings(mode, rope, factor):
    saved = {
        "attn_implementation": "sdpa",
        "rope_scaling": "linear",
        "rope_factor": 8.0,
    }

    resolved = _chat_load_settings(saved, mode)

    assert resolved["context_mode"] == mode
    assert resolved["rope_scaling"] == rope
    assert resolved["rope_factor"] == factor
    assert saved["rope_scaling"] == "linear"


def test_chat_user_message_carries_optional_factorized_address() -> None:
    assert _user_message("plain") == {"role": "user", "content": "plain"}
    assert _user_message("addressed", atxy_t=" voltage ", atxy_x=0, atxy_y="night") == {
        "role": "user",
        "content": "addressed",
        "atxy": {"t": "voltage", "x": "0", "y": "night"},
    }


def test_raw_generation_snapshot_is_one_independent_literal_prompt() -> None:
    history = [
        {"role": "user", "content": "old prompt"},
        {"role": "assistant", "content": "old completion"},
    ]
    current = {
        "role": "user",
        "content": "  #include\n",
        "atxy": {"t": "code", "x": "c", "y": "header"},
    }

    assert _generation_snapshot(history, current, "raw") == [{"content": "  #include\n"}]
    assert _generation_snapshot(history, current, "chat") == [*history, current]


def test_chat_context_telemetry_reports_compaction():
    text = _context_telemetry_text(
        {
            "prompt_tokens": 1200,
            "prompt_budget": 1536,
            "answer_budget": 512,
            "context_limit": 2048,
            "dropped_turns": 2,
            "dropped_messages": 4,
            "compacted": True,
        }
    )

    assert "Prompt 1,200/1,536 tokens" in text
    assert "answer reserved 512" in text
    assert "compacted 2 oldest turns (4 messages)" in text


def test_chat_context_telemetry_reports_nll_selection() -> None:
    text = _context_telemetry_text(
        {
            "prompt_tokens": 900,
            "prompt_budget": 1536,
            "answer_budget": 512,
            "context_limit": 2048,
            "dropped_turns": 3,
            "dropped_messages": 6,
            "compacted": True,
            "strategy": "nll_atxy",
            "retrieved_turns": 2,
            "evaluated_turns": 4,
            "atxy_candidates": 1,
            "nll_gain_nats": 0.25,
        }
    )

    assert "NLL selected 2 retrieved turns / evaluated 4" in text
    assert "omitted 3 prior turns" in text
    assert "ΔNLL +0.250 nats" in text
    assert "ATXY candidates 1" in text


def test_raw_context_telemetry_names_the_literal_prompt() -> None:
    text = _context_telemetry_text(
        {
            "prompt_tokens": 2,
            "prompt_budget": 1536,
            "answer_budget": 512,
            "context_limit": 2048,
            "prompt_mode": "raw",
            "compacted": False,
        }
    )

    assert text.startswith("Raw prompt 2/1,536 tokens")
    assert "completion reserved 512" in text
