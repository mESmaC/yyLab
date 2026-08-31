from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

from yylab.infer.context_retrieval import (
    FactorizedAddress,
    address_similarity,
    rank_context_candidates,
)
from yylab.infer.engine import Engine, LoadSpec


class WordTok:
    eos_token_id = 1
    bos_token_id = 2
    model_max_length = 1_000

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [str(message.get("content") or "") for message in messages]
        if add_generation_prompt:
            parts.append("assistant")
        return " ".join(parts)

    def __call__(self, text, **_):
        ids = []
        for word in str(text).split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 3
            ids.append(self.vocab[word])
        return {"input_ids": ids}


def _engine(context_limit: int) -> Engine:
    engine = Engine()
    engine.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=context_limit))
    engine.tokenizer = WordTok()
    engine.spec = LoadSpec(
        ref="test-model",
        quant="bf16",
        settings={"max_context_tokens": context_limit},
    )
    return engine


def test_factorized_address_is_a_of_txy_and_penalizes_conflicts() -> None:
    query = FactorizedAddress.from_value({"type": "voltage", "entity": 7, "condition": "overnight"})

    assert query == FactorizedAddress("voltage", "7", "overnight")
    assert address_similarity(query, FactorizedAddress("voltage", "7", "overnight")) == 1.0
    assert address_similarity(query, FactorizedAddress("voltage", "8", "overnight")) == pytest.approx(0.1)


def test_atxy_only_proposes_and_query_echo_is_marked() -> None:
    old = [
        [{"role": "user", "content": "unrelated words", "atxy": ("voltage", 7, "night")}],
        [{"role": "user", "content": "where is channel seven"}],
    ]
    query = [
        {
            "role": "user",
            "content": "where is channel seven",
            "metadata": {"atxy": {"t": "voltage", "x": 7, "y": "night"}},
        }
    ]

    ranked = rank_context_candidates(old, query)

    assert ranked[0].index == 0
    assert ranked[0].atxy_score == 1.0
    assert ranked[1].query_echo is True


def test_repeated_question_with_novel_assistant_answer_is_not_an_echo_trap() -> None:
    query = [{"role": "user", "content": "what value belongs to channel seven"}]
    old = [
        [
            {"role": "user", "content": "what value belongs to channel seven"},
            {"role": "assistant", "content": "the retained value is 4.871 volts"},
        ]
    ]

    assert rank_context_candidates(old, query)[0].query_echo is False


def test_nll_context_plan_retrieves_useful_older_turn_over_recent_distractor() -> None:
    engine = _engine(12)
    conversation = [
        {"role": "system", "content": "guard"},
        {"role": "user", "content": "needle project fact"},
        {"role": "assistant", "content": "value alpha"},
        {"role": "user", "content": "recent chatter"},
        {"role": "assistant", "content": "irrelevant reply"},
        {"role": "user", "content": "needle question"},
    ]

    def fake_nll(_self, prefix_messages, _query_group, *, probe_tokens):
        assert probe_tokens == 64
        text = " ".join(str(message.get("content") or "") for message in prefix_messages)
        if "value alpha" in text:
            return 1.0
        if "irrelevant reply" in text:
            return 4.0
        return 3.0

    engine._context_target_nll = MethodType(fake_nll, engine)
    plan = engine.plan_chat_context(
        conversation,
        max_new_tokens=3,
        strategy="nll_atxy",
        recent_turns=0,
        max_nll_probes=4,
        nll_probe_tokens=64,
        min_nll_gain_nats=0.1,
        trigger_ratio=1.0,
    )

    contents = [message["content"] for message in plan.messages]
    assert "value alpha" in contents
    assert "irrelevant reply" not in contents
    assert plan.retrieved_turns == 1
    assert plan.evaluated_turns == 1
    assert plan.nll_gain_nats == 2.0
    assert plan.to_dict()["position_policy"] == "contiguous_rebase"


def test_atxy_controls_one_probe_shortlist_but_nll_still_accepts() -> None:
    engine = _engine(13)
    conversation = [
        {
            "role": "user",
            "content": "stored record",
            "atxy": ("voltage", 7, "overnight"),
        },
        {"role": "assistant", "content": "value alpha"},
        {
            "role": "user",
            "content": "voltage channel 7 overnight distractor",
            "atxy": ("voltage", 8, "overnight"),
        },
        {"role": "assistant", "content": "value beta"},
        {
            "role": "user",
            "content": "voltage channel 7 overnight question",
            "atxy": {"t": "voltage", "x": 7, "y": "overnight"},
        },
    ]

    def fake_nll(_self, prefix_messages, _query_group, *, probe_tokens):
        text = " ".join(str(message.get("content") or "") for message in prefix_messages)
        return 1.0 if "value alpha" in text else 3.0

    engine._context_target_nll = MethodType(fake_nll, engine)
    plan = engine.plan_chat_context(
        conversation,
        max_new_tokens=3,
        strategy="nll_atxy",
        recent_turns=0,
        max_nll_probes=1,
        min_nll_gain_nats=0.1,
        trigger_ratio=1.0,
    )

    assert any(message["content"] == "value alpha" for message in plan.messages)
    assert not any(message["content"] == "value beta" for message in plan.messages)
    assert plan.atxy_candidates == 2
    assert plan.retrieved_turns == 1


def test_nll_probe_failure_falls_back_to_safe_recency() -> None:
    engine = _engine(8)
    conversation = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "new question"},
    ]

    def fail(*_args, **_kwargs):
        raise RuntimeError("probe backend unavailable")

    engine._context_target_nll = fail
    plan = engine.plan_chat_context(
        conversation,
        max_new_tokens=3,
        strategy="nll_atxy",
        recent_turns=0,
        trigger_ratio=1.0,
    )

    assert plan.strategy == "recency_fallback"
    assert plan.fallback_reason == "probe backend unavailable"


def test_weak_nll_gain_cannot_consume_a_large_turn_after_token_cost() -> None:
    engine = _engine(24)
    conversation = [
        {"role": "user", "content": "weak one two three"},
        {"role": "assistant", "content": "weak payload"},
        {"role": "user", "content": "other one two three"},
        {"role": "assistant", "content": "other payload"},
        {"role": "user", "content": "current question"},
    ]

    def weak_gain(_self, prefix_messages, _query_group, *, probe_tokens):
        return 2.99 if prefix_messages else 3.0

    engine._context_target_nll = MethodType(weak_gain, engine)
    plan = engine.plan_chat_context(
        conversation,
        max_new_tokens=3,
        strategy="nll_atxy",
        recent_turns=0,
        max_nll_probes=2,
        min_nll_gain_nats=0.0,
        nll_token_cost_nats=0.01,
        trigger_ratio=0.5,
    )

    assert plan.retrieved_turns == 0
    assert [message["content"] for message in plan.messages] == ["current question"]


def test_nonleading_instruction_is_never_evicted() -> None:
    engine = _engine(9)
    conversation = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old response"},
        {"role": "system", "content": "never drop this"},
        {"role": "user", "content": "latest question"},
    ]

    plan = engine.plan_chat_context(conversation, max_new_tokens=3, strategy="recency")

    assert any(message["role"] == "system" and message["content"] == "never drop this" for message in plan.messages)
