from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from yylab import db
from yylab.train.preference import (
    PREFERENCE_OBJECTIVE,
    DPOConfig,
    PreferenceCollator,
    PreferenceExample,
    dpo_objective,
    load_preference_dataset,
    load_preference_jsonl,
    optimizer_steps,
    sequence_log_probabilities,
)
from yylab.workers.preference import (
    _commit_adapter_artifact,
    _validate_run_registry,
    precompute_reference_logps,
    train_dpo_policy,
)


class TinyChatTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    _roles = {"system": 10, "user": 11, "assistant": 12}
    _content = {
        "policy": [20],
        "question": [21],
        "follow up": [22],
        "good": [30],
        "bad": [31],
        "long answer": [32, 33, 34, 35],
    }

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        ids = [1]
        for message in messages:
            ids.append(self._roles[message["role"]])
            ids.extend(self._content.get(message["content"], [23]))
            ids.append(2)
        if add_generation_prompt:
            ids.append(self._roles["assistant"])
        return ids


class MappingChatTokenizer(TinyChatTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        ids = super().apply_chat_template(*args, **kwargs)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class QwenLikeThinkingTokenizer(TinyChatTokenizer):
    """Generation adds a thinking prefill that plain stored answers do not contain."""

    def apply_chat_template(self, *args, **kwargs):
        ids = super().apply_chat_template(*args, **kwargs)
        if kwargs["add_generation_prompt"]:
            ids.append(39)
        return {"input_ids": ids}


class TinyPreferencePolicy(nn.Module):
    def __init__(self, vocab_size: int = 40) -> None:
        super().__init__()
        self.adapter_logits = nn.Parameter(torch.zeros(vocab_size))
        self.register_buffer("embedding_stub", torch.zeros(1))
        self.adapter_disabled = False

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.embedding_stub)

    @contextmanager
    def disable_adapter(self):
        previous = self.adapter_disabled
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = previous

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        logits = torch.zeros(
            (*input_ids.shape, self.adapter_logits.numel()),
            device=input_ids.device,
        )
        if not self.adapter_disabled:
            logits = logits + self.adapter_logits
        return SimpleNamespace(logits=logits)


def _example(*, chosen: str = "good", rejected: str = "bad") -> PreferenceExample:
    return PreferenceExample.from_dict(
        {
            "id": "pair-1",
            "prompt": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "question"},
            ],
            "chosen": chosen,
            "rejected": rejected,
        }
    )


def _config(**updates) -> DPOConfig:
    values = {
        "checkpoint_id": "ckpt",
        "checkpoint_ref": "base",
        "adapter_id": "adapter",
        "adapter_name": "preference",
        "dataset_path": "pairs.jsonl",
        "quant": "bf16",
        "rank": 8,
        "alpha": 16,
        "dropout": 0,
        "beta": 1,
        "max_length": 32,
        "epochs": 4,
        "micro_batch": 1,
        "grad_accum": 1,
        "learning_rate": 0.1,
        "warmup_ratio": 0,
        "log_every": 1,
        "gradient_checkpointing": False,
    }
    values.update(updates)
    return DPOConfig.from_dict(values)


def test_strict_preference_jsonl_accepts_string_and_multi_turn_prompts(tmp_path) -> None:
    source = tmp_path / "pairs.jsonl"
    source.write_text(
        '{"id":"one","prompt":"question","chosen":"good","rejected":"bad"}\n'
        '{"id":"two","prompt":[{"role":"user","content":"question"},'
        '{"role":"assistant","content":"good"},{"role":"user","content":"follow up"}],'
        '"chosen":"long answer","rejected":"bad","metadata":{"source":"human"}}\n',
        encoding="utf-8",
    )

    examples = load_preference_jsonl(source)

    assert len(examples) == 2
    assert [message.role for message in examples[1].prompt] == ["user", "assistant", "user"]
    assert examples[1].metadata == {"source": "human"}


@pytest.mark.parametrize(
    "row, match",
    [
        ('{"prompt":"question","chosen":"good","rejected":"good"}\n', "must differ"),
        (
            '{"prompt":[{"role":"assistant","content":"bad order"}],'
            '"chosen":"good","rejected":"bad"}\n',
            "end with a user",
        ),
        (
            '{"prompt":"question","prompt":"duplicate","chosen":"good","rejected":"bad"}\n',
            "duplicate JSON key",
        ),
        ("\n", "blank JSONL"),
    ],
)
def test_strict_preference_jsonl_rejects_ambiguous_rows(tmp_path, row: str, match: str) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(row, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_preference_jsonl(source)


def test_directory_mixes_jsonl_and_parquet_and_exact_deduplicates(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "pairs"
    source.mkdir()
    (source / "a.jsonl").write_text(
        '{"id":"same","prompt":"question","chosen":"good","rejected":"bad"}\n',
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "same",
                    "prompt": [{"role": "user", "content": "question"}],
                    "chosen": [{"role": "assistant", "content": "good"}],
                    "rejected": [{"role": "assistant", "content": "bad"}],
                }
            ]
        ),
        source / "b.parquet",
    )

    dataset = load_preference_dataset(source)

    assert len(dataset.examples) == 1
    assert dataset.provenance["source_rows"] == 2
    assert dataset.provenance["duplicate_pairs"] == 1
    assert dataset.provenance["detected_formats"] == ["jsonl", "parquet"]
    assert len(dataset.provenance["canonical_sha256"]) == 64


def test_full_message_branches_preserve_shared_tool_trace_and_reasoning(tmp_path) -> None:
    source = tmp_path / "tool-pair.jsonl"
    common = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "result", "tool_call_id": "c1"},
    ]
    row = {
        "chosen": json.dumps(
            [*common, {"role": "assistant", "content": "fixed", "reasoning_content": "ok"}]
        ),
        "rejected": json.dumps([*common, {"role": "assistant", "content": "guess"}]),
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    example = load_preference_dataset(source).examples[0]

    assert [message.role for message in example.prompt] == ["user", "assistant", "tool"]
    assert example.prompt[1].tool_calls[0]["id"] == "c1"
    assert example.chosen.content == "fixed"
    assert example.chosen.reasoning_content == "ok"


def test_full_message_branches_reject_earlier_divergent_trajectories(tmp_path) -> None:
    source = tmp_path / "divergent.jsonl"
    source.write_text(
        json.dumps(
            {
                "chosen": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "first"},
                    {"role": "user", "content": "again"},
                    {"role": "assistant", "content": "good"},
                ],
                "rejected": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "different"},
                    {"role": "user", "content": "again"},
                    {"role": "assistant", "content": "bad"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="share an identical prompt"):
        load_preference_dataset(source)


def test_scored_normalized_traces_pair_across_jsonl_and_parquet(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "traces"
    source.mkdir()
    preferred = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "good"},
    ]
    rejected = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "bad"},
    ]
    (source / "winner.jsonl").write_text(
        json.dumps({"pair_id": "p1", "messages": preferred, "reward": 0.9}) + "\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "pair_id": "p1",
                    "messages_json": json.dumps(rejected),
                    "reward": -0.2,
                    "raw_event_blob": "unused" * 10_000,
                }
            ]
        ),
        source / "loser.parquet",
    )

    dataset = load_preference_dataset(source, data_format="traces")

    assert len(dataset.examples) == 1
    assert dataset.examples[0].chosen.content == "good"
    assert dataset.examples[0].rejected.content == "bad"
    assert dataset.provenance["trace_rows"] == 2
    assert dataset.provenance["trace_groups"] == 1


def test_trace_groups_never_pair_candidates_across_source_splits(tmp_path) -> None:
    source = tmp_path / "split-traces"
    source.mkdir()
    prompt = [{"role": "user", "content": "question"}]
    (source / "train.jsonl").write_text(
        json.dumps(
            {
                "pair_id": "shared",
                "messages": [*prompt, {"role": "assistant", "content": "good"}],
                "reward": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "test.jsonl").write_text(
        json.dumps(
            {
                "pair_id": "shared",
                "messages": [*prompt, {"role": "assistant", "content": "bad"}],
                "reward": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unpaired trace group"):
        load_preference_dataset(source, data_format="traces")


def test_score_endpoint_ties_expand_deterministically_and_quality_score_nests(tmp_path) -> None:
    source = tmp_path / "ranked.jsonl"
    rows = []
    for response, score in (("best-b", 1), ("low-b", 0), ("best-a", 1), ("low-a", 0)):
        rows.append(
            {
                "group_id": "ranked",
                "split": "train",
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": response},
                ],
                "quality_scores": {"overall": score},
            }
        )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    dataset = load_preference_dataset(source, data_format="traces")
    pairs = [(example.chosen.content, example.rejected.content) for example in dataset.examples]

    assert pairs == [
        ("best-a", "low-a"),
        ("best-a", "low-b"),
        ("best-b", "low-a"),
        ("best-b", "low-b"),
    ]
    assert dataset.provenance["split_counts"] == {"train": 4}


def test_parquet_json_encoded_quality_scores_normalize_like_objects(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "encoded-quality"
    source.mkdir()
    prompt = [{"role": "user", "content": "Explain normalized trace loading."}]
    messages = [
        json.dumps([*prompt, {"role": "assistant", "content": answer}])
        for answer in ("It canonicalizes equivalent storage formats.", "It ignores formats.")
    ]
    pq.write_table(
        pa.table(
            {
                "pair_id": ["quality-json", "quality-json"],
                "messages": messages,
                "quality_scores": [json.dumps({"overall": 0.9}), json.dumps({"overall": 0.1})],
                "split": ["train", "train"],
            }
        ),
        source / "train.parquet",
    )

    dataset = load_preference_dataset(source, data_format="traces")

    assert len(dataset.examples) == 1
    assert dataset.examples[0].chosen.content == "It canonicalizes equivalent storage formats."
    assert dataset.examples[0].rejected.content == "It ignores formats."


def test_unlabelled_or_unpaired_traces_have_actionable_diagnostics(tmp_path) -> None:
    source = tmp_path / "unary.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chat SFT|will not invent"):
        load_preference_dataset(source)

    source.write_text(
        json.dumps(
            {
                "pair_id": "only",
                "label": "chosen",
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unpaired trace group.*only chosen"):
        load_preference_dataset(source)


def test_row_json_unwraps_explicit_pairs_but_rejects_raw_unary_trace(tmp_path) -> None:
    source = tmp_path / "wrapped.jsonl"
    pair = {"prompt": "question", "chosen": "good", "rejected": "bad"}
    source.write_text(json.dumps({"row_json": json.dumps(pair)}) + "\n", encoding="utf-8")
    assert len(load_preference_dataset(source).examples) == 1

    unary = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    source.write_text(json.dumps({"row_json": json.dumps(unary)}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="use chat SFT"):
        load_preference_dataset(source)


def test_collator_masks_prompt_but_labels_response_and_end_of_turn() -> None:
    collator = PreferenceCollator(TinyChatTokenizer(), max_length=32)
    pair = collator.encode(_example(chosen="long answer"))

    assert pair.chosen.labels[: pair.chosen.prompt_tokens] == (-100,) * pair.chosen.prompt_tokens
    assert pair.chosen.labels[pair.chosen.prompt_tokens :] == (12, 32, 33, 34, 35, 2)
    assert pair.rejected.labels[pair.rejected.prompt_tokens :] == (12, 31, 2)
    batch = collator.collate([pair])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[0] == 2
    assert torch.equal(batch["attention_mask"][1, -6:], torch.tensor([1, 1, 1, 0, 0, 0]))
    assert torch.equal(batch["labels"][1, -3:], torch.tensor([-100, -100, -100]))


def test_collator_left_crops_prompt_before_assistant_target() -> None:
    collator = PreferenceCollator(TinyChatTokenizer(), max_length=8)
    pair = collator.encode(_example(chosen="long answer"))

    assert pair.chosen.prompt_tokens_truncated > 0
    assert pair.chosen.target_tokens_truncated == 0
    assert pair.chosen.labels[-6:] == (12, 32, 33, 34, 35, 2)


def test_collator_accepts_transformers_batch_encoding_shape() -> None:
    pair = PreferenceCollator(MappingChatTokenizer(), max_length=32).encode(_example())

    assert pair.chosen.target_tokens == 3


def test_collator_does_not_require_qwen_generation_prefill_to_match_answer() -> None:
    tokenizer = QwenLikeThinkingTokenizer()
    prompt = _example().prompt
    generation_ids = tokenizer.apply_chat_template(
        [message.to_dict() for message in prompt],
        tokenize=True,
        add_generation_prompt=True,
    )["input_ids"]
    pair = PreferenceCollator(tokenizer, max_length=32).encode(_example())

    assert generation_ids[-1] == 39
    assert pair.chosen.labels[pair.chosen.prompt_tokens :] == (12, 30, 2)


def test_sequence_log_probabilities_obeys_causal_shift_and_mask() -> None:
    labels = torch.tensor([[-100, -100, 2, 3]])
    logits = torch.full((1, 4, 5), -20.0)
    logits[0, 1, 2] = 20
    logits[0, 2, 3] = 20

    class FixedModel(nn.Module):
        def forward(self, **_kwargs):
            return SimpleNamespace(logits=logits)

    result = sequence_log_probabilities(
        FixedModel(),
        {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "labels": labels,
        },
    )

    assert result.shape == (1,)
    assert result.item() == pytest.approx(0, abs=1e-6)


def test_dpo_objective_rewards_reference_relative_preference_margin() -> None:
    metrics = dpo_objective(
        torch.tensor([-1.0, -3.0]),
        torch.tensor([-2.0, -2.0]),
        beta=1,
    )

    assert metrics.loss.item() < 0.2
    assert metrics.preference_accuracy.item() == 1
    assert metrics.reward_margin.item() == pytest.approx(2)


def test_config_round_trip_and_optimizer_step_cap() -> None:
    config = _config(
        max_steps=3,
        epochs=8,
        micro_batch=2,
        grad_accum=2,
        dataset_format="traces",
        dataset_checksum="a" * 64,
    )

    assert config.to_dict()["objective"] == PREFERENCE_OBJECTIVE
    assert DPOConfig.from_dict(config.to_dict()) == config
    assert optimizer_steps(5, config) == 3
    with pytest.raises(ValueError, match="unsupported preference objective"):
        DPOConfig.from_dict({**config.to_dict(), "objective": "ppo"})
    with pytest.raises(ValueError, match="dataset_format"):
        _config(dataset_format="guess")
    with pytest.raises(ValueError, match="dataset_checksum"):
        _config(dataset_checksum="not-a-checksum")


def test_reference_cache_uses_disabled_adapter_and_tiny_dpo_training_improves_pair(tmp_path) -> None:
    tokenizer = TinyChatTokenizer()
    collator = PreferenceCollator(tokenizer, max_length=32)
    pairs = [collator.encode(_example())]
    model = TinyPreferencePolicy()
    reference = precompute_reference_logps(
        model,
        pairs,
        collator,
        micro_batch=1,
        device=torch.device("cpu"),
    )
    assert torch.equal(reference[0:1], reference[1:2])

    result = train_dpo_policy(
        model,
        pairs,
        collator,
        reference,
        _config(),
        tmp_path,
        device=torch.device("cpu"),
    )

    assert model.adapter_logits[30].item() > model.adapter_logits[31].item()
    assert result["reward_margin"] > 0
    assert (tmp_path / "metrics.jsonl").read_text("utf-8").count("\n") == 4


def test_completed_adapter_and_quant_lock_commit_together(tmp_path) -> None:
    db.init_db()
    checkpoint_id = db.new_id("pref_ckpt")
    adapter_id = db.new_id("pref_adp")
    db.execute(
        "INSERT INTO checkpoints(id, name, created_at, source, ref) VALUES(?,?,?,?,?)",
        (checkpoint_id, "base", db.now_iso(), "local", "base-ref"),
    )
    db.execute(
        "INSERT INTO adapters(id, name, created_at, checkpoint_id, arch, quant, rank, status) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (adapter_id, "dpo", db.now_iso(), checkpoint_id, "lora", "bf16", 8, "training"),
    )
    config = _config(
        checkpoint_id=checkpoint_id,
        checkpoint_ref="base-ref",
        adapter_id=adapter_id,
    )

    _validate_run_registry(config)
    _commit_adapter_artifact(config, tmp_path, {"objective": PREFERENCE_OBJECTIVE})

    checkpoint = db.q1("SELECT quant_locked FROM checkpoints WHERE id=?", (checkpoint_id,))
    adapter = db.q1("SELECT status, path FROM adapters WHERE id=?", (adapter_id,))
    assert checkpoint["quant_locked"] == "bf16"
    assert adapter == {"status": "trained", "path": str(tmp_path)}
