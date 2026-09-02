from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yylab.export.merge import merge_lora
from yylab.train.antidoom import (
    FTPOPair,
    RepeatHit,
    find_inner_repetition,
    ftpo_objective,
    mine_ftpo_pairs,
    normalize_prompt_row,
    render_prompt_ids,
    safe_single_token_surface,
    select_rejected_token_index_token_aware,
)
from yylab.workers.antidoom import (
    DEFAULT_REASONING_SUFFIX,
    _continuation_ids,
    _is_eval_row,
    _left_pad,
    _pair_from_dict,
    _read_pairs,
    _render_prompt,
    _source_key,
    _training_batch,
    _validate_config,
    _write_pairs,
    classify_diagnostic_result,
)


class TinyAntiDoomTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    surfaces = {
        0: "",
        1: "<prompt>",
        2: "<eos>",
        3: "<second>",
        10: " Loop",
        11: " x",
        12: " ordinary",
        20: " Exit",
        21: "\ufffd",  # one incomplete byte-like token
        22: "!",
        23: " Normalized",
        24: " Continue",
    }

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        assert messages[-1]["role"] == "user"
        return {"input_ids": [1, 30 + len(messages)]}

    def decode(self, ids, **_kwargs):
        return "".join(self.surfaces[int(token_id)] for token_id in ids)

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        exact = {surface: token_id for token_id, surface in self.surfaces.items()}
        if text == "\ufffd":
            return [99]
        if text == " Normalized":
            return [98]
        return [exact[text]] if text in exact else [97]


def _mined_pair(
    *,
    prompt_ids: tuple[int, ...] = (1, 10),
    chosen_ids: tuple[int, ...] = (20,),
    rejected_token_id: int = 10,
) -> FTPOPair:
    return FTPOPair(
        batch_index=0,
        sequence_index=0,
        return_index=0,
        prompt_ids=prompt_ids,
        chosen_ids=chosen_ids,
        rejected_token_id=rejected_token_id,
        source_prompt_ids=(1,),
        generated_ids=(10, 11, 10),
        rejected_generation_index=2,
        repetition=RepeatHit(0, 21, 7, 3, " Loop x"),
        chosen_surfaces=tuple(f" token-{token_id}" for token_id in chosen_ids),
        rejected_surface=" Loop",
        generation_text=" Loop x Loop x Loop",
        metadata={"source_key": "example"},
    )


def test_repeat_detector_and_token_boundary_follow_the_verified_period() -> None:
    tokenizer = TinyAntiDoomTokenizer()
    generated = [10, 11] * 4
    text = tokenizer.decode(generated)

    hit = find_inner_repetition(
        text,
        min_repeats=4,
        min_total_repeated=20,
        max_period=20,
        sample_len=3,
        sample_interval=1,
    )

    assert hit is not None
    assert hit.repeats == 4
    assert hit.repeat_start == len(" Loop x")
    assert select_rejected_token_index_token_aware(generated, tokenizer, hit) == 2
    assert find_inner_repetition("not repetitive", min_total_repeated=4) is None


def test_normalizes_prompt_only_sharegpt_and_renders_native_ids() -> None:
    tokenizer = TinyAntiDoomTokenizer()
    row = {
        "conversations": [
            {"from": "human", "value": "First question"},
            {"from": "gpt", "value": "Earlier response"},
            {"from": "human", "value": "Follow-up"},
        ]
    }

    messages = normalize_prompt_row(row)

    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert render_prompt_ids(row, tokenizer) == [1, 33]
    with pytest.raises(ValueError, match="end with a user"):
        normalize_prompt_row({"conversations": [{"from": "gpt", "value": "answer"}]})


def test_safe_alternative_requires_printable_alphanumeric_exact_roundtrip() -> None:
    tokenizer = TinyAntiDoomTokenizer()

    assert safe_single_token_surface(tokenizer, 20) == " Exit"
    assert safe_single_token_surface(tokenizer, 21) is None
    assert safe_single_token_surface(tokenizer, 22) is None
    assert safe_single_token_surface(tokenizer, 23) is None


def test_mines_batched_hf_generation_using_direct_ids_and_provenance() -> None:
    tokenizer = TinyAntiDoomTokenizer()
    input_ids = torch.tensor([[0, 1], [1, 3]])
    attention_mask = torch.tensor([[0, 1], [1, 1]])
    first = [10, 11] * 4 + [2]
    second = [12, 2] + [0] * 7
    sequences = torch.tensor([[0, 1, *first], [1, 3, *second]])
    scores = []
    for _step in range(9):
        score = torch.full((2, 25), -20.0)
        score[:, 12] = 1.0
        scores.append(score)
    # At the first repeated token, unsafe byte/punctuation/non-roundtrip choices
    # outrank valid alternatives. They must not enter chosen_ids.
    scores[2][0, 10] = 10.0
    scores[2][0, 21] = 9.5
    scores[2][0, 22] = 9.0
    scores[2][0, 23] = 8.5
    scores[2][0, 20] = 8.0
    scores[2][0, 24] = 7.5
    output = SimpleNamespace(sequences=sequences, scores=tuple(scores))

    pairs = mine_ftpo_pairs(
        input_ids,
        attention_mask,
        output,
        tokenizer,
        source_metadata=[{"record_id": "first"}, {"record_id": "second"}],
        max_chosen=2,
        candidate_pool=8,
        min_p=None,
        min_repeats=4,
        min_total_repeated=20,
        max_period=20,
        sample_len=3,
        sample_interval=1,
    )

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.batch_index == 0
    assert pair.sequence_index == 0
    assert pair.return_index == 0
    assert pair.source_prompt_ids == (1,)
    assert pair.prompt_ids == (1, 10, 11)
    assert pair.generated_ids == tuple([10, 11] * 4)
    assert pair.rejected_generation_index == 2
    assert pair.rejected_token_id == 10
    assert pair.chosen_ids == (20, 24)
    assert pair.chosen_surfaces == (" Exit", " Continue")
    assert pair.metadata == {"record_id": "first"}
    assert pair.to_dict()["repetition"]["repeats"] == 4


def test_ftpo_objective_matches_reference_math_and_tethers() -> None:
    policy = torch.tensor([[0.0, 1.0, 3.0, 4.0]], requires_grad=True)
    reference = torch.zeros_like(policy, requires_grad=True)
    chosen = torch.tensor([[1, 2, -1]])
    chosen_mask = torch.tensor([[True, True, False]])

    result = ftpo_objective(
        policy,
        reference,
        chosen,
        torch.tensor([0]),
        chosen_mask,
        epsilon=2.0,
        preference_temperature=1.0,
        lambda_mse=0.4,
        lambda_mse_target=0.05,
        tau_mse_target=0.5,
    )

    expected_preference = torch.nn.functional.softplus(torch.tensor(1.0)) * 0.5 / 2
    expected_target = torch.tensor((0.0 + 0.25 + 6.25) / 3)
    expected = expected_preference + 0.4 * 16.0 + 0.05 * expected_target
    assert result.preference_loss == pytest.approx(expected_preference.item())
    assert result.untargeted_mse == pytest.approx(16.0)
    assert result.target_excess_mse == pytest.approx(expected_target.item())
    assert result.loss.detach() == pytest.approx(expected.item())
    assert result.chosen_win == pytest.approx(1.0)
    assert result.margin_win == pytest.approx(0.5)
    assert result.mean_delta == pytest.approx(2.0)
    assert result.active_weight == pytest.approx(0.25)

    result.loss.backward()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()
    assert reference.grad is None


def test_ftpo_chosen_win_averages_each_row_not_each_choice() -> None:
    policy = torch.tensor(
        [
            [0.0, 1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    chosen = torch.tensor([[1, 0, 0], [1, 2, 3]])
    mask = torch.tensor([[True, False, False], [True, True, True]])

    result = ftpo_objective(
        policy,
        policy.clone(),
        chosen,
        torch.tensor([0, 0]),
        mask,
    )

    assert result.chosen_win == pytest.approx(0.5)


def test_worker_config_validation_resolves_source_and_sets_safe_defaults(tmp_path) -> None:
    source = tmp_path / "prompts.jsonl"
    source.write_text('{"conversations":[{"from":"human","value":"question"}]}\n')

    config = _validate_config(
        {
            "checkpoint_id": "checkpoint",
            "checkpoint_ref": "base-ref",
            "adapter_id": "adapter",
            "prompt_jsonl": str(source),
            "target_pairs": 10,
        }
    )

    assert config["prompt_jsonl"] == str(source.resolve())
    assert config["prompt_field"] == "conversations"
    assert config["reasoning_suffix"] == DEFAULT_REASONING_SUFFIX
    assert config["max_train_examples"] == 7
    assert config["candidate_pool"] >= config["max_chosen"]
    with pytest.raises(ValueError, match="candidate_pool"):
        _validate_config({**config, "candidate_pool": 1, "max_chosen": 2})
    with pytest.raises(ValueError, match="eval_bucket"):
        _validate_config({**config, "eval_bucket": config["eval_modulus"]})


def test_worker_render_prompt_appends_reasoning_suffix_without_mutating_row() -> None:
    class CapturingTokenizer(TinyAntiDoomTokenizer):
        rendered_messages = None

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            self.rendered_messages = messages
            return torch.tensor([[1, 31]])

    tokenizer = CapturingTokenizer()
    row = {"conversations": [{"from": "human", "value": "Question"}]}

    ids = _render_prompt(
        row,
        tokenizer,
        {"prompt_field": "conversations", "reasoning_suffix": " suffix"},
    )

    assert ids == [1, 31]
    assert tokenizer.rendered_messages[-1]["content"] == "Question suffix"
    assert row["conversations"][0]["value"] == "Question"


def test_worker_source_key_and_eval_partition_are_stable() -> None:
    config = {"eval_modulus": 7, "eval_bucket": 3}

    assert _source_key({"id": 42}, b"ignored") == "42"
    assert _source_key({}, b"same row") == _source_key({}, b"same row")
    assert _is_eval_row("stable-key", config) == _is_eval_row("stable-key", config)
    assert sum(
        _is_eval_row("stable-key", {"eval_modulus": 7, "eval_bucket": bucket})
        for bucket in range(7)
    ) == 1


def test_worker_pair_jsonl_roundtrip_preserves_direct_token_provenance(tmp_path) -> None:
    pair = _mined_pair(chosen_ids=(20, 24))
    path = tmp_path / "pairs.jsonl"

    _write_pairs(path, [pair])

    assert _read_pairs(path) == [pair]
    assert _pair_from_dict(pair.to_dict()) == pair


def test_worker_training_batch_left_pads_and_keeps_variable_choice_masks() -> None:
    pairs = [
        _mined_pair(prompt_ids=(1, 10), chosen_ids=(20,)),
        _mined_pair(prompt_ids=(1, 3, 11), chosen_ids=(20, 24), rejected_token_id=11),
    ]

    batch = _training_batch(pairs, pad_token_id=0, device=torch.device("cpu"))

    assert batch["input_ids"].tolist() == [[0, 1, 10], [1, 3, 11]]
    assert batch["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert batch["position_ids"].tolist() == [[0, 0, 1], [0, 1, 2]]
    assert batch["chosen_ids"].tolist() == [[20, 0], [20, 24]]
    assert batch["chosen_mask"].tolist() == [[True, False], [True, True]]
    assert batch["rejected_token_ids"].tolist() == [10, 11]


def test_worker_generation_padding_and_continuation_stops_are_exact() -> None:
    tokenizer = TinyAntiDoomTokenizer()
    inputs, mask = _left_pad(
        [[1], [1, 3]],
        pad_token_id=0,
        device=torch.device("cpu"),
    )
    output = SimpleNamespace(
        sequences=torch.tensor(
            [
                [0, 1, 10, 11, 2, 0],
                [1, 3, 12, 0, 0, 0],
            ]
        )
    )

    assert inputs.tolist() == [[0, 1], [1, 3]]
    assert mask.tolist() == [[0, 1], [1, 1]]
    assert _continuation_ids(output, 2, tokenizer) == [[10, 11], [12]]


def test_output_head_antidoom_adapter_cannot_be_merged(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "yylab_antidoom.json").write_text(
        '{"adapter_only":true,"merge_safe":false}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="intentionally adapter-only"):
        merge_lora("unused-base", str(adapter), tmp_path / "merged")


def test_diagnostic_classifier_separates_latent_gain_from_loop_regression() -> None:
    baseline = {
        "loop_rate": 0.25,
        "mean_repeated_4gram_fraction": 0.5,
        "empty_rate": 0.0,
    }
    nll = {"nll": 3.0}
    baseline_quality = {"scores": {"greedy": {"passed": 0}}}

    latent = classify_diagnostic_result(
        baseline,
        {"loop_rate": 0.1, "mean_repeated_4gram_fraction": 0.3, "empty_rate": 0.0},
        nll,
        {"nll": 3.02},
        baseline_quality,
        {"scores": {"greedy": {"passed": 2}}},
    )
    regressed = classify_diagnostic_result(
        baseline,
        {"loop_rate": 0.34, "mean_repeated_4gram_fraction": 0.53, "empty_rate": 0.0},
        nll,
        {"nll": 3.01},
        baseline_quality,
        {"scores": {"greedy": {"passed": 0}}},
    )

    assert latent["label"] == "latent_capability_unlocked"
    assert latent["recommended_for_inference"] is True
    assert regressed["label"] == "repetition_regression"
    assert regressed["recommended_for_inference"] is False
