from __future__ import annotations

import json
import random
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yylab.train.sprint_quality import (
    default_sprint_quality_prompts,
    evaluate_sprint_generation,
    normalize_sprint_quality_config,
    sprint_repetition_metrics,
)


class ProbeTokenizer:
    eos_token_id = 0
    chat_template = None

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [8, 9] if text != "long" else list(range(20))

    def decode(self, ids, **kwargs):
        words = {0: "<eos>", 1: "blue", 2: "bird", 3: "14", 4: "7"}
        return " ".join(words.get(int(token), f"w{token}") for token in ids)


class ProbeModel(torch.nn.Module):
    def __init__(self, sequence=(1, 2, 0), *, fail=False, uniform=False):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 4)
        self.config = SimpleNamespace(max_position_embeddings=128, eos_token_id=0, attention_type="kda")
        self.sequence, self.fail, self.uniform = sequence, fail, uniform
        self.calls = []

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, *, use_cache, logits_to_keep, return_logits):
        assert not self.training and not self.embed.training
        assert not torch.is_grad_enabled()
        assert use_cache is False and logits_to_keep == 1 and return_logits is True
        assert input_ids.shape[1] <= self.config.max_position_embeddings
        self.calls.append(input_ids.detach().clone())
        # Even a custom forward consuming global RNG must not perturb training.
        random.random()
        np.random.random()
        torch.rand(1)
        if self.fail:
            raise RuntimeError("deliberate model failure")
        logits = torch.full((1, 1, 16), -100.0)
        if self.uniform:
            logits.fill_(0)
            logits[..., 0] = -100.0
        else:
            index = min(input_ids.shape[1] - 2, len(self.sequence) - 1)
            logits[..., self.sequence[index]] = 100.0
        return SimpleNamespace(logits=logits)


def single_config(**kwargs):
    return {"prompts": [{"name": "probe", "prompt": "prefix"}], **kwargs}


def test_ngram_denominator_and_generated_only_scope():
    result = sprint_repetition_metrics([1, 2, 3, 4, 1, 2, 3, 4])
    assert result["ngrams"]["4"] == {"occurrences": 5, "distinct": 4, "duplicates": 1, "duplicate_fraction": .2}
    assert not result["flagged"]
    assert not result["sufficient_length"]
    assert sprint_repetition_metrics([1])["ngrams"]["4"]["duplicate_fraction"] is None


def test_exact_tail_flags_short_but_substantial_loop():
    result = sprint_repetition_metrics([7, 8, 9] * 6)
    assert result["tail_loop"] == {"period": 3, "repeats": 6, "tokens": 18, "fraction": 1.0, "unit_token_ids": [7, 8, 9]}
    assert not result["sufficient_length"]
    assert result["reasons"] == ["repeated_tail"]
    assert not sprint_repetition_metrics([7, 8, 9] * 2)["flagged"]


def test_ngram_alert_catches_loops_longer_than_tail_period():
    result = sprint_repetition_metrics(list(range(20)) * 4)
    assert result["tail_loop"]["tokens"] == 0
    assert result["reasons"] == ["duplicate_ngrams"]
    assert not sprint_repetition_metrics(list(range(64)))["flagged"]


def test_default_prompts_do_not_share_mutable_state_or_assume_chat_markers():
    first = default_sprint_quality_prompts()
    first[2]["check"]["expected"] = "wrong"
    assert default_sprint_quality_prompts()[2]["check"]["expected"] == "blue bird"
    assert default_sprint_quality_prompts()[2]["check"]["type"] == "exact_text"
    assert default_sprint_quality_prompts()[3]["check"]["type"] == "integer_text"
    assert normalize_sprint_quality_config()["prompt_format"] == "plain"


@pytest.mark.parametrize("config", [
    {"max_new_tokens": 97}, {"max_new_tokens": True}, {"max_new_tokens": 1.5},
    {"temperature": float("nan")}, {"top_p": 0}, {"prompt_format": "auto"},
    {"modes": ["sample", "sample"]}, {"prompts": []}, {"repetition_penalty": 1.2},
    {"prompts": [{"prompt": "x", "check": {"type": "integer_first_line", "expected": "maybe"}}]},
])
def test_configuration_rejects_unbounded_or_concealing_options(config):
    with pytest.raises(ValueError):
        normalize_sprint_quality_config(config)


def test_eos_removed_only_from_metrics_and_check_not_receipt():
    model = ProbeModel()
    cfg = single_config(modes=["greedy"])
    cfg["prompts"][0]["check"] = {"type": "exact_first_line", "expected": "blue bird"}
    result = evaluate_sprint_generation(model, ProbeTokenizer(), config=cfg)
    output = result["outputs"][0]
    assert output["generated_token_ids"] == [1, 2, 0]
    assert output["generated_text"] == "blue bird <eos>"
    assert output["repetition"]["num_tokens"] == 2
    assert output["check"]["correct"]
    assert result["summary"]["status"] == "insufficient_length"
    assert result["summary"]["exact_answer_passes"] == 1
    assert "do not establish" in result["limitations"]
    json.dumps(result, allow_nan=False)


def test_short_wrong_answer_is_not_accepted_by_low_repetition():
    cfg = single_config(modes=["greedy"])
    cfg["prompts"][0]["check"] = {"type": "integer_first_line", "expected": "14"}
    result = evaluate_sprint_generation(ProbeModel((4, 0)), ProbeTokenizer(), config=cfg)
    assert not result["outputs"][0]["repetition"]["flagged"]
    assert result["summary"]["exact_answer_passes"] == 0
    assert result["summary"]["status"] == "insufficient_length"
    correct = evaluate_sprint_generation(ProbeModel((3, 0)), ProbeTokenizer(), config=cfg)
    assert correct["summary"]["exact_answer_passes"] == 1


@pytest.mark.parametrize("check_type,expected,text,correct", [
    ("exact_text", "blue bird", "  blue\t bird  \n", True),
    ("exact_text", "blue bird", "blue bird\nHere is an explanation.", False),
    ("exact_text", "blue bird", "blue bird and more", False),
    ("exact_text", "blue bird", "blue\nbird", False),
    ("integer_text", "14", " 14.\n", True),
    ("integer_text", "14", "14\nActually, 9.", False),
    ("integer_text", "14", "14 apples", False),
    ("integer_text", "14", "14\n.", False),
    ("exact_first_line", "blue bird", "blue bird\nUnwanted garbage.", True),
    ("integer_first_line", "14", "14\nActually, 9.", True),
])
def test_completion_checks_reject_extra_content_but_legacy_first_line_is_explicit(check_type, expected, text, correct):
    class TextTokenizer(ProbeTokenizer):
        def decode(self, ids, **kwargs):
            return text + ("<eos>" if ids and ids[-1] == self.eos_token_id else "")

    cfg = single_config(modes=["greedy"])
    cfg["prompts"][0]["check"] = {"type": check_type, "expected": expected}
    result = evaluate_sprint_generation(ProbeModel(), TextTokenizer(), config=cfg)
    output = result["outputs"][0]
    assert output["generated_text"].endswith("<eos>")
    assert output["check"]["observed_text"] == text
    assert output["check"]["correct"] is correct
    assert result["summary"]["exact_answer_passes"] == int(correct)
    if check_type.endswith("_text"):
        assert "whole generated completion" in output["check"]["scope"]
    else:
        assert "later text ignored" in output["check"]["scope"]


@pytest.mark.parametrize("fail", [False, True])
def test_rng_modes_weights_and_gradients_restored_even_on_failure(fail):
    model = ProbeModel(fail=fail)
    model.train()
    model.embed.eval()  # Preserve a mixed module-mode tree, not only the root.
    model.embed.weight.grad = torch.ones_like(model.embed.weight)
    before_weights = model.embed.weight.detach().clone()
    before_grad = model.embed.weight.grad.clone()
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    if fail:
        with pytest.raises(RuntimeError, match="deliberate"):
            evaluate_sprint_generation(model, ProbeTokenizer(), config=single_config())
    else:
        evaluate_sprint_generation(model, ProbeTokenizer(), config=single_config())
    assert model.training and not model.embed.training
    assert random.getstate() == py_state
    after_np = np.random.get_state()
    assert after_np[0] == np_state[0] and np.array_equal(after_np[1], np_state[1]) and after_np[2:] == np_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    torch.testing.assert_close(model.embed.weight, before_weights, rtol=0, atol=0)
    torch.testing.assert_close(model.embed.weight.grad, before_grad, rtol=0, atol=0)


def test_sampling_is_repeatable_without_global_rng_or_penalties():
    model = ProbeModel(uniform=True)
    cfg = single_config(max_new_tokens=32, temperature=.8, top_p=.9)
    first = evaluate_sprint_generation(model, ProbeTokenizer(), config=cfg)
    torch.manual_seed(600)
    second = evaluate_sprint_generation(model, ProbeTokenizer(), config=cfg)
    assert [row["generated_token_ids"] for row in first["outputs"]] == [row["generated_token_ids"] for row in second["outputs"]]
    assert first["outputs"][0]["generated_token_ids"] != first["outputs"][1]["generated_token_ids"]
    assert first["outputs"][0]["settings"]["repetition_penalty"] == 1.0
    assert first["outputs"][0]["settings"]["no_repeat_ngram_size"] == 0
    sample_only = evaluate_sprint_generation(model, ProbeTokenizer(), config={**cfg, "modes": ["sample"]})
    assert sample_only["outputs"][0]["generated_token_ids"] == first["outputs"][1]["generated_token_ids"]


def test_context_is_respected_and_oversize_prompt_is_not_truncated():
    model = ProbeModel((1,))
    model.config.max_position_embeddings = 34
    result = evaluate_sprint_generation(model, ProbeTokenizer(), config=single_config(modes=["greedy"]))
    assert len(result["outputs"][0]["generated_token_ids"]) == 32
    assert result["outputs"][0]["stop_reason"] == "context_limit"
    model.config.max_position_embeddings = 10
    insufficient = evaluate_sprint_generation(model, ProbeTokenizer(), config=single_config(modes=["greedy"]))
    assert insufficient["outputs"][0]["stop_reason"] == "insufficient_context"
    assert insufficient["outputs"][0]["generated_token_ids"] == []
    cfg = {"prompts": [{"prompt": "long"}], "modes": ["greedy"]}
    before_calls = len(model.calls)
    skipped = evaluate_sprint_generation(model, ProbeTokenizer(), config=cfg)
    assert skipped["summary"]["skipped_outputs"] == 1
    assert len(model.calls) == before_calls
    assert len(skipped["outputs"][0]["prompt_token_ids"]) == 20


@pytest.mark.parametrize("style, marker", [("plain", "Question:"), ("inst", "[INST]"), ("roles", "user:")])
def test_format_selection_is_explicit_and_does_not_wrap_prose(style, marker):
    cfg = {"prompt_format": style, "max_new_tokens": 1, "modes": ["greedy"]}
    result = evaluate_sprint_generation(ProbeModel(), ProbeTokenizer(), config=cfg)
    assert result["outputs"][0]["prompt"] == default_sprint_quality_prompts()[0]["prompt"]
    assert result["outputs"][1]["prompt"].startswith(marker)


def test_untrained_chat_template_is_never_selected_implicitly():
    with pytest.raises(ValueError, match="no chat template"):
        evaluate_sprint_generation(ProbeModel(), ProbeTokenizer(), config={"prompt_format": "tokenizer_chat"})


def test_bound_chat_template_overrides_current_template_without_mutation():
    class TemplateTokenizer(ProbeTokenizer):
        chat_template = "current, different template"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            assert tokenize is False and add_generation_prompt is True
            return kwargs.get("chat_template", self.chat_template) + messages[0]["content"]

    tokenizer = TemplateTokenizer()
    cfg = {"prompt_format": "tokenizer_chat", "modes": ["greedy"],
           "prompts": [{"name": "instruction", "kind": "instruction", "prompt": "hello"}]}
    bound = evaluate_sprint_generation(ProbeModel(), tokenizer, config=cfg, chat_template="verified: ")
    assert bound["outputs"][0]["prompt"] == "verified: hello"
    assert tokenizer.chat_template == "current, different template"
    unbound = evaluate_sprint_generation(ProbeModel(), tokenizer, config=cfg)
    assert unbound["outputs"][0]["prompt"] == "current, different templatehello"
    tokenizer.chat_template = None
    bound_without_current = evaluate_sprint_generation(ProbeModel(), tokenizer, config=cfg, chat_template="verified: ")
    assert bound_without_current["outputs"][0]["prompt"] == "verified: hello"
    assert tokenizer.chat_template is None


def test_bound_chat_render_failure_does_not_try_current_template_or_mutate_tokenizer():
    class BrokenTemplateTokenizer(ProbeTokenizer):
        chat_template = "current fallback must not be used"

        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("chat_template") == "broken verified template":
                raise ValueError("cannot render verified template")
            return "fallback"

    tokenizer, model = BrokenTemplateTokenizer(), ProbeModel()
    cfg = {"prompt_format": "tokenizer_chat", "prompts": [{"kind": "instruction", "prompt": "hello"}]}
    with pytest.raises(ValueError, match="cannot render verified template"):
        evaluate_sprint_generation(model, tokenizer, config=cfg, chat_template="broken verified template")
    assert len(tokenizer.calls) == 1 and not model.calls
    assert tokenizer.chat_template == "current fallback must not be used"
    assert model.training


@pytest.mark.parametrize("template", ["", "  \n ", False, 123])
def test_bound_chat_template_requires_nonempty_text(template):
    with pytest.raises(ValueError, match="chat_template must be a nonempty string"):
        evaluate_sprint_generation(ProbeModel(), ProbeTokenizer(), config=single_config(), chat_template=template)


def test_nonfinite_logits_produce_failure_receipt_not_sample():
    class NonfiniteModel(ProbeModel):
        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            output.logits[..., 5] = torch.nan
            return output

    result = evaluate_sprint_generation(NonfiniteModel(), ProbeTokenizer(), config=single_config())
    assert result["summary"]["status"] == "nonfinite_logits"
    assert result["summary"]["nonfinite_outputs"] == 2
    assert all(row["generated_token_ids"] == [] for row in result["outputs"])
    json.dumps(result, allow_nan=False)


def test_context_factory_enters_and_restores_per_output():
    entered = []

    @contextmanager
    def context():
        entered.append("enter")
        try:
            yield
        finally:
            entered.append("exit")

    evaluate_sprint_generation(ProbeModel(), ProbeTokenizer(), config=single_config(), context_factory=context)
    assert entered == ["enter", "exit", "enter", "exit"]


@pytest.mark.parametrize("attention_type", ["sdpa", "kda"])
def test_actual_sprint_eager_generation_is_read_only_on_cpu(attention_type):
    from yylab.models.sprint import SprintConfig, SprintForCausalLM

    model = SprintForCausalLM(SprintConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, max_position_embeddings=32, loss_backend="torch",
        attention_type=attention_type, kda_backend="reference", ple_type="full",
        pad_token_id=0, eos_token_id=0,
    ))
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    result = evaluate_sprint_generation(model, ProbeTokenizer(), config=single_config(max_new_tokens=4))
    assert result["summary"]["nonfinite_outputs"] == 0
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
