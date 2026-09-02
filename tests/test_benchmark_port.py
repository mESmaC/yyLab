from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from yylab.benchmark.backend import (
    GenerationResponse,
    YyLabBenchmarkBackend,
    _format_messages,
    _thinking_token_counts,
)
from yylab.benchmark.models import SuiteError, load_suite
from yylab.benchmark.performance import (
    percentile,
    run_training_performance,
    summarize_step_metrics,
    synthetic_tokens_per_step,
)
from yylab.benchmark.quality import (
    _fingerprint,
    _generation_comparison_fingerprint,
    _oracle_records,
    _workload_contract,
    compare_summaries,
    policies_for_config,
    rescore_quality_artifact,
    run_quality_benchmark,
    summarize_policy,
    summarize_results,
)
from yylab.benchmark.scoring import (
    SCORER_VERSION,
    ScoreSpecError,
    score_response,
    score_response_semantic,
    validate_score_spec,
    visible_answer,
)

SUITE = Path(__file__).parents[1] / "yylab" / "benchmark" / "suites" / "molor-core-v1.json"
SUITE_FINGERPRINT = "0f2be48bcdec9a29d412f4c56f1c671fc276af73df572d78046c841f7afc4de5"


def test_bundled_molor_suite_keeps_its_frozen_identity() -> None:
    suite = load_suite(SUITE)

    assert suite.suite_id == "molor-core-v1"
    assert suite.version == "1.0.0"
    assert suite.fingerprint == SUITE_FINGERPRINT
    assert SCORER_VERSION == "molor.deterministic-scorers.v1.2"
    assert len(suite.cases) == 64
    assert [len(suite.select(profile=name)) for name in ("smoke", "quick", "full")] == [
        10,
        30,
        64,
    ]
    assert tuple(suite.category_weights) == (
        "general_reasoning",
        "quantitative",
        "programming",
        "embedded",
        "terminal",
        "architecture",
        "business",
        "finance",
        "instruction",
        "context",
    )
    assert sum(suite.category_weights.values()) == pytest.approx(1.0)
    assert suite.generation == {
        "max_new_tokens": 48,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "enable_thinking": False,
        "context_token_budget": 8192,
        "cache_mode": "dynamic",
        "seed": 8419,
    }


def test_suite_selection_is_ordered_filtered_and_strict() -> None:
    suite = load_suite(SUITE)
    smoke = suite.select(profile="smoke")

    assert [case.case_id for case in smoke] == list(suite.profiles["smoke"])
    selected = suite.select(profile="full", categories=("finance",), limit=2)
    assert len(selected) == 2
    assert all(case.category == "finance" for case in selected)
    with pytest.raises(SuiteError, match="unknown profile"):
        suite.select(profile="missing")
    with pytest.raises(SuiteError, match="unknown categories"):
        suite.select(profile="full", categories=("missing",))
    with pytest.raises(SuiteError, match="positive integer"):
        suite.select(profile="full", limit=0)


def test_case_messages_prefer_case_system_and_fall_back_to_suite_system() -> None:
    suite = load_suite(SUITE)
    case = suite.cases[0]
    messages = case.messages("default system")

    assert messages[-1] == {"role": "user", "content": case.prompt}
    assert messages[0] == {
        "role": "system",
        "content": case.system.strip() or "default system",
    }


def test_suite_loader_rejects_semantic_drift(tmp_path: Path) -> None:
    raw = json.loads(SUITE.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    bad = tmp_path / "bad-suite.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SuiteError, match="unknown fields"):
        load_suite(bad)

    raw.pop("unexpected")
    raw["profiles"]["smoke"].append(raw["profiles"]["smoke"][0])
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SuiteError, match="duplicates"):
        load_suite(bad)


@pytest.mark.parametrize(
    ("response", "spec", "expected"),
    [
        ("Answer: B", {"type": "choice", "answer": "B"}, 1.0),
        ("Final answer: 3/8", {"type": "numeric", "answer": 0.375}, 1.0),
        (".375", {"type": "numeric", "answer": 0.375}, 1.0),
        ("2000%", {"type": "numeric", "answer": 20}, 0.0),
        (
            "37.5%",
            {"type": "numeric", "answer": 0.375, "percent_mode": "fraction"},
            1.0,
        ),
        (
            '```json\n{"status":"ok","count":3}\n```',
            {"type": "json", "expected": {"status": "ok", "count": 3}},
            1.0,
        ),
        (
            "Get-ChildItem -LiteralPath $targetPath",
            {
                "type": "command",
                "answers": ["Get-ChildItem -LiteralPath $targetPath"],
            },
            1.0,
        ),
        (
            "get-childitem -literalpath $targetpath",
            {
                "type": "command",
                "answers": ["Get-ChildItem -LiteralPath $targetPath"],
            },
            0.0,
        ),
        (
            "The alpha marker and beta marker are present.",
            {"type": "contains", "all": ["alpha"], "any": ["beta", "gamma"]},
            1.0,
        ),
        (
            "Status: READY",
            {"type": "regex", "all": [r"status:\s+ready"], "none": ["failed"]},
            1.0,
        ),
        ("Final!", {"type": "exact", "answer": "final"}, 1.0),
        (
            "Answer: B",
            {
                "type": "composite",
                "components": [
                    {"type": "choice", "answer": "B", "weight": 3},
                    {"type": "contains", "all": ["missing"], "weight": 1},
                ],
            },
            0.75,
        ),
    ],
)
def test_ported_scorers_match_molor_semantics(
    response: str,
    spec: dict[str, object],
    expected: float,
) -> None:
    result = score_response(response, spec)

    assert result.score == pytest.approx(expected)
    assert result.passed is (expected == 1.0)
    assert result.details["scorer_type"] == spec["type"]


def test_reasoning_blocks_are_excluded_from_answer_scoring() -> None:
    response = "<think>Answer: A\nprivate working</think>\nAnswer: C"

    assert visible_answer(response) == "Answer: C"
    assert score_response(response, {"type": "choice", "answer": "C"}).score == 1.0
    assert visible_answer("<think>unfinished private reasoning") == ""


def test_json_and_terminal_scorers_keep_strict_output_contracts() -> None:
    json_spec = {
        "type": "json",
        "expected": {"status": "ok", "count": 3},
        "exact": True,
        "allow_fence": False,
        "minified": True,
    }
    assert score_response('{"status":"ok","count":3}', json_spec).score == 1.0
    assert score_response('{"status":"ok","count":3,"extra":1}', json_spec).score == 0.0
    assert score_response('{"status": "ok", "count": 3}', json_spec).score == 0.0
    assert score_response('```json\n{"status":"ok","count":3}\n```', json_spec).score == 0.0

    command_spec = {"type": "command", "answers": ["rg -c ERROR app.log"]}
    assert score_response("rg -c ERROR app.log", command_spec).score == 1.0
    assert score_response("Command: rg -c ERROR app.log", command_spec).score == 0.0
    assert score_response("`rg -c ERROR app.log`", command_spec).score == 0.0


def test_semantic_choice_credits_explanatory_option_content() -> None:
    spec = {"type": "choice", "answer": "B"}
    prompt = """What happens to a bond's price when yields rise?
A. It rises.
B. It falls.
C. It stays constant.
D. It becomes zero.
"""
    response = "The bond's market price generally falls when yields rise."

    assert score_response(response, spec).score == 0.0
    result = score_response_semantic(response, spec, prompt=prompt)
    assert result.score == 1.0
    assert result.extracted == "B"
    assert result.details["strict_score"] == 0.0


def test_semantic_choice_credits_leading_label_with_explanation() -> None:
    spec = {"type": "choice", "answer": "C"}
    response = "C volatile prevents the compiler from eliding the device access."

    assert score_response(response, spec).score == 0.0
    assert score_response_semantic(response, spec).score == 1.0


def test_semantic_choice_credits_named_option_text() -> None:
    spec = {"type": "choice", "answer": "C"}
    prompt = """Which fallacy is present?
A. Denying the antecedent
B. Circular reasoning
C. Affirming the consequent
D. False dilemma
"""
    response = "This is affirming the consequent because the inference reverses implication."

    assert score_response(response, spec).score == 0.0
    assert score_response_semantic(response, spec, prompt=prompt).score == 1.0


def test_semantic_choice_rejects_prompt_echo_with_all_options() -> None:
    spec = {"type": "choice", "answer": "C"}
    prompt = """Choose one.
A. Alpha
B. Beta
C. Gamma
D. Delta
"""

    result = score_response_semantic(prompt, spec, prompt=prompt)

    assert result.score == 0.0
    assert result.extracted is None
    assert result.details["prompt_echo"] == ["A", "B", "C", "D"]


def test_semantic_choice_rejects_negated_expected_option() -> None:
    spec = {"type": "choice", "answer": "B"}
    prompt = """What happens next?
A. It rises.
B. It falls.
C. It stays constant.
D. It becomes zero.
"""
    response = "The claim that it falls is false."

    assert score_response(response, spec).score == 0.0
    assert score_response_semantic(response, spec, prompt=prompt).score == 0.0


def test_semantic_exact_credits_requested_value_inside_explanation() -> None:
    spec = {"type": "exact", "answer": "9QX-441"}
    response = "The blue diagnostic unit has serial 9QX-441."

    assert score_response(response, spec).score == 0.0
    result = score_response_semantic(response, spec)
    assert result.score == 1.0
    assert result.extracted == "9QX-441"


def test_semantic_exact_rejects_negated_requested_value() -> None:
    spec = {"type": "exact", "answer": "9QX-441"}
    response = "The serial is not 9QX-441; the identifier remains unknown."

    assert score_response(response, spec).score == 0.0
    assert score_response_semantic(response, spec).score == 0.0


def test_semantic_choice_negation_does_not_cross_sentence_boundary() -> None:
    prompt = (
        "The alert did not fire. What follows?\n"
        "A. The server was down\n"
        "B. The server was not down\n"
        "C. The alert fired\n"
        "D. Nothing can be inferred"
    )
    response = (
        "The rule 'If the server is down, the alert fires' is always true. "
        "The alert did not fire. The server was not down."
    )
    spec = {"type": "choice", "answer": "B"}

    result = score_response_semantic(response, spec, prompt=prompt)

    assert result.score == 1.0
    assert result.extracted == "B"


def test_semantic_numeric_credits_supported_value_but_honors_wrong_final() -> None:
    spec = {"type": "numeric", "answer": 42}
    supported = "Multiplying 6 by 7 yields 42, from factors 6 and 7."
    contradicted = "A tentative calculation gave 42, but final answer: 43."

    assert score_response(supported, spec).score == 0.0
    assert score_response_semantic(supported, spec).score == 1.0
    assert score_response_semantic(contradicted, spec).score == 0.0


@pytest.mark.parametrize(
    ("response", "expected", "expected_score", "mode"),
    [
        ("500 * 0.1 + (-40) = 10 -", 10, 0.0, "unfinished-expression"),
        ("500 * 0.1 + (-40) = 10 - 40 = -30", 10, 0.0, "terminal-equation"),
        (r"40 + 30 - 15 = 55; therefore \\boxed{55}", 55, 1.0, "boxed-conclusion"),
        ("0x2F + 0x19 = 0x48, which is 72 in decimal.", 72, 1.0, "supported-conclusion"),
        ("20 * 1 = 20 milliseconds.", 20, 1.0, "terminal-equation"),
    ],
)
def test_semantic_numeric_ranks_conclusions_over_intermediate_values(
    response: str,
    expected: int,
    expected_score: float,
    mode: str,
) -> None:
    result = score_response_semantic(response, {"type": "numeric", "answer": expected})

    assert result.score == expected_score
    assert result.details["mode"] == mode


def test_semantic_exact_ignores_separator_spacing_but_strict_does_not() -> None:
    spec = {"type": "exact", "answer": "1,4,9,16,25", "strip_punctuation": False}

    assert score_response("1, 4, 9, 16, 25", spec).score == 0.0
    assert score_response_semantic("1, 4, 9, 16, 25", spec).score == 1.0


@pytest.mark.parametrize(
    "response",
    [
        "Command: rg -c ERROR app.log",
        "```bash\nrg -c ERROR app.log\n```",
    ],
)
def test_semantic_command_credits_prefixed_or_fenced_command(response: str) -> None:
    spec = {"type": "command", "answers": ["rg -c ERROR app.log"]}

    assert score_response(response, spec).score == 0.0
    assert score_response_semantic(response, spec).score == 1.0


def test_semantic_json_credits_fenced_nonminified_expected_values() -> None:
    spec = {
        "type": "json",
        "expected": {"status": "ok", "count": 3},
        "exact": True,
        "allow_fence": False,
        "minified": True,
    }
    response = """```json
{
  "status": "ok",
  "count": 3,
  "explanation": "The requested values are present."
}
```"""

    assert score_response(response, spec).score == 0.0
    assert score_response_semantic(response, spec).score == 1.0


@pytest.mark.parametrize(
    ("response", "spec", "prompt"),
    [
        (
            "It rises as yields increase.",
            {"type": "choice", "answer": "B"},
            "A. It rises.\nB. It falls.\nC. It stays constant.\nD. It becomes zero.",
        ),
        ("Serial 8QX-442.", {"type": "exact", "answer": "9QX-441"}, None),
        ("The result is 41.", {"type": "numeric", "answer": 42}, None),
        (
            "Command: rg -c WARN app.log",
            {"type": "command", "answers": ["rg -c ERROR app.log"]},
            None,
        ),
    ],
)
def test_semantic_scorers_keep_truly_wrong_answers_at_zero(
    response: str,
    spec: dict[str, object],
    prompt: str | None,
) -> None:
    assert score_response_semantic(response, spec, prompt=prompt).score == 0.0


def test_scorer_validation_accepts_external_declarations_but_not_unknown_types() -> None:
    validate_score_spec({"type": "external", "verifier": "wsl-local-v1"})
    with pytest.raises(ScoreSpecError, match="unsupported scorer type"):
        validate_score_spec({"type": "not-a-scorer"})
    with pytest.raises(ScoreSpecError, match="external.verifier"):
        validate_score_spec({"type": "external", "verifier": "bad verifier"})


def test_benchmark_prompt_format_forwards_qwen_thinking_control() -> None:
    class Tokenizer:
        calls: list[dict[str, object]] = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append({"messages": messages, **kwargs})
            return "rendered prompt"

    tokenizer = Tokenizer()
    messages = [{"role": "user", "content": "question"}]

    assert _format_messages(tokenizer, messages, True) == "rendered prompt"
    assert tokenizer.calls == [
        {
            "messages": messages,
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
        }
    ]


def test_benchmark_prompt_format_supports_templates_without_thinking_argument() -> None:
    class LegacyTokenizer:
        calls = 0

        def apply_chat_template(self, _messages, **kwargs):
            self.calls += 1
            if "enable_thinking" in kwargs:
                raise TypeError("unsupported argument")
            return "legacy prompt"

    tokenizer = LegacyTokenizer()

    assert _format_messages(tokenizer, [{"role": "user", "content": "question"}], False) == ("legacy prompt")
    assert tokenizer.calls == 2


def test_benchmark_prompt_format_fails_when_thinking_control_is_unsupported() -> None:
    class LegacyTokenizer:
        def apply_chat_template(self, _messages, **kwargs):
            if "enable_thinking" in kwargs:
                raise TypeError("unsupported argument")
            return "legacy prompt"

    with pytest.raises(RuntimeError, match="cannot honor enable_thinking=True"):
        _format_messages(
            LegacyTokenizer(),
            [{"role": "user", "content": "question"}],
            True,
        )


def test_benchmark_raw_prompt_uses_plain_message_content_without_role_tokens() -> None:
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            raise AssertionError("raw mode must not use a chat template")

    messages = [
        {"role": "system", "content": "Answer precisely."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    assert _format_messages(Tokenizer(), messages, False, "raw") == ("Answer precisely.\n\nWhat is 2 + 2?")
    with pytest.raises(ValueError, match="thinking requires chat"):
        _format_messages(Tokenizer(), messages, True, "raw")


def test_benchmark_backend_honors_declared_model_family(tmp_path: Path) -> None:
    misleading = tmp_path / "looks-like-spalmer.pt"
    misleading.write_bytes(b"not loaded by this test")

    backend = YyLabBenchmarkBackend(
        checkpoint_ref=str(misleading),
        quant="bf16",
        adapter_path=None,
        adapter_arch="",
        model_family="hf",
    )

    assert backend.model_family == "hf"
    assert backend.identity()["model_family"] == "hf"

    class Engine:
        def __init__(self) -> None:
            self.specs = []

        def load(self, spec):
            self.specs.append(spec)
            return []

    backend.engine = Engine()
    backend.configure_policy("base")
    assert backend.engine.specs[0].model_family == "hf"


@pytest.mark.parametrize(
    ("values", "prompt_open", "expected"),
    [
        ([100, 41, 101, 77, 2], False, (3, 2, 2)),
        ([41, 101, 77, 2], True, (2, 2, 1)),
        ([77, 2], False, (0, 2, None)),
        ([100, 41, 2], False, (3, 0, None)),
        ([41, 2], True, (2, 0, None)),
    ],
)
def test_thinking_token_accounting_handles_natural_direct_and_unclosed_outputs(
    values: list[int],
    prompt_open: bool,
    expected: tuple[int, int, int | None],
) -> None:
    assert (
        _thinking_token_counts(
            values,
            [100],
            [101],
            prompt_has_open_thinking=prompt_open,
        )
        == expected
    )


def test_direct_backend_counts_generated_ids_and_applies_generation_controls() -> None:
    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, _messages, **_kwargs):
            return "prompt"

        def __call__(self, _text, *, return_tensors):
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[10, 11, 12]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return " ".join(str(int(value)) for value in token_ids)

    class Model:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def modules(self):
            return ()

        def generate(self, **kwargs):
            self.kwargs = kwargs
            suffix = torch.tensor([[91, 2]], dtype=torch.long)
            return torch.cat((kwargs["input_ids"], suffix), dim=1)

    class Engine:
        tokenizer = Tokenizer()
        model = Model()
        cuda = False

        @staticmethod
        def is_loaded() -> bool:
            return True

    backend = YyLabBenchmarkBackend(
        checkpoint_ref="fixture",
        quant="bf16",
        adapter_path=None,
        adapter_arch="",
    )
    backend.engine = Engine()
    backend._policy = "base"

    response = backend.generate(
        messages=[{"role": "user", "content": "question"}],
        max_new_tokens=2,
        temperature=0.7,
        top_p=0.8,
        top_k=17,
        enable_thinking=False,
        cache_mode="disabled",
        seed=123,
    )

    assert response.text == "91 2"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 2
    assert response.finish_reason == "stop"
    assert response.tokens_per_second is not None and response.tokens_per_second > 0
    assert Engine.model.kwargs["do_sample"] is True
    assert Engine.model.kwargs["temperature"] == 0.7
    assert Engine.model.kwargs["top_p"] == 0.8
    assert Engine.model.kwargs["top_k"] == 17
    assert Engine.model.kwargs["use_cache"] is False


@pytest.mark.parametrize(
    ("phase_tokens", "thinking_budget", "expected_thinking", "expected_exhausted"),
    [
        ((41, 42, 43, 44), 4, 4, True),
        ((41, 42, 2), 8, 3, False),
        ((41, 101, 2), 8, 2, False),
    ],
)
def test_thinking_backend_preserves_full_answer_budget_without_visible_answer(
    phase_tokens: tuple[int, ...],
    thinking_budget: int,
    expected_thinking: int,
    expected_exhausted: bool,
) -> None:
    open_id = 100
    close_id = 101

    class Tokenizer:
        eos_token_id = 2
        model_max_length = 128

        def apply_chat_template(self, _messages, **_kwargs):
            return "prompt"

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            if text == "<think>":
                return [open_id]
            if text == "<think>\n":
                return [open_id, 90]
            assert text == "</think>"
            return [close_id]

        def __call__(self, text, *, return_tensors, **kwargs):
            assert return_tensors == "pt"
            if text == "prompt":
                assert not kwargs
                return {
                    "input_ids": torch.tensor([[10, 11]]),
                    "attention_mask": torch.ones((1, 2), dtype=torch.long),
                }
            assert kwargs == {
                "return_attention_mask": False,
                "add_special_tokens": False,
            }
            return {"input_ids": torch.tensor([[90, close_id]])}

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            values = [int(value) for value in token_ids]
            pieces = {
                open_id: "<think>",
                41: "reason-a",
                42: "reason-b",
                43: "reason-c",
                44: "reason-d",
                90: "\n",
                close_id: "</think>",
                77: "Answer: B",
                2: "",
            }
            return " ".join(pieces[value] for value in values)

    class Model:
        device = torch.device("cpu")
        config = type("Config", (), {"max_position_embeddings": 128})()

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def modules(self):
            return ()

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                generated = kwargs["input_ids"]
                # This reproduces the real adapter regression: reasoning starts
                # immediately, without generating an opening <think> marker.
                for token in phase_tokens:
                    generated = torch.cat((generated, torch.tensor([[token]])), dim=1)
                    should_stop = kwargs["stopping_criteria"](generated, None)
                assert bool(should_stop[0]) is expected_exhausted
                return generated
            return torch.cat((kwargs["input_ids"], torch.tensor([[77, 2]])), dim=1)

    class Engine:
        tokenizer = Tokenizer()
        model = Model()
        cuda = False

        @staticmethod
        def is_loaded() -> bool:
            return True

    backend = YyLabBenchmarkBackend(
        checkpoint_ref="fixture",
        quant="bf16",
        adapter_path=None,
        adapter_arch="",
    )
    backend.engine = Engine()
    backend._policy = "base"

    response = backend.generate(
        messages=[{"role": "user", "content": "question"}],
        max_new_tokens=3,
        max_thinking_tokens=thinking_budget,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        enable_thinking=True,
        cache_mode="dynamic",
        seed=123,
    )

    assert visible_answer(response.text) == "Answer: B"
    assert response.thinking_tokens == expected_thinking
    assert response.answer_tokens == 2
    assert response.completion_tokens == len(phase_tokens) + 2
    assert response.thinking_budget_exhausted is expected_exhausted
    assert response.finish_reason == "stop"
    assert len(Engine.model.calls) == 2
    assert Engine.model.calls[0]["max_new_tokens"] == thinking_budget + 3
    assert Engine.model.calls[0]["input_ids"][0, -2:].tolist() == [open_id, 90]
    assert Engine.model.calls[1]["max_new_tokens"] == 3
    assert Engine.model.calls[1]["input_ids"][0].tolist() == [10, 11]
    assert Engine.model.calls[1]["do_sample"] is True


def test_route_accumulator_reports_block_token_usage_and_removes_hooks() -> None:
    from yylab.benchmark.backend import _RouteAccumulator
    from yylab.mole.layer import SharedTokenRouter

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = SharedTokenRouter(hidden_size=4, num_experts=3, top_k=2)

    model = Model()
    accumulator = _RouteAccumulator(model)
    model.router(torch.randn(5, 4))

    metrics = accumulator.close()

    assert metrics is not None
    assert metrics["router_calls"] == 1
    assert metrics["routed_token_blocks"] == 5
    assert sum(metrics["expert_weight_share"]) == pytest.approx(1.0, abs=2e-6)
    assert sum(metrics["expert_selection_share"]) == pytest.approx(1.0, abs=2e-6)
    assert not model.router._forward_hooks


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"policy": "base"}, ["base"]),
        (
            {"policy": "adapter", "adapter_path": "adapter", "adapter_arch": "lora"},
            ["adapter"],
        ),
        (
            {
                "policy": "dynamic",
                "adapter_path": "adapter",
                "adapter_arch": "mole",
                "num_experts": 3,
            },
            ["dynamic"],
        ),
        (
            {
                "policy": "matrix",
                "adapter_path": "adapter",
                "adapter_arch": "mole",
                "num_experts": 3,
            },
            ["base", "expert:0", "expert:1", "expert:2", "dynamic"],
        ),
    ],
)
def test_quality_policy_plans_match_yylab_adapter_types(
    config: dict[str, object],
    expected: list[str],
) -> None:
    assert policies_for_config(config) == expected


def test_quality_policy_plan_rejects_missing_or_out_of_range_experts() -> None:
    with pytest.raises(ValueError, match="trained MoLE adapter"):
        policies_for_config({"policy": "matrix"})
    with pytest.raises(ValueError, match="requires a MoLE adapter"):
        policies_for_config({"policy": "dynamic", "adapter_path": "adapter", "adapter_arch": "lora"})
    with pytest.raises(ValueError, match="outside this adapter's 2 slots"):
        policies_for_config(
            {
                "policy": "expert:2",
                "adapter_path": "adapter",
                "adapter_arch": "mole",
                "num_experts": 2,
            }
        )


def test_spalmer_quality_policies_use_native_expert_bank_without_adapter() -> None:
    base = {"model_family": "spalmer", "checkpoint_ref": "checkpoint.pt"}

    assert policies_for_config({**base, "policy": "base"}) == ["base"]
    assert policies_for_config({**base, "policy": "dynamic"}) == ["dynamic"]
    with pytest.raises(ValueError, match="base or native dynamic"):
        policies_for_config({**base, "policy": "matrix"})


def _quality_record(
    *,
    policy: str,
    case_id: str,
    category: str,
    score: float | None,
    semantic_score: float | None = None,
    repetition: int = 0,
    elapsed: float = 1.0,
    throughput: float | None = 10.0,
    error: str | None = None,
) -> dict[str, object]:
    record = {
        "policy": policy,
        "case_id": case_id,
        "category": category,
        "case_weight": 1.0,
        "repetition": repetition,
        "score": score,
        "passed": score == 1.0,
        "error": error,
        "telemetry": {
            "elapsed_seconds": elapsed,
            "completion_tokens": 4,
            "tokens_per_second": throughput,
        },
        "routing": None,
    }
    if semantic_score is not None:
        record["semantic_score"] = semantic_score
        record["semantic_passed"] = semantic_score == 1.0
    return record


def test_policy_summary_uses_category_macro_weights_and_performance_percentiles() -> None:
    suite = load_suite(SUITE)
    records = [
        _quality_record(
            policy="dynamic",
            case_id="G01",
            category="general_reasoning",
            score=1.0,
            elapsed=1.0,
            throughput=20.0,
        ),
        _quality_record(
            policy="dynamic",
            case_id="G02",
            category="general_reasoning",
            score=1.0,
            elapsed=2.0,
            throughput=10.0,
        ),
        _quality_record(
            policy="dynamic",
            case_id="E01",
            category="embedded",
            score=0.0,
            elapsed=3.0,
            throughput=5.0,
        ),
    ]

    summary = summarize_policy(suite, "dynamic", records, expected=3)

    expected_quality = 1000 * 0.10 / (0.10 + 0.16)
    assert summary["quality_index"] == pytest.approx(expected_quality)
    assert summary["coverage"] == {
        "expected": 3,
        "recorded": 3,
        "scored": 3,
        "errors": 0,
        "fraction": 1.0,
    }
    assert summary["performance"]["latency_seconds_median"] == 2.0
    assert summary["performance"]["latency_seconds_p95"] == pytest.approx(2.9)
    assert summary["performance"]["tokens_per_second_median"] == 10.0
    assert summary["performance"]["completion_tokens_total"] == 12
    assert summary["comparable"] is True


def test_policy_summary_keeps_molor_strict_but_leads_with_semantic_correctness() -> None:
    suite = load_suite(SUITE)
    records = [
        _quality_record(
            policy="dynamic",
            case_id="G01",
            category="general_reasoning",
            score=0.0,
            semantic_score=1.0,
        )
    ]

    summary = summarize_policy(suite, "dynamic", records, expected=1)

    assert summary["quality_index"] == 0.0
    assert summary["strict_quality_index"] == 0.0
    assert summary["semantic_quality_index"] == 1000.0
    assert summary["semantic_strict_gap"] == 1000.0
    assert summary["categories"]["general_reasoning"]["semantic_passed"] == 1


def test_correct_explanation_is_never_a_semantic_regression_against_bare_answer() -> None:
    common = {
        "suite_fingerprint": SUITE_FINGERPRINT,
        "profile": "fixture",
        "case_ids": ["G01"],
        "repetitions": 1,
        "generation_comparison_fingerprint": "44" * 32,
        "primary_policy": "dynamic",
    }
    baseline = {
        **common,
        "workload_fingerprint": "45" * 32,
        "quality_index": 1000.0,
        "semantic_quality_index": 1000.0,
        "result_cells": [
            {
                "policy": "dynamic",
                "case_id": "G01",
                "repetition": 0,
                "score": 1.0,
                "semantic_score": 1.0,
            }
        ],
    }
    explanatory = {
        **common,
        "workload_fingerprint": "46" * 32,
        "quality_index": 0.0,
        "semantic_quality_index": 1000.0,
        "result_cells": [
            {
                "policy": "dynamic",
                "case_id": "G01",
                "repetition": 0,
                "score": 0.0,
                "semantic_score": 1.0,
            }
        ],
    }

    comparison = compare_summaries(explanatory, baseline)

    assert comparison["quality_index_delta"] == -1000.0
    assert comparison["regression"] is True
    assert comparison["semantic_quality_index_delta"] == 0.0
    assert comparison["semantic_regression"] is False
    assert (comparison["semantic_wins"], comparison["semantic_ties"], comparison["semantic_losses"]) == (
        0,
        1,
        0,
    )


def test_matrix_summary_builds_deterministic_forced_oracle_and_dynamic_regret() -> None:
    suite = load_suite(SUITE)
    policies = ["base", "expert:0", "expert:1", "dynamic"]
    records = [
        _quality_record(policy="base", case_id="G01", category="general_reasoning", score=0.0),
        _quality_record(policy="expert:0", case_id="G01", category="general_reasoning", score=1.0),
        _quality_record(policy="expert:1", case_id="G01", category="general_reasoning", score=1.0),
        _quality_record(policy="dynamic", case_id="G01", category="general_reasoning", score=0.0),
        _quality_record(policy="base", case_id="Q01", category="quantitative", score=0.0),
        _quality_record(policy="expert:0", case_id="Q01", category="quantitative", score=0.0),
        _quality_record(policy="expert:1", case_id="Q01", category="quantitative", score=1.0),
        _quality_record(policy="dynamic", case_id="Q01", category="quantitative", score=1.0),
    ]

    summary = summarize_results(
        suite,
        records,
        policies,
        profile="fixture",
        repetitions=1,
        case_ids=("G01", "Q01"),
    )

    matrix = summary["matrix"]
    assert summary["primary_policy"] == "dynamic"
    general_weight = suite.category_weights["general_reasoning"]
    quantitative_weight = suite.category_weights["quantitative"]
    assert summary["quality_index"] == pytest.approx(
        1000 * quantitative_weight / (general_weight + quantitative_weight)
    )
    assert matrix["forced_oracle"]["quality_index"] == 1000.0
    assert matrix["dynamic_regret"] == pytest.approx(1000.0 - float(summary["quality_index"]))
    assert matrix["note"] == ("Expert slots are anonymous; this matrix does not claim domain-route accuracy.")
    oracle_cells = {(item["case_id"], item["oracle_policy"]) for item in _oracle_records(records)}
    # Equal scores resolve to the lower expert index, while a strict win selects expert 1.
    assert oracle_cells == {("G01", "expert:0"), ("Q01", "expert:1")}


def test_summary_comparison_is_paired_and_rejects_workload_drift() -> None:
    common = {
        "suite_fingerprint": SUITE_FINGERPRINT,
        "profile": "smoke",
        "case_ids": ["G01", "G02", "G03"],
        "repetitions": 1,
        "workload_fingerprint": "11" * 32,
        "primary_policy": "dynamic",
    }
    baseline = {
        **common,
        "quality_index": 400.0,
        "result_cells": [
            {"policy": "dynamic", "case_id": "G01", "repetition": 0, "score": 0.0},
            {"policy": "dynamic", "case_id": "G02", "repetition": 0, "score": 1.0},
            {"policy": "dynamic", "case_id": "G03", "repetition": 0, "score": 0.5},
        ],
    }
    candidate = {
        **common,
        "quality_index": 500.0,
        "result_cells": [
            {"policy": "dynamic", "case_id": "G01", "repetition": 0, "score": 1.0},
            {"policy": "dynamic", "case_id": "G02", "repetition": 0, "score": 1.0},
            {"policy": "dynamic", "case_id": "G03", "repetition": 0, "score": 0.0},
        ],
    }

    comparison = compare_summaries(candidate, baseline)

    assert comparison["quality_index_delta"] == 100.0
    assert comparison["paired_cells"] == 3
    assert (comparison["wins"], comparison["ties"], comparison["losses"]) == (1, 1, 1)
    assert comparison["mean_cell_delta"] == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="profile"):
        compare_summaries({**candidate, "profile": "full"}, baseline)


def test_summary_comparison_rejects_fingerprint_mismatch_and_partial_cells() -> None:
    common = {
        "suite_fingerprint": SUITE_FINGERPRINT,
        "profile": "smoke",
        "case_ids": ["G01", "G02"],
        "repetitions": 1,
        "workload_fingerprint": "22" * 32,
        "primary_policy": "dynamic",
        "quality_index": 500.0,
        "result_cells": [
            {"policy": "dynamic", "case_id": "G01", "repetition": 0, "score": 1.0},
            {"policy": "dynamic", "case_id": "G02", "repetition": 0, "score": 0.0},
        ],
    }

    with pytest.raises(ValueError, match="workload_fingerprint"):
        compare_summaries({**common, "workload_fingerprint": "33" * 32}, common)
    with pytest.raises(ValueError, match="every primary-policy workload cell"):
        compare_summaries({**common, "result_cells": common["result_cells"][:-1]}, common)
    with pytest.raises(ValueError, match="every primary-policy workload cell"):
        compare_summaries(common, {**common, "result_cells": common["result_cells"][:-1]})


def test_summary_comparison_rejects_different_generation_implementation() -> None:
    common = {
        "suite_fingerprint": SUITE_FINGERPRINT,
        "profile": "smoke",
        "case_ids": ["G01"],
        "repetitions": 1,
        "generation_comparison_fingerprint": "22" * 32,
        "primary_policy": "base",
        "quality_index": 1000.0,
        "result_cells": [{"policy": "base", "case_id": "G01", "repetition": 0, "score": 1.0}],
    }

    with pytest.raises(ValueError, match="generation implementation"):
        compare_summaries(
            {**common, "generation_implementation_fingerprint": "33" * 32},
            {**common, "generation_implementation_fingerprint": "44" * 32},
        )


def test_disabled_thinking_budget_is_a_noop_for_workload_and_legacy_comparison() -> None:
    suite = load_suite(SUITE)
    cases = suite.select(profile="smoke", limit=2)
    args = {
        "profile": "smoke",
        "repetitions": 1,
        "categories": (),
        "limit": 2,
    }
    first = _workload_contract(
        suite,
        cases,
        ["base"],
        {"enable_thinking": False, "max_thinking_tokens": 1024},
        **args,
    )
    second = _workload_contract(
        suite,
        cases,
        ["base"],
        {"enable_thinking": False, "max_thinking_tokens": 4096},
        **args,
    )

    assert _fingerprint(first) == _fingerprint(second)
    legacy = json.loads(json.dumps(first))
    for cell in legacy["cells"]:
        cell["request"].pop("max_thinking_tokens")
        cell["request"].pop("prompt_mode")
    legacy.pop("runtime_context_limit")
    legacy.pop("prompt_mode")
    legacy.pop("context_mode")
    legacy.pop("rope_scaling")
    legacy.pop("rope_factor")
    legacy.pop("max_context_tokens")
    assert _generation_comparison_fingerprint(first) == _generation_comparison_fingerprint(legacy)


def _perfect_response(case) -> str:
    scorer = case.scorer
    scorer_type = scorer["type"]
    if scorer_type == "choice":
        return f"Answer: {scorer['answer']}"
    if scorer_type == "numeric":
        return f"Answer: {scorer['answer']}"
    if scorer_type == "command":
        return str(scorer["answers"][0])
    if scorer_type == "json":
        return json.dumps(scorer["expected"], separators=(",", ":"))
    if scorer_type == "exact":
        return str(scorer.get("answer") or scorer["answers"][0])
    raise AssertionError(f"fixture has no perfect response for {scorer_type}")


class _PerfectFakeBackend:
    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.policies: list[str] = []
        self.requests: list[dict[str, object]] = []
        self.closed = False

    def configure_policy(self, policy: str) -> list[str]:
        self.policies.append(policy)
        return ["fixture backend"]

    def generate(self, *, messages, **request) -> GenerationResponse:
        self.requests.append(request)
        prompt = messages[-1]["content"]
        return GenerationResponse(
            text=self.answers[prompt],
            elapsed_seconds=0.25,
            prompt_tokens=8,
            completion_tokens=2,
            tokens_per_second=8.0,
            finish_reason="stop",
            routing=None,
        )

    def close(self) -> None:
        self.closed = True


def _write_checkpoint(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"fixture weights")


def test_fake_backend_run_writes_self_contained_quality_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = load_suite(SUITE)
    cases = suite.select(profile="smoke", limit=3)
    answers = {case.prompt: _perfect_response(case) for case in cases}

    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    output = tmp_path / "benchmark-output"
    fake = _PerfectFakeBackend(answers)
    monkeypatch.setattr(
        "yylab.benchmark.quality.YyLabBenchmarkBackend",
        lambda **_kwargs: fake,
    )
    progress: list[tuple[str, int, int, float, str]] = []

    summary = run_quality_benchmark(
        {
            "checkpoint_ref": str(checkpoint),
            "quant": "bf16",
            "policy": "base",
            "profile": "smoke",
            "limit": 3,
            "repetitions": 2,
            "label": "fixture quality",
            "enable_thinking": True,
        },
        output,
        progress=lambda *items: progress.append(items),
    )

    assert fake.closed is True
    assert fake.policies == ["base"]
    assert len(fake.requests) == 6
    assert len({request["seed"] for request in fake.requests}) == 6
    assert all(request["enable_thinking"] is True for request in fake.requests)
    assert all(request["max_thinking_tokens"] == 1024 for request in fake.requests)
    assert all(request["temperature"] == 0.6 for request in fake.requests)
    assert all(request["top_p"] == 0.95 for request in fake.requests)
    assert all(request["top_k"] == 20 for request in fake.requests)
    assert summary["quality_index"] == 1000.0
    assert summary["semantic_quality_index"] == 1000.0
    assert summary["coverage"]["scored"] == 6
    assert summary["stack_fully_content_addressed"] is True
    assert len(summary["generation_implementation_fingerprint"]) == 64
    assert progress[-1][1:4] == (6, 6, 100.0)
    expected_artifacts = {
        "suite.snapshot.json",
        "stack.lock.json",
        "run.json",
        "results.jsonl",
        "summary.json",
        "scores.csv",
        "report.md",
    }
    assert expected_artifacts.issubset(path.name for path in output.iterdir())
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(records) == 6
    assert all(record["error"] is None for record in records)
    assert all(record["semantic_score"] == 1.0 for record in records)
    assert all(record["dimensions"]["semantic"]["passed"] is True for record in records)
    lock = json.loads((output / "stack.lock.json").read_text(encoding="utf-8"))
    assert lock["fully_content_addressed"] is True
    assert [item["path"] for item in lock["checkpoint"]["files"]] == [
        "config.json",
        "model.safetensors",
    ]
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["case_ids"] == [case.case_id for case in cases]
    assert manifest["policies"] == ["base"]
    assert len(manifest["generation_comparison_fingerprint"]) == 64
    assert (
        manifest["generation_implementation_fingerprint"] == summary["generation_implementation_fingerprint"]
    )
    assert manifest["semantic_scorer_version"].startswith("yylab.semantic-scorers.")
    assert "Quality Index: 1000.0 / 1000" in (output / "report.md").read_text(encoding="utf-8")

    original_results = (output / "results.jsonl").read_bytes()
    projection = rescore_quality_artifact(output)
    assert projection["semantic_quality_index"] == 1000.0
    assert (output / "results.jsonl").read_bytes() == original_results
    assert (output / "semantic-summary.json").is_file()
    assert (output / "semantic-report.md").is_file()
    assert (output / "semantic-scores.csv").is_file()


def test_quality_run_fails_fast_instead_of_scoring_unfinished_thinking(
    tmp_path: Path,
) -> None:
    suite = load_suite(SUITE)
    case = suite.select(profile="smoke", limit=1)[0]

    class IncompleteThinkingBackend(_PerfectFakeBackend):
        def generate(self, *, messages, **request) -> GenerationResponse:
            self.requests.append(request)
            return GenerationResponse(
                text="<think>unfinished private reasoning",
                elapsed_seconds=1.0,
                prompt_tokens=8,
                completion_tokens=1024,
                thinking_tokens=1024,
                answer_tokens=0,
                thinking_budget_exhausted=True,
                tokens_per_second=1024.0,
                finish_reason="length",
                routing=None,
            )

    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    output = tmp_path / "benchmark-output"
    fake = IncompleteThinkingBackend({case.prompt: _perfect_response(case)})

    with pytest.raises(RuntimeError, match="without a visible final answer"):
        run_quality_benchmark(
            {
                "checkpoint_ref": str(checkpoint),
                "quant": "bf16",
                "policy": "base",
                "profile": "smoke",
                "limit": 1,
                "repetitions": 1,
                "enable_thinking": True,
                "continue_on_error": True,
            },
            output,
            backend=fake,
        )

    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["score"] is None
    assert records[0]["semantic_score"] is None
    assert records[0]["telemetry"]["thinking_tokens"] == 1024
    assert records[0]["telemetry"]["answer_tokens"] == 0
    assert records[0]["telemetry"]["thinking_budget_exhausted"] is True


def test_quality_run_resumes_without_generating_or_duplicating_results(tmp_path: Path) -> None:
    suite = load_suite(SUITE)
    cases = suite.select(profile="smoke", limit=2)
    fake = _PerfectFakeBackend({case.prompt: _perfect_response(case) for case in cases})
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    output = tmp_path / "benchmark-output"
    config = {
        "checkpoint_ref": str(checkpoint),
        "quant": "bf16",
        "policy": "base",
        "profile": "smoke",
        "limit": 2,
        "repetitions": 1,
    }

    first = run_quality_benchmark(config, output, backend=fake)
    original_results = (output / "results.jsonl").read_bytes()
    original_manifest = (output / "run.json").read_bytes()
    second = run_quality_benchmark(config, output, backend=fake)

    assert len(fake.requests) == 2
    assert fake.policies == ["base"]
    assert (output / "results.jsonl").read_bytes() == original_results
    assert (output / "run.json").read_bytes() == original_manifest
    records = [json.loads(line) for line in original_results.decode().splitlines()]
    assert len(records) == 2
    assert len({record["cell_id"] for record in records}) == 2
    assert second["quality_index"] == first["quality_index"] == 1000.0
    assert second["result_cells"] == first["result_cells"]


def test_quality_resume_rejects_workload_and_stack_drift_before_generation(
    tmp_path: Path,
) -> None:
    suite = load_suite(SUITE)
    cases = suite.select(profile="smoke", limit=1)
    fake = _PerfectFakeBackend({case.prompt: _perfect_response(case) for case in cases})
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    output = tmp_path / "benchmark-output"
    config = {
        "checkpoint_ref": str(checkpoint),
        "quant": "bf16",
        "policy": "base",
        "profile": "smoke",
        "limit": 1,
        "repetitions": 1,
    }
    run_quality_benchmark(config, output, backend=fake)
    original_results = (output / "results.jsonl").read_bytes()
    original_manifest = (output / "run.json").read_bytes()

    with pytest.raises(ValueError, match="workload_fingerprint"):
        run_quality_benchmark({**config, "max_new_tokens": 49}, output, backend=fake)

    (checkpoint / "model.safetensors").write_bytes(b"changed fixture weights")
    with pytest.raises(ValueError, match="stack_fingerprint"):
        run_quality_benchmark(config, output, backend=fake)

    assert len(fake.requests) == 1
    assert (output / "results.jsonl").read_bytes() == original_results
    assert (output / "run.json").read_bytes() == original_manifest


def test_incompatible_baseline_fails_before_checkpoint_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"generation_comparison_fingerprint": "00" * 32}),
        encoding="utf-8",
    )
    hashed = False

    def fail_if_hashed(_cfg):
        nonlocal hashed
        hashed = True
        raise AssertionError("checkpoint hashing should not run")

    monkeypatch.setattr("yylab.benchmark.quality._stack_lock", fail_if_hashed)

    with pytest.raises(ValueError, match="baseline generation workload differs"):
        run_quality_benchmark(
            {
                "checkpoint_ref": str(tmp_path / "unused-checkpoint"),
                "quant": "bf16",
                "policy": "base",
                "profile": "smoke",
                "limit": 1,
                "repetitions": 1,
                "baseline_summary_path": str(baseline),
            },
            tmp_path / "output",
            backend=_PerfectFakeBackend({}),
        )

    assert hashed is False


def test_training_performance_accounting_excludes_warmup_and_uses_complete_steps() -> None:
    config = {"seq_len": 512, "micro_batch": 3, "grad_accum": 2}
    measured = [
        {"step_seconds": 1.0, "loss": 2.0},
        {"step_seconds": 2.0, "loss": 1.0},
        {"step_seconds": 3.0, "loss": 0.5},
    ]

    tokens_per_step = synthetic_tokens_per_step(config)
    summary = summarize_step_metrics(measured, useful_tokens_per_step=tokens_per_step)

    assert tokens_per_step == 3_072
    assert summary["measured_steps"] == 3
    assert summary["step_seconds_mean"] == 2.0
    assert summary["step_seconds_median"] == 2.0
    assert summary["step_seconds_p95"] == pytest.approx(2.9)
    assert summary["useful_tokens"] == 9_216
    assert summary["useful_tokens_per_second"] == 1_536.0
    assert summary["loss_mean"] == pytest.approx(7 / 6)
    assert summary["loss_finite"] is True


def test_training_performance_empty_and_nonfinite_receipts_are_explicit() -> None:
    assert percentile([], 0.5) is None
    assert percentile([3.0], 0.95) == 3.0
    empty = summarize_step_metrics([], useful_tokens_per_step=128)
    assert empty == {
        "measured_steps": 0,
        "step_seconds_mean": None,
        "step_seconds_median": None,
        "step_seconds_p95": None,
        "useful_tokens": 0,
        "useful_tokens_per_second": None,
        "loss_mean": None,
        "loss_finite": False,
    }
    nonfinite = summarize_step_metrics(
        [{"step_seconds": 1.0, "loss": float("inf")}],
        useful_tokens_per_step=128,
    )
    assert nonfinite["loss_finite"] is False


def test_training_performance_skips_a_busy_gpu_without_loading_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "yylab.benchmark.performance._gpu_snapshot",
        lambda: {"name": "fixture GPU", "memory_used_gb": 2.5, "memory_total_gb": 12.0},
    )
    monkeypatch.setattr(
        "yylab.models.loading.load_base",
        lambda *_args, **_kwargs: pytest.fail("busy-GPU preflight must precede model loading"),
    )

    summary = run_training_performance(
        {
            "checkpoint_ref": str(checkpoint),
            "quant": "bf16",
            "warmup_steps": 0,
            "measured_steps": 2,
            "require_idle_gpu": True,
            "max_existing_gpu_gb": 1.0,
        },
        tmp_path / "output",
    )

    assert summary["status"] == "busy_gpu"
    assert summary["warmup_steps"] == 0
    assert summary["measured_steps"] == 0
    assert summary["useful_tokens_per_second"] is None
    assert "2.50 GB" in summary["error"]
    assert (tmp_path / "output" / "benchmark-receipt.json").is_file()
    assert (tmp_path / "output" / "summary.json").is_file()
    assert "BUSY_GPU" in (tmp_path / "output" / "report.md").read_text(encoding="utf-8")
