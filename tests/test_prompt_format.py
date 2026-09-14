from __future__ import annotations

import json
from collections.abc import Mapping
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from yylab.infer.engine import Engine, LoadSpec, format_messages
from yylab.infer.prompt_format import (
    PROMPT_FORMAT_FILENAME,
    PromptFormat,
    build_prompt_format_contract,
    checkpoint_prompt_format_contract,
    load_prompt_format,
    resolve_prompt_format,
    validate_prompt_format_contract,
)
from yylab.ui.pages.chat import _prompt_mode_settings


def _tokenizer():
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<s>": 1, "</s>": 2, "user": 3,
                                          "assistant": 4, "hello": 5, "world": 6}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 1)],
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
                                       unk_token="<unk>", pad_token="</s>")
    tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
        "{{ eos_token }}{% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
    )
    return tokenizer


def _save(path, contract):
    path.mkdir(parents=True, exist_ok=True)
    (path / PROMPT_FORMAT_FILENAME).write_text(json.dumps(contract), encoding="utf-8")


def test_causal_contract_does_not_infer_chat_from_generic_tokenizer_template(tmp_path):
    tokenizer = _tokenizer()
    contract = build_prompt_format_contract({"objective": "causal_lm"}, tokenizer)
    _save(tmp_path, contract)
    resolved = load_prompt_format(tmp_path, tokenizer)
    assert resolved.default_mode == "raw"
    assert resolved.supported_modes == ("raw",)
    assert resolved.chat_template is None
    assert format_messages(tokenizer, [{"content": " hello world "}], prompt_format=resolved) == " hello world "
    assert _prompt_mode_settings(resolved)[0] == {"raw": "Text continuation"}


def test_legacy_causal_objective_defaults_raw_without_modifying_bundle(tmp_path):
    path = tmp_path / "sprint_training.json"
    payload = json.dumps({"config": {"objective": "causal_lm", "model_config": {"model_type": "any"}}})
    path.write_text(payload, encoding="utf-8")
    resolved = load_prompt_format(tmp_path, _tokenizer())
    assert resolved.default_mode == "raw"
    assert resolved.source == "legacy_training_objective"
    assert path.read_text(encoding="utf-8") == payload
    assert not (tmp_path / PROMPT_FORMAT_FILENAME).exists()


def test_unknown_checkpoint_retains_legacy_choice_without_claiming_trained_chat(tmp_path):
    # Model name/type and tokenizer template alone are not evidence of training format.
    (tmp_path / "config.json").write_text('{"model_type":"sprint","_name_or_path":"chat"}')
    resolved = load_prompt_format(tmp_path, _tokenizer())
    assert resolved == PromptFormat()
    options, default, note = _prompt_mode_settings(resolved)
    assert default == "chat"
    assert "unverified" in options["chat"]
    assert "unverified" in note


def test_explicit_contract_wins_over_conflicting_legacy_objective_and_current_template(tmp_path):
    tokenizer = _tokenizer()
    contract = build_prompt_format_contract({"objective": "chat_sft"}, tokenizer)
    _save(tmp_path, contract)
    (tmp_path / "sprint_training.json").write_text('{"config":{"objective":"causal_lm"}}')
    expected = tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False,
                                              add_generation_prompt=True)
    tokenizer.chat_template = "wrong untrained generic format"
    resolved = load_prompt_format(tmp_path, tokenizer)
    actual = format_messages(tokenizer, [{"role": "user", "content": "hello"}], prompt_format=resolved)
    assert actual == expected
    assert resolved.source == "saved_contract"


def test_chat_prompt_encoding_matches_training_without_duplicate_bos(tmp_path):
    tokenizer = _tokenizer()
    contract = build_prompt_format_contract({"objective": "chat_sft"}, tokenizer)
    _save(tmp_path, contract)
    engine = Engine()
    engine.tokenizer = tokenizer
    engine.prompt_format = load_prompt_format(tmp_path, tokenizer)
    engine.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100))
    messages = [{"role": "user", "content": "hello"}]
    expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(expected, Mapping):
        expected = expected["input_ids"]
    encoded = engine._encode_messages(messages)["input_ids"]
    assert encoded == expected
    assert encoded.count(tokenizer.bos_token_id) == 1
    assert engine.plan_context(messages, max_new_tokens=10).prompt_tokens == len(expected)


def test_raw_context_and_generation_encoding_share_literal_tokenization(tmp_path):
    tokenizer = _tokenizer()
    _save(tmp_path, build_prompt_format_contract({"objective": "causal_lm"}, tokenizer))
    engine = Engine()
    engine.tokenizer = tokenizer
    engine.prompt_format = load_prompt_format(tmp_path, tokenizer)
    engine.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100))
    messages = [{"role": "user", "content": "hello world"}]
    encoded = engine._encode_messages(messages)["input_ids"]
    assert encoded == [5, 6]
    plan = engine.plan_context(messages, max_new_tokens=10)
    assert plan.prompt_mode == "raw"
    assert plan.prompt_tokens == len(encoded)
    with pytest.raises(ValueError, match="exactly one"):
        engine.plan_context(messages * 2, max_new_tokens=10)


def test_explicit_unverified_chat_override_is_reported_without_changing_default(tmp_path):
    tokenizer = _tokenizer()
    _save(tmp_path, build_prompt_format_contract({"objective": "causal_lm"}, tokenizer))
    engine = Engine()
    engine.tokenizer = tokenizer
    engine.prompt_format = load_prompt_format(tmp_path, tokenizer)
    engine.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100))
    messages = [{"role": "user", "content": "hello"}]
    plan = engine.plan_context(messages, max_new_tokens=10, prompt_mode="chat")
    assert plan.prompt_mode == "chat"
    assert any("explicitly requested" in note for note in engine.notes)
    assert engine.prompt_format.default_mode == "raw"
    assert engine.prompt_format.options() == {"raw": "Text continuation"}
    assert plan.prompt_tokens == len(engine._encode_messages(messages, prompt_mode="chat")["input_ids"])


def test_unknown_raw_model_preserves_its_tokenizer_bos_policy():
    tokenizer = _tokenizer()
    engine = Engine()
    engine.tokenizer = tokenizer
    encoded = engine._encode_messages([{"content": "hello"}], prompt_mode="raw")["input_ids"]
    assert encoded == tokenizer("hello")["input_ids"] == [1, 5]


@pytest.mark.parametrize("mutation,match", [
    ({"schema": 99}, "schema"),
    ({"chat_template_sha256": "bad"}, "checksum"),
    ({"default_mode": "other"}, "default_mode"),
    ({"supported_modes": ["invented"]}, "supported_modes"),
    ({"add_special_tokens": True}, "duplicate"),
])
def test_invalid_authoritative_contract_never_falls_back(mutation, match):
    contract = build_prompt_format_contract({"objective": "chat_sft"}, _tokenizer())
    with pytest.raises(ValueError, match=match):
        validate_prompt_format_contract({**contract, **mutation})


def test_invalid_contract_rejected_before_replacing_loaded_model(tmp_path):
    _save(tmp_path, {"schema": 900})
    engine = Engine()
    sentinel = object()
    engine.model = sentinel
    with pytest.raises(ValueError, match="schema"):
        engine.load(LoadSpec(ref=str(tmp_path), quant="fp32"))
    assert engine.model is sentinel


def test_saved_template_failure_does_not_invent_generic_roles(tmp_path):
    _save(tmp_path, build_prompt_format_contract({"objective": "chat_sft"}, _tokenizer()))
    class BrokenTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise RuntimeError("unsupported template")
    with pytest.raises(ValueError, match="recorded conversation format"):
        format_messages(BrokenTokenizer(), [{"content": "hello"}], prompt_format=load_prompt_format(tmp_path))


def test_adapter_record_wins_and_unrecorded_adapter_does_not_inherit_base_restrictions(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    tokenizer = _tokenizer()
    _save(base, build_prompt_format_contract({"objective": "causal_lm"}, tokenizer))
    assert resolve_prompt_format(str(base), adapter_path=str(adapter)).source == "unknown"
    _save(adapter, build_prompt_format_contract({"objective": "chat_sft"}, tokenizer))
    resolved = resolve_prompt_format(str(base), adapter_path=str(adapter), tokenizer=tokenizer)
    assert resolved.default_mode == "chat"
    assert resolved.source == "saved_contract"


def test_continued_pretraining_can_preserve_explicitly_established_chat_contract():
    tokenizer = _tokenizer()
    original = build_prompt_format_contract({"objective": "chat_sft"}, tokenizer)
    continued = build_prompt_format_contract({"objective": "causal_lm", "prompt_format_contract": original}, tokenizer)
    assert continued == original


def test_new_chat_sft_upgrades_inherited_raw_contract_to_actual_tokenizer_template():
    tokenizer = _tokenizer()
    raw = build_prompt_format_contract({"objective": "causal_lm"}, tokenizer)
    actual = build_prompt_format_contract({"objective": "chat_sft", "prompt_format_contract": raw}, tokenizer)
    assert actual == build_prompt_format_contract({"objective": "chat_sft"}, tokenizer)
    assert actual["chat_template"] == tokenizer.chat_template


def test_chat_sft_rejects_inherited_chat_template_that_differs_from_training_tokenizer():
    tokenizer = _tokenizer()
    old = build_prompt_format_contract({"objective": "chat_sft"}, tokenizer)
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    with pytest.raises(ValueError, match="tokenizer template differs"):
        build_prompt_format_contract({"objective": "chat_sft", "prompt_format_contract": old}, tokenizer)
    # Causal continuation preserves known past chat exposure without claiming
    # that its current plain-text data was rendered with any chat template.
    assert build_prompt_format_contract({"objective": "causal_lm", "prompt_format_contract": old}, tokenizer) == old


def test_parent_contract_read_preserves_provenance_and_only_uses_established_legacy_formats(tmp_path):
    tokenizer = _tokenizer()
    assert checkpoint_prompt_format_contract(tmp_path, tokenizer) is None
    (tmp_path / "sprint_training.json").write_text('{"config":{"objective":"chat_sft"}}')
    expected = build_prompt_format_contract({"objective": "chat_sft"}, tokenizer)
    assert checkpoint_prompt_format_contract(tmp_path, tokenizer) == expected
    expected["training_provenance"] = "parent-record"
    _save(tmp_path, expected)
    assert checkpoint_prompt_format_contract(tmp_path, tokenizer) == expected
