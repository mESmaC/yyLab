from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from yylab.infer.engine import (
    Engine,
    LoadSpec,
    _adaptive_kv_reserve_bytes,
    _cache_kwargs,
    format_messages,
)
from yylab.infer.quantized_offload_cache import QuantizedOffloadedCache
from yylab.models.loading import _merged_rope_parameters
from yylab.mole.inject import MoLEConfig, inject_mole
from yylab.mole.store import save_mole


class FakeTok:
    def __init__(self) -> None:
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = None

    def __call__(self, texts, return_tensors=None, **_):
        if isinstance(texts, str):
            ids = [[(ord(c) % 100) + 2 for c in texts[:64]]]
        else:
            ids = [[(ord(c) % 100) + 2 for c in t[:64]] for t in texts]
        if return_tensors == "pt":
            import torch as _t

            mask = _t.ones(len(ids), max(len(i) for i in ids), dtype=_t.long)
            padded = [i + [1] * (mask.shape[1] - len(i)) for i in ids]
            return {"input_ids": _t.tensor(padded), "attention_mask": mask}
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        raise NotImplementedError("no template")

    def decode(self, ids, **_):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)


class CountingTok:
    """Small tokenizer whose chat-template token counts are deterministic."""

    eos_token_id = 1
    model_max_length = 1_000

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [str(message.get("content") or "") for message in messages]
        if add_generation_prompt:
            parts.append("assistant")
        return " ".join(parts)

    def __call__(self, text, **_):
        return {"input_ids": [list(range(len(str(text).split())))]}


def _planning_engine(context_limit: int) -> Engine:
    engine = Engine()
    engine.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=context_limit))
    engine.tokenizer = CountingTok()
    engine.spec = LoadSpec(
        ref="test-model",
        quant="bf16",
        settings={"max_context_tokens": context_limit},
    )
    return engine


def test_qwen_yarn_uses_trained_context_as_original_length() -> None:
    config = SimpleNamespace(
        model_type="qwen3",
        max_position_embeddings=40_960,
        rope_scaling={"rope_theta": 1_000_000, "rope_type": "default"},
    )

    merged = _merged_rope_parameters(config, {"rope_type": "yarn", "factor": 4.0})

    assert merged == {
        "rope_theta": 1_000_000,
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32_768,
    }


def _tiny_llama_dir(tmp_path: Path) -> Path:
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg)
    out = tmp_path / "tiny_llama"
    model.save_pretrained(out)
    return out


def _fake_loader(tmp_path: Path, monkeypatch):
    from transformers import AutoModelForCausalLM

    import yylab.infer.engine as engine_mod

    model_dir = _tiny_llama_dir(tmp_path)

    def fake_load(
        ref,
        quant,
        attn_implementation="sdpa",
        rope_scaling=None,
        max_position_embeddings=None,
    ):
        m = AutoModelForCausalLM.from_pretrained(ref)
        if max_position_embeddings is not None:
            m.config.max_position_embeddings = max_position_embeddings
        m.config.use_cache = True
        return m, FakeTok(), False, quant, []

    monkeypatch.setattr(engine_mod, "load_base", fake_load, raising=False)
    import yylab.models.loading as loading_mod

    monkeypatch.setattr(loading_mod, "load_base", fake_load, raising=False)
    return model_dir


def test_chat_template_fallback():
    out = format_messages(FakeTok(), [{"role": "user", "content": "hi"}])
    assert "### User" in out and "### Assistant" in out


def test_quantized_offload_cache_kwargs_are_user_cache_not_transformers_alias() -> None:
    kwargs, notes = _cache_kwargs(
        {
            "cache_implementation": "quantized_offloaded",
            "kv_quant": "int4",
        },
        cuda=True,
    )

    assert kwargs == {}
    assert any("packed int4 CPU offload" in note for note in notes)


def test_resident_quantized_cache_is_explicitly_quanto_int4() -> None:
    kwargs, notes = _cache_kwargs(
        {"cache_implementation": "quantized", "kv_quant": "int4"},
        cuda=True,
    )

    assert kwargs["cache_implementation"] == "quantized"
    assert kwargs["cache_config"]["backend"] == "quanto"
    assert kwargs["cache_config"]["nbits"] == 4
    assert any("Optimum Quanto" in note for note in notes)
    with pytest.raises(ValueError, match="supports int4, not int8"):
        _cache_kwargs(
            {"cache_implementation": "quantized", "kv_quant": "int8"},
            cuda=True,
        )


def test_engine_builds_packed_quantized_offload_cache() -> None:
    import torch

    engine = Engine()
    engine.model = SimpleNamespace(
        config=LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    engine.tokenizer = CountingTok()
    engine.spec = LoadSpec(
        ref="test-model",
        quant="bf16",
        settings={
            "max_context_tokens": 64,
            "cache_implementation": "quantized_offloaded",
            "kv_quant": "int4",
            "kv_offload_group_size": 8,
            "kv_offload_residual_length": 0,
            "kv_offload_pin_memory": False,
        },
    )

    kwargs = engine._generate_kwargs(
        torch.ones((1, 4), dtype=torch.long),
        torch.ones((1, 4), dtype=torch.long),
        temperature=0.0,
        top_p=0.9,
        max_new_tokens=8,
    )

    assert isinstance(kwargs["past_key_values"], QuantizedOffloadedCache)
    assert kwargs["past_key_values"].nbits == 4
    assert "cache_implementation" not in kwargs


def test_adaptive_reserve_for_quantized_offload_covers_active_layer_and_residual() -> None:
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        head_dim=8,
    )
    settings = {
        "cache_implementation": "quantized_offloaded",
        "kv_offload_residual_length": 2,
    }

    reserve = _adaptive_kv_reserve_bytes(
        config,
        settings,
        prompt_tokens=8,
        answer_tokens=2,
        cuda=True,
    )

    all_layers_per_token = 2 * 4 * 2 * 8 * 2
    expected = all_layers_per_token / 4 * 10 + all_layers_per_token * 2
    assert reserve == expected


def test_engine_generate_plain(tmp_path, monkeypatch):
    model_dir = _fake_loader(tmp_path, monkeypatch)
    eng = Engine()
    notes = eng.load(LoadSpec(ref=str(model_dir), quant="bf16"))
    assert eng.is_loaded()
    assert isinstance(notes, list)
    chunks = list(
        eng.generate_stream([{"role": "user", "content": "hello"}], temperature=0.0, max_new_tokens=8)
    )
    text = "".join(chunks)
    assert isinstance(text, str)
    eng.unload()
    assert not eng.is_loaded()


def test_engine_applies_selected_yarn_context_target(tmp_path, monkeypatch):
    model_dir = _fake_loader(tmp_path, monkeypatch)
    eng = Engine()

    eng.load(
        LoadSpec(
            ref=str(model_dir),
            quant="bf16",
            settings={
                "rope_scaling": "yarn",
                "rope_factor": 2.0,
                "max_context_tokens": 65_536,
            },
        )
    )

    assert eng.model.config.max_position_embeddings == 65_536
    eng.unload()


@pytest.mark.parametrize(
    ("mode", "expected_context", "expected_rope"),
    [
        ("native", 128, "none"),
        ("yarn_2x", 256, "yarn"),
        ("yarn_4x", 512, "yarn"),
    ],
)
def test_engine_resolves_context_mode_presets(
    tmp_path,
    monkeypatch,
    mode,
    expected_context,
    expected_rope,
):
    model_dir = _fake_loader(tmp_path, monkeypatch)
    eng = Engine()

    eng.load(
        LoadSpec(
            ref=str(model_dir),
            quant="bf16",
            settings={"context_mode": mode, "max_context_tokens": 9_999},
        )
    )

    assert eng.context_limit() == expected_context
    assert eng.spec.settings["max_context_tokens"] == expected_context
    assert eng.spec.settings["rope_scaling"] == expected_rope
    assert eng.model.config.max_position_embeddings == expected_context
    eng.unload()


def test_chat_context_plan_preserves_instructions_and_recent_complete_turns():
    eng = _planning_engine(12)
    conversation = [
        {"role": "system", "content": "guard"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old response"},
        {"role": "user", "content": "middle question"},
        {"role": "assistant", "content": "middle response"},
        {"role": "user", "content": "latest question"},
    ]

    plan = eng.plan_chat_context(conversation, max_new_tokens=4)

    assert list(plan.messages) == [conversation[0], *conversation[3:]]
    assert plan.prompt_tokens == 8
    assert plan.prompt_budget == 8
    assert plan.answer_budget == 4
    assert plan.context_limit == 12
    assert plan.dropped_turns == 1
    assert plan.dropped_messages == 2
    assert plan.compacted is True


def test_chat_context_plan_can_require_explicit_compaction():
    eng = _planning_engine(8)
    conversation = [
        {"role": "user", "content": "older question"},
        {"role": "assistant", "content": "older response"},
        {"role": "user", "content": "latest question"},
    ]

    with pytest.raises(ValueError, match="Enable context compaction or use YaRN"):
        eng.plan_chat_context(conversation, max_new_tokens=3, compact=False)


def test_chat_context_plan_rejects_current_turn_that_cannot_fit():
    eng = _planning_engine(8)

    with pytest.raises(ValueError, match="current turn plus protected instructions"):
        eng.plan_chat_context(
            [{"role": "user", "content": "one two three four five six"}],
            max_new_tokens=2,
        )


def test_engine_with_mole_adapter(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    model_dir = _fake_loader(tmp_path, monkeypatch)

    base = AutoModelForCausalLM.from_pretrained(model_dir)
    mole_cfg = MoLEConfig(num_experts=2, rank=4, alpha=8.0, top_k=2, target_modules=("q_proj", "v_proj"))
    inject_mole(base, mole_cfg)
    adp_dir = tmp_path / "adp"
    save_mole(adp_dir, base, mole_cfg, base_ref=str(model_dir))

    eng = Engine()
    notes = eng.load(
        LoadSpec(ref=str(model_dir), quant="bf16", adapter_path=str(adp_dir), adapter_arch="mole")
    )
    assert any("MoLE" in n for n in notes)
    chunks = list(
        eng.generate_stream([{"role": "user", "content": "hi"}], temperature=0.8, top_p=0.9, max_new_tokens=6)
    )
    assert isinstance("".join(chunks), str)


def test_engine_with_native_per_token_mole_adapter(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    model_dir = _fake_loader(tmp_path, monkeypatch)
    base = AutoModelForCausalLM.from_pretrained(model_dir)
    mole_cfg = MoLEConfig(
        num_experts=3,
        rank=2,
        alpha=4.0,
        top_k=2,
        target_modules=("q_proj", "v_proj"),
        router_type="per_token",
    )
    inject_mole(base, mole_cfg)
    adp_dir = tmp_path / "token-adapter"
    save_mole(adp_dir, base, mole_cfg, base_ref=str(model_dir))

    eng = Engine()
    notes = eng.load(
        LoadSpec(ref=str(model_dir), quant="bf16", adapter_path=str(adp_dir), adapter_arch="mole")
    )
    assert any("MoLE" in note for note in notes)
    text = "".join(
        eng.generate_stream(
            [{"role": "user", "content": "route these tokens"}],
            temperature=0.0,
            max_new_tokens=4,
        )
    )
    assert isinstance(text, str)

    with pytest.raises(ValueError, match="cannot override"):
        Engine().load(
            LoadSpec(
                ref=str(model_dir),
                quant="bf16",
                adapter_path=str(adp_dir),
                adapter_arch="mole",
                router_path=str(tmp_path / "legacy-router"),
            )
        )


def test_engine_cancel(tmp_path, monkeypatch):
    model_dir = _fake_loader(tmp_path, monkeypatch)
    eng = Engine()
    eng.load(LoadSpec(ref=str(model_dir), quant="bf16"))
    gen = eng.generate_stream([{"role": "user", "content": "hi"}], temperature=0.0, max_new_tokens=200)
    got = 0
    for _chunk in gen:
        got += 1
        if got >= 2:
            eng.cancel()
    assert got < 50


def test_engine_context_budget(tmp_path, monkeypatch):
    model_dir = _fake_loader(tmp_path, monkeypatch)
    eng = Engine()
    eng.load(LoadSpec(ref=str(model_dir), quant="bf16", settings={"max_context_tokens": 8}))
    try:
        list(eng.generate_stream([{"role": "user", "content": "x" * 40}], max_new_tokens=8))
        raise AssertionError("expected context budget error")
    except ValueError as e:
        assert "context" in str(e).lower()
