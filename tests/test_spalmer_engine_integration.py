from __future__ import annotations

from types import SimpleNamespace

import pytest

from yylab.benchmark.quality import _stack_lock
from yylab.infer.engine import Engine, LoadSpec
from yylab.ui.pages.checkpoints import _spalmer_base_recipe


class _Facade:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(eos_token_id=2, model_max_length=4096)
        self.notes = ["native SPALMER"]
        self.cuda_available = False
        self.config = SimpleNamespace(max_position_embeddings=4096)

    def routing_telemetry(self):
        return {"routing_mode": "adaptive_surprise", "active_experts": 4}


def test_engine_dispatches_spalmer_before_hf_context_loading(monkeypatch):
    facade = _Facade()
    calls = []

    def load(ref, *, dynamic_residency, max_context_tokens):
        calls.append((ref, dynamic_residency, max_context_tokens))
        return facade

    monkeypatch.setattr("yylab.models.spalmer.load_spalmer_runtime", load)
    engine = Engine()

    notes = engine.load(
        LoadSpec(
            ref="fixture.pt",
            quant="auto",
            model_family="spalmer",
            settings={
                "context_mode": "native",
                "max_context_tokens": 3072,
                "spalmer_dynamic_residency": False,
            },
        )
    )

    assert calls == [("fixture.pt", False, 3072)]
    assert engine.model is facade
    assert engine.tokenizer is facade.tokenizer
    assert engine.spec.model_family == "spalmer"
    assert engine.spec.quant == "checkpoint"
    assert notes == ["native SPALMER"]
    assert engine.routing_telemetry()["active_experts"] == 4


def test_engine_rejects_hf_only_options_for_spalmer(monkeypatch):
    monkeypatch.setattr(
        "yylab.models.spalmer.load_spalmer_runtime",
        lambda *_args, **_kwargs: pytest.fail("runtime should not load"),
    )
    engine = Engine()

    with pytest.raises(ValueError, match="does not use RoPE or YaRN"):
        engine.load(
            LoadSpec(
                ref="fixture.pt",
                quant="checkpoint",
                model_family="spalmer",
                settings={"context_mode": "yarn_2x", "rope_scaling": "yarn"},
            )
        )
    with pytest.raises(ValueError, match="native micro-expert bank"):
        engine.load(
            LoadSpec(
                ref="fixture.pt",
                quant="checkpoint",
                model_family="spalmer",
                adapter_path="adapter",
            )
        )


def test_engine_never_constructs_hf_quantized_cache_for_spalmer(monkeypatch):
    facade = _Facade()
    engine = Engine()
    engine.model = facade
    engine.tokenizer = facade.tokenizer
    engine.spec = LoadSpec(
        ref="fixture.pt",
        quant="checkpoint",
        model_family="spalmer",
        settings={"cache_implementation": "quantized_offloaded", "kv_quant": "int4"},
    )

    kwargs = engine._generate_kwargs(
        input_ids=SimpleNamespace(shape=(1, 8)),
        attention_mask=object(),
        temperature=0.0,
        top_p=0.9,
        max_new_tokens=16,
    )

    assert "past_key_values" not in kwargs
    assert "cache_implementation" not in kwargs
    assert any("native recurrent KDA/MLA state" in note for note in engine.notes)


def test_spalmer_context_overflow_never_recommends_yarn():
    class Tokenizer:
        model_max_length = 8

        def __call__(self, text, **_kwargs):
            return {"input_ids": list(range(len(text)))}

        @staticmethod
        def apply_chat_template(_messages, **_kwargs):
            return "12345"

    engine = Engine()
    engine.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8))
    engine.tokenizer = Tokenizer()
    engine.spec = LoadSpec(
        ref="fixture.pt",
        quant="checkpoint",
        model_family="spalmer",
    )

    with pytest.raises(ValueError) as raw_error:
        engine.plan_context(
            [{"content": "12345"}],
            max_new_tokens=4,
            prompt_mode="raw",
        )
    with pytest.raises(ValueError) as chat_error:
        engine.plan_context(
            [{"role": "user", "content": "prompt"}],
            max_new_tokens=4,
            prompt_mode="chat",
            compact=False,
        )

    for error in (raw_error.value, chat_error.value):
        assert "native SPALMER context limit" in str(error)
        assert "YaRN" not in str(error)


def test_spalmer_recipe_reads_trained_and_scanned_metadata_shapes():
    assert _spalmer_base_recipe(
        {
            "precision": {
                "expert_weights": "nvfp4",
                "expert_activations": "mxfp8",
                "master_weights": "bfloat16",
            }
        }
    ) == "SPALMER · NVFP4/MXFP8 · BF16 master"
    assert _spalmer_base_recipe(
        {
            "experts_config": {
                "expert_weight_format": "mxfp4",
                "expert_activation_format": "mxfp8",
                "expert_master_dtype": "bfloat16",
            }
        }
    ) == "SPALMER · MXFP4/MXFP8 · BF16 master"


def test_stack_lock_infers_spalmer_family_for_direct_configs(tmp_path):
    checkpoint = tmp_path / "spalmer.pt"
    checkpoint.write_bytes(b"static fixture")

    lock = _stack_lock({"checkpoint_ref": str(checkpoint), "quant": "checkpoint"})

    assert lock["model_family"] == "spalmer"
