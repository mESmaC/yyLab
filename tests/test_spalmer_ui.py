from __future__ import annotations

import pytest

from yylab.ui.pages import train_spalmer


def _resolved_plan() -> dict:
    return {
        "target_parameters": 10_000_000,
        "parameter_error": 0,
        "relative_error": 0.0,
        "fingerprint": "a" * 64,
        "config": {
            "vocab_size": 4096,
            "expert_weight_format": "mxfp4",
            "d_model": 128,
            "n_layers": 8,
        },
        "parameters": {"total": 10_000_000},
        "memory": {},
    }


def _run_config(monkeypatch, **overrides):
    monkeypatch.setattr(
        train_spalmer,
        "attention_backend_status",
        lambda backend: {
            "requested_backend": backend,
            "available": True,
            "fla_available": False,
            "detail": "test attention capability",
        },
    )
    monkeypatch.setattr(
        train_spalmer,
        "qat_backend_status",
        lambda weight_format, backend: {
            "requested_backend": backend,
            "selected_backend": "reference",
            "native_w4a8_available": False,
            "emulated": True,
            "torchao_available": False,
            "detail": f"test {weight_format} capability",
        },
    )
    values = {
        "checkpoint_name": "spalmer-10m",
        "tokenizer_ref": "tok_4096",
        "tokenizer_source": "registered",
        "tokenizer_registry_id": "tok_4096",
        "tokenizer_preview_vocab_size": 4096,
        "target_scale": "10M",
        "target_parameters": 10_000_000,
        "resolved_plan": _resolved_plan(),
        "corpora": ["code", "english", "code"],
        "domains": ["code", "language"],
        "splits": ["train", "unspecified"],
        "corpus_mix": "token_balanced",
        "cross_corpus_dedup": True,
        "source_max_ratio": 0.5,
        "sequence_length": 512,
        "token_budget": 1_000_001,
        "max_steps": 0,
        "micro_batch": 2,
        "gradient_accumulation": 4,
        "learning_rate": 3e-4,
        "minimum_lr_ratio": 0.1,
        "warmup_steps": 10,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-8,
        "seed": 42,
        "gradient_clip": 1.0,
        "auxiliary_loss_weight": 0.01,
        "surprise_calibration_weight": 0.05,
        "log_every": 10,
        "save_every": 100,
        "activation_checkpointing": False,
        "expert_weight_format": "mxfp4",
        "attention_backend": "auto",
        "expert_qat_backend": "auto",
        "directional": {
            "enabled": True,
            "num_feature_groups": 4,
            "lateral_rank": 16,
        },
        "atxy": {"enabled": False},
        "out_checkpoint_id": "ckpt_spalmer",
    }
    values.update(overrides)
    return train_spalmer.build_spalmer_run_config(**values)


def test_spalmer_ui_config_preserves_canonical_architecture_and_budget(monkeypatch) -> None:
    config = _run_config(monkeypatch)
    from yylab.train.spalmer import normalize_spalmer_config

    normalized = normalize_spalmer_config(config, require_runtime=True)

    assert config["architecture_type"] == "spalmer_v1"
    assert config["model_family"] == "spalmer"
    assert config["initialization"] == {
        "mode": "scratch",
        "tokenizer_ref": "tok_4096",
    }
    assert config["corpora"] == ["code", "english"]
    assert config["max_steps"] == 245
    assert config["tokens_per_optimizer_step"] == 4096
    assert config["scheduled_training_tokens"] == 1_003_520
    assert config["spalmer"]["plan_fingerprint"] == "a" * 64
    assert config["spalmer"]["num_experts"] == 200
    assert config["spalmer"]["experts"]["active_experts"] == 2
    assert config["spalmer"]["experts"]["max_resident_experts"] == 20
    assert config["training"]["parameter_dtype"] == "bfloat16"
    assert config["training"]["optimizer_state_dtype"] == "float32"
    assert config["training"]["activation_checkpointing"] is False
    assert config["training"]["validation_interval_steps"] == 246
    assert normalized["max_steps"] == 245
    assert normalized["activation_checkpointing"] is False
    assert normalized["spalmer"]["require_fla"] is True


def test_spalmer_ui_budget_requires_exactly_one_limit() -> None:
    with pytest.raises(ValueError, match="either"):
        train_spalmer._resolve_budget(
            token_budget=1000,
            max_steps=5,
            sequence_length=128,
            micro_batch=1,
            gradient_accumulation=1,
        )
    with pytest.raises(ValueError, match="enter"):
        train_spalmer._resolve_budget(
            token_budget=0,
            max_steps=0,
            sequence_length=128,
            micro_batch=1,
            gradient_accumulation=1,
        )


def test_spalmer_ui_strict_native_qat_fails_closed(monkeypatch) -> None:
    with pytest.raises(ValueError, match="test mxfp4 capability"):
        _run_config(monkeypatch, expert_qat_backend="native")


def test_spalmer_ui_rejects_unimplemented_activation_checkpointing(monkeypatch) -> None:
    with pytest.raises(ValueError, match="not implemented"):
        _run_config(monkeypatch, activation_checkpointing=True)
