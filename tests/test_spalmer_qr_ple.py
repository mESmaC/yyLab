"""QR-PLE backend wiring through the SPALMER bridge and the Train · SPALMER page."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yylab.train import spalmer as bridge
from yylab.ui.pages import train_spalmer


def _scratch_config(**spalmer_overrides):
    spalmer = {
        "target_parameters": 10_000_000,
        "num_experts": 200,
        "expert_weight_format": "bfloat16",
        "expert_activation_format": "bfloat16",
        "attention_backend": "fla",
        "expert_qat_backend": "auto",
        "ple_backend": "qr",
        "directional": {"enabled": False},
        "atxy": {"enabled": False},
        "recurrence": {"enabled": False},
    }
    spalmer.update(spalmer_overrides)
    return {
        "model_family": "spalmer",
        "method": "full",
        "objective": "causal_lm",
        "initialization": {"mode": "scratch", "tokenizer_ref": "fake-tokenizer"},
        "spalmer": spalmer,
        "corpora": ["alpha"],
        "corpus_mix": "concatenate",
        "seq_len": 8,
        "micro_batch": 1,
        "grad_accum": 1,
        "max_steps": 2,
        "seed": 0,
    }


def test_normalize_ple_backend_defaults_to_qr_and_rejects_fake_qat() -> None:
    default = bridge.normalize_spalmer_config(_scratch_config())
    assert default["spalmer"]["ple_backend"] == "qr"

    qr = bridge.normalize_spalmer_config(_scratch_config(ple_backend="QR"))
    assert qr["spalmer"]["ple_backend"] == "qr"

    root_alias = bridge.normalize_spalmer_config({**_scratch_config(), "ple_backend": "qr"})
    assert root_alias["spalmer"]["ple_backend"] == "qr"

    with pytest.raises(ValueError, match="fake-QAT"):
        bridge.normalize_spalmer_config(_scratch_config(ple_backend="fake_qat"))


def test_checkpoint_mode_rejects_a_ple_backend_key() -> None:
    with pytest.raises(ValueError, match="inherits its architecture"):
        bridge._normalize_checkpoint_execution(
            {"attention_backend": "fla", "ple_backend": "qr"}
        )


def _fake_api(calls: dict):
    def plan_configuration(target, vocab, **kwargs):
        calls.clear()
        calls.update(kwargs)
        return SimpleNamespace(fingerprint="plan")

    return SimpleNamespace(
        DirectionalScaleConfig=lambda **_kwargs: object(),
        ATXYScaleConfig=lambda **_kwargs: object(),
        RecurrenceScaleConfig=lambda **_kwargs: object(),
        ScaleSearchSpace=lambda **_kwargs: object(),
        plan_configuration=plan_configuration,
    )


def test_planner_always_passes_qr_backend_and_exact_precision_pair() -> None:
    calls: dict = {}
    api = _fake_api(calls)

    bridge.plan_spalmer_configuration(_scratch_config(), 4096, api=api)
    assert calls["ple_backend"] == "qr"
    assert calls["expert_weight_format"] == "bfloat16"
    assert calls["expert_activation_format"] == "bfloat16"


def test_real_planner_carries_the_qr_ple_block() -> None:
    pytest.importorskip("spalmer.experiment.planning")

    qr = bridge.plan_spalmer_configuration(_scratch_config(), 4096)
    block = qr.to_dict()["ple"]
    assert qr.config.ple_backend == "qr"
    assert block["backend"] == "qr"
    assert block["exact_input_parameters"] == 4096 * qr.config.d_model
    assert block["total_parameters"] == qr.parameters.ple_lookup + qr.parameters.ple_controls
    assert abs(qr.relative_error) <= 0.02


def test_real_bundle_does_not_replay_plan_owned_precision_as_overrides() -> None:
    pytest.importorskip("spalmer.experiment.planning")

    config = _scratch_config()
    plan = bridge.plan_spalmer_configuration(config, 4096)
    tokenizer = SimpleNamespace(identity=SimpleNamespace(fingerprint="f" * 64))

    bundle = bridge.build_spalmer_bundle(config, tokenizer, plan)

    assert bundle.experts.expert_weight_format == plan.config.expert_weight_format
    assert bundle.experts.expert_activation_format == plan.config.expert_activation_format
    assert bundle.experts.expert_master_dtype == plan.config.expert_master_dtype


def test_ple_metadata_reports_qr_ownership_and_rejects_legacy() -> None:
    qr = bridge.PreparedSpalmerRun(
        config={},
        tokenizer=None,
        plan=SimpleNamespace(
            config=SimpleNamespace(ple_expansion=2, ple_backend="qr"),
            parameters=SimpleNamespace(
                ple_lookup=1_783_104,
                ple_controls=9,
                ple_exact_input=1_572_864,
                ple_qr_refresh=210_240,
            ),
            to_dict=lambda: {"ple": {"moduli": [97, 101]}},
        ),
        bundle=SimpleNamespace(model=SimpleNamespace(ple_backend="qr")),
        training=None,
    )
    assert bridge._ple_metadata(qr) == {
        "backend": "qr",
        "lanes": 2,
        "moduli": [97, 101],
        "exact_input_parameters": 1_572_864,
        "qr_refresh_parameters": 210_240,
        "lookup_parameters": 1_783_104,
        "control_parameters": 9,
        "total_parameters": 1_783_113,
    }

    legacy = bridge.PreparedSpalmerRun(
        config={},
        tokenizer=None,
        plan=SimpleNamespace(
            config=SimpleNamespace(ple_expansion=2),
            parameters=SimpleNamespace(ple_lookup=2_097_152, ple_controls=24),
            to_dict=lambda: {},
        ),
        bundle=SimpleNamespace(),
        training=None,
    )
    with pytest.raises(RuntimeError, match="fake-QAT PLE"):
        bridge._ple_metadata(legacy)


def _resolved_plan(**config_overrides) -> dict:
    config = {
        "vocab_size": 4096,
        "expert_weight_format": "bfloat16",
        "expert_activation_format": "bfloat16",
        "ple_backend": "qr",
        "d_model": 128,
        "n_layers": 8,
    }
    config.update(config_overrides)
    return {
        "target_parameters": 10_000_000,
        "parameter_error": 0,
        "relative_error": 0.0,
        "fingerprint": "a" * 64,
        "config": config,
        "parameters": {"total": 10_000_000},
        "memory": {},
    }


def _ui_config(monkeypatch, **overrides) -> dict:
    capability_available = bool(overrides.pop("_capability_available", True))
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
        lambda weight_format, backend, activation_format=None: {
            "requested_backend": backend,
            "selected_backend": "bf16_torch" if capability_available else None,
            "native_available": capability_available,
            "verified": capability_available,
            "selectable": capability_available,
            "forward_available": capability_available,
            "backward_available": capability_available,
            "grouped_available": capability_available,
            "emulated": False,
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
        "corpora": ["code"],
        "domains": None,
        "splits": None,
        "corpus_mix": "token_balanced",
        "cross_corpus_dedup": True,
        "source_max_ratio": None,
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
        "expert_weight_format": "bfloat16",
        "expert_activation_format": "bfloat16",
        "attention_backend": "fla",
        "expert_qat_backend": "auto",
        "directional": {"enabled": True, "num_feature_groups": 4, "lateral_rank": 16},
        "atxy": {"enabled": False},
        "out_checkpoint_id": "ckpt_spalmer",
    }
    values.update(overrides)
    return train_spalmer.build_spalmer_run_config(**values)


def test_ui_config_defaults_to_qr_without_emulation_controls(monkeypatch) -> None:
    config = _ui_config(monkeypatch)

    assert config["recipe_version"] == 6
    assert config["spalmer"]["ple_backend"] == "qr"
    assert config["precision"]["ple_backend"] == "qr"
    assert "ple_fake_quantization" not in config["precision"]
    assert "expert_fake_quantization" not in config["precision"]
    normalized = bridge.normalize_spalmer_config(config, require_runtime=True)
    assert normalized["spalmer"]["ple_backend"] == "qr"


def test_ui_config_emits_qr_ple_backend(monkeypatch) -> None:
    config = _ui_config(
        monkeypatch,
        resolved_plan=_resolved_plan(ple_backend="qr"),
        ple_backend="qr",
    )

    assert config["spalmer"]["ple_backend"] == "qr"
    assert config["precision"]["ple_backend"] == "qr"
    assert config["precision"]["ple_parameter_dtype"] == "bfloat16"
    assert "ple_quant_bits" not in config["precision"]
    assert config["architecture"]["config"]["ple_backend"] == "qr"
    normalized = bridge.normalize_spalmer_config(config, require_runtime=True)
    assert normalized["spalmer"]["ple_backend"] == "qr"


def test_ui_config_rejects_plan_and_backend_mismatch(monkeypatch) -> None:
    with pytest.raises(ValueError, match="fake-QAT"):
        _ui_config(monkeypatch, ple_backend="fake_qat")
    with pytest.raises(ValueError, match="activation format"):
        _ui_config(
            monkeypatch,
            resolved_plan=_resolved_plan(expert_activation_format="mxfp8"),
        )


def test_ui_checkpoint_mode_inherits_the_ple_backend(monkeypatch) -> None:
    checkpoint_values = {
        "initialization_mode": "checkpoint",
        "parent_checkpoint_id": "ckpt_parent",
        "tokenizer_ref": "",
        "tokenizer_source": "checkpoint",
        "tokenizer_registry_id": None,
        "resolved_plan": {},
        "target_parameters": 0,
        "attention_backend": "fla",
    }
    inherited = _ui_config(monkeypatch, **checkpoint_values)
    assert "ple_backend" not in inherited["spalmer"]
    assert "precision" not in inherited

    with pytest.raises(ValueError, match="inherits the PLE backend"):
        _ui_config(monkeypatch, ple_backend="qr", **checkpoint_values)


def test_plan_preview_cache_key_includes_ple_backend(monkeypatch) -> None:
    seen: list[tuple] = []
    monkeypatch.setattr(
        train_spalmer,
        "_cached_plan_json",
        lambda *args: seen.append(args) or "{}",
    )

    train_spalmer.plan_spalmer_configuration(
        target_parameters=10_000_000,
        vocab_size=4096,
        expert_weight_format="bfloat16",
        expert_activation_format="bfloat16",
        directional={"enabled": False},
        atxy={"enabled": False},
        ple_backend="qr",
    )
    train_spalmer.plan_spalmer_configuration(
        target_parameters=10_000_000,
        vocab_size=4096,
        expert_weight_format="bfloat16",
        expert_activation_format="bfloat16",
        directional={"enabled": False},
        atxy={"enabled": False},
    )

    assert seen[0][:5] == (10_000_000, 4096, "bfloat16", "bfloat16", "qr")
    assert seen[1][:5] == (10_000_000, 4096, "bfloat16", "bfloat16", "qr")
    assert seen[0][-5:] == (False, 1, 1, 8, 1.0)

    with pytest.raises(ValueError, match="QR-PLE"):
        train_spalmer.plan_spalmer_configuration(
            target_parameters=10_000_000,
            vocab_size=4096,
            expert_weight_format="bfloat16",
            expert_activation_format="bfloat16",
            directional={"enabled": False},
            atxy={"enabled": False},
            ple_backend="fake_qat",
        )


def test_plan_preview_always_names_qr_backend(monkeypatch) -> None:
    planning = pytest.importorskip("spalmer.experiment.planning")
    calls: list[dict] = []

    def fake_plan_configuration(target, vocab, **kwargs):
        calls.append(dict(kwargs))
        return SimpleNamespace(to_dict=lambda: {"target_parameters": target})

    monkeypatch.setattr(planning, "plan_configuration", fake_plan_configuration)
    train_spalmer._cached_plan_json.cache_clear()
    common = {
        "target_parameters": 123_457,
        "vocab_size": 999,
        "expert_weight_format": "bfloat16",
        "expert_activation_format": "bfloat16",
        "directional": {"enabled": False},
        "atxy": {"enabled": False},
    }
    train_spalmer.plan_spalmer_configuration(**common)
    train_spalmer._cached_plan_json.cache_clear()
    train_spalmer.plan_spalmer_configuration(**common, ple_backend="qr")
    train_spalmer._cached_plan_json.cache_clear()

    assert calls[0]["ple_backend"] == "qr"
    assert calls[1]["ple_backend"] == "qr"


def test_checkpoint_ple_reads_every_metadata_shape() -> None:
    produced = {"meta_json": json.dumps({"ple": {"backend": "qr", "exact_input_parameters": 5}})}
    assert train_spalmer._checkpoint_ple(produced) == {
        "backend": "qr",
        "exact_input_parameters": 5,
    }
    scanned = {"meta_json": json.dumps({"ple_backend": "fake_qat"})}
    assert train_spalmer._checkpoint_ple(scanned) == {"backend": "fake_qat"}
    nested = {"meta_json": json.dumps({"metadata": {"ple": {"backend": "qr"}}})}
    assert train_spalmer._checkpoint_ple(nested) == {"backend": "qr"}
    bundle = {"meta_json": json.dumps({"model_config": {"ple_backend": "qr"}})}
    assert train_spalmer._checkpoint_ple(bundle) == {"backend": "qr"}
    legacy = {"meta_json": json.dumps({"model_family": "spalmer"})}
    assert train_spalmer._checkpoint_ple(legacy) == {"backend": "fake_qat"}
    assert train_spalmer._checkpoint_ple({"meta_json": "{}"}) is None
    assert train_spalmer._checkpoint_ple({"meta_json": "not json"}) is None


def test_checkpoint_expert_precision_reads_scanner_native_and_blocked_contracts() -> None:
    native = {
        "expert_precision": {
            "weight_format": "nvfp4",
            "activation_format": "nvfp4",
            "master_dtype": "bfloat16",
            "backend": "auto",
            "execution": "loop",
            "executable": True,
        },
        "precision_executable": True,
    }
    assert train_spalmer._checkpoint_expert_precision(native) == {
        "weight_format": "nvfp4",
        "activation_format": "nvfp4",
        "master_dtype": "bfloat16",
        "execution": "loop",
        "backend": "auto",
        "fake_quantization": False,
        "stochastic_quantization": False,
        "executable": True,
        "rejection_reason": "",
    }

    blocked = {
        **native,
        "precision_executable": False,
        "precision_rejection_reason": "retired fake-QAT PLE",
    }
    inherited = train_spalmer._checkpoint_expert_precision(blocked)
    assert inherited is not None
    assert inherited["executable"] is False
    assert inherited["rejection_reason"] == "retired fake-QAT PLE"


def test_ple_backend_names_agree_across_ui_bridge_and_planner() -> None:
    planning = pytest.importorskip("spalmer.experiment.planning")
    assert set(train_spalmer.PLE_BACKENDS) == bridge.PLE_BACKENDS == set(planning.PLE_BACKENDS)
    assert train_spalmer.DEFAULT_PLE_BACKEND == bridge.DEFAULT_PLE_BACKEND == "qr"


def test_plan_preview_explains_an_older_spalmer_planner(monkeypatch) -> None:
    planning = pytest.importorskip("spalmer.experiment.planning")

    def legacy_only_plan_configuration(target, vocab, **kwargs):
        if "ple_backend" in kwargs:
            raise TypeError("plan_configuration() got an unexpected keyword argument 'ple_backend'")
        return SimpleNamespace(to_dict=lambda: {"target_parameters": target})

    monkeypatch.setattr(planning, "plan_configuration", legacy_only_plan_configuration)
    train_spalmer._cached_plan_json.cache_clear()
    common = {
        "target_parameters": 123_459,
        "vocab_size": 997,
        "expert_weight_format": "bfloat16",
        "expert_activation_format": "bfloat16",
        "directional": {"enabled": False},
        "atxy": {"enabled": False},
    }
    with pytest.raises(RuntimeError, match="predates QR-PLE"):
        train_spalmer.plan_spalmer_configuration(**common)
    train_spalmer._cached_plan_json.cache_clear()
