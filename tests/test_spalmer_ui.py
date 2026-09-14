from __future__ import annotations

import pytest
import spalmer.precision as spalmer_precision

from yylab.ui.pages import train_spalmer


def test_native_precision_capabilities_probes_exact_yylab_device(monkeypatch) -> None:
    seen: list[object] = []

    monkeypatch.setattr(train_spalmer, "model_execution_device", lambda: "cuda:7")
    monkeypatch.setattr(
        spalmer_precision,
        "detect_precision_capabilities",
        lambda device: seen.append(device)
        or {"expert_precisions": [], "selectable_pairs": [], "detail": "test"},
    )
    train_spalmer.native_precision_capabilities.cache_clear()
    try:
        report = train_spalmer.native_precision_capabilities()
    finally:
        train_spalmer.native_precision_capabilities.cache_clear()

    assert seen == ["cuda:7"]
    assert report["detail"] == "test"


def test_pipeline_controls_are_persisted_in_run_recipe(monkeypatch):
    cfg = _run_config(monkeypatch, optimizer_state_offload="none", token_shard_mib=16, data_prefetch_batches=3)
    assert cfg["training"]["optimizer_state_offload"] == "none"
    assert cfg["token_cache"] is True
    assert cfg["token_shard_bytes"] == 16 * 2**20
    assert cfg["data_prefetch_batches"] == 3


def _resolved_plan() -> dict:
    return {
        "target_parameters": 10_000_000,
        "parameter_error": 0,
        "relative_error": 0.0,
        "fingerprint": "a" * 64,
        "config": {
            "vocab_size": 4096,
            "expert_weight_format": "bfloat16",
            "expert_activation_format": "bfloat16",
            "ple_backend": "qr",
            "d_model": 128,
            "n_layers": 8,
        },
        "parameters": {"total": 10_000_000},
        "memory": {},
    }


def _run_config(monkeypatch, **overrides):
    capability_available = bool(overrides.pop("_capability_available", True))
    grouped_available = bool(overrides.pop("_grouped_available", True))
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
            "grouped_available": capability_available and grouped_available,
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
        "expert_weight_format": "bfloat16",
        "expert_activation_format": "bfloat16",
        "attention_backend": "fla",
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
    assert config["recipe_version"] == 6
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
    assert config["training"]["optimizer_state_offload"] == "cpu"
    assert config["training"]["activation_checkpointing"] is False
    assert config["training"]["validation_interval_steps"] == 246
    assert normalized["max_steps"] == 245
    assert normalized["activation_checkpointing"] is False
    assert normalized["optimizer_state_offload"] == "cpu"
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


def test_native_precision_options_hide_unverified_or_incomplete_pairs() -> None:
    report = {
        "expert_precisions": [
            {
                "provider_id": "bf16_torch",
                "weight_format": "bfloat16",
                "activation_format": "bfloat16",
                "forward_available": True,
                "backward_available": True,
                "grouped_available": True,
                "verified": True,
                "selectable": True,
            },
            {
                "provider_id": "cutlass_mxfp4",
                "weight_format": "mxfp4",
                "activation_format": "mxfp8",
                "forward_available": True,
                "backward_available": False,
                "grouped_available": False,
                "verified": True,
                "selectable": False,
            },
        ]
    }

    assert train_spalmer.native_expert_precision_options(report) == {
        "bfloat16:bfloat16": "BF16 weights / BF16 activations · bf16_torch · grouped"
    }
    with pytest.raises(ValueError, match="enter"):
        train_spalmer._resolve_budget(
            token_budget=0,
            max_steps=0,
            sequence_length=128,
            micro_batch=1,
            gradient_accumulation=1,
        )


def test_qat_status_skips_an_unavailable_duplicate_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        train_spalmer,
        "native_precision_capabilities",
        lambda: {
            "expert_precisions": [
                {
                    "provider_id": "native_broken",
                    "weight_format": "nvfp4",
                    "activation_format": "nvfp4",
                    "forward_available": True,
                    "backward_available": False,
                    "verified": True,
                    "selectable": False,
                    "detail": "no backward",
                },
                {
                    "provider_id": "native_working",
                    "weight_format": "nvfp4",
                    "activation_format": "nvfp4",
                    "forward_available": True,
                    "backward_available": True,
                    "grouped_available": False,
                    "verified": True,
                    "selectable": True,
                    "detail": "native dense",
                },
            ]
        },
    )

    status = train_spalmer.qat_backend_status("nvfp4", "auto", "nvfp4")
    assert status["native_available"] is True
    assert status["selected_backend"] == "native_working"


def test_spalmer_ui_strict_native_qat_fails_closed(monkeypatch) -> None:
    with pytest.raises(ValueError, match="test bfloat16 capability"):
        _run_config(
            monkeypatch,
            expert_qat_backend="native",
            _capability_available=False,
        )


def test_spalmer_ui_uses_loop_dispatch_for_native_dense_provider(monkeypatch) -> None:
    plan = _resolved_plan()
    plan["config"].update(
        expert_weight_format="nvfp4",
        expert_activation_format="nvfp4",
    )
    config = _run_config(
        monkeypatch,
        resolved_plan=plan,
        expert_weight_format="nvfp4",
        expert_activation_format="nvfp4",
        _grouped_available=False,
    )

    assert config["spalmer"]["experts"]["expert_execution"] == "loop"
    assert config["spalmer"]["experts"]["potentiation_budget"] == 2
    assert config["spalmer"]["experts"]["expert_promotion_format"] == "bfloat16"
    assert "expert_stochastic_rounding" not in config["spalmer"]["experts"]
    assert config["precision"]["expert_weight_format"] == "nvfp4"
    assert config["precision"]["expert_activation_format"] == "nvfp4"


def test_spalmer_ui_rejects_unimplemented_activation_checkpointing(monkeypatch) -> None:
    with pytest.raises(ValueError, match="not implemented"):
        _run_config(monkeypatch, activation_checkpointing=True)


def test_spalmer_ui_config_carries_epochs_for_scratch_runs(monkeypatch) -> None:
    config = _run_config(monkeypatch, epochs=3)

    assert config["recipe_version"] == 6
    assert config["epochs"] == 3
    assert config["initialization"] == {"mode": "scratch", "tokenizer_ref": "tok_4096"}


def test_spalmer_ui_config_defaults_to_one_epoch(monkeypatch) -> None:
    assert _run_config(monkeypatch)["epochs"] == 1


def test_spalmer_ui_config_rejects_non_positive_epochs(monkeypatch) -> None:
    with pytest.raises(ValueError, match="epochs"):
        _run_config(monkeypatch, epochs=0)


def test_spalmer_ui_config_continues_from_checkpoint(monkeypatch) -> None:
    config = _run_config(
        monkeypatch,
        initialization_mode="checkpoint",
        parent_checkpoint_id="ckpt_parent",
        tokenizer_ref="",
        tokenizer_source="checkpoint",
        tokenizer_registry_id=None,
        resolved_plan={},
        target_parameters=0,
        attention_backend="fla",
        epochs=2,
    )

    assert config["initialization"] == {"mode": "checkpoint", "checkpoint_id": "ckpt_parent"}
    assert set(config["spalmer"]) <= {
        "attention_backend",
        "require_fla",
    }
    assert config["spalmer"]["attention_backend"] == "fla"
    assert "architecture" not in config
    assert "precision" not in config
    assert "parameter_dtype" not in config["training"]
    assert "stochastic_parameter_rounding" not in config["training"]
    assert config["tokenizer"] == {
        "source": "checkpoint",
        "registry_id": None,
        "ref": "",
        "preview_vocab_size": None,
    }
    assert config["epochs"] == 2
    assert config["max_steps"] > 0


def test_spalmer_ui_checkpoint_mode_requires_parent(monkeypatch) -> None:
    with pytest.raises(ValueError, match="parent checkpoint"):
        _run_config(monkeypatch, initialization_mode="checkpoint", parent_checkpoint_id=None)


@pytest.mark.parametrize(("dtype", "policy"), [("bfloat16", "cpu"), ("float32", "none")])
def test_spalmer_ui_checkpoint_mode_normalizes(monkeypatch, tmp_path, dtype, policy) -> None:
    import torch

    from yylab import db
    from yylab.train.spalmer import normalize_spalmer_config

    bundle = tmp_path / "parent" / "spalmer.pt"
    bundle.parent.mkdir()
    torch.save({
        "format": "spalmer.prototype.checkpoint",
        "model_parameter_dtype": dtype,
        "experts_config": {"expert_master_dtype": dtype},
    }, bundle)
    monkeypatch.setattr(
        db,
        "q1",
        lambda sql, args=(): {
            "id": "ckpt_parent",
            "ref": str(bundle),
            "path": str(bundle.parent),
            "meta_json": '{"model_family": "spalmer"}',
        },
    )
    config = _run_config(
        monkeypatch,
        initialization_mode="checkpoint",
        parent_checkpoint_id="ckpt_parent",
        tokenizer_ref="",
        tokenizer_source="checkpoint",
        tokenizer_registry_id=None,
        resolved_plan={},
        target_parameters=0,
        attention_backend="fla",
        optimizer_state_offload=policy,
    )

    normalized = normalize_spalmer_config(config, require_runtime=True)

    assert normalized["initialization"]["checkpoint_ref"] == str(bundle.resolve())
    assert normalized["spalmer"] == {"attention_backend": "fla", "require_fla": True}
    assert normalized["training"]["parameter_dtype"] == dtype
    assert normalized["optimizer_state_offload"] == policy


def test_spalmer_checkpoint_rows_filter_by_family(monkeypatch) -> None:
    from yylab import db

    rows = [
        {"id": "a", "name": "spal", "params_b": 0.05, "meta_json": '{"model_family": "spalmer"}'},
        {"id": "b", "name": "scan", "params_b": 0.05, "meta_json": '{"family": "spalmer"}'},
        {"id": "c", "name": "hf", "params_b": 0.5, "meta_json": '{"model_family": "hf"}'},
        {"id": "d", "name": "broken", "params_b": 0.5, "meta_json": "not json"},
    ]
    monkeypatch.setattr(db, "q", lambda sql, args=(): rows)

    assert [row["id"] for row in train_spalmer._spalmer_checkpoint_rows()] == ["a", "b"]


def test_parent_plan_reads_nested_and_flat_metadata() -> None:
    flat = {"plan": {"fingerprint": "f", "config": {"vocab_size": 8192}}}
    nested = {"metadata": {"plan": {"fingerprint": "n"}}}

    assert train_spalmer._parent_plan(flat)["fingerprint"] == "f"
    assert train_spalmer._parent_plan(nested)["fingerprint"] == "n"
    assert train_spalmer._parent_plan({}) == {}


def _recurrent_plan() -> dict:
    plan = _resolved_plan()
    plan["recurrence"] = {
        "prelude_layers": 1,
        "core_layers": 6,
        "coda_layers": 1,
        "default_steps": 8,
        "effective_depth": 50,
        "core_mixers": {"kda": 5, "mla": 1},
    }
    return plan


def _recurrent_run_config(monkeypatch, **overrides):
    values = {
        "resolved_plan": _recurrent_plan(),
        "recurrence": {
            "enabled": True,
            "prelude_layers": 1,
            "coda_layers": 1,
            "default_steps": 8,
            "latent_init_std": 1.0,
        },
        "mean_recurrence": 8.0,
        "backprop_depth": 4,
        "recurrence_sigma": 0.5,
    }
    values.update(overrides)
    return _run_config(monkeypatch, **values)


def test_spalmer_ui_config_emits_recurrence_recipe(monkeypatch) -> None:
    from yylab.train.spalmer import normalize_spalmer_config

    config = _recurrent_run_config(monkeypatch)
    normalized = normalize_spalmer_config(config, require_runtime=True)

    assert config["recipe_version"] == 6
    assert config["spalmer"]["recurrence"] == {
        "enabled": True,
        "prelude_layers": 1,
        "coda_layers": 1,
        "default_steps": 8,
        "latent_init_std": 1.0,
    }
    assert config["training"]["mean_recurrence"] == 8.0
    assert config["training"]["mean_backprop_depth"] == 4
    assert config["training"]["recurrence_sigma"] == 0.5
    assert config["training"]["recurrence_sampling"] == "poisson_lognormal"
    assert config["mean_recurrence"] == 8.0
    assert config["mean_backprop_depth"] == 4
    assert config["recurrence_sigma"] == 0.5
    assert normalized["spalmer"]["recurrence"]["enabled"] is True
    assert normalized["recurrence_sampling"] == "poisson_lognormal"


def test_spalmer_ui_config_omits_recurrence_when_disabled(monkeypatch) -> None:
    config = _run_config(monkeypatch)

    assert config["spalmer"]["recurrence"] == {"enabled": False}
    assert "mean_recurrence" not in config
    assert "mean_recurrence" not in config["training"]


@pytest.mark.parametrize("scheme", ["fixed", "uniform", "poisson_lognormal"])
def test_recurrence_depth_controls_survive_recipe_normalization(monkeypatch, scheme):
    from yylab.train.spalmer import normalize_spalmer_config

    config = _recurrent_run_config(monkeypatch, recurrence_sampling=scheme, max_recurrence=12)
    normalized = normalize_spalmer_config(config, require_runtime=True)
    for values in (config, config["training"], normalized):
        assert values["recurrence_sampling"] == scheme
        assert values["max_recurrence"] == 12


@pytest.mark.parametrize("controls", [
    {"mean_recurrence": float("nan")}, {"recurrence_sigma": float("inf")},
    {"recurrence_sampling": "invalid"},
    {"recurrence_sampling": "fixed", "mean_recurrence": 4.5},
    {"max_recurrence": 4}, {"max_recurrence": 8.5},
])
def test_recurrence_ui_rejects_invalid_sampling_controls(monkeypatch, controls):
    with pytest.raises(ValueError):
        _recurrent_run_config(monkeypatch, **controls)


def test_spalmer_ui_config_rejects_inconsistent_recurrence(monkeypatch) -> None:
    with pytest.raises(ValueError, match="plan is not recurrent"):
        _recurrent_run_config(monkeypatch, resolved_plan=_resolved_plan())
    with pytest.raises(ValueError, match="at least one layer beyond"):
        _recurrent_run_config(
            monkeypatch,
            recurrence={
                "enabled": True,
                "prelude_layers": 4,
                "coda_layers": 4,
                "default_steps": 8,
            },
        )
    with pytest.raises(ValueError, match="at least one block each"):
        _recurrent_run_config(
            monkeypatch,
            recurrence={"enabled": True, "prelude_layers": 0, "coda_layers": 1},
        )
    with pytest.raises(ValueError, match="recurrent-depth architecture"):
        _run_config(monkeypatch, mean_recurrence=8.0)


def test_spalmer_ui_checkpoint_mode_recurrence_keys_follow_parent(monkeypatch) -> None:
    checkpoint_values = {
        "initialization_mode": "checkpoint",
        "parent_checkpoint_id": "ckpt_parent",
        "tokenizer_ref": "",
        "tokenizer_source": "checkpoint",
        "tokenizer_registry_id": None,
        "resolved_plan": {},
        "target_parameters": 0,
        "attention_backend": "fla",
        "mean_recurrence": 6.0,
        "backprop_depth": 3,
        "recurrence_sigma": 0.25,
    }
    inherited = _run_config(monkeypatch, parent_recurrent=True, **checkpoint_values)
    flat = _run_config(monkeypatch, parent_recurrent=False, **checkpoint_values)

    assert inherited["training"]["mean_recurrence"] == 6.0
    assert inherited["training"]["mean_backprop_depth"] == 3
    assert inherited["training"]["recurrence_sigma"] == 0.25
    assert inherited["mean_recurrence"] == 6.0
    assert "recurrence" not in inherited["spalmer"]
    assert "mean_recurrence" not in flat["training"]
    assert "mean_recurrence" not in flat


def test_spalmer_ui_checkpoint_mode_rejects_a_new_recurrent_core(monkeypatch) -> None:
    with pytest.raises(ValueError, match="inherits the recurrent core"):
        _run_config(
            monkeypatch,
            initialization_mode="checkpoint",
            parent_checkpoint_id="ckpt_parent",
            tokenizer_ref="",
            tokenizer_source="checkpoint",
            tokenizer_registry_id=None,
            resolved_plan={},
            target_parameters=0,
            attention_backend="fla",
            recurrence={"enabled": True, "prelude_layers": 1, "coda_layers": 1},
        )


def test_plan_preview_cache_key_includes_recurrence(monkeypatch) -> None:
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
        recurrence={
            "enabled": True,
            "prelude_layers": 2,
            "coda_layers": 3,
            "default_steps": 5,
            "latent_init_std": 0.0,
        },
    )
    train_spalmer.plan_spalmer_configuration(
        target_parameters=10_000_000,
        vocab_size=4096,
        expert_weight_format="bfloat16",
        expert_activation_format="bfloat16",
        directional={"enabled": False},
        atxy={"enabled": False},
    )

    assert seen[0][-5:] == (True, 2, 3, 5, 0.0)
    assert seen[1][-5:] == (False, 1, 1, 8, 1.0)


def test_checkpoint_recurrence_reads_every_metadata_shape() -> None:
    run_row = {"meta_json": '{"recurrence": {"prelude_layers": 1, "core_layers": 2}}'}
    scanned_row = {"meta_json": '{"recurrence_config": {"core_layers": 4}}'}
    nested_row = {"meta_json": '{"metadata": {"recurrence": {"core_layers": 6}}}'}

    assert train_spalmer._checkpoint_recurrence(run_row)["core_layers"] == 2
    assert train_spalmer._checkpoint_recurrence(scanned_row)["core_layers"] == 4
    assert train_spalmer._checkpoint_recurrence(nested_row)["core_layers"] == 6
    assert train_spalmer._checkpoint_recurrence({"meta_json": "{}"}) is None
    assert train_spalmer._checkpoint_recurrence({"meta_json": "not json"}) is None
    # A scratch recipe records the disabled block verbatim; it is not a core.
    assert (
        train_spalmer._checkpoint_recurrence({"meta_json": '{"recurrence": {"enabled": false}}'})
        is None
    )


def test_spalmer_base_recipe_mentions_recurrence() -> None:
    from yylab.ui.pages import checkpoints

    registry_receipt = {
        "precision": {
            "expert_weights": "nvfp4",
            "expert_activations": "nvfp4",
            "master_weights": "bfloat16",
        },
        "recurrence": {
            "prelude_layers": 1,
            "core_layers": 2,
            "coda_layers": 1,
            "default_steps": 8,
            "effective_depth": 18,
        },
        "effective_depth": 18,
    }
    scanned_bundle = {
        "experts_config": {
            "expert_weight_format": "mxfp8",
            "expert_activation_format": "mxfp8",
            "expert_master_dtype": "bfloat16",
        },
        "recurrence_config": {
            "prelude_layers": 2,
            "core_layers": 4,
            "coda_layers": 2,
            "default_steps": 6,
        },
        "effective_depth": 28,
    }
    flat = {
        "precision": {
            "expert_weights": "bfloat16",
            "expert_activations": "bfloat16",
            "master_weights": "bfloat16",
        }
    }

    assert "recurrent 1/2/1 × 8" in checkpoints._spalmer_base_recipe(registry_receipt)
    assert "recurrent 2/4/2 × 6" in checkpoints._spalmer_base_recipe(scanned_bundle)
    assert "recurrent" not in checkpoints._spalmer_base_recipe(flat)
    assert checkpoints._spalmer_effective_depth(registry_receipt) == 18
    assert checkpoints._spalmer_effective_depth(scanned_bundle) == 28
    assert checkpoints._spalmer_effective_depth(flat) is None
