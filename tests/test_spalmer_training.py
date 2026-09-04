from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from yylab.train import spalmer as bridge
from yylab.train.data import PackedBatches


def _config(**updates):
    value = {
        "model_family": "spalmer",
        "method": "full",
        "objective": "causal_lm",
        "initialization": {"mode": "scratch", "tokenizer_ref": "fake-tokenizer"},
        "spalmer": {
            "target_parameters": 10_000_000,
            "num_experts": 200,
            "expert_weight_format": "mxfp4",
            "attention_backend": "reference",
            "expert_qat_backend": "reference",
            "directional": {
                "enabled": True,
                "num_feature_groups": 4,
                "lateral_rank": 16,
            },
            "atxy": {"enabled": False},
            # These shape values are UI plan evidence and must not be passed as
            # protected ConfigBundle overrides.
            "experts": {
                "num_experts": 200,
                "active_experts": 2,
                "min_active_experts": 2,
                "max_active_experts": 20,
                "min_resident_experts": 2,
                "max_resident_experts": 20,
                "expert_execution": "grouped",
                "potentiation_budget": 2,
                "residency_increment": 2,
            },
        },
        "corpora": ["alpha"],
        "corpus_mix": "concatenate",
        "cross_corpus_dedup": True,
        "seq_len": 8,
        "micro_batch": 1,
        "grad_accum": 1,
        "max_steps": 2,
        "epochs": 1,
        "seed": 0,
        "log_every": 1,
        "save_every": 1,
        "out_checkpoint_id": "ckpt-test",
        "device": "cpu",
        "require_cuda": False,
    }
    value.update(updates)
    return value


def test_normalize_spalmer_config_accepts_ui_contract_without_mutation():
    original = _config()
    normalized = bridge.normalize_spalmer_config(original, require_runtime=True)

    assert original["spalmer"]["experts"]["num_experts"] == 200
    assert normalized["model_family"] == "spalmer"
    assert normalized["objective"] == "causal_lm"
    assert normalized["seed"] == 0
    assert normalized["spalmer"]["num_experts"] == 200
    assert normalized["spalmer"]["experts"] == {
        "expert_execution": "grouped",
        "potentiation_budget": 2,
        "residency_increment": 2,
        "expert_qat_backend": "reference",
    }
    assert normalized["spalmer"]["expert_plan_evidence"]["max_active_experts"] == 20
    assert normalized["optimizer_state_offload"] == "cpu"


def test_normalize_spalmer_config_requires_cpu_optimizer_state_offload():
    config = _config(
        training={
            "compute_dtype": "bfloat16",
            "parameter_dtype": "bfloat16",
            "optimizer_state_dtype": "float32",
            "optimizer_state_offload": "none",
        }
    )
    with pytest.raises(ValueError, match="optimizer_state_offload='cpu'"):
        bridge.normalize_spalmer_config(config)


def test_training_config_enables_cpu_optimizer_state_offload():
    api = SimpleNamespace(
        TrainingConfig=lambda **values: SimpleNamespace(**values),
    )

    training = bridge.build_spalmer_training_config(_config(), api=api)

    assert training.parameter_dtype == "bfloat16"
    assert training.optimizer_state_dtype == "float32"
    assert training.optimizer_state_offload == "cpu"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_family": "hf"}, "model_family='spalmer'"),
        ({"objective": "chat_sft"}, "only causal_lm"),
        ({"method": "lora"}, "full-parameter"),
    ],
)
def test_normalize_spalmer_config_rejects_incompatible_base_modes(updates, message):
    with pytest.raises(ValueError, match=message):
        bridge.normalize_spalmer_config(_config(**updates))


def test_planner_recomputes_with_vocab_custom_experts_directional_and_atxy():
    calls = {}

    class Directional:
        def __init__(self, **kwargs):
            self.values = kwargs

    class ATXY:
        def __init__(self, **kwargs):
            self.values = kwargs

    class Search:
        def __init__(self, **kwargs):
            self.values = kwargs

    plan = SimpleNamespace(fingerprint="f" * 64)

    def plan_configuration(target, vocab, **kwargs):
        calls.update(target=target, vocab=vocab, **kwargs)
        return plan

    api = SimpleNamespace(
        DirectionalScaleConfig=Directional,
        ATXYScaleConfig=ATXY,
        ScaleSearchSpace=Search,
        plan_configuration=plan_configuration,
    )
    config = _config()
    config["spalmer"] = {
        **config["spalmer"],
        "plan_fingerprint": "f" * 64,
        "num_experts": 37,
        "expert_weight_format": "nvfp4",
        "atxy": {
            "enabled": True,
            "value_dim": 32,
            "a_cardinality": 3,
            "t_cardinality": 5,
            "x_cardinality": 7,
            "y_cardinality": 11,
            "injection_layer": 0,
        },
    }

    assert bridge.plan_spalmer_configuration(config, 8192, api=api) is plan
    assert calls["target"] == 10_000_000
    assert calls["vocab"] == 8192
    assert calls["expert_weight_format"] == "nvfp4"
    assert calls["search_space"].values == {"num_experts": 37}
    assert calls["directional"].values == {
        "num_feature_groups": 4,
        "lateral_rank": 16,
    }
    assert calls["atxy"].values["y_cardinality"] == 11


def test_planner_rejects_stale_ui_fingerprint():
    api = SimpleNamespace(
        DirectionalScaleConfig=lambda **_kwargs: object(),
        ATXYScaleConfig=lambda **_kwargs: object(),
        ScaleSearchSpace=lambda **_kwargs: object(),
        plan_configuration=lambda *_args, **_kwargs: SimpleNamespace(
            fingerprint="new-plan"
        ),
    )
    config = _config()
    config["spalmer"] = {**config["spalmer"], "plan_fingerprint": "old-plan"}
    with pytest.raises(ValueError, match="plan fingerprint mismatch"):
        bridge.plan_spalmer_configuration(config, 4096, api=api)


@dataclass
class _CausalBatch:
    input_ids: torch.Tensor


def _batch_source(*, sequence_length=3, accumulation_steps=2, token_count=32):
    def factory():
        docs = [{"text": "x", "_yylab_corpus_id": "alpha"}]

        def tokenize(_texts):
            return [list(range(1, token_count + 1))]

        return PackedBatches(
            docs,
            tokenize,
            seq_len=sequence_length + 1,
            eos_token_id=0,
            micro_batch=1,
            max_batches=20,
            tokenize_chunk=1,
        )

    return bridge.YYLabSPALMERBatchSource(
        factory,
        causal_batch_type=_CausalBatch,
        micro_batch_size=1,
        sequence_length=sequence_length,
        accumulation_steps=accumulation_steps,
    )


def test_batch_source_uses_n_plus_one_and_replays_consumed_batches():
    source = _batch_source()
    first = source.next_batch(batch_size=1, sequence_length=3)
    source.next_batch(batch_size=1, sequence_length=3)
    state = source.state_dict()
    expected_third = source.next_batch(batch_size=1, sequence_length=3).input_ids.clone()

    resumed = _batch_source()
    resumed.load_state_dict(state)
    actual_third = resumed.next_batch(batch_size=1, sequence_length=3).input_ids

    assert first.input_ids.shape == (1, 4)
    assert state["consumed_batches"] == 2
    assert torch.equal(actual_third, expected_third)
    assert resumed.data_stats()["tokens"]["model_target_slots"] == 9


def test_batch_source_does_not_emit_partial_accumulation_window():
    source = _batch_source(sequence_length=3, accumulation_steps=2, token_count=4)
    with pytest.raises(bridge.CorpusExhausted, match="complete gradient-accumulation"):
        source.next_batch(batch_size=1, sequence_length=3)
    assert source.consumed_batches == 0
    assert source.discarded_incomplete_batches == 1


class _FakeIdentity:
    fingerprint = "tokenizer-fingerprint"

    def to_dict(self):
        return {
            "backend": "huggingface",
            "source": "fake-tokenizer",
            "fingerprint": self.fingerprint,
            "vocab_size": 4096,
        }


class _FakeTokenizer:
    vocab_size = 4096
    identity = _FakeIdentity()


class _FakePlan:
    target_parameters = 10_000_000
    relative_error = 0.0
    fingerprint = "plan-fingerprint"
    config = SimpleNamespace(
        num_experts=200,
        expert_weight_format="mxfp4",
        d_model=64,
        n_layers=4,
    )
    parameters = SimpleNamespace(total=10_000_000)

    def to_dict(self):
        return {
            "target_parameters": self.target_parameters,
            "fingerprint": self.fingerprint,
            "config": {
                "num_experts": 200,
                "expert_weight_format": "mxfp4",
                "d_model": 64,
                "n_layers": 4,
            },
            "parameters": {"total": 10_000_000},
        }


class _ConfigObject:
    def __init__(self, **values):
        self.__dict__.update(values)

    def to_dict(self):
        return dict(self.__dict__)


class _FakeModel:
    def parameter_accounting(self):
        return SimpleNamespace(
            total=10_000_000,
            expert_pool=7_000_000,
            shared_base=2_000_000,
            resident_parameters=3_000_000,
            per_token_active_parameters=2_100_000,
        )


class _FakeBatches:
    docs_seen = 3
    tokens_seen = 18
    delivered_sequences = 2
    discarded_incomplete_batches = 0

    def state_dict(self):
        return {"version": 1, "consumed_batches": 2}

    def data_stats(self):
        return {
            "docs_seen": 3,
            "rendered_tokens": 18,
            "tokens": {
                "rendered_token_positions": 18,
                "covered_rendered_positions": 18,
                "forwarded_model_slots": 18,
                "forwarded_real_inputs": 18,
                "padding_inputs": 0,
                "packing_replay_inputs": 0,
                "model_target_slots": 16,
                "optimizer_supervised_targets": 16,
                "ignored_target_slots": 0,
            },
            "per_corpus": {"alpha": {"documents_seen": 3}},
        }


def _step_metric(**updates):
    values = {
        "step": 1,
        "tokens_seen": 8,
        "objective": 2.5,
        "model_loss": 2.4,
        "auxiliary_loss": 0.1,
        "surprise_calibration_loss": 0.05,
        "predictive_entropy": 1.7,
        "learning_rate": 3e-4,
        "gradient_norm": 0.8,
        "elapsed_seconds": 0.1,
        "promoted_experts": (3,),
        "average_surprise": 2.3,
        "max_expert_group_load": 17,
        "max_expert_group_padding_amplification": 1.375,
        "max_expert_group_global_max_counterfactual_padding_amplification": 13.625,
    }
    values.update(updates)
    return SimpleNamespace(**values)


class _FakeTrainer:
    metric_updates: dict = {}

    def __init__(self, model, batches, config, *, on_checkpoint, on_telemetry):
        self.model = model
        self.batches = batches
        self.config = config
        self.on_checkpoint = on_checkpoint
        self.on_telemetry = on_telemetry
        self.progress = SimpleNamespace(completed_steps=0, tokens_seen=0)
        self.optimizer = SimpleNamespace(
            zero_grad=lambda **_kwargs: None,
            state_placement=lambda: {
                "dense": {
                    "policy": "cpu",
                    "moment_dtype": "float32",
                    "initialized": True,
                    "moment_bytes": 128,
                    "devices": ["cpu"],
                    "cpu_pinned": False,
                    "parameter_devices": ["cuda:0"],
                    "parameter_dtypes": ["bfloat16"],
                },
                "sparse": None,
            },
        )

    def run(self):
        metric = _step_metric(**type(self).metric_updates)
        self.progress.completed_steps = 1
        self.progress.tokens_seen = 8
        self.on_checkpoint(self, metric)
        yield metric


def _native_run(
    tmp_path,
    monkeypatch,
    *,
    recurrence=None,
    training_updates=None,
    metric_updates=None,
    api_updates=None,
):
    """Drive one fake run and hand back everything it wrote."""

    output = tmp_path / "checkpoint"
    batches = _FakeBatches()
    training = _ConfigObject(
        max_steps=2,
        sequence_length=8,
        seed=0,
        deterministic_algorithms=False,
        device="cpu",
        optimizer_update_chunk_size=1_048_576,
        **(training_updates or {}),
    )
    experts = _ConfigObject(
        expert_activation_format="mxfp8",
        expert_master_dtype="bfloat16",
        expert_qat_backend="reference",
    )
    model_config = _ConfigObject(vocab_size=4096, d_model=64, n_layers=4)
    if recurrence is not None:
        model_config.recurrence = recurrence
    bundle = SimpleNamespace(
        build=lambda: _FakeModel(),
        model=model_config,
        kda=_ConfigObject(backend="reference"),
        mla=_ConfigObject(num_heads=4),
        experts=experts,
        directional=None,
        atxy=None,
    )
    prepared = bridge.PreparedSpalmerRun(
        _config(), _FakeTokenizer(), _FakePlan(), bundle, training
    )
    saved = {}

    def save_checkpoint(path, *_args, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mock-spalmer-checkpoint")
        saved[path] = kwargs.get("metadata")
        return path

    trainer = type(
        "_ScopedFakeTrainer", (_FakeTrainer,), {"metric_updates": dict(metric_updates or {})}
    )
    api = SimpleNamespace(
        CausalBatch=_CausalBatch,
        ExperimentTrainer=trainer,
        assert_accounting_matches_plan=lambda *_args: None,
        save_checkpoint=save_checkpoint,
        seed_everything=lambda *_args, **_kwargs: None,
        **(api_updates or {}),
    )
    database_calls = []
    monkeypatch.setattr(bridge, "_load_spalmer_api", lambda: api)
    monkeypatch.setattr(bridge, "prepare_spalmer_run", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(bridge, "build_spalmer_batch_source", lambda *_args, **_kwargs: batches)
    monkeypatch.setattr(bridge, "checkpoint_dir", lambda _checkpoint_id: output)
    monkeypatch.setattr(
        bridge,
        "corpus_training_preflight",
        lambda *_args, **_kwargs: {
            "errors": [],
            "warnings": [],
            "provenance": {"schema_version": 3, "corpus_ids": ["alpha"]},
        },
    )
    monkeypatch.setattr(
        bridge.db,
        "execute",
        lambda sql, args=(): database_calls.append((sql, args)) or 1,
    )

    summary = bridge.run_spalmer_base(_config(), tmp_path / "run")
    run_directory = tmp_path / "run"
    return {
        "output": output,
        "saved": saved,
        "summary": summary,
        "database_calls": database_calls,
        "manifest": json.loads((output / "spalmer_manifest.json").read_text("utf-8")),
        "recipe": json.loads((output / "yylab_recipe.json").read_text("utf-8")),
        "token_view": json.loads(
            (output / "tokenized_view_manifest.json").read_text("utf-8")
        ),
        "progress": json.loads((run_directory / "progress.json").read_text("utf-8")),
        "telemetry": [
            json.loads(line)
            for line in (run_directory / "metrics.jsonl")
            .read_text("utf-8")
            .splitlines()
        ],
        "placement_receipt": json.loads(
            (run_directory / "optimizer_state_placement.json").read_text("utf-8")
        ),
    }


def test_run_writes_native_artifact_metadata_without_real_training(tmp_path, monkeypatch):
    artifacts = _native_run(tmp_path, monkeypatch, api_updates={"FORMAT_VERSION": 6})

    output = artifacts["output"]
    saved = artifacts["saved"]
    summary = artifacts["summary"]
    database_calls = artifacts["database_calls"]
    manifest = artifacts["manifest"]
    recipe = artifacts["recipe"]
    token_view = artifacts["token_view"]
    progress = artifacts["progress"]
    telemetry = artifacts["telemetry"][-1]
    placement_receipt = artifacts["placement_receipt"]

    assert output / "spalmer.pt" in saved
    assert output / "steps" / "step-00000001" / "spalmer.pt" in saved
    assert manifest["checkpoint_format"] == "spalmer.prototype.checkpoint.v6"
    assert manifest["recurrence"] is None
    assert manifest["effective_depth"] == 4
    assert saved[output / "spalmer.pt"]["recurrence"] is None
    assert telemetry["recurrence_steps"] is None
    assert manifest["model_family"] == "spalmer"
    assert manifest["checkpoint_file"] == "spalmer.pt"
    assert manifest["plan_fingerprint"] == "plan-fingerprint"
    assert recipe["model_family"] == "spalmer"
    assert token_view["input_width"] == 9
    assert summary["model_family"] == "spalmer"
    assert progress["max_expert_group_load"] == 17
    assert progress["max_expert_group_padding_amplification"] == pytest.approx(1.375)
    assert progress[
        "max_expert_group_global_max_counterfactual_padding_amplification"
    ] == pytest.approx(13.625)
    assert "expert group max=17" in telemetry["msg"]
    assert "bucket padding=1.375x" in telemetry["msg"]
    assert "legacy global-max=13.625x" in telemetry["msg"]
    _, update_args = database_calls[-1]
    metadata = json.loads(update_args[-2])
    assert update_args[0].endswith("spalmer.pt")
    assert metadata["model_family"] == "spalmer"
    assert metadata["precision"] == {
        "expert_weights": "mxfp4",
        "expert_activations": "mxfp8",
        "master_weights": "bfloat16",
        "expert_qat_backend": "reference",
        "ple_backend": "fake_qat",
        "ple_quant_bits": 4,
        "ple_stochastic_rounding": True,
        "optimizer_moments": "float32_cpu_offload",
    }
    assert manifest["optimizer_state_placement"]["dense"]["devices"] == ["cpu"]
    assert placement_receipt["verified_after_step"] == 1
    assert placement_receipt["optimizer_state_placement"]["dense"]["devices"] == ["cpu"]


def _parent_row(bundle: Path, *, family: str = "spalmer") -> dict:
    return {
        "id": "ckpt_parent",
        "name": "spal50m",
        "kind": "base",
        "ref": str(bundle),
        "path": str(bundle.parent),
        "params_b": 0.05,
        "hidden_size": 192,
        "num_layers": 4,
        "meta_json": json.dumps({"model_family": family, "plan": {"fingerprint": "parent-plan"}}),
    }


def _checkpoint_config(**updates):
    config = _config(
        initialization={"mode": "checkpoint", "checkpoint_id": "ckpt_parent"},
        spalmer={"attention_backend": "reference"},
    )
    config.update(updates)
    return config


def test_normalize_spalmer_config_checkpoint_mode_inherits_architecture(tmp_path, monkeypatch):
    bundle = tmp_path / "parent" / "spalmer.pt"
    bundle.parent.mkdir()
    bundle.write_bytes(b"bundle")
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(bundle))

    normalized = bridge.normalize_spalmer_config(_checkpoint_config(epochs=3), require_runtime=True)

    assert normalized["initialization"] == {
        "mode": "checkpoint",
        "checkpoint_id": "ckpt_parent",
        "checkpoint_ref": str(bundle.resolve()),
    }
    assert normalized["tokenizer_ref"] is None
    assert normalized["spalmer"] == {"attention_backend": "reference", "require_fla": False}
    assert "target_parameters" not in normalized
    assert normalized["epochs"] == 3


@pytest.mark.parametrize(
    ("row_factory", "updates", "message"),
    [
        (lambda bundle: None, {}, "unknown parent checkpoint"),
        (lambda bundle: _parent_row(bundle, family="hf"), {}, "not a SPALMER checkpoint"),
        (
            lambda bundle: _parent_row(bundle),
            {"spalmer": {"attention_backend": "reference", "num_experts": 100}},
            "num_experts",
        ),
        (lambda bundle: _parent_row(bundle), {"epochs": 0}, "epochs must be positive"),
        (
            lambda bundle: _parent_row(bundle),
            {"initialization": {"mode": "checkpoint"}},
            "initialization.checkpoint_id",
        ),
    ],
)
def test_normalize_spalmer_config_checkpoint_mode_rejections(
    tmp_path, monkeypatch, row_factory, updates, message
):
    bundle = tmp_path / "parent" / "spalmer.pt"
    bundle.parent.mkdir()
    bundle.write_bytes(b"bundle")
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): row_factory(bundle))

    with pytest.raises(ValueError, match=message):
        bridge.normalize_spalmer_config(_checkpoint_config(**updates))


def test_normalize_spalmer_config_checkpoint_mode_requires_bundle_file(tmp_path, monkeypatch):
    missing = tmp_path / "parent" / "spalmer.pt"
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(missing))

    with pytest.raises(ValueError, match="bundle is unavailable"):
        bridge.normalize_spalmer_config(_checkpoint_config())


def test_normalize_spalmer_config_rejects_unknown_initialization_mode():
    with pytest.raises(ValueError, match="initialization mode"):
        bridge.normalize_spalmer_config(_config(initialization={"mode": "lora"}))


def _parent_bundle(
    path: Path, *, tokenizer_source: str, format_version: int = 5, recurrence=None
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = {"vocab_size": 32, "d_model": 8, "n_layers": 4}
    if format_version >= 7:
        model_config["ple_backend"] = "qr"
    if recurrence is not None:
        model_config["recurrence"] = recurrence
    torch.save(
        {
            "format": "spalmer.prototype.checkpoint",
            "format_version": format_version,
            "model_parameter_dtype": "bfloat16",
            "model_config": model_config,
            "kda_config": {"hidden_size": 8, "num_heads": 2},
            "mla_config": {"hidden_size": 8, "num_heads": 2},
            "experts_config": {"num_experts": 200, "expert_weight_format": "mxfp4"},
            "directional_config": {"d_model": 8, "enabled": True},
            "atxy_config": None,
            "tokenizer_binding": {
                "kind": "huggingface",
                "identity": {
                    "backend": "huggingface",
                    "source": tokenizer_source,
                    "artifact_fingerprint": "artifact-expected",
                    "vocab_size": 32,
                    "special_tokens": {"bos_id": None, "eos_id": 0, "eod_id": None, "pad_id": 0},
                    "revision": None,
                    "fingerprint": "identity-expected",
                },
            },
            "model_state": {"w": torch.zeros(2, dtype=torch.bfloat16)},
            "metadata": {
                "run_id": "run_parent",
                "plan": {
                    "target_parameters": 50_000_000,
                    "relative_error": 0.0001,
                    "fingerprint": "parent-plan",
                    "config": {
                        "num_experts": 200,
                        "expert_weight_format": "mxfp4",
                        "d_model": 8,
                        "n_layers": 4,
                    },
                    "parameters": {"total": 50_006_136},
                },
            },
        },
        path,
    )
    return path


def _checkpoint_api(load_checkpoint, **extra):
    return SimpleNamespace(
        load_checkpoint=load_checkpoint,
        SPALMERConfig=_ConfigObject,
        KDAConfig=_ConfigObject,
        MLAConfig=_ConfigObject,
        MicroExpertsConfig=_ConfigObject,
        DirectionalConfig=_ConfigObject,
        ATXYConfig=_ConfigObject,
        TrainingConfig=lambda **values: SimpleNamespace(**values),
        fla_available=lambda: True,
        plan_configuration=lambda *_args, **_kwargs: pytest.fail("plan must not be recomputed"),
        build_configs=lambda *_args, **_kwargs: pytest.fail("bundle must not be rebuilt"),
        **extra,
    )


def test_prepare_spalmer_run_continues_from_parent_bundle(tmp_path, monkeypatch):
    bundle = _parent_bundle(
        tmp_path / "parent" / "spalmer.pt", tokenizer_source=str(tmp_path / "gone")
    )
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(bundle))
    override = SimpleNamespace(identity=SimpleNamespace(source="relocated"))
    seen_bindings = []
    monkeypatch.setattr(
        "yylab.models.spalmer.relocated_spalmer_tokenizer",
        lambda binding: seen_bindings.append(binding) or override,
    )
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    calls = {}

    def load_checkpoint(path, **kwargs):
        calls["path"] = Path(path)
        calls["kwargs"] = kwargs
        plan = torch.load(path, weights_only=True)["metadata"]["plan"]
        return model, tokenizer, {"run_id": "run_parent", "plan": plan}

    prepared = bridge.prepare_spalmer_run(_checkpoint_config(), api=_checkpoint_api(load_checkpoint))

    assert prepared.model is model
    assert prepared.tokenizer is tokenizer
    assert calls["path"] == bundle.resolve()
    assert calls["kwargs"] == {"map_location": "cpu", "tokenizer_override": override}
    assert seen_bindings[0]["kind"] == "huggingface"
    assert prepared.plan.fingerprint == "parent-plan"
    assert prepared.plan.config.num_experts == 200
    assert prepared.plan.parameters.total == 50_006_136
    assert prepared.plan.to_dict()["target_parameters"] == 50_000_000
    assert prepared.bundle.kda.hidden_size == 8
    assert prepared.bundle.model.to_dict()["d_model"] == 8
    assert prepared.bundle.directional.enabled is True
    assert prepared.bundle.atxy is None
    assert prepared.parent == {
        "checkpoint_id": "ckpt_parent",
        "checkpoint_path": str(bundle.resolve()),
        "run_id": "run_parent",
        "plan_fingerprint": "parent-plan",
    }
    assert prepared.training.max_steps == 2


def test_prepare_spalmer_run_skips_override_when_tokenizer_path_exists(tmp_path, monkeypatch):
    present = tmp_path / "tok"
    present.mkdir()
    (present / "tokenizer.json").write_text("{}", encoding="utf-8")
    bundle = _parent_bundle(tmp_path / "parent" / "spalmer.pt", tokenizer_source=str(present))
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(bundle))
    monkeypatch.setattr("yylab.models.spalmer.relocated_spalmer_tokenizer", lambda binding: None)
    calls = {}

    def load_checkpoint(path, **kwargs):
        calls["kwargs"] = kwargs
        return (
            _FakeModel(),
            _FakeTokenizer(),
            {"run_id": "run_parent", "plan": {"fingerprint": "parent-plan"}},
        )

    prepared = bridge.prepare_spalmer_run(_checkpoint_config(), api=_checkpoint_api(load_checkpoint))

    assert calls["kwargs"] == {"map_location": "cpu"}
    assert prepared.parent["run_id"] == "run_parent"


def test_prepare_spalmer_run_requires_parent_plan_metadata(tmp_path, monkeypatch):
    bundle = _parent_bundle(
        tmp_path / "parent" / "spalmer.pt", tokenizer_source=str(tmp_path / "gone")
    )
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(bundle))
    monkeypatch.setattr("yylab.models.spalmer.relocated_spalmer_tokenizer", lambda binding: None)
    api = _checkpoint_api(lambda path, **kwargs: (_FakeModel(), _FakeTokenizer(), {"run_id": "x"}))

    with pytest.raises(ValueError, match="scale plan"):
        bridge.prepare_spalmer_run(_checkpoint_config(), api=api)


class _FakeLoadedModel(_FakeModel):
    def __init__(self):
        self.moves = []

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))
        return self


def test_run_uses_loaded_model_and_records_parent(tmp_path, monkeypatch):
    output = tmp_path / "checkpoint"
    bundle_path = tmp_path / "parent" / "spalmer.pt"
    bundle_path.parent.mkdir()
    bundle_path.write_bytes(b"bundle")
    batches = _FakeBatches()
    training = _ConfigObject(
        max_steps=2,
        sequence_length=8,
        seed=0,
        deterministic_algorithms=False,
        device="cpu",
        optimizer_update_chunk_size=1_048_576,
    )
    experts = _ConfigObject(
        expert_activation_format="mxfp8",
        expert_master_dtype="bfloat16",
        expert_qat_backend="reference",
    )
    bundle = SimpleNamespace(
        build=lambda: pytest.fail("continued runs must not build a fresh model"),
        model=_ConfigObject(vocab_size=4096, d_model=64, n_layers=4),
        kda=_ConfigObject(backend="reference"),
        mla=_ConfigObject(num_heads=4),
        experts=experts,
        directional=None,
        atxy=None,
    )
    loaded = _FakeLoadedModel()
    parent = {
        "checkpoint_id": "ckpt_parent",
        "checkpoint_path": str(bundle_path),
        "run_id": "run_parent",
        "plan_fingerprint": "plan-fingerprint",
    }
    prepared = bridge.PreparedSpalmerRun(
        _checkpoint_config(),
        _FakeTokenizer(),
        _FakePlan(),
        bundle,
        training,
        model=loaded,
        parent=parent,
    )
    saved = {}

    def save_checkpoint(path, *_args, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mock-spalmer-checkpoint")
        saved[path] = kwargs["metadata"]
        return path

    api = SimpleNamespace(
        CausalBatch=_CausalBatch,
        ExperimentTrainer=_FakeTrainer,
        assert_accounting_matches_plan=lambda *_args: pytest.fail("accounting is not replanned"),
        save_checkpoint=save_checkpoint,
        seed_everything=lambda *_args, **_kwargs: None,
    )
    database_calls = []
    monkeypatch.setattr(bridge, "_load_spalmer_api", lambda: api)
    monkeypatch.setattr(bridge, "prepare_spalmer_run", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(bridge, "build_spalmer_batch_source", lambda *_args, **_kwargs: batches)
    monkeypatch.setattr(bridge, "checkpoint_dir", lambda _checkpoint_id: output)
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(bundle_path))
    monkeypatch.setattr(
        bridge,
        "corpus_training_preflight",
        lambda *_args, **_kwargs: {
            "errors": [],
            "warnings": [],
            "provenance": {"schema_version": 3, "corpus_ids": ["alpha"]},
        },
    )
    monkeypatch.setattr(
        bridge.db,
        "execute",
        lambda sql, args=(): database_calls.append((sql, args)) or 1,
    )

    summary = bridge.run_spalmer_base(_checkpoint_config(), tmp_path / "run")

    assert loaded.moves, "loaded model must be moved to the training device"
    assert saved[output / "spalmer.pt"]["parent_checkpoint"] == parent
    manifest = json.loads((output / "spalmer_manifest.json").read_text("utf-8"))
    assert manifest["parent_checkpoint"] == parent
    assert manifest["data_provenance"]["initialization"] == "checkpoint"
    registry_meta = json.loads(database_calls[-1][1][6])
    assert registry_meta["initialization"] == "checkpoint"
    assert registry_meta["parent_checkpoint_id"] == "ckpt_parent"
    assert summary["parent_checkpoint"] == parent


def _recurrent_config(**updates):
    config = _config()
    config["spalmer"] = {
        **config["spalmer"],
        "recurrence": {"enabled": True, "prelude_layers": 1, "coda_layers": 1},
    }
    config["training"] = {"mean_recurrence": 8, "mean_backprop_depth": 4}
    config.update(updates)
    return config


def test_normalize_scratch_recurrence_block_and_training_keys():
    normalized = bridge.normalize_spalmer_config(_recurrent_config(), require_runtime=True)

    assert normalized["spalmer"]["recurrence"] == {
        "enabled": True,
        "prelude_layers": 1,
        "coda_layers": 1,
        "default_steps": 8,
        "latent_init_std": 1.0,
    }
    # The training keys are hoisted to the root exactly like every other alias.
    assert normalized["mean_recurrence"] == 8.0
    assert normalized["mean_backprop_depth"] == 4
    assert normalized["recurrence_sigma"] == 0.5
    assert normalized["recurrence_sampling"] == "poisson_lognormal"
    assert normalized["max_recurrence"] is None


def test_normalize_scratch_recurrence_defaults_to_disabled():
    normalized = bridge.normalize_spalmer_config(_config())

    assert normalized["spalmer"]["recurrence"] == {"enabled": False}
    assert "mean_recurrence" not in normalized


@pytest.mark.parametrize(
    "updates",
    [
        {"training": {"mean_recurrence": 4}},
        {"mean_recurrence": 4},
    ],
)
def test_normalize_rejects_recurrence_training_keys_without_a_core(updates):
    with pytest.raises(ValueError, match="spalmer.recurrence.enabled"):
        bridge.normalize_spalmer_config(_config(**updates))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"training": {}}, "requires training.mean_recurrence"),
        ({"training": {"mean_recurrence": 0.5}}, "mean_recurrence must be at least 1"),
        (
            {"training": {"mean_recurrence": 4, "mean_backprop_depth": 0}},
            "mean_backprop_depth must be positive",
        ),
        (
            {"training": {"mean_recurrence": 4, "recurrence_sigma": -1}},
            "recurrence_sigma must be at least 0",
        ),
        (
            {"training": {"mean_recurrence": 4, "recurrence_sampling": "beta"}},
            "recurrence_sampling must be one of",
        ),
        (
            {"training": {"mean_recurrence": 4.5, "recurrence_sampling": "fixed"}},
            "integer mean_recurrence",
        ),
        (
            {"training": {"mean_recurrence": 8, "max_recurrence": 4}},
            "max_recurrence must be at least",
        ),
    ],
)
def test_normalize_rejects_bad_recurrence_training_ranges(updates, message):
    with pytest.raises(ValueError, match=message):
        bridge.normalize_spalmer_config(_recurrent_config(**updates))


def test_normalize_rejects_unknown_recurrence_architecture_keys():
    config = _config()
    config["spalmer"] = {
        **config["spalmer"],
        "recurrence": {"enabled": True, "core_layers": 2},
    }
    with pytest.raises(ValueError, match="unsupported SPALMER recurrence keys: core_layers"):
        bridge.normalize_spalmer_config(config)


def test_planner_passes_recurrence_scale_config():
    calls = {}

    class Recurrence:
        def __init__(self, **kwargs):
            self.values = kwargs

    def plan_configuration(target, vocab, **kwargs):
        calls.update(kwargs)
        return SimpleNamespace(fingerprint="plan")

    api = SimpleNamespace(
        DirectionalScaleConfig=lambda **_kwargs: object(),
        ATXYScaleConfig=lambda **_kwargs: object(),
        RecurrenceScaleConfig=Recurrence,
        ScaleSearchSpace=lambda **_kwargs: object(),
        plan_configuration=plan_configuration,
    )

    bridge.plan_spalmer_configuration(_recurrent_config(), 4096, api=api)
    assert calls["recurrence"].values == {
        "prelude_layers": 1,
        "coda_layers": 1,
        "default_steps": 8,
        "latent_init_std": 1.0,
    }

    calls.clear()
    bridge.plan_spalmer_configuration(_config(), 4096, api=api)
    assert calls["recurrence"] is None


def test_training_config_carries_recurrence_fields():
    api = SimpleNamespace(TrainingConfig=lambda **values: SimpleNamespace(**values))

    recurrent = bridge.build_spalmer_training_config(_recurrent_config(), api=api)
    flat = bridge.build_spalmer_training_config(_config(), api=api)

    assert recurrent.mean_recurrence == 8.0
    assert recurrent.mean_backprop_depth == 4
    assert recurrent.recurrence_sigma == 0.5
    assert recurrent.recurrence_sampling == "poisson_lognormal"
    assert recurrent.max_recurrence is None
    assert flat.mean_recurrence is None


@pytest.mark.parametrize("format_version", [5, 6, 7])
def test_prepare_checkpoint_run_accepts_supported_bundle_versions(
    tmp_path, monkeypatch, format_version
):
    bundle = _parent_bundle(
        tmp_path / "parent" / "spalmer.pt",
        tokenizer_source=str(tmp_path / "gone"),
        format_version=format_version,
    )
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(bundle))
    monkeypatch.setattr("yylab.models.spalmer.relocated_spalmer_tokenizer", lambda binding: None)
    api = _checkpoint_api(
        lambda path, **kwargs: (
            _FakeModel(),
            _FakeTokenizer(),
            {"run_id": "parent", "plan": {"fingerprint": "parent-plan"}},
        )
    )

    prepared = bridge.prepare_spalmer_run(_checkpoint_config(), api=api)

    assert prepared.training.mean_recurrence is None


def test_prepare_checkpoint_run_rejects_unsupported_bundle_version(tmp_path, monkeypatch):
    bundle = _parent_bundle(
        tmp_path / "parent" / "spalmer.pt",
        tokenizer_source=str(tmp_path / "gone"),
        format_version=8,
    )
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(bundle))
    monkeypatch.setattr("yylab.models.spalmer.relocated_spalmer_tokenizer", lambda binding: None)
    api = _checkpoint_api(lambda path, **kwargs: pytest.fail("must not load"))

    with pytest.raises(ValueError, match="format-version 5, 6, or 7"):
        bridge.prepare_spalmer_run(_checkpoint_config(), api=api)


def _recurrent_parent_api(recurrence):
    model = _FakeModel()
    model.config = SimpleNamespace(recurrence=recurrence)
    return _checkpoint_api(
        lambda path, **kwargs: (
            model,
            _FakeTokenizer(),
            {"run_id": "parent", "plan": {"fingerprint": "parent-plan"}},
        )
    )


def test_prepare_checkpoint_run_checks_parent_recurrence(tmp_path, monkeypatch):
    bundle = _parent_bundle(
        tmp_path / "parent" / "spalmer.pt",
        tokenizer_source=str(tmp_path / "gone"),
        format_version=6,
        recurrence={
            "prelude_layers": 1,
            "core_layers": 2,
            "coda_layers": 1,
            "default_steps": 8,
            "latent_init_std": 1.0,
        },
    )
    monkeypatch.setattr(bridge.db, "q1", lambda sql, args=(): _parent_row(bundle))
    monkeypatch.setattr("yylab.models.spalmer.relocated_spalmer_tokenizer", lambda binding: None)
    recurrent_api = _recurrent_parent_api(SimpleNamespace(prelude_layers=1, core_layers=2))
    flat_api = _recurrent_parent_api(None)

    with pytest.raises(ValueError, match="parent checkpoint is recurrent"):
        bridge.prepare_spalmer_run(_checkpoint_config(), api=recurrent_api)
    with pytest.raises(ValueError, match="no recurrent core"):
        bridge.prepare_spalmer_run(
            _checkpoint_config(training={"mean_recurrence": 8}), api=flat_api
        )

    prepared = bridge.prepare_spalmer_run(
        _checkpoint_config(training={"mean_recurrence": 8}), api=recurrent_api
    )
    assert prepared.training.mean_recurrence == 8.0


def test_metric_payload_and_checkpoint_metadata_include_recurrence(tmp_path, monkeypatch):
    artifacts = _native_run(
        tmp_path,
        monkeypatch,
        recurrence={
            "prelude_layers": 1,
            "core_layers": 2,
            "coda_layers": 1,
            "default_steps": 8,
            "latent_init_std": 1.0,
        },
        training_updates={
            "mean_recurrence": 8.0,
            "mean_backprop_depth": 4,
            "recurrence_sigma": 0.5,
            "recurrence_sampling": "poisson_lognormal",
            "max_recurrence": None,
        },
        metric_updates={
            "recurrence_steps": 6,
            "backprop_steps": 4,
            "effective_depth": 14,
            "latent_delta_final": 0.125,
            "latent_position_correlation": 0.5,
        },
        api_updates={"FORMAT_VERSION": 6},
    )
    manifest = artifacts["manifest"]
    telemetry = artifacts["telemetry"][-1]
    expected_block = {
        "prelude_layers": 1,
        "core_layers": 2,
        "coda_layers": 1,
        "default_steps": 8,
        "latent_init_std": 1.0,
        "effective_depth": 18,
        "mean_recurrence": 8.0,
        "mean_backprop_depth": 4,
        "recurrence_sigma": 0.5,
        "recurrence_sampling": "poisson_lognormal",
        "max_recurrence": None,
    }

    assert manifest["checkpoint_format"] == "spalmer.prototype.checkpoint.v6"
    assert manifest["recurrence"] == expected_block
    assert manifest["effective_depth"] == 18
    assert artifacts["saved"][artifacts["output"] / "spalmer.pt"]["recurrence"] == expected_block
    assert telemetry["recurrence_steps"] == 6
    assert telemetry["backprop_steps"] == 4
    assert telemetry["effective_depth"] == 14
    assert telemetry["latent_delta_final"] == pytest.approx(0.125)
    assert telemetry["latent_position_correlation"] == pytest.approx(0.5)
    assert "· r=6 k=4 Δ=0.125 ρ=0.50" in telemetry["msg"]
    _, update_args = artifacts["database_calls"][-1]
    registry_metadata = json.loads(update_args[-2])
    assert registry_metadata["recurrence"] == expected_block
    assert registry_metadata["effective_depth"] == 18
    # The registry column stays physical; the walked depth lives in meta_json.
    assert update_args[5] == 4
