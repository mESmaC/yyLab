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


class _FakeTrainer:
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
                },
                "sparse": None,
            },
        )

    def run(self):
        metric = SimpleNamespace(
            step=1,
            tokens_seen=8,
            objective=2.5,
            model_loss=2.4,
            auxiliary_loss=0.1,
            surprise_calibration_loss=0.05,
            predictive_entropy=1.7,
            learning_rate=3e-4,
            gradient_norm=0.8,
            elapsed_seconds=0.1,
            promoted_experts=(3,),
            average_surprise=2.3,
        )
        self.progress.completed_steps = 1
        self.progress.tokens_seen = 8
        self.on_checkpoint(self, metric)
        yield metric


def test_run_writes_native_artifact_metadata_without_real_training(tmp_path, monkeypatch):
    output = tmp_path / "checkpoint"
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
        build=lambda: _FakeModel(),
        model=_ConfigObject(vocab_size=4096, d_model=64, n_layers=4),
        kda=_ConfigObject(backend="reference"),
        mla=_ConfigObject(num_heads=4),
        experts=experts,
        directional=None,
        atxy=None,
    )
    prepared = bridge.PreparedSpalmerRun(
        _config(), _FakeTokenizer(), _FakePlan(), bundle, training
    )
    saved = []

    def save_checkpoint(path, *_args, **_kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mock-spalmer-checkpoint")
        saved.append(path)
        return path

    api = SimpleNamespace(
        CausalBatch=_CausalBatch,
        ExperimentTrainer=_FakeTrainer,
        assert_accounting_matches_plan=lambda *_args: None,
        save_checkpoint=save_checkpoint,
        seed_everything=lambda *_args, **_kwargs: None,
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

    assert output / "spalmer.pt" in saved
    assert output / "steps" / "step-00000001" / "spalmer.pt" in saved
    manifest = json.loads((output / "spalmer_manifest.json").read_text("utf-8"))
    recipe = json.loads((output / "yylab_recipe.json").read_text("utf-8"))
    token_view = json.loads((output / "tokenized_view_manifest.json").read_text("utf-8"))
    assert manifest["model_family"] == "spalmer"
    assert manifest["checkpoint_file"] == "spalmer.pt"
    assert manifest["plan_fingerprint"] == "plan-fingerprint"
    assert recipe["model_family"] == "spalmer"
    assert token_view["input_width"] == 9
    assert summary["model_family"] == "spalmer"
    _, update_args = database_calls[-1]
    metadata = json.loads(update_args[-2])
    assert update_args[0].endswith("spalmer.pt")
    assert metadata["model_family"] == "spalmer"
    assert metadata["precision"] == {
        "expert_weights": "mxfp4",
        "expert_activations": "mxfp8",
        "master_weights": "bfloat16",
        "expert_qat_backend": "reference",
        "optimizer_moments": "float32_cpu_offload",
    }
    assert manifest["optimizer_state_placement"]["dense"]["devices"] == ["cpu"]
