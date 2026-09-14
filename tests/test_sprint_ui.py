from __future__ import annotations

import json

import pytest

from yylab import db
from yylab.ui.pages import train_sprint
from yylab.ui.shell import NAV


def recipe(**overrides) -> dict:
    values = {
        "checkpoint_name": "tiny-sprint",
        "tokenizer_ref": "tok_tiny",
        "tokenizer_id": "tok_tiny",
        "model_config": {
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 2,
            "max_position_embeddings": 256,
        },
        "corpora": ["english", "code", "english"],
        "domains": ["language", "code"],
        "splits": ["train", "unspecified"],
        "seq_len": 64, "micro_batch": 2, "grad_accum": 4,
        "token_budget": 1001,
    }
    values.update(overrides)
    return train_sprint.build_sprint_run_config(values)


def test_sprint_navigation_and_route() -> None:
    assert ("Train · Sprint", "/train/sprint", "bolt") in NAV
    from nicegui import app

    assert any(route.path == "/train/sprint" for route in app.routes)


def test_sprint_recipe_retains_data_architecture_and_rounded_budget() -> None:
    cfg = recipe(corpus_mix="token_balanced", corpus_weights={"code": 2}, source_max_ratio=0.75, objective="pretrain")
    assert cfg["objective"] == "causal_lm"
    assert cfg["corpora"] == ["english", "code"]
    assert cfg["corpus_weights"] == {"code": 2}
    assert cfg["splits"] == ["train", "unspecified"]
    assert cfg["initialization"] == {"mode": "scratch", "tokenizer_ref": "tok_tiny"}
    assert cfg["model_config"]["hidden_size"] == 64
    assert cfg["model_config"]["tie_word_embeddings"] is True
    assert cfg["max_steps"] == 2
    assert cfg["tokens_per_optimizer_step"] == 512
    assert cfg["scheduled_training_tokens"] == 960
    assert cfg["token_budget_rounding"] == -41
    assert cfg["unused_token_budget"] == 41
    assert cfg["max_rows"] == 15
    assert cfg["token_cache"] is False
    assert cfg["torch_compile"] is True
    assert cfg["activation_checkpointing"] is False
    assert cfg["gradient_checkpointing"] is False
    assert cfg["parameter_dtype"] == "bfloat16"
    assert cfg["optimizer_state_offload"] == "cpu"
    assert "cpu_threads" not in cfg
    assert cfg["offload_bucket_mb"] == 32
    assert cfg["attention_backend"] == "auto"
    assert cfg["cpu_optimizer_kernel"] == "auto"
    assert cfg["eval_interval"] == 0
    assert not {"recurrence", "qat", "expert_precision"} & cfg.keys()


@pytest.mark.parametrize("changes,match", [
    ({"token_budget": 0}, "either"),
    ({"max_steps": 2}, "either"),
    ({"token_budget": -1}, "either"),
    ({"token_budget": 63}, "complete sequence"),
    ({"micro_batch": 0}, "positive"),
    ({"seq_len": 1}, "at least 2"),
    ({"seq_len": 512}, "context"),
    ({"corpora": []}, "corpus"),
    ({"tokenizer_ref": ""}, "tokenizer"),
    ({"lr": float("nan")}, "finite"),
    ({"log_interval": 0}, "positive"),
    ({"corpus_weights": {"other": 1}}, "selected corpora"),
    ({"source_max_ratio": 0.4}, "too small"),
    ({"corpus_mix": "concatenate", "corpus_weights": {"code": 2}}, "token-balanced"),
    ({"eval_interval": 5, "eval_splits": []}, "held-out"),
    ({"eval_interval": 5, "eval_splits": ["train"]}, "disjoint"),
    ({"eval_interval": 5, "splits": []}, "disjoint"),
    ({"initialization": {"mode": "checkpoint"}}, "checkpoint"),
    ({"initialization": {"mode": "resume"}}, "resume"),
    ({"parameter_dtype": "float16"}, "BF16 weights"),
    ({"parameter_dtype": "bfloat16", "optimizer_state_offload": "none"}, "BF16 weights"),
    ({"cpu_threads": 0}, "CPU threads"),
    ({"cpu_threads": 1.5}, "CPU threads"),
    ({"offload_bucket_mb": 0}, "transfer size"),
    ({"attention_backend": "unknown"}, "attention implementation"),
    ({"cpu_optimizer_kernel": "unknown"}, "CPU optimizer update"),
    ({"cpu_optimizer_kernel": "compiled", "parameter_dtype": "float32", "optimizer_state_offload": "none"}, "require BF16"),
])
def test_sprint_recipe_rejects_invalid_training_inputs(changes: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        recipe(**changes)


def test_continue_and_resume_inherit_tokenizer_and_architecture() -> None:
    continued = recipe(
        tokenizer_ref="", model_config={},
        initialization={"mode": "checkpoint", "checkpoint_ref": "saved-sprint"},
    )
    assert continued["model_config"] == {}
    assert continued["initialization"]["checkpoint_ref"] == "saved-sprint"
    resumed = recipe(
        tokenizer_ref="", model_config={}, initialization={"mode": "resume"}, resume_from="run/checkpoint",
        token_budget=0, max_steps=100,
    )
    assert resumed["resume_from"] == "run/checkpoint"
    assert resumed["max_steps"] == 100


def test_benchmark_uses_fixed_effective_batch_without_data_dependency() -> None:
    cfg = train_sprint.build_sprint_benchmark_config(
        model_config=recipe()["model_config"], seq_len=64, batch_sizes="1,2,4,2",
        compile_modes=["eager", "default"], effective_batch=4, warmup_steps=2, measure_steps=5, device="cpu",
    )
    assert cfg["batch_sizes"] == [1, 2, 4]
    assert cfg["effective_batch"] == 4
    assert cfg["parameter_dtype"] == "bfloat16"
    assert cfg["optimizer_state_offload"] == "cpu"
    assert cfg["gradient_checkpointing"] is False
    assert cfg["cpu_optimizer_kernel"] == "auto"
    assert "corpora" not in cfg and "tokenizer_ref" not in cfg
    with pytest.raises(ValueError, match="divide"):
        train_sprint.build_sprint_benchmark_config(
            model_config=cfg["model_config"], seq_len=64, batch_sizes="3",
            compile_modes=["eager"], effective_batch=4, warmup_steps=2, measure_steps=5,
        )


def test_training_and_benchmark_share_selected_execution_settings() -> None:
    training = recipe(parameter_dtype="float32", optimizer_state_offload="none", activation_checkpointing=True,
                      cpu_threads="4", offload_bucket_mb=64, attention_backend="math",
                      compile_mode="max-autotune-no-cudagraphs", cpu_optimizer_kernel="native")
    benchmark = train_sprint.build_sprint_benchmark_config(
        model_config=training["model_config"], seq_len=64, batch_sizes="1,2",
        compile_modes=["eager"], effective_batch=2, warmup_steps=1, measure_steps=2, device="cpu",
        parameter_dtype="float32", optimizer_state_offload="none", gradient_checkpointing=True,
        cpu_threads="4", offload_bucket_mb=64, attention_backend="math",
        cpu_optimizer_kernel="native",
    )
    for key in ("parameter_dtype", "optimizer_state_offload", "gradient_checkpointing",
                "cpu_threads", "offload_bucket_mb", "attention_backend", "cpu_optimizer_kernel"):
        assert benchmark[key] == training[key]
    assert training["cpu_threads"] == 4
    assert training["compile_mode"] == "max-autotune-no-cudagraphs"


def test_compiled_cpu_optimizer_is_preserved_for_bf16_offload_training_and_benchmark() -> None:
    training = recipe(cpu_optimizer_kernel="compiled")
    assert training["cpu_optimizer_kernel"] == "compiled"
    benchmark = train_sprint.build_sprint_benchmark_config(
        model_config=training["model_config"], seq_len=64, batch_sizes="1", compile_modes=["eager"],
        effective_batch=1, warmup_steps=1, measure_steps=1, cpu_optimizer_kernel="compiled",
    )
    assert benchmark["cpu_optimizer_kernel"] == "compiled"
    with pytest.raises(ValueError, match="require BF16"):
        train_sprint.build_sprint_benchmark_config(
            model_config=training["model_config"], seq_len=64, batch_sizes="1", compile_modes=["eager"],
            effective_batch=1, warmup_steps=1, measure_steps=1, cpu_optimizer_kernel="compiled",
            parameter_dtype="float32", optimizer_state_offload="none",
        )


def test_benchmark_ui_passes_optional_cpu_and_attention_sweeps() -> None:
    cfg = train_sprint.build_sprint_benchmark_config(
        model_config=recipe()["model_config"], seq_len=64, batch_sizes="1,2",
        compile_modes=["eager", "max-autotune-no-cudagraphs"], effective_batch=2, warmup_steps=1,
        measure_steps=2, cpu_threads="", cpu_thread_counts="4,8,12,20,4",
        offload_bucket_sizes_mb="16,32,64", attention_backends=["auto", "efficient", "auto"],
    )
    assert "cpu_threads" not in cfg
    assert cfg["cpu_thread_counts"] == [4, 8, 12, 20]
    assert cfg["offload_bucket_sizes_mb"] == [16, 32, 64]
    assert cfg["attention_backends"] == ["auto", "efficient"]
    assert cfg["offload_bucket_mb"] == 32
    assert cfg["attention_backend"] == "auto"


@pytest.mark.parametrize("updates", [
    {"cpu_thread_counts": "4,0"}, {"offload_bucket_sizes_mb": "32,"},
    {"attention_backends": ["unknown"]},
    {"offload_bucket_sizes_mb": "16,32", "optimizer_state_offload": "none", "parameter_dtype": "float32"},
])
def test_benchmark_ui_rejects_invalid_sweep_inputs(updates) -> None:
    with pytest.raises(ValueError):
        train_sprint.build_sprint_benchmark_config(
            model_config=recipe()["model_config"], seq_len=64, batch_sizes="1,2", compile_modes=["eager"],
            effective_batch=2, warmup_steps=1, measure_steps=2, **updates,
        )


def test_memory_preview_separates_cuda_and_host_training_state() -> None:
    assert train_sprint.sprint_memory_estimate(1_000_000_000) == {
        "gpu_bytes": 4_000_000_000, "cpu_bytes": 12_000_000_000,
    }
    assert train_sprint.sprint_memory_estimate(
        1_000_000_000, parameter_dtype="float32", optimizer_state_offload="none",
    ) == {"gpu_bytes": 16_000_000_000, "cpu_bytes": 0}


def test_planner_preview_matches_allocated_free_estimator() -> None:
    from yylab.models.sprint import SprintConfig, estimate_parameter_count

    plan = train_sprint.resolve_sprint_shape(vocab_size=512, context_length=256, target_parameters=1_000_000)
    config = SprintConfig(**plan["model_config"])
    assert estimate_parameter_count(config) == plan["parameters"]
    assert config.max_position_embeddings == 256
    assert config.vocab_size >= 512
    assert plan["shape_profile"] == "throughput"


def test_sizing_profile_separates_planner_cache_and_persists_in_recipes(monkeypatch) -> None:
    from yylab.models import sprint

    calls = []

    def planner(target, vocab_size, *, max_position_embeddings, profile, ple_type, ple_lanes,
                attention_type, kda_backend, kda_disable_recompute):
        calls.append((target, profile))
        return sprint.SprintConfig(vocab_size=vocab_size, hidden_size=64 if profile == "throughput" else 32,
                                    intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
                                    max_position_embeddings=max_position_embeddings)

    monkeypatch.setattr(sprint, "plan_sprint_model", planner)
    train_sprint._planned_config.cache_clear()
    try:
        throughput = train_sprint.resolve_sprint_shape(vocab_size=128, context_length=256,
                                                       target_parameters=50_000_000, profile="throughput")
        balanced = train_sprint.resolve_sprint_shape(vocab_size=128, context_length=256,
                                                     target_parameters=50_000_000, profile="balanced")
        cached = train_sprint.resolve_sprint_shape(vocab_size=128, context_length=256,
                                                   target_parameters=50_000_000, profile="throughput")
        assert cached == throughput
        assert calls == [(50_000_000, "throughput"), (50_000_000, "balanced")]
        assert throughput["model_config"]["hidden_size"] != balanced["model_config"]["hidden_size"]
        training = recipe(**balanced)
        assert training["shape_profile"] == "balanced"
        assert training["model_config"]["hidden_size"] == 32
        benchmark = train_sprint.build_sprint_benchmark_config(
            model_config=balanced["model_config"], shape_profile=balanced["shape_profile"], seq_len=64,
            batch_sizes="1", compile_modes=["eager"], effective_batch=1, warmup_steps=1, measure_steps=1,
        )
        assert benchmark["shape_profile"] == "balanced"
    finally:
        train_sprint._planned_config.cache_clear()


def test_manual_and_checkpoint_shapes_are_unaffected_by_sizing_profile() -> None:
    manual = recipe()["model_config"]
    throughput = train_sprint.resolve_sprint_shape(vocab_size=128, context_length=256,
                                                   model_config=manual, profile="throughput")
    balanced = train_sprint.resolve_sprint_shape(vocab_size=128, context_length=256,
                                                 model_config=manual, profile="balanced")
    assert throughput == balanced
    assert "shape_profile" not in throughput
    assert "shape_profile" not in recipe(model_config=manual, shape_profile="throughput")
    continued = recipe(tokenizer_ref="", model_config={}, shape_profile="balanced",
                       initialization={"mode": "checkpoint", "checkpoint_ref": "saved-sprint"})
    assert "shape_profile" not in continued
    assert continued["model_config"] == {}


def test_target_loss_selection_does_not_mutate_cached_plan(monkeypatch) -> None:
    from yylab.models import sprint

    calls = []

    def planner(target, vocab_size, *, max_position_embeddings, profile, ple_type, ple_lanes,
                attention_type, kda_backend, kda_disable_recompute):
        calls.append((target, vocab_size, max_position_embeddings, profile))
        return sprint.SprintConfig(vocab_size=vocab_size, hidden_size=64, intermediate_size=128,
                                    num_hidden_layers=2, num_attention_heads=2,
                                    max_position_embeddings=max_position_embeddings)

    monkeypatch.setattr(sprint, "plan_sprint_model", planner)
    train_sprint._planned_config.cache_clear()
    try:
        values = {"vocab_size": 128, "context_length": 256, "target_parameters": 1_000_000}
        original = train_sprint.resolve_sprint_shape(**values)
        cached_json = train_sprint._planned_config(1_000_000, 128, 256, "throughput", "none", 4, "sdpa", "auto", True)
        for backend in ("torch", "liger", "auto"):
            selected = train_sprint.resolve_sprint_shape(**values, loss_backend=backend)
            assert selected["model_config"]["loss_backend"] == backend
            assert selected["parameters"] == original["parameters"]
            assert train_sprint._planned_config(1_000_000, 128, 256, "throughput", "none", 4, "sdpa", "auto", True) == cached_json
        assert train_sprint.resolve_sprint_shape(**values) == original
        assert original["model_config"]["loss_backend"] == "auto"
        assert len(calls) == 1
    finally:
        train_sprint._planned_config.cache_clear()


def test_manual_loss_selection_preserves_explicit_config_when_omitted() -> None:
    manual = {**recipe()["model_config"], "loss_backend": "chunked"}
    original = dict(manual)
    values = {"vocab_size": 128, "context_length": 256, "model_config": manual}
    assert train_sprint.resolve_sprint_shape(**values)["model_config"]["loss_backend"] == "chunked"
    for backend in ("torch", "liger", "auto"):
        selected = train_sprint.resolve_sprint_shape(**values, loss_backend=backend)
        assert selected["model_config"]["loss_backend"] == backend
        assert manual == original
    assert train_sprint.resolve_sprint_shape(**values)["model_config"]["loss_backend"] == "chunked"


@pytest.mark.parametrize("target", [0, 1_000_000])
@pytest.mark.parametrize("backend", ["auto", "torch", "liger"])
def test_selected_loss_is_shared_by_training_and_benchmark(target, backend) -> None:
    shape = train_sprint.resolve_sprint_shape(
        vocab_size=128, context_length=256, target_parameters=target,
        model_config=recipe()["model_config"], loss_backend=backend,
    )
    training = recipe(**shape)
    benchmark = train_sprint.build_sprint_benchmark_config(
        model_config=shape["model_config"], seq_len=64, batch_sizes="1", compile_modes=["eager"],
        effective_batch=1, warmup_steps=1, measure_steps=1,
    )
    assert training["model_config"] == benchmark["model_config"]
    assert training["model_config"]["loss_backend"] == backend


@pytest.mark.parametrize("target", [0, 1_000_000])
def test_loss_selection_rejects_invalid_choice(target) -> None:
    with pytest.raises(ValueError, match="loss implementation"):
        train_sprint.resolve_sprint_shape(
            vocab_size=128, context_length=256, target_parameters=target,
            model_config=recipe()["model_config"], loss_backend="unknown",
        )


def test_target_profile_defaults_and_validation() -> None:
    assert recipe(target_parameters=50_000_000)["shape_profile"] == "throughput"
    with pytest.raises(ValueError, match="sizing profile"):
        recipe(target_parameters=50_000_000, shape_profile="unknown")


def test_ple_choices_separate_cached_plans_and_preserve_allocation():
    values = dict(vocab_size=128, context_length=256, target_parameters=1_000_000)
    train_sprint._planned_config.cache_clear()
    try:
        plans = {kind: train_sprint.resolve_sprint_shape(**values, ple_type=kind) for kind in ("none", "full", "qr")}
        other_lanes = train_sprint.resolve_sprint_shape(**values, ple_type="qr", ple_lanes=2)
        assert train_sprint._planned_config.cache_info().misses == 4
        for kind, plan in plans.items():
            assert train_sprint.resolve_sprint_shape(**values, ple_type=kind) == plan
            assert plan["model_config"]["ple_type"] == kind
            allocation = plan["parameter_breakdown"]
            assert allocation["total"] == plan["parameters"] <= values["target_parameters"]
            assert allocation["ple"] == allocation["ple_tables"] + allocation["ple_controls"]
            assert (allocation["ple"] > 0) == (kind != "none")
            training = recipe(**plan)
            benchmark = train_sprint.build_sprint_benchmark_config(
                model_config=plan["model_config"], seq_len=64, batch_sizes="1", compile_modes=["eager"],
                effective_batch=1, warmup_steps=1, measure_steps=1)
            assert training["model_config"] == benchmark["model_config"] == plan["model_config"]
            assert training["parameter_breakdown"] == benchmark["parameter_breakdown"] == allocation
        assert other_lanes["model_config"]["ple_lanes"] == 2
        assert other_lanes["parameter_breakdown"]["ple"] != plans["qr"]["parameter_breakdown"]["ple"]
    finally:
        train_sprint._planned_config.cache_clear()


def test_ple_recommendations_require_type_and_lanes_and_accept_legacy_none(monkeypatch):
    config = recipe()["model_config"]
    legacy = {key: value for key, value in config.items() if key not in {"ple_type", "ple_lanes"}}
    report = {"model_config": legacy, "sequence_length": 64, "device": "cpu",
              "execution": train_sprint.sprint_execution_config(),
              "recommended": {"micro_batch": 2, "grad_accum": 2, "torch_compile": False, "compile_mode": "default"}}
    monkeypatch.setattr(db, "q", lambda *_: [{"id": "legacy", "metrics_json": json.dumps(report)}])
    assert train_sprint.latest_sprint_recommendation(config, 64, device="cpu")["run_id"] == "legacy"
    for updates in ({"ple_type": "full"}, {"ple_type": "qr"}, {"ple_lanes": 2}):
        with pytest.raises(ValueError, match="No completed benchmark"):
            train_sprint.latest_sprint_recommendation({**config, **updates}, 64, device="cpu")
    report["model_config"] = {**config, "ple_type": "qr", "ple_lanes": 2}
    assert train_sprint.latest_sprint_recommendation(report["model_config"], 64, device="cpu")["run_id"] == "legacy"
    with pytest.raises(ValueError, match="No completed benchmark"):
        train_sprint.latest_sprint_recommendation({**config, "ple_type": "qr"}, 64, device="cpu")


def test_ple_queue_metadata_preserves_architecture_and_allocation(monkeypatch):
    monkeypatch.setattr(train_sprint, "launch_worker", lambda _: 123)
    shape = train_sprint.resolve_sprint_shape(vocab_size=128, context_length=256,
                                             model_config=recipe()["model_config"], ple_type="full")
    run_id = train_sprint.queue_sprint_job(recipe(**shape))
    config = json.loads(db.q1("SELECT config_json FROM runs WHERE id=?", (run_id,))["config_json"])
    metadata = json.loads(db.q1("SELECT meta_json FROM checkpoints WHERE id=?", (config["out_checkpoint_id"],))["meta_json"])
    assert config["model_config"]["ple_type"] == metadata["model_config"]["ple_type"] == "full"
    assert config["parameter_breakdown"] == metadata["parameter_breakdown"] == shape["parameter_breakdown"]


def test_queue_creates_run_and_output_atomically_and_preserves_worker_metrics(monkeypatch) -> None:
    def launch(run_id):
        db.execute("UPDATE runs SET metrics_json=? WHERE id=?", ('{"result":"already-done"}', run_id))
        return 321

    monkeypatch.setattr(train_sprint, "launch_worker", launch)
    run_id = train_sprint.queue_sprint_job(recipe())
    run = db.q1("SELECT * FROM runs WHERE id=?", (run_id,))
    cfg = json.loads(run["config_json"])
    checkpoint = db.q1("SELECT * FROM checkpoints WHERE id=?", (cfg["out_checkpoint_id"],))
    assert run["kind"] == "sprint"
    assert json.loads(run["metrics_json"]) == {"pid": 321, "result": "already-done"}
    assert checkpoint["kind"] == "pending"
    assert checkpoint["arch"] == "SprintForCausalLM"


def test_launch_failure_removes_only_pending_output(monkeypatch) -> None:
    seen = []

    def launch(run_id):
        seen.append(run_id)
        raise OSError("launch unavailable")

    monkeypatch.setattr(train_sprint, "launch_worker", launch)
    with pytest.raises(OSError, match="launch unavailable"):
        train_sprint.queue_sprint_job(recipe())
    run = db.q1("SELECT * FROM runs WHERE id=?", (seen[0],))
    assert run["status"] == "failed"
    assert "launch unavailable" in run["error"]
    cfg = json.loads(run["config_json"])
    assert db.q1("SELECT id FROM checkpoints WHERE id=?", (cfg["out_checkpoint_id"],)) is None


def test_benchmark_queue_never_creates_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(train_sprint, "launch_worker", lambda _: 123)
    before = db.q1("SELECT COUNT(*) AS n FROM checkpoints")["n"]
    run_id = train_sprint.queue_sprint_job({"model_config": recipe()["model_config"]}, benchmark=True)
    assert db.q1("SELECT COUNT(*) AS n FROM checkpoints")["n"] == before
    assert db.q1("SELECT kind FROM runs WHERE id=?", (run_id,))["kind"] == "sprint_benchmark"


def test_worker_failure_cleans_sprint_placeholder_but_keeps_benchmarks() -> None:
    from yylab.workers.train import mark_training_artifacts_failed

    checkpoint_id = db.new_id("ckpt")
    db.execute("INSERT INTO checkpoints(id,name,created_at,kind,source,ref) VALUES(?,?,?,?,?,?)",
               (checkpoint_id, "pending-sprint", db.now_iso(), "pending", "pending", ""))
    mark_training_artifacts_failed("sprint_benchmark", {"out_checkpoint_id": checkpoint_id})
    assert db.q1("SELECT id FROM checkpoints WHERE id=?", (checkpoint_id,))
    mark_training_artifacts_failed("sprint", {"out_checkpoint_id": checkpoint_id})
    assert db.q1("SELECT id FROM checkpoints WHERE id=?", (checkpoint_id,)) is None


def test_apply_benchmark_matches_shape_context_and_device(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model_config = recipe()["model_config"]
    recommended = {"micro_batch": 4, "grad_accum": 2, "torch_compile": False, "compile_mode": "default"}
    report = {"model_config": model_config, "sequence_length": 64, "device": "cpu",
              "recommended": recommended, "best_supervised_tokens_per_second": 1234,
              "execution": train_sprint.sprint_execution_config()}
    other = {**report, "model_config": {**model_config, "hidden_size": 128}}
    # A newer benchmark with different shape must not replace the applicable result.
    monkeypatch.setattr(db, "q", lambda *_: [
        {"id": "newer-other-model", "metrics_json": json.dumps(other)},
        {"id": "matching", "metrics_json": json.dumps(report)},
    ])
    match = train_sprint.latest_sprint_recommendation(model_config, 64, device="cpu")
    assert match["run_id"] == "matching"
    assert match["recommended"] == {**recommended, "offload_bucket_mb": 32, "attention_backend": "auto",
                                     "cpu_optimizer_kernel": "native"}
    assert match["tokens_per_second"] == 1234
    with pytest.raises(ValueError, match="No completed benchmark"):
        train_sprint.latest_sprint_recommendation(model_config, 128)
    with pytest.raises(ValueError, match="No completed benchmark"):
        train_sprint.latest_sprint_recommendation(model_config, 64, device="cuda")


def test_auto_recommendation_requires_current_cuda_device_and_hardware(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA Test GPU")
    model_config = recipe()["model_config"]
    report = {"model_config": model_config, "sequence_length": 64, "device": "cpu", "hardware": "CPU",
              "execution": train_sprint.sprint_execution_config(),
              "recommended": {"micro_batch": 4, "grad_accum": 2, "torch_compile": False, "compile_mode": "default"}}
    monkeypatch.setattr(db, "q", lambda *_: [{"id": "candidate", "metrics_json": json.dumps(report)}])
    with pytest.raises(ValueError, match="No completed benchmark"):
        train_sprint.latest_sprint_recommendation(model_config, 64, device="auto")
    report.update(device="cuda", hardware="Different GPU")
    with pytest.raises(ValueError, match="No completed benchmark"):
        train_sprint.latest_sprint_recommendation(model_config, 64, device="auto")
    report.update(device="cuda:1", hardware="NVIDIA Test GPU")
    with pytest.raises(ValueError, match="No completed benchmark"):
        train_sprint.latest_sprint_recommendation(model_config, 64, device="auto")
    report.update(device="cuda:0", hardware=" nvidia   TEST GPU ")
    assert train_sprint.latest_sprint_recommendation(model_config, 64, device="auto")["run_id"] == "candidate"


@pytest.mark.parametrize("execution", [
    {},
    {"parameter_dtype": "float32", "optimizer_state_offload": "none", "gradient_checkpointing": False},
    {"parameter_dtype": "bfloat16", "optimizer_state_offload": "cpu", "gradient_checkpointing": True},
])
def test_recommendation_rejects_incompatible_or_unknown_execution(monkeypatch, execution) -> None:
    model_config = recipe()["model_config"]
    recommended = {"micro_batch": 2, "grad_accum": 2, "torch_compile": False, "compile_mode": "default"}
    report = {"model_config": model_config, "sequence_length": 64, "device": "cpu",
              "execution": execution, "recommended": recommended}
    monkeypatch.setattr(db, "q", lambda *_: [{"id": "candidate", "metrics_json": json.dumps(report)}])
    with pytest.raises(ValueError, match="optimizer/memory settings"):
        train_sprint.latest_sprint_recommendation(model_config, 64, device="cpu")
    report["execution"] = train_sprint.sprint_execution_config()
    assert train_sprint.latest_sprint_recommendation(model_config, 64, device="cpu")["run_id"] == "candidate"
    report["recommended"] = {**recommended, "parameter_dtype": "float32"}
    with pytest.raises(ValueError, match="optimizer/memory settings"):
        train_sprint.latest_sprint_recommendation(model_config, 64, device="cpu")


def test_recommendation_preserves_measured_cpu_transfer_and_attention_settings(monkeypatch) -> None:
    model_config = recipe()["model_config"]
    recommended = {"micro_batch": 2, "grad_accum": 2, "torch_compile": True,
                   "compile_mode": "max-autotune-no-cudagraphs", "cpu_threads": 8,
                   "offload_bucket_mb": 64, "attention_backend": "efficient", "cpu_optimizer_kernel": "compiled"}
    report = {"model_config": model_config, "sequence_length": 64, "device": "cpu",
              "execution": train_sprint.sprint_execution_config(), "recommended": recommended}
    monkeypatch.setattr(db, "q", lambda *_: [{"id": "candidate", "metrics_json": json.dumps(report)}])
    match = train_sprint.latest_sprint_recommendation(model_config, 64, device="cpu")
    assert match["recommended"] == recommended
    replayed = recipe(**match["recommended"])
    for key, value in recommended.items():
        assert replayed[key] == value


def test_page_constructs_with_empty_registry(monkeypatch) -> None:
    from nicegui import Client
    from nicegui.page import page

    monkeypatch.setattr(train_sprint, "list_tokenizers", lambda **_: [])
    monkeypatch.setattr(train_sprint, "sprint_checkpoint_rows", lambda: [])
    monkeypatch.setattr(train_sprint, "corpus_options", lambda: {})
    with Client(page("/__sprint_test")) as client:
        train_sprint.train_sprint_page()
        texts = [getattr(element, "text", "") for element in client.elements.values()]
        assert "Start Sprint training" in texts
        assert "Benchmark training throughput" in texts
        assert "Apply latest matching benchmark" in texts
        loss_select = next(element for element in client.elements.values()
                           if element._props.get("label") == "Loss implementation")
        assert loss_select.value == "auto"
        assert loss_select.options == train_sprint.LOSS_OPTIONS
        assert set(loss_select.options) == {"auto", "torch", "liger"}
        ple_select = next(element for element in client.elements.values()
                          if element._props.get("label") == "Per-layer embeddings (PLE)")
        assert ple_select.value == "none"
        assert ple_select.options == {"none": "No PLE", "full": "Regular PLE", "qr": "QR PLE"}
