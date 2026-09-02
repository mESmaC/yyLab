from __future__ import annotations

import pytest

from yylab.ui.pages import benchmarks
from yylab.ui.shell import NAV


def _checkpoint() -> dict:
    return {
        "id": "ckpt-1",
        "ref": "local-model",
        "quant_locked": "",
        "params_b": 0.6,
    }


def _adapter() -> dict:
    return {
        "id": "adapter-1",
        "path": "adapter-path",
        "arch": "mole",
        "quant": "bf16",
        "num_experts": 8,
    }


def test_quality_job_config_uses_non_training_adapter_identifier():
    config = benchmarks._quality_job_config(
        _checkpoint(),
        _adapter(),
        quant="bf16",
        policy="expert:3",
        profile="smoke",
        repetitions=2,
        label="forced expert",
        categories=["embedded"],
        limit=4,
        baseline_run_id="run-base",
        max_new_tokens=32,
        enable_thinking=True,
        max_thinking_tokens=2048,
        context_mode="yarn_4x",
        continue_on_error=False,
    )

    assert "adapter_id" not in config
    assert config == {
        "checkpoint_id": "ckpt-1",
        "checkpoint_ref": "local-model",
        "model_family": "hf",
        "quant": "bf16",
        "adapter_benchmark_id": "adapter-1",
        "adapter_path": "adapter-path",
        "adapter_arch": "mole",
        "num_experts": 8,
        "policy": "expert:3",
        "profile": "smoke",
        "repetitions": 2,
        "label": "forced expert",
        "categories": ["embedded"],
        "limit": 4,
        "baseline_run_id": "run-base",
        "max_new_tokens": 32,
        "prompt_mode": "chat",
        "enable_thinking": True,
        "max_thinking_tokens": 2048,
        "context_mode": "yarn_4x",
        "rope_scaling": "yarn",
        "rope_factor": 4.0,
        "max_context_tokens": 131072,
        "continue_on_error": False,
    }


def test_quality_job_config_disables_thinking_by_default():
    config = benchmarks._quality_job_config(
        _checkpoint(),
        None,
        quant="bf16",
        policy="base",
        profile="smoke",
        repetitions=1,
        label="frozen-compatible",
    )

    assert config["enable_thinking"] is False
    assert config["prompt_mode"] == "chat"
    assert config["max_thinking_tokens"] == 1024
    assert config["context_mode"] == "native"
    assert config["rope_scaling"] == "none"
    assert config["max_context_tokens"] is None


def test_quality_job_config_tags_spalmer_family_and_native_precision():
    checkpoint = {
        **_checkpoint(),
        "ref": "spalmer.pt",
        "meta_json": '{"model_family":"spalmer"}',
    }

    config = benchmarks._quality_job_config(
        checkpoint,
        None,
        quant=benchmarks._resolve_quant(checkpoint, "auto"),
        policy="base",
        profile="smoke",
        repetitions=1,
        label="spalmer",
    )

    assert config["model_family"] == "spalmer"
    assert config["quant"] == "checkpoint"
    assert config["spalmer_expert_offload"] is True
    assert config["spalmer_expert_paging"] is True
    assert config["spalmer_expert_cache_size"] is None
    assert "paged full-pool expert offload (native cap)" in benchmarks._quality_contract_text(
        config
    )

    bounded = benchmarks._quality_job_config(
        checkpoint,
        None,
        quant="checkpoint",
        policy="base",
        profile="smoke",
        repetitions=1,
        label="spalmer-bounded",
        spalmer_expert_cache_size=8,
    )
    assert bounded["spalmer_expert_cache_size"] == 8
    assert "paged full-pool expert offload (8 cap)" in benchmarks._quality_contract_text(
        bounded
    )

    legacy = benchmarks._quality_job_config(
        checkpoint,
        None,
        quant="checkpoint",
        policy="dynamic",
        profile="smoke",
        repetitions=1,
        label="spalmer-legacy",
    )
    assert legacy["spalmer_expert_paging"] is False
    assert "legacy request-global (experimental)" in benchmarks._quality_contract_text(legacy)


def test_quality_job_config_supports_raw_completion_without_thinking():
    config = benchmarks._quality_job_config(
        _checkpoint(),
        None,
        quant="bf16",
        policy="base",
        profile="smoke",
        repetitions=1,
        label="raw-base",
        prompt_mode="raw",
    )

    assert config["prompt_mode"] == "raw"
    with pytest.raises(ValueError, match="thinking requires chat"):
        benchmarks._quality_job_config(
            _checkpoint(),
            None,
            quant="bf16",
            policy="base",
            profile="smoke",
            repetitions=1,
            label="invalid",
            prompt_mode="raw",
            enable_thinking=True,
        )


def test_quality_contract_labels_distinguish_legacy_shared_thinking_cap():
    assert (
        benchmarks._quality_contract_text({"max_new_tokens": 256, "enable_thinking": True})
        == "chat · answer 256 · legacy shared-cap · native"
    )
    assert (
        benchmarks._quality_contract_text(
            {
                "max_new_tokens": 256,
                "enable_thinking": True,
                "max_thinking_tokens": 2048,
                "context_mode": "yarn_4x",
            }
        )
        == "chat · answer 256 · think 2048 · yarn 4x"
    )


def test_training_performance_job_config_matches_worker_contract():
    config = benchmarks._train_perf_job_config(
        _checkpoint(),
        quant="nf4",
        num_experts=16,
        rank=4,
        top_k=2,
        seq_len=1024,
        micro_batch=2,
        grad_accum=8,
        warmup_steps=0,
        measured_steps=7,
        optimizer_state="cpu_paged",
        lr=1e-4,
        target_modules=["q_proj", "v_proj"],
        grad_ckpt=True,
        liger_fused_ce=False,
        cudnn_sdpa=True,
        label="paged rack",
    )

    assert config["warmup_steps"] == 0
    assert config["measured_steps"] == 7
    assert config["optimizer_state"] == "cpu_paged"
    assert config["target_modules"] == ["q_proj", "v_proj"]
    assert config["num_experts"] * config["rank"] == 64


def test_summary_helpers_fall_back_to_molor_score_for_legacy_shape():
    summary = {
        "primary_policy": "dynamic",
        "policies": {
            "dynamic": {
                "quality_index": 735.25,
                "coverage": {"scored": 30, "expected": 30},
                "performance": {"tokens_per_second_median": 41.75},
                "categories": {
                    "embedded": {
                        "label": "Embedded + CAN",
                        "score_percent": 80.0,
                        "passed": 8,
                        "scored": 10,
                    }
                },
            }
        },
    }

    assert benchmarks._policy_rows(summary) == [
        {
            "policy": "dynamic",
            "semantic_quality": 735.2,
            "canonical_quality": 735.2,
            "coverage": "30/30",
            "speed": 41.8,
        }
    ]
    assert benchmarks._category_rows(summary) == [
        {
            "category": "Embedded + CAN",
            "semantic_score": 80.0,
            "canonical_score": 80.0,
            "semantic_passed": 8,
            "canonical_passed": 8,
            "scored": 10,
        }
    ]


def test_summary_helpers_lead_with_semantic_score_and_keep_molor_score():
    summary = {
        "primary_policy": "dynamic",
        "policies": {
            "dynamic": {
                "quality_index": 425.67,
                "semantic_quality_index": 501.92,
                "coverage": {"scored": 64, "expected": 64},
                "performance": {"tokens_per_second_median": 15.21},
                "categories": {
                    "finance": {
                        "label": "Finance",
                        "score_percent": 40.0,
                        "semantic_score_percent": 60.0,
                        "passed": 2,
                        "semantic_passed": 3,
                        "scored": 5,
                    }
                },
            }
        },
    }

    assert benchmarks._policy_rows(summary) == [
        {
            "policy": "dynamic",
            "semantic_quality": 501.9,
            "canonical_quality": 425.7,
            "coverage": "64/64",
            "speed": 15.2,
        }
    ]
    assert benchmarks._category_rows(summary) == [
        {
            "category": "Finance",
            "semantic_score": 60.0,
            "canonical_score": 40.0,
            "semantic_passed": 3,
            "canonical_passed": 2,
            "scored": 5,
        }
    ]


def test_summary_for_prefers_derived_semantic_summary(monkeypatch, tmp_path):
    run_path = tmp_path / "run-quality"
    run_path.mkdir()
    (run_path / "summary.json").write_text('{"quality_index": 425.67}', encoding="utf-8")
    (run_path / "semantic-summary.json").write_text(
        '{"quality_index": 425.67, "semantic_quality_index": 501.92}',
        encoding="utf-8",
    )
    monkeypatch.setattr(benchmarks, "run_dir", lambda _run_id: run_path)

    summary = benchmarks._summary_for({"id": "run-quality", "metrics_json": '{"quality_index": 400.0}'})

    assert summary == {"quality_index": 425.67, "semantic_quality_index": 501.92}


def test_quality_history_uses_semantic_summary(monkeypatch):
    monkeypatch.setattr(
        benchmarks,
        "_benchmark_runs",
        lambda: [
            {
                "id": "run-quality",
                "kind": benchmarks.QUALITY_KIND,
                "status": "done",
                "config_json": '{"label": "YIN-v1", "policy": "dynamic"}',
                "metrics_json": '{"quality_index": 425.67}',
                "created_at": "2026-08-29T20:00:00",
            }
        ],
    )
    monkeypatch.setattr(
        benchmarks,
        "_summary_for",
        lambda _run: {
            "primary_policy": "dynamic",
            "quality_index": 425.67,
            "semantic_quality_index": 501.92,
        },
    )

    assert benchmarks._history_rows()[0]["result"] == "501.9 semantic · 425.7 Molor"


def test_launch_refuses_while_any_run_is_active(monkeypatch):
    monkeypatch.setattr(
        benchmarks.db,
        "q1",
        lambda *_args, **_kwargs: {
            "id": "run-active",
            "kind": "mole_stage1",
            "status": "running",
        },
    )
    monkeypatch.setattr(
        benchmarks.db,
        "new_id",
        lambda *_args, **_kwargs: pytest.fail("must not create a run while GPU is busy"),
    )

    with pytest.raises(benchmarks.ActiveRunError, match="run-active"):
        benchmarks._launch_benchmark(benchmarks.QUALITY_KIND, {"label": "blocked"})


def test_launch_refuses_while_chat_owns_the_model(monkeypatch):
    monkeypatch.setattr(benchmarks.ENGINE, "is_loaded", lambda: True)
    monkeypatch.setattr(
        benchmarks.db,
        "q1",
        lambda *_args, **_kwargs: pytest.fail("run query must follow the chat GPU guard"),
    )

    with pytest.raises(benchmarks.ActiveRunError, match="Unload it from Chat"):
        benchmarks._launch_benchmark(benchmarks.TRAIN_PERF_KIND, {"label": "blocked"})


def test_benchmarks_page_is_registered_and_bundled_suite_loads():
    assert any(path == "/benchmarks" for _label, path, _icon in NAV)
    suite = benchmarks._load_bundled_suite()
    assert suite.suite_id == "molor-core-v1"
    assert len(suite.profiles["smoke"]) == 10
    assert len(suite.profiles["quick"]) == 30
    assert len(suite.profiles["full"]) == 64
