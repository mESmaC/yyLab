import json

import pytest
import torch

from yylab.models.sprint import SprintConfig
from yylab.train import benchmark_sprint
from yylab.train.benchmark_sprint import run_sprint_benchmark


def recipe():
    return {
        "model_config": SprintConfig(vocab_size=31, hidden_size=16, intermediate_size=32,
                                     num_hidden_layers=1, num_attention_heads=1,
                                     max_position_embeddings=32).to_dict(),
        "seq_len": 8, "effective_batch": 2, "batch_sizes": [1, 2],
        "compile_modes": ["eager"], "warmup_steps": 1, "measure_steps": 2,
        "device": "cpu",
    }


def test_complete_step_benchmark_preserves_effective_batch_and_writes_receipt(tmp_path):
    result = run_sprint_benchmark(recipe(), tmp_path)
    assert len(result["cases"]) == 2
    for row in result["cases"]:
        assert row["status"] == "ok"
        assert row["micro_batch"] * row["grad_accum"] == 2
        assert row["tokens"] == 2 * 2 * 8
        assert row["supervised_tokens"] == 2 * 2 * 7
        assert row["supervised_tokens_per_second"] > 0
        assert row["warmup_and_compile_seconds"] > 0
        assert row["cpu_threads"] == torch.get_num_threads()
        assert row["offload_bucket_mb"] == 32
        assert row["offload_bucket_bytes"] == 32 * 2**20
        assert row["attention_backend"] == "auto"
    # Same initial weights, input order, effective gradient and Adam schedule.
    assert result["cases"][0]["last_loss"] == pytest.approx(result["cases"][1]["last_loss"], abs=1e-5)
    saved = json.loads((tmp_path / "benchmark.json").read_text("utf-8"))
    assert saved["recommended"] == result["recommended"]
    assert saved["measurement"].startswith("Synthetic")
    assert saved["recommended"]["cpu_threads"] == torch.get_num_threads()
    assert saved["recommended"]["offload_bucket_mb"] == 32
    assert saved["recommended"]["attention_backend"] == "auto"
    assert saved["recommended"]["cpu_optimizer_kernel"] == "native"
    assert saved["execution"]["attention_backend"] == "auto"


@pytest.mark.parametrize("updates", [
    {"batch_sizes": [3]}, {"warmup_steps": 0}, {"measure_steps": 0},
    {"seq_len": 1}, {"compile_modes": ["unknown"]}, {"effective_batch": 0},
    {"cpu_thread_counts": []}, {"cpu_thread_counts": [0]}, {"cpu_thread_counts": [True]},
    {"cpu_thread_counts": "4,8"}, {"cpu_threads": -1}, {"cpu_threads": None},
    {"offload_bucket_sizes_mb": []}, {"offload_bucket_sizes_mb": [None]},
    {"offload_bucket_sizes_mb": [16, 32], "optimizer_state_offload": "none"},
    {"attention_backends": []}, {"attention_backends": ["unknown"]}, {"attention_backends": "auto,math"},
])
def test_invalid_benchmark_fails_before_model_allocation(tmp_path, updates):
    with pytest.raises(ValueError):
        run_sprint_benchmark({**recipe(), **updates}, tmp_path)


def _stub_cpu_threads(monkeypatch, original=6):
    state = {"current": original, "changes": []}

    def set_threads(value):
        state["current"] = value
        state["changes"].append(value)

    monkeypatch.setattr(torch, "get_num_threads", lambda: state["current"])
    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    return state


def _fake_success(cfg, batch, mode, inputs):
    rate = cfg["cpu_threads"] * 100 + cfg["offload_bucket_mb"] + batch
    return {"status": "ok", "micro_batch": batch, "grad_accum": len(inputs) // batch,
            "torch_compile": mode != "eager", "compile_mode": "default" if mode == "eager" else mode,
            "parameter_dtype": cfg.get("parameter_dtype", "float32"), "optimizer_state_offload": "cpu",
            "gradient_checkpointing": False, "cpu_threads": torch.get_num_threads(),
            "offload_bucket_mb": cfg["offload_bucket_mb"], "attention_backend": cfg["attention_backend"],
            "cpu_optimizer_kernel": "native",
            "supervised_tokens_per_second": rate,
            "projected_seconds_per_billion_targets": 1e9 / rate}


def test_sweeps_cover_cartesian_cases_with_same_model_inputs_and_effective_batch(tmp_path, monkeypatch):
    state = _stub_cpu_threads(monkeypatch)
    seen = []

    def measure(config, cfg, batch, mode, device, inputs):
        seen.append((id(config), id(inputs), cfg["cpu_threads"], cfg["offload_bucket_mb"],
                     cfg["attention_backend"], batch, mode))
        assert torch.get_num_threads() == cfg["cpu_threads"]
        return _fake_success(cfg, batch, mode, inputs)

    monkeypatch.setattr(benchmark_sprint, "_measure_case", measure)
    result = run_sprint_benchmark({**recipe(), "cpu_thread_counts": [4, 8, 4],
                                   "offload_bucket_sizes_mb": [16, 32],
                                   "compile_modes": ["eager", "default"],
                                   "attention_backends": ["auto", "math", "auto"]}, tmp_path)
    assert len(result["cases"]) == 32
    assert len({row[0] for row in seen}) == len({row[1] for row in seen}) == 1
    assert len({row[2:] for row in seen}) == 32
    assert all(row["micro_batch"] * row["grad_accum"] == 2 for row in result["cases"])
    assert result["recommended"]["cpu_threads"] == 8
    assert result["recommended"]["offload_bucket_mb"] == 32
    assert result["sweeps"] == {"cpu_thread_counts": [4, 8], "offload_bucket_sizes_mb": [16, 32],
                                "attention_backends": ["auto", "math"]}
    assert "attention_backend" not in result["execution"]
    assert state["current"] == 6
    assert state["changes"] == [4] * 16 + [8] * 16 + [6]
    assert json.loads((tmp_path / "summary.json").read_text())["recommended"] == result["recommended"]


def test_single_cpu_thread_and_bucket_settings_are_recorded_without_expanding_default_cases(tmp_path, monkeypatch):
    state = _stub_cpu_threads(monkeypatch)
    monkeypatch.setattr(benchmark_sprint, "_measure_case", lambda config, cfg, batch, mode, device, inputs:
                        _fake_success(cfg, batch, mode, inputs))
    result = run_sprint_benchmark({**recipe(), "cpu_threads": 4, "offload_bucket_mb": 64,
                                  "attention_backend": "efficient"}, tmp_path)
    assert len(result["cases"]) == 2
    assert result["sweeps"] == {"cpu_thread_counts": [4], "offload_bucket_sizes_mb": [64],
                                "attention_backends": ["efficient"]}
    assert result["recommended"]["cpu_threads"] == 4
    assert result["recommended"]["offload_bucket_mb"] == 64
    assert result["recommended"]["attention_backend"] == "efficient"
    assert result["execution"]["attention_backend"] == "efficient"
    assert state["current"] == 6


@pytest.mark.parametrize("raises", [False, True])
def test_benchmark_restores_thread_count_after_failed_candidates_or_unexpected_exception(tmp_path, monkeypatch, raises):
    state = _stub_cpu_threads(monkeypatch)

    def fail(*_):
        if raises:
            raise RuntimeError("unexpected failure")
        return {"status": "error", "error": "candidate failed"}

    monkeypatch.setattr(benchmark_sprint, "_measure_case", fail)
    with pytest.raises(RuntimeError, match="unexpected failure" if raises else "All Sprint benchmark candidates failed"):
        run_sprint_benchmark({**recipe(), "cpu_thread_counts": [4, 8],
                              "offload_bucket_sizes_mb": [16, 32]}, tmp_path)
    assert state["current"] == 6
    assert state["changes"][-1] == 6
