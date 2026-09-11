from __future__ import annotations

import json
import math
from datetime import datetime, timedelta

import pytest
import torch

from yylab.models.sprint import SprintConfig
from yylab.train import benchmark_sprint_precision as benchmark


def recipe(**updates):
    result = {
        "model_config": SprintConfig(vocab_size=31, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=1,
                                      max_position_embeddings=32).to_dict(),
        "seq_len": 8, "effective_batch": 2, "batch_sizes": [1, 2],
        "precisions": ["bf16", "bitnet_b158"], "compile_modes": ["eager"],
        "warmup_steps": 1, "measure_steps": 2, "device": "cpu", "cpu_threads": 1,
        "cpu_optimizer_kernel": "native", "offload_bucket_mb": 1,
    }
    result.update(updates)
    return result


def test_real_cpu_bf16_and_bitnet_comparison_verifies_updates_state_and_metadata(tmp_path, monkeypatch):
    apply = benchmark.apply_sprint_precision
    initial_decoder_weights = []

    def checked_adapter(model, precision, device):
        assert device.type == "cpu"
        assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
        initial_decoder_weights.append(next(parameter.detach().clone() for name, parameter in model.named_parameters()
                                            if name.startswith("layers.") and parameter.ndim == 2))
        return apply(model, precision, device)

    monkeypatch.setattr(benchmark, "apply_sprint_precision", checked_adapter)
    original_threads = torch.get_num_threads()
    result = benchmark.run_sprint_precision_benchmark(recipe(), tmp_path)
    assert result["status"] == "ok"
    assert result["successful_cases"] == 4 and result["failed_cases"] == 0
    assert torch.get_num_threads() == original_threads
    assert all(torch.equal(initial_decoder_weights[0], weight) for weight in initial_decoder_weights)
    for row in result["cases"]:
        assert row["status"] == "ok"
        assert row["micro_batch"] * row["grad_accum"] == 2
        assert row["tokens"] == 2 * 2 * 8
        assert row["supervised_tokens"] == 2 * 2 * 7
        assert row["supervised_tokens_per_second"] > 0
        assert row["initialization_seconds"] > 0 and row["warmup_and_compile_seconds"] > 0
        assert math.isfinite(row["last_loss"]) and math.isfinite(row["final_grad_norm"])
        assert row["parameter_update"]["verified"] is True
        assert row["parameter_update"]["example"].startswith("layers.")
        assert row["cpu_optimizer_state_bytes"] == row["parameters"] * 12
        assert row["optimizer_state_bytes"] == row["cpu_optimizer_state_bytes"]
        assert row["offload_buffer_bytes"] > 0
        assert row["memory"]["parameters"]["bytes"] == row["parameters"] * 2
        assert row["runtime"]["parameter_dtype"] == "bfloat16"
        assert row["precision_metadata"]["precision"] == row["precision"]
        if row["precision"] == "bf16":
            assert row["parameters"] == result["planned_parameters"]
        else:
            assert row["parameters"] != row["parameters_before_conversion"]
            assert row["precision_metadata"]["native_low_bit_gemm"] is False
    saved = json.loads((tmp_path / "benchmark.json").read_text("utf-8"))
    assert saved == json.loads((tmp_path / "summary.json").read_text("utf-8"))
    assert saved["status"] == "ok" and "BF16-rounded" in saved["initialization"]
    assert datetime.fromisoformat(saved["created_at_utc"]).utcoffset() == timedelta(0)
    assert saved["runtime_versions"]["torch"] == str(torch.__version__)
    assert set(saved["runtime_versions"]) == {"torch", "torchao", "transformers", "triton"}
    assert saved["source_sha256"] == benchmark._source_sha256()
    assert len(saved["source_sha256"]["yylab/train/benchmark_sprint_precision.py"]) == 64
    assert saved["recipe"]["model_config"] == saved["model_config"]
    assert saved["recipe"]["precisions"] == ["bf16", "bitnet_b158"]
    assert saved["recipe"]["batch_sizes"] == [1, 2]
    assert saved["recipe"]["device"] == "cpu"
    assert saved["recipe"]["parameter_dtype"] == "bfloat16"
    assert saved["recipe"]["optimizer_state_offload"] == "cpu"
    assert saved["recipe"]["measure_steps"] == 2
    assert saved["recipe"]["lr"] == 3e-4
    assert saved["recipe"]["betas"] == [0.9, 0.95]
    assert saved["optimizer_schedule"] == {"max_steps": 1_000_000, "warmup_steps": 0}


def test_source_provenance_hashes_existing_files_only(monkeypatch):
    import hashlib
    from pathlib import Path

    relative = "yylab/train/benchmark_sprint_precision.py"
    monkeypatch.setattr(benchmark, "PROVENANCE_SOURCE_PATHS", (relative, "missing-precision-source.py"))
    assert benchmark._source_sha256() == {relative: hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest()}


def test_runtime_versions_allow_missing_packages_and_windows_triton(monkeypatch):
    def version(distribution):
        if distribution == "triton-windows":
            return "3.6.0"
        raise benchmark.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(benchmark.metadata, "version", version)
    assert benchmark._runtime_versions() == {
        "torch": str(torch.__version__), "torchao": None, "transformers": None, "triton": "3.6.0",
    }


def test_native_precision_failure_remains_visible_and_does_not_fall_back(tmp_path, monkeypatch):
    apply = benchmark.apply_sprint_precision
    seen = []

    def selectively_available(model, precision, device):
        seen.append(precision)
        if precision == "nvfp4_bf16":
            raise RuntimeError("native kernel unavailable")
        return apply(model, precision, device)

    monkeypatch.setattr(benchmark, "apply_sprint_precision", selectively_available)
    result = benchmark.run_sprint_precision_benchmark(
        recipe(precisions=["bf16", "nvfp4_bf16"], batch_sizes=[2]), tmp_path,
    )
    assert seen == ["bf16", "nvfp4_bf16"]
    assert result["status"] == "partial"
    assert result["successful_cases"] == result["failed_cases"] == 1
    failure = result["cases"][1]
    assert failure["precision"] == "nvfp4_bf16"
    assert failure["status"] == "error" and "native kernel unavailable" in failure["error"]
    assert "supervised_tokens_per_second" not in failure
    assert result["fastest_measured"]["precision"] == "bf16"
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "partial"


def test_all_precision_failures_write_failed_report_and_raise(tmp_path, monkeypatch):
    def unavailable(*_):
        raise RuntimeError("no native provider")

    monkeypatch.setattr(benchmark, "apply_sprint_precision", unavailable)
    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="All Sprint precision candidates failed"):
        benchmark.run_sprint_precision_benchmark(recipe(precisions=["mxfp8"], batch_sizes=[2]), tmp_path)
    assert torch.get_num_threads() == threads
    saved = json.loads((tmp_path / "benchmark.json").read_text())
    assert saved["status"] == "failed" and saved["failed_cases"] == 1
    assert "no native provider" in saved["cases"][0]["error"]
    assert "fastest_measured" not in saved


@pytest.mark.parametrize("updates", [
    {"precisions": ["unknown"]}, {"precisions": []}, {"precisions": "bf16"},
    {"batch_sizes": [3]}, {"batch_sizes": []}, {"effective_batch": 0},
    {"warmup_steps": 0}, {"measure_steps": 0}, {"seq_len": 1}, {"seq_len": 64},
    {"cpu_threads": 0}, {"offload_bucket_mb": 0}, {"compile_modes": ["unknown"]},
    {"optimizer_state_offload": "none"}, {"parameter_dtype": "float32"},
])
def test_invalid_comparison_fails_before_model_allocation(tmp_path, monkeypatch, updates):
    monkeypatch.setattr(benchmark, "SprintForCausalLM", lambda *_: pytest.fail("model allocated for invalid recipe"))
    with pytest.raises(ValueError):
        benchmark.run_sprint_precision_benchmark(recipe(**updates), tmp_path)


@pytest.mark.parametrize("failure", ["no_update", "nonfinite_gradient", "wrong_precision"])
def test_validation_failures_cannot_be_reported_as_success(tmp_path, monkeypatch, failure):
    if failure == "no_update":
        monkeypatch.setattr(benchmark, "_decoder_samples", lambda _: {"layers.0.fake.weight": torch.ones(2)})
    elif failure == "nonfinite_gradient":
        monkeypatch.setattr(benchmark, "_final_gradient_norm", lambda _: float("nan"))
    else:
        monkeypatch.setattr(benchmark, "apply_sprint_precision", lambda *_: {"precision": "mxfp8"})
    with pytest.raises(RuntimeError, match="All Sprint precision candidates failed"):
        benchmark.run_sprint_precision_benchmark(recipe(precisions=["bf16"], batch_sizes=[2]), tmp_path)
    saved = json.loads((tmp_path / "benchmark.json").read_text())
    assert saved["cases"][0]["status"] == "error"
    if failure == "nonfinite_gradient":
        assert saved["cases"][0]["final_grad_norm"] is None
        assert saved["cases"][0]["final_grad_norm_nonfinite"] == "nan"


def test_storage_accounting_deduplicates_offload_aliases():
    buffer = torch.ones(8, dtype=torch.float32)
    stats = benchmark._storage_inventory({"staging": buffer, "alias": buffer[:4], "buffers": [buffer]})
    assert stats == {"bytes": 32, "storages": 1, "by_device": {"cpu": 32}, "by_dtype": {"torch.float32": 32}}


def test_partial_cli_result_has_nonzero_exit_code(tmp_path, monkeypatch):
    import sys

    cfg = tmp_path / "recipe.json"
    cfg.write_text(json.dumps(recipe()))
    monkeypatch.setattr(sys, "argv", ["benchmark", "--config", str(cfg), "--output", str(tmp_path)])
    monkeypatch.setattr(benchmark, "run_sprint_precision_benchmark", lambda *_: {"status": "partial"})
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
