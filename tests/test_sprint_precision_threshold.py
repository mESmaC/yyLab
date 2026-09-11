"""CPU-only contract checks for the optional precision-benchmark Adam cutoff."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from yylab.train import benchmark_sprint_precision as benchmark
from yylab.train.sprint import SprintCPUOffloadAdamW

KEY = "experimental_cpu_adam_min_elements"


class SizeOnlyParameter:
    def __init__(self, size):
        self.size = size

    def numel(self):
        return self.size


class FakeOptimizer:
    _buckets = SprintCPUOffloadAdamW._buckets

    def __init__(self, prepared=True):
        self._compiled_cpu_adam = object() if prepared else None
        self._min_compiled_elements = 6
        self.bucket_elements = 8
        self.param_groups = [{"params": [SizeOnlyParameter(n) for n in sizes]}
                             for sizes in ([5, 5, 9], [3])]


def request(value=4, **updates):
    return {KEY: value, "cpu_optimizer_kernel": "compiled", **updates}


def test_absent_override_leaves_default_engine_completely_untouched():
    assert benchmark._requested_cpu_adam_threshold({}) is None
    assert benchmark._configure_cpu_adam_threshold(SimpleNamespace(), {}) is None


@pytest.mark.parametrize("value", [None, 0, -1, True, 4.0, "4", float("nan"), float("inf")])
def test_explicit_override_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="explicit positive integer"):
        benchmark._requested_cpu_adam_threshold(request(value))


@pytest.mark.parametrize("kernel", [None, "auto", "native"])
def test_override_requires_explicit_compiled_selection(kernel):
    with pytest.raises(ValueError, match="cpu_optimizer_kernel"):
        benchmark._requested_cpu_adam_threshold(request(cpu_optimizer_kernel=kernel))


def test_override_rejects_gpu_resident_optimizer():
    with pytest.raises(ValueError, match="CPU optimizer offload"):
        benchmark._requested_cpu_adam_threshold(request(optimizer_state_offload="none"))


def test_coverage_uses_real_per_group_bucketing_and_partial_parameter_slices():
    optimizer = FakeOptimizer()
    receipt = benchmark._configure_cpu_adam_threshold(SimpleNamespace(optimizer=optimizer), request())
    assert optimizer._min_compiled_elements == receipt["effective_min_elements"] == 4
    assert receipt["requested_min_elements"] == 4 and receipt["default_min_elements"] == 6
    assert receipt["bucket_elements"] == 8
    # Group1 slices are [5,3], [2,6], [3]; group2 begins a fresh bucket [3].
    assert receipt["eligible_compiled_parameters"] == receipt["eligible_native_parameters"] == 11
    assert receipt["eligible_compiled_slices"] == 2 and receipt["eligible_native_slices"] == 4
    assert receipt["eligible_compiled_fraction"] == 0.5


@pytest.mark.parametrize("prepared,threshold,error", [(False, 4, RuntimeError), (True, 9, ValueError)])
def test_unprepared_or_oversized_request_fails_without_clamping_or_mutation(prepared, threshold, error):
    optimizer = FakeOptimizer(prepared)
    with pytest.raises(error):
        benchmark._configure_cpu_adam_threshold(SimpleNamespace(optimizer=optimizer), request(threshold))
    assert optimizer._min_compiled_elements == 6


@pytest.mark.parametrize("updates", [request(0), request(cpu_optimizer_kernel="native")])
def test_invalid_public_request_fails_before_device_probe_or_model_allocation(monkeypatch, tmp_path, updates):
    def forbidden(*args, **kwargs):
        raise AssertionError("device/model allocation happened before threshold validation")

    monkeypatch.setattr(benchmark.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(benchmark, "SprintConfig", forbidden)
    with pytest.raises(ValueError):
        benchmark.run_sprint_precision_benchmark(updates, tmp_path)
    assert not list(tmp_path.iterdir())


def test_public_config_preserves_override_and_case_runtime_receipt(monkeypatch, tmp_path):
    seen = []

    def measured(config, cfg, precision, batch, mode, device, inputs):
        optimizer = FakeOptimizer()
        runtime = benchmark._configure_cpu_adam_threshold(SimpleNamespace(optimizer=optimizer), cfg)
        seen.append((cfg[KEY], optimizer._min_compiled_elements))
        return {"status": "ok", "precision": precision, "micro_batch": batch, "grad_accum": 1,
                "torch_compile": False, "compile_mode": "default", "parameters": 100,
                "supervised_tokens_per_second": 1.0,
                "runtime": {"experimental_cpu_adam_threshold": runtime}}

    monkeypatch.setattr(benchmark, "_measure_precision_case", measured)
    report = benchmark.run_sprint_precision_benchmark({
        **request(), "device": "cpu", "cpu_threads": 1, "seq_len": 2,
        "effective_batch": 1, "batch_sizes": [1], "precisions": ["bf16"],
        "compile_modes": ["eager"], "warmup_steps": 1, "measure_steps": 1,
        "model_config": {"vocab_size": 17, "hidden_size": 8, "intermediate_size": 16,
                         "num_hidden_layers": 1, "num_attention_heads": 1},
    }, tmp_path)
    assert seen == [(4, 4)] and report["recipe"][KEY] == 4
    assert report["fastest_measured"]["runtime"]["experimental_cpu_adam_threshold"]["effective_min_elements"] == 4
    assert json.loads((tmp_path / "benchmark.json").read_text())["recipe"][KEY] == 4


@pytest.mark.parametrize("with_config", [False, True])
def test_cli_explicit_flag_reaches_benchmark_and_overrides_recipe(monkeypatch, tmp_path, with_config):
    seen = []
    arguments = ["benchmark_sprint_precision", "--output", str(tmp_path / "output"),
                 "--cpu-optimizer-min-elements", "262144"]
    if with_config:
        recipe = tmp_path / "recipe.json"
        recipe.write_text(json.dumps(request(1048576)))
        arguments += ["--config", str(recipe)]
    else:
        arguments += ["--cpu-optimizer-kernel", "compiled"]

    def run(cfg, output):
        seen.append((benchmark._requested_cpu_adam_threshold(cfg), output))
        return {"status": "ok"}

    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(benchmark, "run_sprint_precision_benchmark", run)
    benchmark.main()
    assert seen == [(262144, tmp_path / "output")]
