from copy import deepcopy

import pytest
import torch

from yylab.train.benchmark_spalmer import compare_pairs, summarize_performance, tensor_state_hash


def sample(step, seconds=1):
    return {
        "step": step,
        "warmup": step == 1,
        "diagnostic": step == 2,
        "training_seconds": seconds,
        "target_tokens": 100,
        "peak_cuda_allocated_bytes": 200,
        "peak_cuda_reserved_bytes": 300,
        "peak_working_set_bytes": 400,
        "callback_seconds": 20,
        "phase_seconds": {"forward": 0.2},
        "recurrence_steps": 2,
        "input_sha256": str(step),
        "learning_rate": 0.001,
        "model_loss": 1.25,
    }


def test_throughput_excludes_warmup_profiling_and_validation():
    result = summarize_performance([sample(1, 50), sample(2, 50), sample(3), sample(4)])
    assert result["measured_steps"] == 2
    assert result["target_tokens_per_second"] == 100
    assert result["callback_seconds"] == 80
    assert result["diagnostic_total_seconds"] == 50


def test_optimizer_comparison_rejects_different_tokens_or_depth():
    cpu = [sample(3), sample(4)]
    gpu = deepcopy(cpu)
    gpu[0]["input_sha256"] = "wrong"
    with pytest.raises(ValueError, match="input_sha256"):
        compare_pairs(cpu, gpu)
    gpu = deepcopy(cpu)
    gpu[0]["recurrence_steps"] = 8
    with pytest.raises(ValueError, match="recurrence_steps"):
        compare_pairs(cpu, gpu)
    assert compare_pairs(cpu, cpu)["gpu_speedup"] == 1


def test_state_identity_supports_scalar_bf16_and_detects_changes():
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16)))
    model.register_buffer("counter", torch.tensor(0))
    before = tensor_state_hash(model)
    assert tensor_state_hash(model) == before
    model.counter.add_(1)
    assert tensor_state_hash(model) != before
