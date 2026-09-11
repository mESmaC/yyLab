"""CPU admission and methodology checks for disposable precision comparisons."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from yylab.models.sprint import SprintConfig, SprintForCausalLM
from yylab.train import low_precision, mxfp8
from yylab.train.bitnet_benchmark import BitNetQATLinear
from yylab.train.sprint_precision import PRECISION_MODES, apply_sprint_precision


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model():
    return SprintForCausalLM(SprintConfig(
        vocab_size=19, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=8, loss_backend="torch",
        tie_word_embeddings=True)).bfloat16()


@pytest.mark.parametrize("precision", ["fp8", "int8", "nvfp4", "emulated", ""])
def test_unknown_precision_is_rejected_before_model_validation(precision):
    with pytest.raises(ValueError, match="unknown Sprint benchmark precision"):
        apply_sprint_precision(nn.Identity(), precision, "cpu")


@pytest.mark.parametrize("precision", PRECISION_MODES)
def test_every_precision_requires_all_bf16_parameters(precision):
    model = nn.Sequential(nn.Linear(4, 4, dtype=torch.bfloat16), nn.Linear(4, 4))
    with pytest.raises(ValueError, match="BF16 parameters in every mode"):
        apply_sprint_precision(model, precision, "cpu")


def test_parameterless_model_is_rejected():
    with pytest.raises(ValueError, match="BF16 parameters"):
        apply_sprint_precision(nn.Identity(), "bf16", "cpu")


@pytest.mark.parametrize("precision", PRECISION_MODES)
def test_device_mismatch_is_rejected_before_native_preflight(precision, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("device validation must precede backend preflight")

    monkeypatch.setattr(mxfp8, "native_mxfp8_status", unexpected)
    monkeypatch.setattr(low_precision, "native_nvfp4_status", unexpected)
    with pytest.raises(ValueError, match="place all model parameters"):
        apply_sprint_precision(tiny_model(), precision, "cuda:0")


def test_bf16_receipt_preserves_the_exact_baseline_model():
    model = tiny_model()
    identities = {name: id(module) for name, module in model.named_modules()}
    parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    metadata = apply_sprint_precision(model, "bf16", "cpu")
    assert metadata["precision"] == "bf16"
    assert metadata["forward_gemm"] == metadata["backward_gemm"] == "bf16"
    assert metadata["converted_linear_count"] == 0
    assert metadata["parameters_before"] == metadata["parameters_after"]
    assert not metadata["training_state_is_low_bit"]
    assert not metadata["native_low_bit_gemm"] and not metadata["native_low_bit_backward"]
    assert identities == {name: id(module) for name, module in model.named_modules()}
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters[name])
    json.dumps(metadata)


def test_bitnet_receipt_reports_dense_training_and_extra_norm_parameters():
    model = tiny_model()
    embedding, head = model.embed_tokens, model.lm_head
    metadata = apply_sprint_precision(model, "bitnet_b158", "cpu")
    assert metadata["provider"] == "bitnet_b158_ste_dense_bf16"
    assert metadata["forward_gemm"] == metadata["backward_gemm"] == "bf16"
    assert not metadata["native_low_bit_gemm"] and not metadata["native_low_bit_backward"]
    assert not metadata["training_state_is_low_bit"]
    assert metadata["converted_linear_count"] == 4
    assert metadata["parameters_after"] - metadata["parameters_before"] == 16 + 32
    assert metadata["bitnet_recipe"]["added_norm_parameters"] == 16 + 32
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert model.embed_tokens is embedding and model.lm_head is head
    assert head.weight is embedding.weight
    assert sum(isinstance(module, BitNetQATLinear) for module in model.modules()) == 4
    json.dumps(metadata)


def test_mxfp8_unavailable_receipt_fails_closed_before_mutation(monkeypatch):
    model = tiny_model()
    modules = dict(model.named_modules())
    classes = {name: type(module) for name, module in modules.items()}
    monkeypatch.setattr(mxfp8, "native_mxfp8_status", lambda device: (False, "injected native cast failure"))
    with pytest.raises(RuntimeError, match="native MXFP8 training is unavailable: injected native cast failure"):
        apply_sprint_precision(model, "mxfp8", "cpu")
    assert modules == dict(model.named_modules())
    assert classes == {name: type(module) for name, module in model.named_modules()}


@pytest.mark.parametrize("precision", ["nvfp4_bf16", "nvfp4_mxfp8"])
def test_native_nvfp4_unavailable_fails_closed_before_conversion(precision, monkeypatch):
    model = tiny_model()
    modules = dict(model.named_modules())
    classes = {name: type(module) for name, module in modules.items()}
    monkeypatch.setattr(low_precision, "native_nvfp4_status", lambda device: (False, "injected unsupported GPU"))
    with pytest.raises(RuntimeError, match="native NVFP4 training is unavailable: injected unsupported GPU"):
        apply_sprint_precision(model, precision, "cpu")
    assert modules == dict(model.named_modules())
    assert classes == {name: type(module) for name, module in model.named_modules()}


def test_nvfp4_mxfp8_backward_requires_its_own_native_receipt(monkeypatch):
    model = tiny_model()
    modules = dict(model.named_modules())
    monkeypatch.setattr(low_precision, "native_nvfp4_status", lambda device: (True, "forward available"))
    monkeypatch.setattr(mxfp8, "native_mxfp8_status", lambda device: (False, "gradient GEMM unavailable"))
    with pytest.raises(RuntimeError, match="MXFP8.*gradient GEMM unavailable"):
        apply_sprint_precision(model, "nvfp4_mxfp8", "cpu")
    assert modules == dict(model.named_modules())
