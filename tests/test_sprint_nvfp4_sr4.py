"""CPU-only admission, metadata, and cross-case isolation for portable SR4."""

import copy
import inspect
import json

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from yylab.models.sprint import SprintConfig, SprintForCausalLM
from yylab.train import sprint_nvfp4_portable as portable
from yylab.train.sprint_nvfp4_training import apply_torchao_nvfp4_training
from yylab.train.sprint_precision import DEFAULT_PRECISION_MODES, PRECISION_MODES, apply_sprint_precision


def _function(linear_class):
    return inspect.getclosurevars(linear_class.forward).nonlocals["function_cls"]


def _operation(linear_class):
    return _function(linear_class).forward.__globals__["triton_rht_quantize_row_col"]


def test_sr4_is_explicit_and_requires_cuda_for_execution():
    assert "nvfp4_all_portable_sr4" in PRECISION_MODES
    assert "nvfp4_all_portable_sr4" not in DEFAULT_PRECISION_MODES
    assert "nvfp4_all_portable" in PRECISION_MODES
    model = torch.nn.Linear(128, 128, dtype=torch.bfloat16)
    original_class = type(model)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        apply_sprint_precision(model, "nvfp4_all_portable_sr4", "cpu")
    assert type(model) is original_class


def test_sr4_has_independent_class_operation_and_kernel_globals():
    from torchao.prototype.moe_training.nvfp4_training import hadamard_quantize_row_col_triton as provider
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import nvfp4_mm_triton

    original_kernel = provider._hadamard_quantize_row_col_kernel
    original_operation = provider.triton_rht_quantize_row_col
    original_pack = portable._portable_pack_fp4
    scalar = portable.portable_nvfp4_linear_class()
    before = copy.deepcopy(scalar._portable_provider_provenance)
    scalar_op = _operation(scalar)
    sr4 = portable.portable_nvfp4_sr4_linear_class()
    sr4_op = _operation(sr4)
    assert scalar is portable.portable_nvfp4_linear_class()
    assert sr4 is portable.portable_nvfp4_sr4_linear_class()
    assert scalar is not sr4 and scalar_op is not sr4_op
    for linear, operation, pack in ((scalar, scalar_op, original_pack),
                                   (sr4, sr4_op, portable._portable_pack_fp4_sr4)):
        function = _function(linear)
        assert function.forward.__globals__ is not nvfp4_mm_triton.forward.__globals__
        assert function.backward.__globals__["triton_rht_quantize_row_col"] is operation
        kernel = operation._init_fn.__globals__["_hadamard_quantize_row_col_kernel"]
        assert kernel.fn.fn.__globals__["_pack_fp4"] is pack
    assert scalar._portable_provider_provenance == before
    assert "stochastic_rounding_recipe" not in before
    assert portable._portable_pack_fp4 is original_pack
    assert provider._hadamard_quantize_row_col_kernel is original_kernel
    assert provider.triton_rht_quantize_row_col is original_operation
    assert nvfp4_mm_triton.backward.__globals__["triton_rht_quantize_row_col"] is original_operation
    assert "nvfp4_portable_sr4_quantize" in str(sr4_op)
    assert "nvfp4_portable_sr_quantize" in str(scalar_op)


def test_sr4_fake_operation_keeps_provider_output_contract():
    operation = _operation(portable.portable_nvfp4_sr4_linear_class())
    with FakeTensorMode():
        values = torch.empty((384, 640), dtype=torch.bfloat16)
        amax = torch.empty((), dtype=torch.float32)
        result = operation(values, amax, amax, [1] * 16)
        assert [tuple(tensor.shape) for tensor in result] == [
            (640, 192), (5, 6, 32, 16), (384, 320), (3, 10, 32, 16)]
        assert [tensor.dtype for tensor in result] == [
            torch.uint8, torch.float8_e4m3fn, torch.uint8, torch.float8_e4m3fn]


def test_sr4_conversion_preserves_parameters_and_reports_distinct_random_layout():
    model = SprintForCausalLM(SprintConfig(
        vocab_size=257, hidden_size=128, intermediate_size=192,
        num_hidden_layers=1, num_attention_heads=2,
    )).to(dtype=torch.bfloat16).eval()
    parameters = {name: (id(parameter), parameter.detach().clone()) for name, parameter in model.named_parameters()}
    head = model.lm_head
    metadata = apply_torchao_nvfp4_training(model, stochastic_rounding="portable_sr4", pad_to_alignment=True)
    for name, parameter in model.named_parameters():
        identity, before = parameters[name]
        assert id(parameter) == identity
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert not model.training and all(not module.training for module in model.modules())
    assert model.lm_head is head and model.lm_head.weight is model.embed_tokens.weight
    assert metadata["precision"] == "nvfp4_all_gemm_portable_sr4"
    assert metadata["stochastic_rounding"] == "portable_sr4"
    assert metadata["parameters_before"] == metadata["parameters_after"]
    assert len(metadata["padded_projections"]) == 2
    recipe = metadata["stochastic_rounding_recipe"]
    assert recipe["philox_rounds"] == 10
    assert recipe["random_values_per_call"] == 4
    assert recipe["uniform_bits_per_element"] == 24
    assert recipe["baseline_bitwise_equivalent"] is False
    assert recipe["hardware_bitwise_equivalent"] is False
    assert recipe == metadata["source_provenance"]["stochastic_rounding_recipe"]
    assert len(recipe["reviewed_candidate_source_sha256"]) == 64
    assert metadata["source_provenance"]["upstream_source_sha256"] == portable._REVIEWED_PROVIDER_SHA256
    json.dumps(metadata)
    rng = torch.get_rng_state().clone()
    repeated = apply_torchao_nvfp4_training(model, stochastic_rounding="portable_sr4", pad_to_alignment=True)
    assert repeated == metadata
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    repeated["stochastic_rounding_recipe"]["philox_rounds"] = 1
    assert model._torchao_nvfp4_training_benchmark_metadata["stochastic_rounding_recipe"]["philox_rounds"] == 10
    with pytest.raises(ValueError, match="Cannot change stochastic rounding"):
        apply_torchao_nvfp4_training(model, stochastic_rounding="portable", pad_to_alignment=True)
