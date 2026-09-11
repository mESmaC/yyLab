"""CPU-only adapter checks; native CUDA numerics belong to the GPU benchmark."""

import copy
import json

import pytest
import torch
from torch import nn

from yylab.models.sprint import SprintConfig, SprintForCausalLM
from yylab.train import sprint_nvfp4_training as adapter


def _model(**kwargs):
    return SprintForCausalLM(SprintConfig(
        vocab_size=257, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=2, **kwargs,
    )).to(dtype=torch.bfloat16)


def test_preserves_parameters_head_tie_and_mode_without_running_kernels():
    model = _model().eval()
    parameters = list(model.parameters())
    saved = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    head = model.lm_head
    metadata = adapter.apply_torchao_nvfp4_training(model)
    assert {id(p) for p in model.parameters()} == {id(p) for p in parameters}
    assert model.lm_head is head
    assert model.lm_head.weight is model.embed_tokens.weight
    assert all(not module.training for module in model.modules())
    for name, tensor in saved.items():
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)
    assert metadata["converted_linear_count"] == 4
    assert metadata["parameters_before"] == metadata["parameters_after"]
    assert metadata["forward_gemm"] == metadata["backward_gemm"] == "nvfp4"
    assert metadata["training_state_is_low_bit"] is False
    assert metadata["cuda_graph_workspace_prepared"] is False
    json.dumps(metadata)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())


def test_repeat_conversion_preserves_random_state_and_returns_metadata_copy():
    model = _model()
    first = adapter.apply_torchao_nvfp4_training(model)
    state = torch.get_rng_state().clone()
    module_ids = [id(module) for module in model.modules()]
    second = adapter.apply_torchao_nvfp4_training(model)
    assert first == second
    assert [id(module) for module in model.modules()] == module_ids
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
    second["converted_modules"].clear()
    assert adapter.apply_torchao_nvfp4_training(model)["converted_modules"] == first["converted_modules"]


def test_unaligned_shape_rejected_before_any_replacement_or_provider_import(monkeypatch):
    model = _model()
    model.layers[0].up_proj = nn.Linear(128, 192, bias=False, dtype=torch.bfloat16)
    modules = list(model.modules())
    monkeypatch.setattr(adapter, "_load_provider", lambda: pytest.fail("provider imported before validation"))
    with pytest.raises(ValueError, match="up_proj.*out_features=192"):
        adapter.apply_torchao_nvfp4_training(model)
    assert list(model.modules()) == modules


def test_requires_bf16_and_sprint():
    with pytest.raises(TypeError, match="SprintForCausalLM"):
        adapter.apply_torchao_nvfp4_training(nn.Linear(128, 128, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="BF16 parameters"):
        adapter.apply_torchao_nvfp4_training(_model().float())


def test_rejects_existing_projection_adapter_without_partial_conversion():
    class CustomLinear(nn.Linear):
        pass

    model = _model()
    model.layers[0].down_proj.__class__ = CustomLinear
    modules = list(model.modules())
    with pytest.raises(ValueError, match="unconverted.*down_proj"):
        adapter.apply_torchao_nvfp4_training(model)
    assert list(model.modules()) == modules


def test_seed_and_hadamard_buffers_survive_state_dict_restore():
    model = _model()
    adapter.apply_torchao_nvfp4_training(model)
    saved = copy.deepcopy(model.state_dict())
    target = _model()
    adapter.apply_torchao_nvfp4_training(target)
    target.load_state_dict(saved)
    for name in model._torchao_nvfp4_training_benchmark_metadata["converted_modules"]:
        left, right = model.get_submodule(name), target.get_submodule(name)
        assert left.rht_sign_vector == right.rht_sign_vector
        torch.testing.assert_close(left._sr_seed, right._sr_seed, rtol=0, atol=0)
        torch.testing.assert_close(left._rht_sign_vector, right._rht_sign_vector, rtol=0, atol=0)


@pytest.mark.parametrize("capability", [(12, 0), (11, 0), (9, 0)])
def test_stock_rounding_rejects_architectures_without_cvt_rs(capability):
    with pytest.raises(RuntimeError, match="cvt.rs requires SM100/SM103"):
        adapter._validate_hardware_rounding(capability)


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_stock_rounding_accepts_documented_cvt_rs_architectures(capability):
    adapter._validate_hardware_rounding(capability)


def test_portable_provider_isolated_from_stock_globals_and_explicitly_labeled():
    from torchao.prototype.moe_training.nvfp4_training import (
        hadamard_quantize_row_col_triton as provider,
    )
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import nvfp4_mm_triton

    old_kernel = provider._hadamard_quantize_row_col_kernel
    old_quantize = nvfp4_mm_triton.backward.__globals__["triton_rht_quantize_row_col"]
    model = _model()
    metadata = adapter.apply_torchao_nvfp4_training(model, stochastic_rounding="portable")
    assert metadata["stochastic_rounding"] == "portable"
    assert metadata["precision"] == "nvfp4_all_gemm_portable_sr"
    assert provider._hadamard_quantize_row_col_kernel is old_kernel
    assert nvfp4_mm_triton.backward.__globals__["triton_rht_quantize_row_col"] is old_quantize
    with pytest.raises(ValueError, match="Cannot change stochastic rounding"):
        adapter.apply_torchao_nvfp4_training(model)


def test_portable_rounding_oracle_representable_values_signs_and_saturation():
    from yylab.train.sprint_nvfp4_portable import stochastic_e2m1_reference

    magnitudes = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    values = torch.cat([magnitudes, -magnitudes, torch.tensor([float("inf"), -float("inf"), float("nan")])])
    for uniform_value in (0., .25, .5, 1 - 2**-24):
        codes = stochastic_e2m1_reference(values, torch.full_like(values, uniform_value))
        torch.testing.assert_close(codes, torch.tensor(list(range(16)) + [7, 15, 7], dtype=torch.uint8))


def test_portable_rounding_oracle_unbiased_over_uniform_grid():
    from yylab.train.sprint_nvfp4_portable import stochastic_e2m1_reference

    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    values = torch.tensor([.125, .375, .625, 1.25, 1.75, 2.25, 2.75, 3.25, 4.5, 5.5])
    values = torch.cat([values, -values])
    uniforms = ((torch.arange(256) + .5) / 256).expand(len(values), -1)
    codes = stochastic_e2m1_reference(values[:, None].expand_as(uniforms), uniforms)
    decoded = levels[(codes & 7).long()] * torch.where(codes >= 8, -1., 1.)
    torch.testing.assert_close(decoded.mean(1), values, rtol=0, atol=0)
    assert (decoded.min(1).values < decoded.max(1).values).all()


def test_portable_source_manifest_is_reported_and_changed_source_is_rejected(monkeypatch):
    from yylab.train import sprint_nvfp4_portable as portable

    provenance = portable._provider_source_provenance()
    assert provenance["upstream_source_sha256"] == portable._REVIEWED_PROVIDER_SHA256
    assert provenance["torchao_version"]
    assert len(provenance["portable_source_sha256"]) == 64
    monkeypatch.setitem(portable._REVIEWED_PROVIDER_SHA256, "nvfp4_linear.py", "0" * 64)
    with pytest.raises(RuntimeError, match="source guard rejected changed TorchAO file nvfp4_linear.py"):
        portable._provider_source_provenance()


def test_explicit_padding_preserves_original_model_shape_and_parameters():
    model = SprintForCausalLM(SprintConfig(
        vocab_size=257, hidden_size=128, intermediate_size=192,
        num_hidden_layers=1, num_attention_heads=2,
    )).to(dtype=torch.bfloat16)
    parameters = list(model.parameters())
    parameter_shapes = [parameter.shape for parameter in parameters]
    metadata = adapter.apply_torchao_nvfp4_training(
        model, stochastic_rounding="portable", pad_to_alignment=True)
    assert {id(p) for p in model.parameters()} == {id(p) for p in parameters}
    assert [parameter.shape for parameter in parameters] == parameter_shapes
    assert model.config.intermediate_size == 192
    assert model.layers[0].up_proj.out_features == 192
    assert model.layers[0].down_proj.in_features == 192
    assert model.lm_head.weight is model.embed_tokens.weight
    assert metadata["parameters_before"] == metadata["parameters_after"]
    assert metadata["pad_to_alignment"]
    assert {item["name"] for item in metadata["padded_projections"]} == {
        "layers.0.up_proj", "layers.0.down_proj"}
    assert metadata["padded_projections"][0]["gemm_out_features"] == 256
    assert metadata["padded_projections"][1]["gemm_in_features"] == 256
    assert len(metadata["source_provenance"]["upstream_source_sha256"]) == 6
    with pytest.raises(ValueError, match="Cannot change operand padding"):
        adapter.apply_torchao_nvfp4_training(model, stochastic_rounding="portable")


def test_padding_is_explicitly_portable_only():
    with pytest.raises(ValueError, match="only by the portable"):
        adapter.apply_torchao_nvfp4_training(_model(), pad_to_alignment=True)
