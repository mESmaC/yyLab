from __future__ import annotations

import copy

import pytest
import torch

from yylab.train.base_config import (
    TRAINING_RECIPE_PRESETS,
    projected_training_memory_gb,
)
from yylab.train.loop import CPUOffloadAdamW, make_optimizer
from yylab.train.low_precision import apply_qat, qat_metadata
from yylab.train.potentiated import (
    FP8_ADAPTIVE_MASTER,
    FP8_POTENTIATED_MASTER,
    adaptive_master_bytes,
    encode_adaptive_potentiated_fp8,
    encode_potentiated_fp8,
    potentiated_master_bytes,
)


def test_potentiated_fp8_improves_on_bare_e4m3() -> None:
    torch.manual_seed(7)
    value = torch.randn(10_003, dtype=torch.float32) * 0.02
    state = encode_potentiated_fp8(value, stochastic=False)
    restored = state.dequantize().reshape(-1)
    bare = state.base.float() * state.scale.repeat_interleave(state.block_size)[: value.numel()]

    assert state.base.dtype == torch.float8_e4m3fn
    assert state.potentiation.dtype == torch.uint8
    assert state.storage_bytes == potentiated_master_bytes(value.numel())
    assert 0.5 < state.promoted_fraction <= 1.0
    assert (restored - value).abs().mean() < (bare - value).abs().mean() / 5
    assert torch.isfinite(restored).all()


def test_potentiated_fp8_stochastic_encoding_is_seeded() -> None:
    value = torch.linspace(-0.15, 0.15, 513)
    first_generator = torch.Generator().manual_seed(123)
    second_generator = torch.Generator().manual_seed(123)
    first = encode_potentiated_fp8(value, generator=first_generator)
    second = encode_potentiated_fp8(value, generator=second_generator)

    assert torch.equal(first.base.view(torch.uint8), second.base.view(torch.uint8))
    assert torch.equal(first.potentiation, second.potentiation)
    assert torch.equal(first.scale, second.scale)


def test_potentiated_fp8_rejects_nonfinite_and_hysterizes_scales() -> None:
    with pytest.raises(FloatingPointError, match="NaN or infinity"):
        encode_potentiated_fp8(torch.tensor([0.0, float("nan")]))

    value = torch.full((32,), 1.0)
    initial = encode_potentiated_fp8(value, stochastic=False)
    nearby = encode_potentiated_fp8(
        value * 1.01,
        stochastic=False,
        previous_scale=initial.scale,
    )
    overflow = encode_potentiated_fp8(
        torch.full((32,), float(initial.scale.item()) * 500.0),
        stochastic=False,
        previous_scale=initial.scale,
    )

    assert torch.equal(nearby.scale, initial.scale)
    assert torch.all(overflow.scale > initial.scale)
    assert torch.isfinite(overflow.dequantize()).all()


def test_adaptive_fp8_stores_residuals_only_for_promoted_blocks() -> None:
    torch.manual_seed(41)
    value = torch.randn(320, dtype=torch.float32) * 0.02
    state = encode_adaptive_potentiated_fp8(
        value,
        stochastic=False,
        max_promoted_fraction=0.10,
        promote_threshold=0.0,
        demote_threshold=0.0,
    )

    assert state.base.dtype == torch.float8_e4m3fn
    assert state.promoted_blocks.dtype == torch.int32
    assert state.promoted_blocks.numel() == 1
    assert state.potentiation.shape == (1, state.block_size // 2)
    assert state.stability.shape == (10,)
    assert state.promoted_fraction == pytest.approx(0.10)
    assert state.nbytes == adaptive_master_bytes(value.numel(), promoted_blocks=1)
    assert state.potentiation.numel() < (value.numel() + 1) // 2
    assert torch.isfinite(state.dequantize()).all()


def test_adaptive_fp8_hysteresis_demotes_a_stable_block() -> None:
    value = torch.linspace(-0.15, 0.15, 320)
    promoted = encode_adaptive_potentiated_fp8(
        value,
        stochastic=False,
        max_promoted_fraction=0.10,
        promote_threshold=0.0,
        demote_threshold=0.0,
        demote_patience=2,
    )
    exact = torch.zeros_like(value)
    first_stable = encode_adaptive_potentiated_fp8(
        exact,
        stochastic=False,
        previous=promoted,
        max_promoted_fraction=0.10,
        demote_patience=2,
    )
    demoted = encode_adaptive_potentiated_fp8(
        exact,
        stochastic=False,
        previous=first_stable,
        max_promoted_fraction=0.10,
        demote_patience=2,
    )

    assert first_stable.promoted_blocks.numel() == 1
    retained = int(first_stable.promoted_blocks.item())
    assert int(first_stable.stability[retained].item()) == 1
    assert demoted.promoted_blocks.numel() == 0
    assert demoted.potentiation.numel() == 0


def test_adaptive_fp8_promotes_when_bare_fp8_would_lose_an_update() -> None:
    value = torch.ones(320)
    baseline = encode_adaptive_potentiated_fp8(
        value,
        stochastic=False,
        max_promoted_fraction=0.10,
        promote_threshold=2.0,
    )
    assert baseline.promoted_blocks.numel() == 0

    requested = baseline.dequantize() + 0.01
    updated = encode_adaptive_potentiated_fp8(
        requested,
        stochastic=False,
        previous=baseline,
        max_promoted_fraction=0.10,
    )

    assert updated.promoted_blocks.numel() == 1
    promoted_block = int(updated.promoted_blocks.item())
    block = slice(promoted_block * updated.block_size, (promoted_block + 1) * updated.block_size)
    bare = updated.base.float() * updated.scale.repeat_interleave(updated.block_size)
    assert (updated.dequantize()[block] - requested[block]).abs().mean() < (
        bare[block] - requested[block]
    ).abs().mean()


def test_adaptive_fp8_stochastic_encoding_and_sparse_order_are_seeded() -> None:
    value = torch.randn(640, generator=torch.Generator().manual_seed(9)) * 0.02
    first = encode_adaptive_potentiated_fp8(
        value,
        generator=torch.Generator().manual_seed(123),
        max_promoted_fraction=0.10,
        promote_threshold=0.0,
        demote_threshold=0.0,
    )
    second = encode_adaptive_potentiated_fp8(
        value,
        generator=torch.Generator().manual_seed(123),
        max_promoted_fraction=0.10,
        promote_threshold=0.0,
        demote_threshold=0.0,
    )

    assert torch.equal(first.base.view(torch.uint8), second.base.view(torch.uint8))
    assert torch.equal(first.promoted_blocks, second.promoted_blocks)
    assert torch.equal(first.potentiation, second.potentiation)
    assert torch.all(first.promoted_blocks[1:] > first.promoted_blocks[:-1])


def test_cpu_offload_adamw_uses_potentiated_fp8_master() -> None:
    torch.manual_seed(11)
    parameter = torch.nn.Parameter(torch.randn(257, dtype=torch.bfloat16) * 0.02)
    before = parameter.detach().clone()
    optimizer = CPUOffloadAdamW(
        [parameter],
        lr=1e-3,
        master_weight_storage=FP8_POTENTIATED_MASTER,
        master_chunk_size=32,
    )
    parameter.grad = torch.randn_like(parameter) * 0.01
    optimizer.step()

    state = optimizer.state[parameter]
    summary = optimizer.master_state_summary()
    assert "master_param" not in state
    assert state["master_fp8"].dtype == torch.float8_e4m3fn
    assert state["master_potentiation"].dtype == torch.uint8
    assert summary["format"] == FP8_POTENTIATED_MASTER
    assert int(summary["bytes"]) == potentiated_master_bytes(parameter.numel())
    assert 0.0 <= float(summary["promoted_fraction"]) <= 1.0
    assert torch.isfinite(parameter.float()).all()
    assert not torch.equal(parameter, before)


def test_cpu_offload_adamw_uses_sparse_adaptive_fp8_master() -> None:
    torch.manual_seed(53)
    parameter = torch.nn.Parameter(torch.randn(320, dtype=torch.bfloat16) * 0.02)
    optimizer = CPUOffloadAdamW(
        [parameter],
        lr=1e-3,
        master_weight_storage=FP8_ADAPTIVE_MASTER,
        adaptive_max_promoted_fraction=0.10,
        adaptive_promote_threshold=0.0,
        adaptive_demote_threshold=0.0,
    )
    parameter.grad = torch.randn_like(parameter) * 0.01
    optimizer.step()

    state = optimizer.state[parameter]
    summary = optimizer.master_state_summary()
    assert "master_param" not in state
    assert "master_potentiation" not in state
    assert state["master_fp8"].dtype == torch.float8_e4m3fn
    assert state["master_promoted_blocks"].dtype == torch.int32
    assert state["master_promoted_blocks"].numel() <= 1
    assert state["master_promoted_potentiation"].shape[1] == 16
    assert summary["format"] == FP8_ADAPTIVE_MASTER
    assert summary["moment_format"] == "fp32"
    assert int(summary["moment_bytes"]) == parameter.numel() * 8
    assert int(summary["bytes"]) == adaptive_master_bytes(
        parameter.numel(),
        promoted_blocks=state["master_promoted_blocks"].numel(),
    )
    assert float(summary["promoted_fraction"]) <= 0.10
    assert torch.isfinite(parameter.float()).all()


def test_adaptive_fp8_optimizer_state_resumes_exactly_on_cpu(monkeypatch) -> None:
    torch.manual_seed(59)
    first_parameter = torch.nn.Parameter(torch.randn(320, dtype=torch.bfloat16) * 0.02)
    options = {
        "lr": 1e-3,
        "master_weight_storage": FP8_ADAPTIVE_MASTER,
        "adaptive_max_promoted_fraction": 0.10,
        "adaptive_promote_threshold": 0.0,
        "adaptive_demote_threshold": 0.0,
    }
    first = CPUOffloadAdamW([first_parameter], **options)
    first_parameter.grad = torch.randn_like(first_parameter) * 0.01
    first.step()

    second_parameter = torch.nn.Parameter(first_parameter.detach().clone())
    second = CPUOffloadAdamW([second_parameter], **options)
    monkeypatch.setattr(
        torch.optim.Optimizer,
        "load_state_dict",
        lambda *_args, **_kwargs: pytest.fail(
            "offloaded adaptive state must stay on CPU during restore"
        ),
    )
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    next_gradient = torch.randn_like(first_parameter) * 0.01
    first_parameter.grad = next_gradient.clone()
    second_parameter.grad = next_gradient.clone()
    first.step()
    second.step()

    assert torch.equal(first_parameter, second_parameter)
    first_state = first.state[first_parameter]
    second_state = second.state[second_parameter]
    for name in (
        "master_fp8",
        "master_promoted_blocks",
        "master_promoted_potentiation",
        "master_scale",
        "master_promotion_stability",
        "exp_avg",
        "exp_avg_sq",
    ):
        assert torch.equal(first_state[name], second_state[name])
        assert not first_state[name].is_cuda


def test_potentiated_optimizer_state_resumes_exactly_on_cpu(monkeypatch) -> None:
    torch.manual_seed(31)
    first_parameter = torch.nn.Parameter(torch.randn(129, dtype=torch.bfloat16) * 0.02)
    first = CPUOffloadAdamW(
        [first_parameter],
        lr=1e-3,
        master_weight_storage=FP8_POTENTIATED_MASTER,
        master_chunk_size=32,
    )
    first_parameter.grad = torch.randn_like(first_parameter) * 0.01
    first.step()

    second_parameter = torch.nn.Parameter(first_parameter.detach().clone())
    second = CPUOffloadAdamW(
        [second_parameter],
        lr=1e-3,
        master_weight_storage=FP8_POTENTIATED_MASTER,
        master_chunk_size=32,
    )
    monkeypatch.setattr(
        torch.optim.Optimizer,
        "load_state_dict",
        lambda *_args, **_kwargs: pytest.fail(
            "offloaded state must not pass through the accelerator-casting base loader"
        ),
    )
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    restored = second.state[second_parameter]
    assert restored["master_fp8"].dtype == torch.float8_e4m3fn
    assert restored["master_potentiation"].dtype == torch.uint8
    assert restored["master_scale"].dtype == torch.float32
    assert restored["exp_avg"].dtype == torch.float32
    assert restored["exp_avg_sq"].dtype == torch.float32
    assert all(not value.is_cuda for value in restored.values() if isinstance(value, torch.Tensor))

    next_gradient = torch.randn_like(first_parameter) * 0.01
    first_parameter.grad = next_gradient.clone()
    second_parameter.grad = next_gradient.clone()
    first.step()
    second.step()

    assert torch.equal(first_parameter, second_parameter)
    first_state = first.state[first_parameter]
    second_state = second.state[second_parameter]
    for name in (
        "master_fp8",
        "master_potentiation",
        "master_scale",
        "exp_avg",
        "exp_avg_sq",
    ):
        assert torch.equal(first_state[name], second_state[name])


def test_potentiated_master_requires_cpu_offload_and_preset_is_exact() -> None:
    model = torch.nn.Linear(8, 8)
    with pytest.raises(ValueError, match="CPU-offload AdamW"):
        make_optimizer(
            model,
            1e-3,
            cuda=False,
            optimizer_state="gpu",
            master_weight_storage=FP8_POTENTIATED_MASTER,
        )

    preset = TRAINING_RECIPE_PRESETS["nvfp4_paired_4to8_fp8_master"]
    assert preset["qat"] == "nvfp4"
    assert preset["qat_backend"] == "native"
    assert preset["sparsity"] == "paired_4:8"
    assert preset["optimizer_state"] == "cpu_offload"
    assert preset["optimizer_impl"] == "cpu_offload_adamw"
    assert preset["master_weight_storage"] == FP8_POTENTIATED_MASTER

    _, fp32_cpu = projected_training_memory_gb(
        1_000_000,
        optimizer_state="cpu_offload",
        master_weight_storage="fp32",
    )
    _, fp8_cpu = projected_training_memory_gb(
        1_000_000,
        optimizer_state="cpu_offload",
        master_weight_storage=FP8_POTENTIATED_MASTER,
    )
    assert fp8_cpu < fp32_cpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_nvfp4_sparse_step_with_potentiated_master() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        use_cache=False,
    )
    model = Qwen3ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    metadata = qat_metadata(
        "nvfp4",
        parameter_dtype="bf16",
        backend="native",
        sparsity="paired_4:8",
    )
    assert metadata is not None
    apply_qat(model, metadata)
    optimizer = CPUOffloadAdamW(
        model.parameters(),
        lr=1e-3,
        master_weight_storage=FP8_POTENTIATED_MASTER,
    )
    input_ids = torch.randint(0, config.vocab_size, (1, 32), device="cuda")
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert optimizer.master_state_summary()["format"] == FP8_POTENTIATED_MASTER
    assert all(
        torch.isfinite(parameter.float()).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
