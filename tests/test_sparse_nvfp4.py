from __future__ import annotations

import pytest
import torch

from yylab.train.sparse_nvfp4 import (
    _EXECUTION_CACHE,
    _padded_paired_weight,
    _padded_shape,
    _quantize_group32,
    clear_sparse_nvfp4_cache,
    paired_4to8,
    sparse_nvfp4_forward,
    sparse_nvfp4_linear,
    sparse_nvfp4_status,
)


def test_sparse_nvfp4_padding_obeys_descriptor_multiples() -> None:
    shape = _padded_shape(3, 35, 47)

    assert (shape.m, shape.n, shape.k) == (3, 35, 47)
    assert (shape.padded_m, shape.padded_n, shape.padded_k) == (16, 64, 64)
    assert shape.padded_m % 16 == 0
    assert shape.padded_n % 32 == 0
    assert shape.padded_k % 64 == 0

    tiny = _padded_shape(1, 1, 1)
    assert (tiny.padded_m, tiny.padded_n, tiny.padded_k) == (16, 32, 64)


def test_paired_4to8_keeps_exactly_two_complete_pairs() -> None:
    value = torch.arange(1, 65, dtype=torch.float32).reshape(4, 16)

    sparse = paired_4to8(value)
    sparse_again = paired_4to8(value)
    pair_nonzero = sparse.ne(0).reshape(-1, 4, 2)

    assert torch.equal(sparse, sparse_again)
    assert torch.equal(pair_nonzero[..., 0], pair_nonzero[..., 1])
    assert torch.all(pair_nonzero[..., 0].sum(dim=-1) == 2)
    assert sparse.count_nonzero() * 2 == sparse.numel()


def test_sparse_nvfp4_rejects_invalid_public_inputs() -> None:
    with pytest.raises(ValueError, match="2-D"):
        sparse_nvfp4_forward(torch.zeros(2, 3, 4), torch.zeros(5, 4))
    with pytest.raises(ValueError, match="does not match"):
        sparse_nvfp4_forward(
            torch.zeros(2, 8, dtype=torch.bfloat16),
            torch.zeros(4, 16, dtype=torch.bfloat16),
        )
    with pytest.raises(TypeError, match="BF16"):
        sparse_nvfp4_forward(torch.zeros(2, 8), torch.zeros(4, 8))


@pytest.fixture(scope="module")
def native_sparse_backend() -> str:
    available, note = sparse_nvfp4_status(force_refresh=True)
    if not available:
        pytest.skip(note)
    yield note
    clear_sparse_nvfp4_cache()


def test_group32_quantization_uses_packed_values_and_blocked_scales(
    native_sparse_backend: str,
) -> None:
    del native_sparse_backend
    value = torch.linspace(
        -0.75,
        0.75,
        129 * 160,
        device="cuda",
        dtype=torch.bfloat16,
    ).reshape(129, 160)

    packed, blocked_scales, global_scale = _quantize_group32(value)

    assert packed.shape == (129, 80)
    assert packed.dtype == torch.uint8
    assert packed.is_contiguous()
    # Logical scales are [129, 5]. NVIDIA's layout pads that to [256, 8].
    assert blocked_scales.numel() == 256 * 8
    assert blocked_scales.dtype == torch.float8_e4m3fn
    assert blocked_scales.is_contiguous()
    assert global_scale.shape == ()
    assert global_scale.dtype == torch.float32
    assert bool(torch.isfinite(global_scale)) and bool(global_scale > 0)


def test_sparse_nvfp4_real_irregular_gemm_is_finite_nonzero_and_numerical(
    native_sparse_backend: str,
) -> None:
    assert "cuSPARSELt 0.9.1" in native_sparse_backend
    clear_sparse_nvfp4_cache()
    generator = torch.Generator(device="cpu").manual_seed(91)
    input = (
        torch.randn((19, 47), generator=generator, dtype=torch.float32)
        .mul(0.2)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    weight = (
        torch.randn((35, 47), generator=generator, dtype=torch.float32)
        .mul(0.2)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    sparse_weight = _padded_paired_weight(weight, 64)[:, :47]

    output = sparse_nvfp4_forward(input, sparse_weight)
    torch.cuda.synchronize()
    reference = input.float().mm(sparse_weight.float().t())
    normalized_error = (output.float() - reference).norm() / reference.norm()

    assert output.shape == (19, 35)
    assert output.dtype == torch.bfloat16
    assert output.is_contiguous()
    assert bool(torch.isfinite(output).all())
    assert bool(output.abs().max() > 0)
    assert float(normalized_error) < 0.30


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(1, 1, 1), (7, 9, 31), (16, 32, 32), (17, 33, 33), (33, 65, 96)],
)
def test_sparse_nvfp4_kernel_safe_padding_grid(
    native_sparse_backend: str,
    m: int,
    n: int,
    k: int,
) -> None:
    del native_sparse_backend
    generator = torch.Generator(device="cpu").manual_seed(1000 + m + n + k)
    input = (
        torch.randn((m, k), generator=generator, dtype=torch.float32)
        .mul(0.2)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    weight = (
        torch.randn((n, k), generator=generator, dtype=torch.float32)
        .mul(0.2)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    padded_k = _padded_shape(m, n, k).padded_k
    sparse_weight = _padded_paired_weight(weight, padded_k)[:, :k]

    output = sparse_nvfp4_forward(input, sparse_weight)
    torch.cuda.synchronize()
    reference = input.float().mm(sparse_weight.float().t())
    normalized_error = (output.float() - reference).norm() / reference.norm().clamp_min(1e-12)

    assert output.shape == (m, n)
    assert output.dtype == torch.bfloat16
    assert bool(torch.isfinite(output).all())
    assert bool(output.abs().max() > 0)
    assert float(normalized_error) < 0.35


def test_sparse_nvfp4_autograd_and_current_stream_cache(
    native_sparse_backend: str,
) -> None:
    del native_sparse_backend
    clear_sparse_nvfp4_cache()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        input = (torch.randn(2, 3, 47, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
        weight = (torch.randn(35, 47, device="cuda", dtype=torch.bfloat16) * 0.2).requires_grad_()
        bias = torch.randn(35, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        output = sparse_nvfp4_linear(input, weight, bias)
        output.float().square().mean().backward()
        # The same padded shape/current stream reuses its host plan and GPU buffers.
        repeated = sparse_nvfp4_linear(input.detach(), weight.detach(), bias.detach())
    torch.cuda.current_stream().wait_stream(stream)

    assert len(_EXECUTION_CACHE) == 1
    assert output.shape == (2, 3, 35)
    assert repeated.shape == output.shape
    assert output.dtype == torch.bfloat16
    assert all(
        gradient is not None
        and gradient.dtype == torch.bfloat16
        and bool(torch.isfinite(gradient).all())
        for gradient in (input.grad, weight.grad, bias.grad)
    )
