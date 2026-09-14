"""Independent Philox integer and E2M1 code oracles for production portable SR4."""

import pytest
import torch
import triton
import triton.language as tl

from yylab.train.sprint_nvfp4_portable import (
    _portable_pack_fp4_sr4,
    _sr4_random_bits,
    stochastic_e2m1_reference,
)


def _philox4_cpu(seed: int, offset: int) -> tuple[int, int, int, int]:
    """Scalar integer oracle, independent of Triton's device implementation."""
    mask = 0xFFFFFFFF
    c0, c1, c2, c3 = offset & mask, (offset >> 32) & mask, 0, 0
    k0, k1 = seed & mask, (seed >> 32) & mask
    for _ in range(10):
        p0, p2 = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((p2 >> 32) ^ c1 ^ k0) & mask, p2 & mask, ((p0 >> 32) ^ c3 ^ k1) & mask, p0 & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return c0, c1, c2, c3


def test_cpu_philox_known_answer():
    assert _philox4_cpu(0, 0) == (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)


@triton.jit
def _rng_kernel(output, seed, offset, ROWS: tl.constexpr, COLUMNS: tl.constexpr):
    tile = tl.program_id(0)
    bits = _sr4_random_bits(ROWS, COLUMNS, tl.load(seed), tl.load(offset), tile)
    rows = tl.arange(0, ROWS)[:, None]
    columns = tl.arange(0, COLUMNS)[None, :]
    tl.store(output + tile * ROWS * COLUMNS + rows * COLUMNS + columns, bits.to(tl.int64))


@triton.jit
def _pack_kernel(values, output, seed, offset, ROWS: tl.constexpr, COLUMNS: tl.constexpr):
    tile = tl.program_id(0)
    rows = tl.arange(0, ROWS)[:, None]
    columns = tl.arange(0, COLUMNS)[None, :]
    scaled = tl.load(values + tile * ROWS * COLUMNS + rows * COLUMNS + columns)
    packed = _portable_pack_fp4_sr4(scaled, ROWS, COLUMNS, True, seed, offset, tile)
    packed_columns = tl.arange(0, COLUMNS // 2)[None, :]
    tl.store(output + tile * ROWS * COLUMNS // 2 + rows * (COLUMNS // 2) + packed_columns, packed)


@pytest.mark.parametrize("rows,columns", [(128, 128), (128, 256), (256, 128)])
def test_cuda_sr4_random_bits_and_boundary_codes_match_independent_cpu_oracle(rows, columns):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    tiles = 3
    count = tiles * rows * columns
    seed_value, call_offset = 0x12345678000006C1, 91
    expected_bits = torch.tensor([
        value for group in range(count // 4)
        for value in _philox4_cpu(seed_value, (group << 32) | call_offset)
    ], dtype=torch.int64)
    seed = torch.tensor([seed_value], device="cuda", dtype=torch.int64)
    offset = torch.tensor([call_offset], device="cuda", dtype=torch.int64)
    actual_bits = torch.empty(count, device="cuda", dtype=torch.int64)
    _rng_kernel[(tiles,)](actual_bits, seed, offset, rows, columns)
    torch.testing.assert_close(actual_bits.cpu(), expected_bits, rtol=0, atol=0)

    boundaries = torch.tensor([
        -float("inf"), -7., -6., -5.5, -4.5, -4., -3.5, -3., -2.5, -2.,
        -1.75, -1.5, -1.25, -1., -.75, -.5, -.25, -0., 0., .125, .25,
        .375, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 3.5, 4.,
        4.5, 5.5, 6., 7., float("inf"), float("nan"),
    ])
    values = boundaries[torch.arange(count) % len(boundaries)]
    uniforms = (expected_bits >> 8).float() * (1.0 / 16777216.0)
    expected_codes = stochastic_e2m1_reference(values, uniforms)
    packed = torch.empty(count // 2, device="cuda", dtype=torch.uint8)
    _pack_kernel[(tiles,)](values.cuda(), packed, seed, offset, rows, columns)
    result = packed.cpu()
    unpacked = torch.stack([result & 15, result >> 4], dim=-1).flatten()
    torch.testing.assert_close(unpacked, expected_codes, rtol=0, atol=0)

    # The caller uses seed XOR 1 for a separate direction; the layout must
    # preserve that separation and changes to the lower per-call counter.
    seed.fill_(seed_value ^ 1)
    _rng_kernel[(tiles,)](actual_bits, seed, offset, rows, columns)
    assert not torch.equal(actual_bits.cpu(), expected_bits)
    seed.fill_(seed_value)
    offset.fill_(call_offset + 1)
    _rng_kernel[(tiles,)](actual_bits, seed, offset, rows, columns)
    assert not torch.equal(actual_bits.cpu(), expected_bits)
