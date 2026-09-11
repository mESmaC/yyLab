"""Optional CUDA validation; run before accepting portable-NVFP4 throughput."""

import pytest
import torch
import triton
import triton.language as tl

from yylab.train.sprint_nvfp4_portable import (
    _portable_e2m1_codes,
    _portable_pack_fp4,
    portable_nvfp4_linear_class,
    stochastic_e2m1_reference,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@triton.jit
def _codes_oracle_kernel(values, uniforms, output, N: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(values + indices, indices < N, other=0.0)
    uniform = tl.load(uniforms + indices, indices < N, other=0.0)
    tl.store(output + indices, _portable_e2m1_codes(value, uniform), indices < N)


@triton.jit
def _pack_validation_kernel(values, output, seed, offset, ROWS: tl.constexpr, COLUMNS: tl.constexpr):
    rows = tl.arange(0, ROWS)[:, None]
    columns = tl.arange(0, COLUMNS)[None, :]
    scaled = tl.load(values + rows * COLUMNS + columns)
    packed = _portable_pack_fp4(scaled, ROWS, COLUMNS, True, seed, offset, tl.program_id(0))
    packed_columns = tl.arange(0, COLUMNS // 2)[None, :]
    tl.store(output + rows * (COLUMNS // 2) + packed_columns, packed)


def test_cuda_portable_codes_match_cpu_oracle_at_boundaries_and_special_values():
    values = torch.tensor([
        -float("inf"), -7., -6., -5.5, -4.5, -4., -3.5, -3., -2.5, -2., -1.75,
        -1.5, -1.25, -1., -.75, -.5, -.25, -0., 0., .125, .25, .375, .5, .75,
        1., 1.25, 1.5, 1.75, 2., 2.5, 3., 3.5, 4., 4.5, 5.5, 6., 7., float("inf"), float("nan"),
    ])
    uniforms = torch.cat([torch.arange(16) / 16, torch.tensor([1 - 2**-24])])
    values = values[:, None].expand(-1, uniforms.numel()).contiguous().flatten()
    uniforms = uniforms.repeat(values.numel() // uniforms.numel()).contiguous()
    expected = stochastic_e2m1_reference(values, uniforms)
    output = torch.empty_like(expected, device="cuda")
    _codes_oracle_kernel[(triton.cdiv(values.numel(), 128),)](
        values.cuda(), uniforms.cuda(), output, values.numel(), 128)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_cuda_portable_pack_repeats_seed_offset_and_changes_with_new_offset():
    values = torch.full((128, 128), .25, device="cuda", dtype=torch.float32)
    seed = torch.tensor([1729], device="cuda", dtype=torch.int64)
    offset = torch.tensor([91], device="cuda", dtype=torch.int64)

    def draw():
        result = torch.empty((128, 64), dtype=torch.uint8, device="cuda")
        _pack_validation_kernel[(1,)](values, result, seed, offset, 128, 128)
        return result.cpu()

    first, repeated = draw(), draw()
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    offset.fill_(92)
    assert (draw() != first).any()
    codes = torch.stack([first & 15, first >> 4], dim=-1).flatten()
    assert set(codes.tolist()) == {0, 1}
    assert abs(float(codes.float().mean()) - .5) < .03


@pytest.mark.parametrize("in_features,out_features,padded", [(256, 128, False), (192, 320, True)])
def test_cuda_portable_native_linear_forward_backward_and_gradient_cropping(in_features, out_features, padded):
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native NVFP4 GEMMs require Blackwell")
    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import prepare_for_cuda_graph

    generator = torch.Generator(device="cpu").manual_seed(834)
    input_cpu = torch.randn((128, in_features), generator=generator).to(torch.bfloat16)
    weight_cpu = (torch.randn((out_features, in_features), generator=generator) * .05).to(torch.bfloat16)
    grad_cpu = torch.randn((128, out_features), generator=generator).to(torch.bfloat16)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(833)
        linear = portable_nvfp4_linear_class()(
            in_features, out_features, bias=True, device="cuda", dtype=torch.bfloat16)
        linear._sprint_nvfp4_pad_to_alignment = padded
        with torch.no_grad():
            linear.weight.copy_(weight_cpu)
            linear.bias.zero_()
        prepare_for_cuda_graph(linear.weight.device, sign_vectors=(linear.rht_sign_vector,))
        inputs = input_cpu.cuda().requires_grad_()
        output = linear(inputs)
        output.backward(grad_cpu.cuda())
        assert output.shape == (128, out_features)
        assert inputs.grad.shape == inputs.shape
        assert linear.weight.grad.shape == weight_cpu.shape
        assert linear.bias.grad.shape == (out_features,)
        actual = [output.detach().float().cpu(), inputs.grad.float().cpu(), linear.weight.grad.float().cpu()]
        expected = [input_cpu.float().mm(weight_cpu.float().t()), grad_cpu.float().mm(weight_cpu.float()),
                    grad_cpu.float().t().mm(input_cpu.float())]
        for result, reference in zip(actual, expected, strict=True):
            assert result.isfinite().all()
            assert result.count_nonzero() > 0
            relative_error = (result - reference).norm() / reference.norm().clamp_min(1e-12)
            assert relative_error < .4, f"relative NVFP4 error {float(relative_error):.6f}"
        torch.testing.assert_close(linear.bias.grad.float().cpu(), grad_cpu.float().sum(0).to(torch.bfloat16).float())


def test_cuda_portable_sr_rng_advances_across_compiled_default_backward():
    """Fresh SR across default-compiled executions; explicit CUDA graphs are not covered."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native NVFP4 GEMMs require Blackwell")
    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import prepare_for_cuda_graph

    generator = torch.Generator(device="cpu").manual_seed(936)
    input_cpu = torch.randn((128, 256), generator=generator).to(torch.bfloat16)
    weight_cpu = (torch.randn((128, 256), generator=generator) * .05).to(torch.bfloat16)
    output_gradient_cpu = torch.randn((128, 128), generator=generator).to(torch.bfloat16)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(937)
        linear = portable_nvfp4_linear_class()(256, 128, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.copy_(weight_cpu)
        prepare_for_cuda_graph(linear.weight.device, sign_vectors=(linear.rht_sign_vector,))
        inputs = input_cpu.cuda().requires_grad_()
        output_gradient = output_gradient_cpu.cuda()
        compiled = torch.compile(linear, mode="default")

        def execute():
            # Prevent accumulated gradients from masquerading as fresh SR noise.
            inputs.grad = None
            linear.zero_grad(set_to_none=True)
            output = compiled(inputs)
            output.backward(output_gradient)
            assert output.isfinite().all()
            assert inputs.grad.isfinite().all()
            assert linear.weight.grad.isfinite().all()
            return (output.detach().cpu(), inputs.grad.detach().cpu(), linear.weight.grad.detach().cpu())

        # Both AOT backward compilation and provider autotuning finish before
        # samples are collected. Do not reset the generator between executions.
        for _ in range(3):
            execute()
        samples = [execute() for _ in range(3)]
        for output, _, _ in samples[1:]:
            torch.testing.assert_close(output, samples[0][0], rtol=0, atol=0)
        assert any(not torch.equal(sample[1], samples[0][1]) for sample in samples[1:]), (
            "dgrad stochastic rounding did not advance across compiled backward executions"
        )
        assert any(not torch.equal(sample[2], samples[0][2]) for sample in samples[1:]), (
            "wgrad stochastic rounding did not advance across compiled backward executions"
        )
        # No optimizer runs: variation must come from backward quantization.
        torch.testing.assert_close(linear.weight.detach().cpu(), weight_cpu, rtol=0, atol=0)
        torch.testing.assert_close(inputs.detach().cpu(), input_cpu, rtol=0, atol=0)
        torch.testing.assert_close(output_gradient.cpu(), output_gradient_cpu, rtol=0, atol=0)
