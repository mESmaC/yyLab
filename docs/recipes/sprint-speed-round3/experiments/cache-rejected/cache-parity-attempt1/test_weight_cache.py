"""Optional strict CUDA parity checks. Run only in an allocated GPU test slot.

These are correctness tests, not throughput measurements. No tolerance is
introduced for cached-vs-fresh weight codes: provider operations must agree.
"""
from __future__ import annotations

import copy
import io
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weight_cache import BUFFER_PREFIX, ForwardWeightCache, cached_engine_class, provider, provider_utils
from yylab.models.sprint import SprintConfig, SprintForCausalLM
from yylab.train.sprint import SprintStepEngine
from yylab.train.sprint_precision import FastLowPrecisionLinear, apply_sprint_precision

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA slot required")


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.cuda.synchronize()
    torch.set_num_threads(previous)


def projection(n=384, k=128):
    torch.manual_seed(3801)
    model = nn.Module()
    model.proj = FastLowPrecisionLinear(k, n, bias=False).to(device="cuda", dtype=torch.bfloat16)
    model.proj.forward_format = "nvfp4"
    return model


def fresh_payload(weight):
    scale = provider.per_tensor_amax_to_scale(weight.detach().abs().amax())
    packed = provider.NVFP4Tensor.to_nvfp4(weight, per_tensor_scale=scale)
    return packed.qdata, provider_utils.to_blocked(packed.scale), scale


@pytest.mark.parametrize("n,k", [(384, 128), (128, 192)])
def test_compiled_refresh_exact_packed_parity_and_stable_storage(n, k):
    model = projection(n, k)
    parameter_ids = [id(p) for p in model.parameters()]
    keys = set(model.state_dict())
    cache = ForwardWeightCache(model)
    assert [id(p) for p in model.parameters()] == parameter_ids
    pointers = tuple(buffer.data_ptr() for buffer in model.buffers())
    expected_fn = torch.compile(fresh_payload, fullgraph=True, dynamic=False,
                                options={"triton.cudagraphs": False})
    for update in range(3):
        with torch.no_grad():
            model.proj.weight.add_(0.03125 * update)
            expected = expected_fn(model.proj.weight)
        with cache.update_scope():
            cache.validate()
            actual = (model.proj._update_nvfp4_codes, model.proj._update_nvfp4_scales,
                      model.proj._update_nvfp4_tensor_scale)
            torch.cuda.synchronize()
            assert torch.equal(actual[0], expected[0])
            assert torch.equal(actual[1].view(torch.uint8).flatten(), expected[1].view(torch.uint8).flatten())
            assert torch.equal(actual[2], expected[2])
        assert tuple(buffer.data_ptr() for buffer in model.buffers()) == pointers
    assert cache.refresh_scopes == 3 and not cache.active
    assert set(model.state_dict()) == keys
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    assert all(BUFFER_PREFIX not in key for key in torch.load(buffer, weights_only=True))


def test_scope_rejects_stale_forward_mutation_and_load_then_refreshes():
    model = projection()
    cache = ForwardWeightCache(model)
    inputs = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    saved = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="active update scope"):
        model.proj(inputs)
    with pytest.raises(RuntimeError, match="weight mutation"):
        with cache.update_scope():
            with torch.no_grad():
                model.proj.weight.add_(1)
            cache.validate()
    assert not cache.active
    with pytest.raises(RuntimeError, match="load_state_dict"):
        with cache.update_scope():
            model.load_state_dict(saved)
    assert not cache.active
    model.load_state_dict(saved)
    with cache.update_scope():
        cache.validate()
        output = model.proj(inputs)
        assert torch.isfinite(output).all()
    with pytest.raises(RuntimeError, match="active update scope"):
        model.proj(inputs)
    with pytest.raises(RuntimeError, match="intentional"):
        with cache.update_scope():
            raise RuntimeError("intentional scope failure")
    assert not cache.active
    model.proj.weight = nn.Parameter(model.proj.weight.detach().clone())
    with pytest.raises(RuntimeError, match="identity changed"):
        with cache.update_scope():
            pass


def tiny_model():
    torch.manual_seed(3802)
    model = SprintForCausalLM(SprintConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=1, max_position_embeddings=128, loss_backend="torch"))
    model.to(device="cuda", dtype=torch.bfloat16)
    apply_sprint_precision(model, "nvfp4_mxfp8_fast", "cuda")
    return model


@pytest.mark.parametrize("compile_model", [False, True])
def test_losses_gradients_and_cpu_optimizer_updates_match_across_microbatches(compile_model):
    baseline, cached = tiny_model(), tiny_model()
    cfg = {"parameter_dtype": "bfloat16", "optimizer_state_offload": "cpu",
           "cpu_optimizer_kernel": "native", "cpu_threads": 2, "offload_bucket_mb": 1,
           "torch_compile": compile_model, "compile_mode": "default", "attention_backend": "cudnn",
           "gradient_checkpointing": False, "grad_clip": 1.0, "lr": 3e-4,
           "weight_decay": 0.1, "max_steps": 20, "warmup_steps": 0}
    reference = SprintStepEngine(baseline, cfg, device="cuda")
    candidate = cached_engine_class(SprintStepEngine)(cached, cfg, device="cuda")
    saved_model = saved_optimizer = None
    for step in range(4):
        if step == 3:
            # Restore both model and full optimizer state; cached buffers are
            # absent from the snapshot and must be regenerated for this step.
            baseline.load_state_dict(saved_model)
            cached.load_state_dict(saved_model)
            # The offloader adopts CPU state tensors without cloning. Keep
            # independent snapshots so one optimizer cannot mutate the other.
            reference.optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
            candidate.optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        generator = torch.Generator(device="cpu").manual_seed(3900 + step)
        batches = [{"input_ids": torch.randint(256, (2, 128), generator=generator)} for _ in range(2)]
        expected = reference.train_step(batches, step=step)
        actual = candidate.train_step(batches, step=step)
        reference.optimizer.synchronize()
        candidate.optimizer.synchronize()
        torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
        assert actual.tokens == expected.tokens == 512
        assert actual.supervised_tokens == expected.supervised_tokens == 508
        assert not candidate.weight_cache.active
        for (name, left), (other_name, right) in zip(baseline.named_parameters(), cached.named_parameters(), strict=True):
            assert name == other_name
            assert left.grad is not None and right.grad is not None
            torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)
            torch.testing.assert_close(right, left, rtol=0, atol=0)
            left_state, right_state = reference.optimizer.state[left], candidate.optimizer.state[right]
            assert left_state["step"] == right_state["step"]
            for field in ("master_param", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(right_state[field], left_state[field], rtol=0, atol=0)
        if step == 1:
            saved_model = copy.deepcopy(baseline.state_dict())
            saved_optimizer = copy.deepcopy(reference.optimizer.state_dict())
    assert candidate.weight_cache.refresh_scopes == 4
