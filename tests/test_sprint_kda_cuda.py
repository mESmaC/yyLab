"""CUDA numerical and resumable-training gate for native PLE + KDA."""
import copy
from contextlib import contextmanager

import pytest
import torch

from yylab.models.sprint import SprintConfig, SprintForCausalLM
from yylab.train.sprint import SprintStepEngine, save_sprint_checkpoint

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def config():
    return SprintConfig(vocab_size=128, hidden_size=128, intermediate_size=256,
                        num_attention_heads=2, num_hidden_layers=4, max_position_embeddings=128,
                        attention_type="kda", kda_backend="auto", ple_type="full", loss_backend="torch")


@pytest.mark.parametrize("layout", ["plain", "padding", "segments"])
def test_fla_matches_reference_loss_logits_and_gradients(layout):
    torch.set_num_threads(2)
    torch.manual_seed(924)
    fast = SprintForCausalLM(config()).cuda().train()
    reference = copy.deepcopy(fast)
    for layer in reference.layers:
        if hasattr(layer.self_attn, "backend"):
            layer.self_attn.backend = "reference"
    ids = torch.randint(3, 128, (2, 16), device="cuda")
    kwargs = {}
    labels = ids.clone()
    if layout == "padding":
        mask = torch.ones_like(ids)
        mask[0, :3] = 0
        mask[1, -3:] = 0
        labels[mask == 0] = -100
        kwargs["attention_mask"] = mask
    elif layout == "segments":
        kwargs["position_ids"] = torch.arange(8, device="cuda").repeat(2).unsqueeze(0).expand(2, -1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = fast(ids, labels=labels, return_logits=True, use_cache=False, **kwargs)
        b = reference(ids, labels=labels, return_logits=True, use_cache=False, **kwargs)
    torch.testing.assert_close(a.logits, b.logits, rtol=.025, atol=.015)
    torch.testing.assert_close(a.loss, b.loss, rtol=.003, atol=.003)
    a.loss.backward()
    b.loss.backward()
    error_squared, reference_squared = 0.0, 0.0
    for (name, p), (other_name, q) in zip(fast.named_parameters(), reference.named_parameters(), strict=True):
        assert name == other_name and p.grad is not None and q.grad is not None
        assert torch.isfinite(p.grad).all(), name
        error = (p.grad.float() - q.grad.float()).norm()
        scale = q.grad.float().norm().clamp_min(1e-8)
        # A nearly cancelled scalar PLE gate gradient needs an absolute bound;
        # a relative-only criterion is ill-conditioned around zero. Matrix and
        # vector gradients retain the original relative criterion. The direct
        # kernel test below independently constrains every KDA gradient to 1%.
        absolute = .001 if p.numel() == 1 else 0.0
        assert float(error) < .08 * float(scale) + absolute, (name, float(error / scale))
        error_squared += float(error) ** 2
        reference_squared += float(scale) ** 2
    assert (error_squared / reference_squared) ** .5 < .04


@pytest.mark.parametrize("head_dim", [64, 128])
def test_fla_recurrence_all_gradients_match_fp32_reference(head_dim):
    from fla.ops.kda import chunk_kda

    from yylab.models.sprint_kda import reference_kda

    torch.manual_seed(926)
    shape = (2, 64, 2, head_dim)
    values = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    values += [torch.randn((2, 64, 2), device="cuda", dtype=torch.bfloat16),
               torch.tensor([0., 1.], device="cuda"), torch.full((2 * head_dim,), -3., device="cuda")]
    left = [value.detach().clone().requires_grad_() for value in values]
    right = [value.detach().clone().requires_grad_() for value in values]
    q, k, v, g, beta, a_log, dt_bias = left
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual, _ = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=a_log, dt_bias=dt_bias,
                             use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                             use_beta_sigmoid_in_kernel=True, disable_recompute=True)
        expected = reference_kda(*right)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    for a, b in [(actual, expected), *((x.grad, y.grad) for x, y in zip(left, right, strict=True))]:
        assert torch.isfinite(a).all()
        assert float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)) < .01


def test_compiled_fla_optimizer_and_checkpoint_resume(tmp_path):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    torch.set_num_threads(2)
    torch.manual_seed(925)
    cfg = {"parameter_dtype": "float32", "optimizer_state_offload": "none", "compute_dtype": "bfloat16",
           "torch_compile": True, "compile_mode": "default", "cpu_threads": 2,
           "lr": .001, "warmup_steps": 0, "max_steps": 4, "grad_clip": 1.0}
    engine = SprintStepEngine(SprintForCausalLM(config()), cfg, "cuda")
    ids = torch.randint(3, 128, (2, 64))
    batch = {"input_ids": ids, "labels": ids}
    before = {name: p.detach().clone() for name, p in engine.model.named_parameters()}
    first = engine.train_step([batch], step=0)
    assert torch.isfinite(first.loss) and first.tokens == 128
    for name, p in engine.model.named_parameters():
        assert torch.isfinite(p).all(), name
        assert not torch.equal(p, before[name]), name
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(
        vocab={f"t{i}": i for i in range(128)}, unk_token="t0")), unk_token="t0", eos_token="t2")
    path = save_sprint_checkpoint(tmp_path / "saved", engine, tokenizer, {"step": 1})
    loaded = SprintForCausalLM.from_pretrained(path, local_files_only=True, dtype="auto")
    restored = SprintStepEngine(loaded, cfg, "cuda")
    state = torch.load(path / "trainer.pt", map_location="cpu", weights_only=True)
    restored.optimizer.load_state_dict(state["optimizer"])
    # The health check uses the eager model between compiled training updates.
    # It must leave both state and the subsequent fused-optimizer update intact.
    from yylab.train.sprint_quality import evaluate_sprint_generation

    @contextmanager
    def inference_context():
        with engine._attention_context(), engine._autocast():
            yield

    rng_before = torch.cuda.get_rng_state().clone()
    weights_before = {name: p.detach().clone() for name, p in engine.model.named_parameters()}
    grads_before = {name: p.grad.detach().clone() for name, p in engine.model.named_parameters()}
    receipt = evaluate_sprint_generation(engine.model, tokenizer,
        config={"max_new_tokens": 8, "min_evidence_tokens": 8,
                "prompts": [{"name": "cuda", "kind": "prose", "prompt": "t3"}]},
        context_factory=inference_context)
    assert receipt["summary"]["outputs"] == 2
    assert receipt["summary"]["nonfinite_outputs"] == 0
    assert torch.equal(torch.cuda.get_rng_state(), rng_before)
    assert engine.model.training
    for name, parameter in engine.model.named_parameters():
        assert torch.equal(parameter, weights_before[name]), name
        assert torch.equal(parameter.grad, grads_before[name]), name
    a = engine.train_step([batch], step=1)
    b = restored.train_step([batch], step=1)
    torch.testing.assert_close(a.loss, b.loss, rtol=1e-5, atol=1e-5)
    for name, p in engine.model.named_parameters():
        torch.testing.assert_close(p, dict(restored.model.named_parameters())[name], rtol=1e-4, atol=1e-5)
