import pytest
import torch
from torch.nn import functional as F

from yylab.models.sprint import SprintConfig, SprintForCausalLM, estimate_parameter_count, plan_sprint_model


def tiny(**kwargs):
    return SprintConfig(vocab_size=47, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=2,
                        max_position_embeddings=64, **kwargs)


@pytest.mark.parametrize("gated", ["gelu", "swiglu"])
@pytest.mark.parametrize("tied", [True, False])
def test_exact_parameter_accounting(gated, tied):
    config = tiny(mlp_type=gated, tie_word_embeddings=tied)
    model = SprintForCausalLM(config)
    assert sum(p.numel() for p in model.parameters()) == estimate_parameter_count(config)
    assert (model.lm_head.weight is model.embed_tokens.weight) == tied


def test_causal_no_future_leakage_and_shifted_masked_loss():
    torch.manual_seed(3)
    model = SprintForCausalLM(tiny()).eval()
    x = torch.randint(0, 47, (2, 13))
    changed = x.clone()
    changed[:, 7:] = (changed[:, 7:] + 1) % 47
    reference = model(x, use_cache=False).logits
    torch.testing.assert_close(reference[:, :7], model(changed, use_cache=False).logits[:, :7])
    labels = x.clone()
    labels[:, :4] = -100
    out = model(x, labels=labels)
    expected = F.cross_entropy(reference[:, :-1].reshape(-1, 47), labels[:, 1:].reshape(-1))
    assert out.logits is None
    torch.testing.assert_close(out.loss, expected)
    out.loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


def test_chunked_loss_and_gradients_match_dense():
    torch.manual_seed(5)
    dense = SprintForCausalLM(tiny(loss_backend="torch"))
    chunked = SprintForCausalLM(tiny(loss_backend="chunked", loss_chunk_size=7))
    chunked.load_state_dict(dense.state_dict())
    x = torch.randint(0, 47, (2, 13))
    labels = x.clone()
    labels[0, 1:5] = -100
    left, right = dense(x, labels=labels).loss, chunked(x, labels=labels).loss
    torch.testing.assert_close(left, right)
    left.backward()
    right.backward()
    for a, b in zip(dense.parameters(), chunked.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-7, rtol=3e-5)


def test_all_ignored_loss_is_zero_and_differentiable():
    model = SprintForCausalLM(tiny())
    x = torch.ones(2, 8, dtype=torch.long)
    loss = model(x, labels=torch.full_like(x, -100)).loss
    assert loss == 0
    loss.backward()
    assert all(p.grad.isfinite().all() for p in model.parameters())


def test_checkpointing_preserves_gradients():
    torch.manual_seed(3)
    left, right = SprintForCausalLM(tiny()), SprintForCausalLM(tiny())
    right.load_state_dict(left.state_dict())
    right.gradient_checkpointing_enable()
    x = torch.randint(0, 47, (2, 11))
    left(x, labels=x).loss.backward()
    right(x, labels=x).loss.backward()
    for a, b in zip(left.parameters(), right.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize("target", [100000, 1000000, 10000000, 50000000, 500000000, 1000000000])
def test_planner_continuous_budgets(target):
    cfg = plan_sprint_model(target, 257)
    assert 0.98 * target <= estimate_parameter_count(cfg) <= target
    assert cfg.hidden_size % 16 == 0
    assert cfg.intermediate_size % 16 == 0


def test_vocabulary_floor_and_invalid_shapes():
    with pytest.raises(ValueError, match="vocabulary"):
        plan_sprint_model(100, 128000)
    with pytest.raises(ValueError, match="even head"):
        SprintConfig(hidden_size=63, num_attention_heads=3)
    with pytest.raises(ValueError, match="context"):
        tiny(rope_scaling={"rope_type": "linear", "factor": 2})


def test_cpu_autocast_forward():
    model = SprintForCausalLM(tiny())
    x = torch.randint(0, 47, (2, 11))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(x, labels=x).loss
    loss.backward()
    assert loss.isfinite()
