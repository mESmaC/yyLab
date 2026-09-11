"""CPU equivalence tests for Sprint's single-microbatch execution shortcut."""

from __future__ import annotations

import copy

import pytest
import torch

from yylab.models.sprint import SprintConfig, SprintForCausalLM
from yylab.train import sprint


@pytest.fixture(autouse=True)
def _restore_cpu_settings():
    threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.set_default_dtype(dtype)


def _engines(parameter_dtype="float32", grad_clip=0.0):
    torch.manual_seed(81)
    model = SprintForCausalLM(SprintConfig(
        vocab_size=17, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
        loss_backend="torch",
    ))
    cfg = {"lr": 0.0015, "max_steps": 9, "warmup_steps": 2, "weight_decay": 0.1,
           "grad_clip": grad_clip, "parameter_dtype": parameter_dtype,
           "optimizer_state_offload": "cpu" if parameter_dtype == "bfloat16" else "none",
           "cpu_optimizer_kernel": "native"}
    return (sprint.SprintStepEngine(copy.deepcopy(model), cfg, "cpu"),
            sprint.SprintStepEngine(copy.deepcopy(model), cfg, "cpu"))


def _batch(rows, *, masked=False, explicit_counts=False):
    generator = torch.Generator(device="cpu").manual_seed(100 + rows)
    ids = torch.randint(0, 17, (rows, 8), generator=generator)
    labels = ids.clone()
    if masked:
        labels[0, 1:5] = -100
        labels[-1, -2:] = -100
    batch = {"input_ids": ids, "labels": labels}
    if explicit_counts:
        batch.update(num_targets=int((labels[:, 1:] != -100).sum()), real_tokens=ids.numel())
    return batch


def _legacy_single_step(engine, batch, step):
    """Freeze the previous one-batch scaling/accumulation as a regression oracle."""
    original = engine._cpu_batch(batch)
    total = original["num_targets"]
    assert total > 0
    engine.execution_model.train()
    engine.optimizer.zero_grad(set_to_none=True)
    for group in engine.optimizer.param_groups:
        group["lr"] = sprint.sprint_learning_rate(engine.cfg, step)
    loss_sum = torch.zeros((), device=engine.device)
    with engine._attention_context():
        for transferred in engine._device_batches([original]):
            with engine._autocast():
                loss = engine.execution_model(**transferred).loss
                weighted = loss * (original["num_targets"] / total)
            weighted.backward()
            loss_sum.add_(weighted.detach())
    raw_norm = sum(parameter.grad.float().square().sum()
                   for parameter in engine.model.parameters() if parameter.grad is not None).sqrt()
    clip = float(engine.cfg.get("grad_clip", 1.0))
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(engine.model.parameters(), clip, foreach=True)
    engine.optimizer.step()
    result = sprint.SprintStepResult(loss_sum, original["num_tokens"], total, original["real_tokens"])
    return result, raw_norm


@pytest.mark.parametrize("parameter_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("grad_clip", [0.0, 0.05])
def test_single_microbatch_matches_legacy_loss_gradients_and_adam_state(parameter_dtype, grad_clip):
    candidate, reference = _engines(parameter_dtype, grad_clip)
    batches = [_batch(3), _batch(2, masked=True), _batch(1, masked=True, explicit_counts=True)]
    for step, batch in enumerate(batches):
        actual = candidate.train_step([batch], step=step)
        expected, raw_norm = _legacy_single_step(reference, batch, step)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
        assert actual.loss.dtype == expected.loss.dtype
        assert not actual.loss.requires_grad and actual.loss.grad_fn is None
        assert (actual.tokens, actual.supervised_tokens, actual.real_tokens) == (
            expected.tokens, expected.supervised_tokens, expected.real_tokens)
        torch.testing.assert_close(candidate.model.state_dict(), reference.model.state_dict(), rtol=0, atol=0)
        for left, right in zip(candidate.model.parameters(), reference.model.parameters(), strict=True):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
        torch.testing.assert_close(candidate.optimizer.state_dict(), reference.optimizer.state_dict(),
                                   rtol=0, atol=0)
        if grad_clip:
            assert raw_norm > grad_clip  # This case must exercise actual clipping.


def test_single_microbatch_preserves_legacy_accumulation_dtype():
    torch.set_default_dtype(torch.float64)
    candidate, reference = _engines()
    batch = _batch(2, masked=True)
    actual = candidate.train_step([batch])
    expected, _ = _legacy_single_step(reference, batch, 0)
    assert actual.loss.dtype == expected.loss.dtype == torch.float64
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)


def test_returned_loss_does_not_alias_the_model_output_buffer():
    engine, _ = _engines()
    model_losses = []
    handle = engine.model.register_forward_hook(lambda _module, _args, output: model_losses.append(output.loss))
    try:
        result = engine.train_step([_batch(2)])
    finally:
        handle.remove()
    saved_result = result.loss.clone()
    # Simulate a compiled output buffer being reused after its backward is done.
    model_losses[-1].detach().add_(123.0)
    torch.testing.assert_close(result.loss, saved_result, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["empty", "ignored", "declared_zero"])
def test_no_target_guard_preserves_existing_model_gradients_and_optimizer_state(invalid):
    engine, _ = _engines()
    engine.train_step([_batch(2)])
    saved_model = copy.deepcopy(engine.model.state_dict())
    saved_optimizer = copy.deepcopy(engine.optimizer.state_dict())
    saved_gradients = [parameter.grad.clone() for parameter in engine.model.parameters()]
    batch = _batch(1)
    if invalid == "ignored":
        batch["labels"].fill_(-100)
    elif invalid == "declared_zero":
        batch["num_targets"] = 0
    batches = [] if invalid == "empty" else [batch]
    with pytest.raises(ValueError, match="no supervised shifted targets"):
        engine.train_step(batches, step=1)
    torch.testing.assert_close(engine.model.state_dict(), saved_model, rtol=0, atol=0)
    torch.testing.assert_close(engine.optimizer.state_dict(), saved_optimizer, rtol=0, atol=0)
    for parameter, saved in zip(engine.model.parameters(), saved_gradients, strict=True):
        torch.testing.assert_close(parameter.grad, saved, rtol=0, atol=0)
