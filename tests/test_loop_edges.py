from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from yylab.train.loop import CPUOffloadAdamW, LoopOpts, make_optimizer, train_loop
from yylab.workers.train import _warmup_steps


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)

    def forward(self, input_ids=None, labels=None):
        logits = self.head(self.embed(input_ids))
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        return SimpleNamespace(loss=loss)


def _run_update(model: nn.Module, batches: list[dict], grad_accum: int) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    train_loop(
        model,
        iter(batches),
        optimizer,
        scheduler,
        LoopOpts(grad_accum=grad_accum, max_steps=1, grad_clip=0, device="cpu"),
    )


def test_partial_accumulation_window_keeps_mean_gradient_scale() -> None:
    torch.manual_seed(8)
    partial = _TinyLM()
    combined = _TinyLM()
    combined.load_state_dict(partial.state_dict())
    first = [2, 3, 4, 5]
    second = [6, 7, 8, 9]

    _run_update(
        partial,
        [{"input_ids": [first]}, {"input_ids": [second]}],
        grad_accum=4,
    )
    _run_update(combined, [{"input_ids": [first, second]}], grad_accum=1)

    for partial_value, combined_value in zip(partial.parameters(), combined.parameters(), strict=True):
        assert torch.allclose(partial_value, combined_value, atol=1e-7, rtol=1e-6)


def test_one_step_schedule_has_no_zero_lr_warmup() -> None:
    assert _warmup_steps(1) == 0
    assert _warmup_steps(2) == 1


def test_large_joint_optimizer_auto_selects_paged_cpu_backing(monkeypatch) -> None:
    sentinel = object()
    calls: list[dict] = []

    class FakeParameter:
        requires_grad = True
        is_cuda = True

        def numel(self):
            return 60_000_000

    class FakeModel:
        def parameters(self):
            return [FakeParameter()]

    def fake_paged(_params, **kwargs):
        calls.append(kwargs)
        return sentinel

    fake_bnb = SimpleNamespace(optim=SimpleNamespace(PagedAdamW8bit=fake_paged))
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)
    assert make_optimizer(FakeModel(), 2e-4, cuda=True, optimizer_state="auto") is sentinel
    assert calls == [{"lr": 2e-4, "weight_decay": 0.0}]


def test_cpu_offload_adamw_matches_torch_and_keeps_state_on_cpu() -> None:
    torch.manual_seed(12)
    reference = nn.Linear(4, 3)
    offloaded = nn.Linear(4, 3)
    offloaded.load_state_dict(reference.state_dict())
    expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=2e-3, weight_decay=0.1)
    actual_optimizer = make_optimizer(
        offloaded,
        2e-3,
        weight_decay=0.1,
        cuda=False,
        optimizer_state="cpu_offload",
    )
    assert isinstance(actual_optimizer, CPUOffloadAdamW)

    for _ in range(3):
        inputs = torch.randn(5, 4)
        reference(inputs).square().mean().backward()
        offloaded(inputs).square().mean().backward()
        expected_optimizer.step()
        actual_optimizer.step()
        expected_optimizer.zero_grad(set_to_none=True)
        actual_optimizer.zero_grad(set_to_none=True)

    for expected, actual in zip(reference.parameters(), offloaded.parameters(), strict=True):
        assert torch.allclose(expected, actual, atol=2e-7, rtol=2e-6)
    state_tensors = [
        value
        for state in actual_optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    assert state_tensors
    assert all(value.device.type == "cpu" and value.dtype == torch.float32 for value in state_tensors)
