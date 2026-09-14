from __future__ import annotations

import math

import pytest
import torch

from yylab.train.sprint_unlikelihood import (
    UNLIKELIHOOD_VARIANT,
    repeat_unlikelihood_mask,
    sequence_unlikelihood_loss,
)


def test_third_generated_ngram_masks_only_its_final_token():
    ids = torch.tensor([8, 9] + [1, 2, 3, 4] * 3)
    mask = repeat_unlikelihood_mask(ids, 2, eos_token_id=0)
    assert mask.dtype == torch.bool and mask.shape == ids.shape
    assert mask.nonzero().flatten().tolist() == [13]
    assert "third-occurrence-final-token" in UNLIKELIHOOD_VARIANT


def test_later_overlapping_ngram_occurrences_are_counted_independently():
    ids = torch.tensor([9] + [1, 2, 3, 4] * 4)
    mask = repeat_unlikelihood_mask(ids, 1, None)
    assert mask.nonzero().flatten().tolist() == [12, 13, 14, 15, 16]
    mono = repeat_unlikelihood_mask(torch.ones(8, dtype=torch.long), 0, None)
    assert mono.nonzero().flatten().tolist() == [5, 6, 7]


def test_prompt_repetitions_and_ngrams_crossing_boundary_never_count():
    ids = torch.tensor([1, 2, 3, 4] * 8)
    assert not repeat_unlikelihood_mask(ids, 28, None).any()
    assert not repeat_unlikelihood_mask(ids, 24, None).any()
    assert not repeat_unlikelihood_mask(ids, 30, None).any()
    assert not repeat_unlikelihood_mask(ids, 32, None).any()


def test_first_generated_eos_excludes_itself_and_all_following_tokens():
    ids = torch.tensor([0, 8] + [1, 2, 3, 4] * 3 + [0] + [1, 2, 3, 4] * 4)
    mask = repeat_unlikelihood_mask(ids, 2, eos_token_id=0)
    assert mask.nonzero().flatten().tolist() == [13]
    multi = repeat_unlikelihood_mask(ids, 2, eos_token_id=[0, 3])
    assert not multi.any()
    early = repeat_unlikelihood_mask(torch.tensor([8, 0, 1, 1, 1]), 1, 0, ngram_size=1)
    assert not early.any()


def test_per_row_prefixes_and_counts_are_independent():
    ids = torch.tensor([[1, 2, 3, 4] * 4, [1, 2, 3, 4] * 4])
    mask = repeat_unlikelihood_mask(ids, torch.tensor([4, 12]), 0)
    assert mask[0].nonzero().flatten().tolist() == [15]
    assert not mask[1].any()
    assert not repeat_unlikelihood_mask(torch.tensor([[1, 2, 3, 4]] * 3), 0, 0).any()


@pytest.mark.parametrize("changes", [
    {"prefix_length": -1}, {"prefix_length": 5}, {"prefix_length": True}, {"prefix_length": [1, 2]},
    {"ngram_size": 0}, {"ngram_size": 1.5}, {"min_occurrences": 1}, {"eos_token_id": -1},
])
def test_invalid_mask_contracts_are_rejected(changes):
    args = {"prefix_length": 0, "eos_token_id": 0, **changes}
    with pytest.raises(ValueError):
        repeat_unlikelihood_mask(torch.tensor([1, 2, 3, 4]), **args)


def test_loss_matches_independent_probability_formula_and_causal_alignment():
    scores = torch.tensor([[1., 2., 3.], [3., 1., 2.], [2., 3., 1.], [4., 2., 1.]],
                          dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([0, 2, 1, 0])
    mask = torch.tensor([False, True, False, True])
    loss = sequence_unlikelihood_loss(scores, ids, mask)
    probabilities = scores.softmax(-1)
    expected = (-torch.log1p(-probabilities[0, 2]) - torch.log1p(-probabilities[2, 0])) / 2
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(loss, scores, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, scores)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)
    assert not actual_grad[1].any() and not actual_grad[3].any()
    assert actual_grad[0, 2] > 0  # Gradient descent reduces the repeated target logit.
    assert actual_grad[0, 0] < 0


def test_loss_denominator_is_actual_negative_positions_not_rows_or_length():
    logits = torch.zeros(2, 4, 2, dtype=torch.float64, requires_grad=True)
    logits.data[0, 0] = torch.tensor([0., math.log(3)])
    ids = torch.ones(2, 4, dtype=torch.long)
    mask = torch.tensor([[False, True, False, False], [False, True, True, True]])
    loss = sequence_unlikelihood_loss(logits, ids, mask)
    assert loss.item() == pytest.approx((math.log(4) + 3 * math.log(2)) / 4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_near_probability_one_has_finite_loss_and_useful_finite_gradients(dtype):
    logits = torch.tensor([[0., 100., -10.], [0., 0., 0.]], dtype=dtype, requires_grad=True)
    ids, mask = torch.tensor([0, 1]), torch.tensor([False, True])
    assert logits[0].float().softmax(-1)[1].item() == 1.0
    loss = sequence_unlikelihood_loss(logits, ids, mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert loss.item() == pytest.approx(100., abs=.001)
    assert logits.grad[0, 1].item() == pytest.approx(1.)
    assert logits.grad[0, 0].item() < -.99
    assert not logits.grad[1].any()
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)


@pytest.mark.parametrize("length", [0, 1, 8])
def test_no_negatives_returns_graph_connected_zero_and_zero_gradients(length):
    logits = torch.randn(2, length, 5, requires_grad=True)
    ids = torch.zeros(2, length, dtype=torch.long)
    loss = sequence_unlikelihood_loss(logits, ids, torch.zeros_like(ids, dtype=torch.bool))
    assert loss.requires_grad and loss.item() == 0
    loss.backward()
    assert logits.grad is not None and not logits.grad.any()


def test_gradient_check_double_precision():
    scores = torch.randn(1, 5, 4, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 1, 2, 3, 0]])
    mask = torch.tensor([[False, False, True, False, True]])
    assert torch.autograd.gradcheck(lambda values: sequence_unlikelihood_loss(values, ids, mask), (scores,))


def test_rejects_invalid_loss_alignment_and_nonfinite_selected_logits():
    logits = torch.zeros(2, 4, 3, requires_grad=True)
    ids = torch.ones(2, 4, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    with pytest.raises(ValueError, match="boolean"):
        sequence_unlikelihood_loss(logits, ids, mask.long())
    with pytest.raises(ValueError, match="shape"):
        sequence_unlikelihood_loss(logits[:, :-1], ids, mask)
    mask[:, 0] = True
    with pytest.raises(ValueError, match="position zero"):
        sequence_unlikelihood_loss(logits, ids, mask)
    mask[:, 0], mask[:, 1] = False, True
    bad_ids = ids.clone()
    bad_ids[:, 1] = 3
    with pytest.raises(ValueError, match="vocabulary"):
        sequence_unlikelihood_loss(logits, bad_ids, mask)
    invalid = logits.detach().clone()
    invalid[0, 0, 0] = torch.inf
    with pytest.raises(FloatingPointError, match="not finite"):
        sequence_unlikelihood_loss(invalid, ids, mask)


def test_objective_never_modifies_logits_ids_or_mask_in_place():
    logits = torch.randn(1, 20, 6, requires_grad=True)
    ids = torch.tensor([[5, 5, 5, 5] + [1, 2, 3, 4] * 4])
    mask = repeat_unlikelihood_mask(ids, 4, 0)
    snapshots = [value.detach().clone() for value in (logits, ids, mask)]
    sequence_unlikelihood_loss(logits, ids, mask).backward()
    for actual, expected in zip((logits, ids, mask), snapshots, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
