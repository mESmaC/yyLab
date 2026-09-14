"""Experimental unlikelihood objective for self-generated repetition.

Independently implemented from the objective in Welleck et al., *Neural Text
Generation with Unlikelihood Training*, https://arxiv.org/abs/1908.04319.
This is a conservative variant, not a faithful replication of that paper: an
exact ngram must occur three times inside the generated suffix before only its
final token becomes a negative. Prompt text and tokens at/after generated EOS
are excluded. Positive cross entropy on real corpus examples remains the
caller's separate objective; these helpers do not train, generate or decode.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import torch
import torch.nn.functional as F

UNLIKELIHOOD_VARIANT = "generated-ngram-third-occurrence-final-token-v1"
_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = int(value)
        valid = result == float(value) and result >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _batched_ids(input_ids: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not isinstance(input_ids, torch.Tensor) or input_ids.dtype not in _INTEGER_DTYPES:
        raise ValueError("input_ids must be an integer tensor")
    if input_ids.ndim not in {1, 2}:
        raise ValueError("input_ids must have shape [time] or [batch, time]")
    single = input_ids.ndim == 1
    return (input_ids.unsqueeze(0) if single else input_ids), single


def repeat_unlikelihood_mask(
    input_ids: torch.Tensor,
    prefix_length: int | Sequence[int] | torch.Tensor,
    eos_token_id: int | Sequence[int] | None,
    ngram_size: int = 4,
    min_occurrences: int = 3,
) -> torch.Tensor:
    """Return a boolean negative mask aligned to input token positions.

    The ngram ending at position ``t`` must lie entirely after the prefix and
    have appeared ``min_occurrences`` times, including this occurrence. Only
    ``t`` is masked, not the whole ngram. Overlapping occurrences count, and
    counting is independent for each row. The first EOS *in the generated
    suffix* terminates counting; EOS separators inside a prompt do not.

    Prefix length can be one scalar for the batch or one length per row. This
    bounded diagnostic uses a CPU copy for exact tuple counting and returns the
    mask on the input device. Do not use it as a corpus repetition filter.
    """
    ids, single = _batched_ids(input_ids)
    ngram_size = _integer(ngram_size, "ngram_size", 1)
    min_occurrences = _integer(min_occurrences, "min_occurrences", 2)
    batch, length = ids.shape
    if isinstance(prefix_length, torch.Tensor):
        prefix_length = prefix_length.detach().cpu().tolist()
    if isinstance(prefix_length, Sequence) and not isinstance(prefix_length, (str, bytes)):
        if len(prefix_length) != batch:
            raise ValueError("prefix_length must contain one length per batch row")
        prefixes = [_integer(value, "prefix_length", 0) for value in prefix_length]
    else:
        prefixes = [_integer(prefix_length, "prefix_length", 0)] * batch
    if any(prefix > length for prefix in prefixes):
        raise ValueError("prefix_length cannot exceed the input length")
    if eos_token_id is None:
        eos = set()
    elif isinstance(eos_token_id, Sequence) and not isinstance(eos_token_id, (str, bytes)):
        eos = {_integer(value, "eos_token_id", 0) for value in eos_token_id}
    else:
        eos = {_integer(eos_token_id, "eos_token_id", 0)}
    result = torch.zeros(ids.shape, dtype=torch.bool)
    for row_index, (row, prefix) in enumerate(zip(ids.detach().cpu().tolist(), prefixes, strict=True)):
        counts: Counter[tuple[int, ...]] = Counter()
        for position in range(prefix, length):
            if row[position] in eos:
                break
            start = position - ngram_size + 1
            if start < prefix:
                continue
            ngram = tuple(row[start:position + 1])
            counts[ngram] += 1
            if counts[ngram] >= min_occurrences:
                result[row_index, position] = True
    result = result.to(device=input_ids.device)
    return result.squeeze(0) if single else result


def sequence_unlikelihood_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, negative_mask: torch.Tensor,
) -> torch.Tensor:
    """Average ``-log(1 - p(token[t] | prefix[:t])))`` over negative positions.

    ``logits`` has shape [batch, time, vocabulary] (or [time, vocabulary] for
    one unbatched input). Its position t-1 predicts input token t: the shift is
    applied exactly once here. ``negative_mask`` has the input shape; it must
    not mark position zero, which has no preceding prediction.

    Numerically, -log(1-p) equals
    ``softplus(target_logit - logsumexp(all_other_logits))``. Computing the
    complement this way avoids cancellation when softmax rounds p to one and
    avoids a probability clamp's zero-gradient plateau. FP16/BF16/FP32 logits
    accumulate in FP32, while FP64 is preserved for numerical verification.
    A vocabulary of at least two tokens and finite selected logits are required.
    With no negatives, an empty differentiable sum returns zero with zero grads.
    """
    ids, single = _batched_ids(input_ids)
    if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
            or logits.ndim != input_ids.ndim + 1 or logits.shape[:-1] != input_ids.shape):
        raise ValueError("logits must be floating point with shape input_ids.shape + [vocabulary]")
    if logits.shape[-1] < 2:
        raise ValueError("unlikelihood requires at least two vocabulary tokens")
    if (not isinstance(negative_mask, torch.Tensor) or negative_mask.dtype != torch.bool
            or negative_mask.shape != input_ids.shape):
        raise ValueError("negative_mask must be a boolean tensor matching input_ids")
    if logits.device != input_ids.device or logits.device != negative_mask.device:
        raise ValueError("logits, input_ids and negative_mask must share a device")
    mask = negative_mask.unsqueeze(0) if single else negative_mask
    scores = logits.unsqueeze(0) if single else logits
    if ids.shape[1] and bool(mask[:, 0].any()):
        raise ValueError("negative_mask cannot mark position zero without a preceding logit")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    selected = scores[:, :-1, :][mask[:, 1:]].to(dtype=dtype)
    if not selected.shape[0]:
        return logits.reshape(-1)[:0].to(dtype=dtype).sum()
    targets = ids[:, 1:][mask[:, 1:]].long()
    if bool((targets < 0).any()) or bool((targets >= logits.shape[-1]).any()):
        raise ValueError("negative token ID is outside the logits vocabulary")
    if not bool(torch.isfinite(selected).all()):
        raise FloatingPointError("unlikelihood selected logits are not finite")
    target_scores = selected.gather(1, targets[:, None]).squeeze(1)
    others = selected.scatter(1, targets[:, None], -torch.inf)
    complement_scores = torch.logsumexp(others, dim=-1)
    return F.softplus(target_scores - complement_scores).mean()
