"""Native AntiDoom pair mining and Final Token Preference Optimization.

This is a clean adaptation of Liquid AI's Apache-2.0 ``Liquid4All/antidoom``
reference implementation.  yyLab keeps the detector and FTPO loss semantics,
but mines directly from Transformers token IDs so byte-level tokens are never
reconstructed from decoded text.

Upstream method: https://github.com/Liquid4All/antidoom
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

ANTIDOOM_METHOD = "Liquid AI AntiDoom / Final Token Preference Optimization"
ANTIDOOM_UPSTREAM_URL = "https://github.com/Liquid4All/antidoom"
ANTIDOOM_UPSTREAM_LICENSE = "Apache-2.0"


@dataclass(frozen=True)
class RepeatHit:
    """A verified consecutive repetition in decoded character space."""

    start: int
    end: int
    period: int
    repeats: int
    snippet: str

    @property
    def repeat_start(self) -> int:
        """Character offset at which the second copy begins."""

        return self.start + self.period


def _verify_repetition_at(
    text: str,
    start: int,
    period: int,
    *,
    min_repeats: int,
    min_total_repeated: int,
) -> RepeatHit | None:
    if period < 1 or start < 0 or start + period > len(text):
        return None
    pattern = text[start : start + period]
    repeats = 0
    end = start
    while end + period <= len(text) and text[end : end + period] == pattern:
        repeats += 1
        end += period
    previous = start - period
    while previous >= 0 and text[previous : previous + period] == pattern:
        repeats += 1
        start = previous
        previous -= period
    if repeats < min_repeats or repeats * period < min_total_repeated:
        return None
    snippet = pattern if len(pattern) <= 100 else pattern[:100] + "..."
    return RepeatHit(start=start, end=end, period=period, repeats=repeats, snippet=snippet)


def find_inner_repetition(
    text: str,
    *,
    min_repeats: int = 4,
    max_period: int = 1024,
    min_period: int = 1,
    min_total_repeated: int = 60,
    sample_len: int = 16,
    sample_interval: int = 128,
) -> RepeatHit | None:
    """Find a repeated inner span using Liquid AI's fingerprint verifier.

    The sampling makes detection essentially linear for long generations; a
    candidate is returned only after exact consecutive-repeat verification.
    """

    if min_repeats < 2:
        raise ValueError("min_repeats must be at least 2")
    if min_period < 1 or max_period < min_period:
        raise ValueError("period bounds must satisfy 1 <= min_period <= max_period")
    if min_total_repeated < 1 or sample_len < 1 or sample_interval < 1:
        raise ValueError("repeat lengths and sample interval must be positive")
    if not text or len(text) < min_total_repeated:
        return None

    stop = max(1, len(text) - sample_len + 1)
    for sample_pos in range(0, stop, sample_interval):
        fingerprint = text[sample_pos : sample_pos + sample_len]
        if len(fingerprint) < sample_len:
            continue
        after = text.find(fingerprint, sample_pos + sample_len)
        if after >= 0:
            period = after - sample_pos
            if min_period <= period <= max_period:
                hit = _verify_repetition_at(
                    text,
                    sample_pos,
                    period,
                    min_repeats=min_repeats,
                    min_total_repeated=min_total_repeated,
                )
                if hit is not None:
                    return hit
        before = text.rfind(fingerprint, 0, sample_pos)
        if before >= 0:
            period = sample_pos - before
            if min_period <= period <= max_period:
                hit = _verify_repetition_at(
                    text,
                    before,
                    period,
                    min_repeats=min_repeats,
                    min_total_repeated=min_total_repeated,
                )
                if hit is not None:
                    return hit
    return None


_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
}


def normalize_prompt_row(
    row: Mapping[str, Any],
    *,
    prompt_field: str | None = None,
) -> list[dict[str, str]]:
    """Normalize a prompt-only string, OpenAI row, or ShareGPT row.

    Multi-turn prompts may contain earlier assistant messages, but must finish
    with a user turn.  That invariant prevents a corpus answer from silently
    becoming part of the generation prompt.
    """

    if not isinstance(row, Mapping):
        raise TypeError("prompt row must be a mapping")
    if prompt_field is not None:
        if prompt_field not in row:
            raise ValueError(f"prompt field {prompt_field!r} is missing")
        value = row[prompt_field]
    else:
        present = [name for name in ("messages", "conversations", "prompt") if name in row]
        if not present:
            raise ValueError("prompt row needs messages, conversations, or prompt")
        value = row[present[0]]

    if isinstance(value, str):
        if not value.strip():
            raise ValueError("prompt text cannot be blank")
        return [{"role": "user", "content": value}]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("prompt value must be text or a message list")

    messages: list[dict[str, str]] = []
    for index, turn in enumerate(value):
        if not isinstance(turn, Mapping):
            raise ValueError(f"prompt turn {index} must be an object")
        if "role" in turn or "content" in turn:
            raw_role, content = turn.get("role"), turn.get("content")
        else:
            raw_role, content = turn.get("from"), turn.get("value")
        role = _ROLE_MAP.get(str(raw_role).strip().lower())
        if role is None:
            raise ValueError(f"prompt turn {index} has unsupported role {raw_role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"prompt turn {index} needs non-blank text")
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("prompt message list cannot be empty")
    if messages[-1]["role"] != "user":
        raise ValueError("prompt-only conversations must end with a user turn")
    return messages


def _unwrap_input_ids(value: Any) -> Any:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if isinstance(value, Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one rendered prompt")
        value = value[0]
    return value


def render_prompt_ids(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    prompt_field: str | None = None,
) -> list[int]:
    """Render a normalized row with the tokenizer's native chat template."""

    messages = normalize_prompt_row(row, prompt_field=prompt_field)
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise ValueError("AntiDoom prompt mining requires a tokenizer chat template")
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = _unwrap_input_ids(rendered)
    if not isinstance(ids, list) or not ids or not all(isinstance(item, int) for item in ids):
        raise ValueError("chat template did not return a non-empty token ID list")
    return ids


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    try:
        return str(tokenizer.decode(list(ids), **kwargs))
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces")
        return str(tokenizer.decode(list(ids), **kwargs))


def _stable_token_ends(token_ids: Sequence[int], tokenizer: Any) -> tuple[str, list[int]]:
    """Map decoded characters to IDs without encoding text back into tokens.

    Prefix decodes can shrink when byte-fallback fragments form one Unicode
    character.  Longest-common-prefix offsets attribute that character to the
    token that completes it and remain monotonic.
    """

    decoded = _decode(tokenizer, token_ids)
    ends: list[int] = []
    stable = 0
    for index in range(1, len(token_ids) + 1):
        prefix = _decode(tokenizer, token_ids[:index])
        common = 0
        for left, right in zip(prefix, decoded, strict=False):
            if left != right:
                break
            common += 1
        stable = max(stable, common)
        ends.append(stable)
    if ends:
        ends[-1] = len(decoded)
    return decoded, ends


def _char_to_token_index(ends: Sequence[int], char_pos: int) -> int | None:
    if char_pos < 0:
        return None
    for index, end in enumerate(ends):
        if char_pos < end:
            return index
    return None


def _single_token_surface(tokenizer: Any, token_id: int) -> str:
    return _decode(tokenizer, [int(token_id)])


def _boundary_only(surface: str) -> bool:
    return bool(surface) and "\ufffd" not in surface and not any(
        char.isalnum() or char == "_" for char in surface
    )


def select_rejected_token_index_token_aware(
    token_ids: Sequence[int],
    tokenizer: Any,
    hit: RepeatHit,
) -> int | None:
    """Refine a character repetition boundary to its direct generation ID.

    The boundary is slid left over an exact repeated token period, then harmless
    whitespace/punctuation tokens are skipped.  No decoded substring is ever
    re-tokenized.
    """

    if not token_ids:
        return None
    _decoded, ends = _stable_token_ends(token_ids, tokenizer)
    seed_index = _char_to_token_index(ends, hit.start)
    repeat_index = _char_to_token_index(ends, hit.repeat_start)
    if seed_index is None or repeat_index is None or repeat_index <= seed_index:
        return None
    period_tokens = repeat_index - seed_index
    while seed_index > 0:
        previous = seed_index - 1
        repeated = previous + period_tokens
        if repeated >= len(token_ids) or token_ids[previous] != token_ids[repeated]:
            break
        seed_index = previous
    target = seed_index + period_tokens
    while target < len(token_ids) and _boundary_only(
        _single_token_surface(tokenizer, int(token_ids[target]))
    ):
        target += 1
    return target if target < len(token_ids) else None


def _encode_without_specials(tokenizer: Any, text: str) -> list[int]:
    if callable(getattr(tokenizer, "encode", None)):
        encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    encoded = _unwrap_input_ids(encoded)
    return [int(item) for item in encoded]


def safe_single_token_surface(tokenizer: Any, token_id: int) -> str | None:
    """Return a display-safe, exact single-token surface or ``None``.

    This deliberately rejects replacement characters, byte fragments, control
    characters, punctuation-only alternatives, and tokenizer-normalized text.
    """

    surface = _single_token_surface(tokenizer, token_id)
    if not surface or "\ufffd" in surface or not surface.isprintable():
        return None
    if not any(char.isalnum() for char in surface):
        return None
    try:
        if _encode_without_specials(tokenizer, surface) != [int(token_id)]:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    if _single_token_surface(tokenizer, token_id) != surface:
        return None
    return surface


@dataclass(frozen=True)
class FTPOPair:
    """One local final-token preference mined without token round-tripping."""

    batch_index: int
    sequence_index: int
    return_index: int
    prompt_ids: tuple[int, ...]
    chosen_ids: tuple[int, ...]
    rejected_token_id: int
    source_prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    rejected_generation_index: int
    repetition: RepeatHit
    chosen_surfaces: tuple[str, ...]
    rejected_surface: str
    generation_text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready training/provenance row."""

        value = asdict(self)
        value["metadata"] = dict(self.metadata)
        return value


def _as_rank_two_long(name: str, value: Any) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a rank-two tensor")
    return tensor


def _trim_generated_ids(
    ids: Sequence[int],
    *,
    eos_token_id: int | Sequence[int] | None,
    pad_token_id: int | None,
) -> list[int]:
    eos_ids = {int(eos_token_id)} if isinstance(eos_token_id, int) else {
        int(item) for item in (eos_token_id or [])
    }
    trimmed: list[int] = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id in eos_ids:
            break
        if pad_token_id is not None and token_id == int(pad_token_id):
            break
        trimmed.append(token_id)
    return trimmed


def _chosen_alternatives(
    logits: Tensor,
    tokenizer: Any,
    rejected_id: int,
    *,
    max_chosen: int,
    candidate_pool: int,
    min_p: float | None,
) -> tuple[list[int], list[str]]:
    logits = logits.detach().float().cpu()
    finite = torch.isfinite(logits)
    if not finite.any():
        return [], []
    count = min(int(candidate_pool), logits.numel())
    values, ids = torch.topk(logits.masked_fill(~finite, -torch.inf), count)
    floor = -math.inf
    if min_p is not None:
        if not 0 <= min_p <= 1:
            raise ValueError("min_p must lie in [0, 1]")
        floor = float(values[0]) + math.log(max(float(min_p), torch.finfo(torch.float32).tiny))
    rejected_surface = _single_token_surface(tokenizer, rejected_id).casefold()
    chosen: list[int] = []
    surfaces: list[str] = []
    for value, token_id in zip(values.tolist(), ids.tolist(), strict=True):
        token_id = int(token_id)
        if value < floor:
            break
        if token_id == rejected_id:
            continue
        surface = safe_single_token_surface(tokenizer, token_id)
        if surface is None or surface.casefold() == rejected_surface:
            continue
        chosen.append(token_id)
        surfaces.append(surface)
        if len(chosen) >= max_chosen:
            break
    return chosen, surfaces


def mine_ftpo_pairs(
    input_ids: Any,
    attention_mask: Any,
    generation_output: Any,
    tokenizer: Any,
    *,
    source_metadata: Sequence[Mapping[str, Any]] | None = None,
    max_chosen: int = 20,
    candidate_pool: int = 128,
    min_p: float | None = 0.01,
    min_repeats: int = 4,
    max_period: int = 1024,
    min_period: int = 1,
    min_total_repeated: int = 60,
    sample_len: int = 16,
    sample_interval: int = 128,
) -> list[FTPOPair]:
    """Mine FTPO rows from a batched Transformers ``generate`` result.

    ``generation_output`` must contain ``sequences`` and per-step ``scores``
    (use ``return_dict_in_generate=True, output_scores=True``).  Beam-search
    ancestry is intentionally rejected; use greedy or sampled generation so
    each score row corresponds directly to each returned sequence.
    """

    if max_chosen < 1 or candidate_pool < max_chosen:
        raise ValueError("candidate_pool must be at least max_chosen >= 1")
    inputs = _as_rank_two_long("input_ids", input_ids)
    mask = _as_rank_two_long("attention_mask", attention_mask).bool()
    if inputs.shape != mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape")
    sequences_value = getattr(generation_output, "sequences", None)
    scores = getattr(generation_output, "scores", None)
    if sequences_value is None or scores is None:
        raise ValueError("generation output needs sequences and scores")
    sequences = _as_rank_two_long("generation sequences", sequences_value)
    scores = tuple(scores)
    if not scores:
        return []
    batch_size, input_width = inputs.shape
    if sequences.shape[0] % batch_size:
        raise ValueError("generation rows are not divisible by prompt batch size")
    returns_per_prompt = sequences.shape[0] // batch_size
    if getattr(generation_output, "beam_indices", None) is not None:
        raise ValueError("beam generation is unsupported; use greedy or sampled generation")
    for step, score in enumerate(scores):
        if score.ndim != 2 or score.shape[0] != sequences.shape[0]:
            raise ValueError(f"generation score step {step} does not align with returned sequences")
    if source_metadata is not None and len(source_metadata) != batch_size:
        raise ValueError("source_metadata must contain one mapping per input row")

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    pairs: list[FTPOPair] = []
    for sequence_index, sequence in enumerate(sequences):
        batch_index = sequence_index // returns_per_prompt
        return_index = sequence_index % returns_per_prompt
        source_prompt = inputs[batch_index][mask[batch_index]].tolist()
        generated = _trim_generated_ids(
            sequence[input_width : input_width + len(scores)].tolist(),
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        if not generated:
            continue
        generation_text, _ends = _stable_token_ends(generated, tokenizer)
        hit = find_inner_repetition(
            generation_text,
            min_repeats=min_repeats,
            max_period=max_period,
            min_period=min_period,
            min_total_repeated=min_total_repeated,
            sample_len=sample_len,
            sample_interval=sample_interval,
        )
        if hit is None:
            continue
        rejected_index = select_rejected_token_index_token_aware(generated, tokenizer, hit)
        if rejected_index is None or rejected_index >= len(scores):
            continue
        rejected_id = int(generated[rejected_index])
        chosen_ids, chosen_surfaces = _chosen_alternatives(
            scores[rejected_index][sequence_index],
            tokenizer,
            rejected_id,
            max_chosen=max_chosen,
            candidate_pool=candidate_pool,
            min_p=min_p,
        )
        if not chosen_ids:
            continue
        context_ids = tuple(int(item) for item in source_prompt + generated[:rejected_index])
        metadata = dict(source_metadata[batch_index]) if source_metadata is not None else {}
        pairs.append(
            FTPOPair(
                batch_index=batch_index,
                sequence_index=sequence_index,
                return_index=return_index,
                prompt_ids=context_ids,
                chosen_ids=tuple(chosen_ids),
                rejected_token_id=rejected_id,
                source_prompt_ids=tuple(int(item) for item in source_prompt),
                generated_ids=tuple(generated),
                rejected_generation_index=rejected_index,
                repetition=hit,
                chosen_surfaces=tuple(chosen_surfaces),
                rejected_surface=_single_token_surface(tokenizer, rejected_id),
                generation_text=generation_text,
                metadata=metadata,
            )
        )
    return pairs


@dataclass(frozen=True)
class FTPOResult:
    """Differentiable FTPO loss and detached monitoring metrics."""

    loss: Tensor
    preference_loss: Tensor
    untargeted_mse: Tensor
    target_excess_mse: Tensor
    chosen_win: Tensor
    margin_win: Tensor
    mean_delta: Tensor
    median_delta: Tensor
    active_weight: Tensor


def ftpo_objective(
    policy_logits: Tensor,
    reference_logits: Tensor,
    chosen_ids: Tensor,
    rejected_token_ids: Tensor,
    chosen_mask: Tensor | None = None,
    *,
    epsilon: float = 2.0,
    preference_temperature: float = 1.0,
    lambda_mse: float = 0.4,
    lambda_mse_target: float = 0.05,
    tau_mse_target: float = 0.5,
) -> FTPOResult:
    """Compute Liquid AI's local FTPO preference loss and logit tethers."""

    if policy_logits.ndim != 2 or reference_logits.shape != policy_logits.shape:
        raise ValueError("policy and reference logits must have matching [batch, vocab] shapes")
    if chosen_ids.ndim != 2 or chosen_ids.shape[0] != policy_logits.shape[0]:
        raise ValueError("chosen_ids must have shape [batch, choices]")
    if rejected_token_ids.shape != (policy_logits.shape[0],):
        raise ValueError("rejected_token_ids must have shape [batch]")
    if epsilon <= 0 or preference_temperature <= 0 or tau_mse_target < 0:
        raise ValueError("epsilon and preference_temperature must be positive; target tau nonnegative")
    if lambda_mse < 0 or lambda_mse_target < 0:
        raise ValueError("MSE weights cannot be negative")
    chosen_ids = chosen_ids.to(device=policy_logits.device)
    rejected_token_ids = rejected_token_ids.to(device=policy_logits.device)
    if chosen_mask is None:
        chosen_mask = torch.ones_like(chosen_ids, dtype=torch.bool)
    else:
        chosen_mask = chosen_mask.to(device=policy_logits.device, dtype=torch.bool)
        if chosen_mask.shape != chosen_ids.shape:
            raise ValueError("chosen_mask must match chosen_ids")
    if not chosen_mask.any(dim=1).all():
        raise ValueError("every FTPO row needs at least one chosen token")

    batch, vocab = policy_logits.shape
    safe_chosen = torch.where(chosen_mask, chosen_ids, torch.zeros_like(chosen_ids)).long()
    rejected = rejected_token_ids.long()
    active_chosen = safe_chosen[chosen_mask]
    if (active_chosen < 0).any() or (active_chosen >= vocab).any():
        raise ValueError("chosen token ID is outside the vocabulary")
    if (rejected < 0).any() or (rejected >= vocab).any():
        raise ValueError("rejected token ID is outside the vocabulary")
    rows = torch.arange(batch, device=policy_logits.device).unsqueeze(1)
    rejected_logits = policy_logits.gather(1, rejected.unsqueeze(1))
    delta = policy_logits[rows, safe_chosen] - rejected_logits
    gap = epsilon - delta
    weight = torch.clamp(gap / epsilon, min=0.0, max=1.0) * chosen_mask
    per_choice = F.softplus(gap / preference_temperature)
    counts = chosen_mask.sum(dim=1).clamp(min=1)
    preference_loss = ((per_choice * weight).sum(dim=1) / counts).mean()

    reference = reference_logits.detach().to(
        device=policy_logits.device,
        dtype=policy_logits.dtype,
    )
    difference = policy_logits - reference
    target_mask = torch.zeros_like(policy_logits, dtype=torch.bool)
    expanded_rows = rows.expand_as(safe_chosen)
    target_mask[expanded_rows[chosen_mask], safe_chosen[chosen_mask]] = True
    target_mask.scatter_(1, rejected.unsqueeze(1), True)
    untargeted_mask = ~target_mask
    if untargeted_mask.any():
        untargeted_mse = difference.square()[untargeted_mask].mean()
    else:
        untargeted_mse = policy_logits.new_zeros(())
    excess = (difference.abs() - tau_mse_target).clamp_min(0).square()
    target_excess_mse = excess[target_mask].mean()
    loss = (
        preference_loss
        + lambda_mse * untargeted_mse
        + lambda_mse_target * target_excess_mse
    )

    active_delta = delta[chosen_mask]
    active_weight = weight[chosen_mask]
    chosen_win = (
        (((delta > 0) & chosen_mask).float().sum(dim=1) / counts).mean().detach()
    )
    margin_win = (active_delta >= epsilon).float().mean().detach()
    return FTPOResult(
        loss=loss,
        preference_loss=preference_loss.detach(),
        untargeted_mse=untargeted_mse.detach(),
        target_excess_mse=target_excess_mse.detach(),
        chosen_win=chosen_win,
        margin_win=margin_win,
        mean_delta=active_delta.mean().detach(),
        median_delta=active_delta.median().detach(),
        active_weight=active_weight.mean().detach(),
    )
