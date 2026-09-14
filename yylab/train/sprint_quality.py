"""Bounded, read-only generation diagnostics for Sprint checkpoints.

These probes detect some conspicuous failures that teacher-forced loss misses.
Absence of repetition is *not* a quality score or permission to accept a model.
Prompt formats must be chosen from the actual training contract: a tokenizer
having a chat template does not establish that its markers were trained.
"""

from __future__ import annotations

import math
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np
import torch

QUALITY_SCHEMA = "sprint-generation-health-v1"
_FORMATS = {"plain", "inst", "roles", "tokenizer_chat"}
_CHECK_TYPES = {"exact_text", "integer_text", "exact_first_line", "integer_first_line"}
_LIMITATION = (
    "Short fixed probes are diagnostic examples, not benchmark accuracy. "
    "Low repetition and successful exact-answer probes do not establish usable model quality."
)


def default_sprint_quality_prompts() -> list[dict[str, Any]]:
    """Fresh prompt dictionaries; prose is always presented as plain completion."""
    return [
        {"name": "prose", "kind": "prose",
         "prompt": "The rain stopped just before sunrise. Mara opened the window and"},
        {"name": "instruction", "kind": "instruction",
         "prompt": "Write three short sentences about a rainy morning."},
        {"name": "copy", "kind": "instruction",
         "prompt": "Copy these two words exactly, with no explanation: blue bird",
         "check": {"type": "exact_text", "expected": "blue bird"}},
        {"name": "math", "kind": "instruction",
         "prompt": "Mia has 3 bags with 4 apples in each bag. She buys 2 more apples. "
                   "How many apples does she have? Answer with just the number.",
         "check": {"type": "integer_text", "expected": "14"}},
    ]


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    try:
        result = int(value)
        valid = result == float(value) and low <= result <= high
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return result


def normalize_sprint_quality_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize the nested ``generation_health`` config passed by a caller.

    Scheduling, checkpoint acceptance and stopping policy belong to the trainer.
    The evaluator never trains, saves, stops a run, or rewrites a chat template.
    """
    supplied = dict(config or {})
    defaults = {
        "max_new_tokens": 64, "max_prompt_tokens": 128, "seed": 1729,
        "prompt_format": "plain", "modes": ["greedy", "sample"],
        "temperature": 1.0, "top_p": 1.0, "min_evidence_tokens": 24,
        "repeat_ngram_size": 4, "repeat_fraction_threshold": 0.30,
        "min_tail_repeats": 3, "min_tail_tokens": 16, "max_tail_period": 16,
    }
    unknown = supplied.keys() - defaults.keys() - {"prompts"}
    if unknown:
        raise ValueError(f"unknown Sprint generation-health options: {sorted(unknown)}")
    cfg = defaults | supplied
    for key, low, high in (
        ("max_new_tokens", 1, 96), ("max_prompt_tokens", 1, 1024), ("seed", 0, 2**32 - 1),
        ("min_evidence_tokens", 8, 96), ("repeat_ngram_size", 2, 16),
        ("min_tail_repeats", 3, 32), ("min_tail_tokens", 8, 96), ("max_tail_period", 1, 32),
    ):
        cfg[key] = _integer(cfg[key], key, low, high)
    for key, low, high in (("temperature", 0.0, 5.0), ("top_p", 0.0, 1.0),
                           ("repeat_fraction_threshold", 0.0, 1.0)):
        value = float(cfg[key])
        if not math.isfinite(value) or not low < value <= high:
            raise ValueError(f"{key} must be finite and in ({low}, {high}]")
        cfg[key] = value
    if cfg["prompt_format"] not in _FORMATS:
        raise ValueError(f"prompt_format must be one of {sorted(_FORMATS)}")
    if (not isinstance(cfg["modes"], (list, tuple)) or not cfg["modes"]
            or any(mode not in {"greedy", "sample"} for mode in cfg["modes"])
            or len(set(cfg["modes"])) != len(cfg["modes"])):
        raise ValueError("modes must be a nonempty, nonduplicated list of greedy and/or sample")
    cfg["modes"] = list(cfg["modes"])
    prompts = supplied.get("prompts", default_sprint_quality_prompts())
    if not isinstance(prompts, (list, tuple)) or not 1 <= len(prompts) <= 8:
        raise ValueError("prompts must contain between one and eight prompt objects")
    cfg["prompts"] = []
    names = set()
    for index, raw in enumerate(prompts):
        if not isinstance(raw, Mapping):
            raise ValueError("each prompt must be an object")
        prompt = dict(raw)
        if prompt.keys() - {"name", "kind", "prompt", "check"}:
            raise ValueError("prompt supports only name, kind, prompt and check")
        prompt.setdefault("name", f"probe_{index}")
        prompt.setdefault("kind", "prose")
        if (not isinstance(prompt["name"], str) or not prompt["name"] or prompt["name"] in names
                or not isinstance(prompt.get("prompt"), str) or not prompt["prompt"].strip()
                or prompt["kind"] not in {"prose", "instruction"}):
            raise ValueError("prompts require unique names, nonempty text and a prose/instruction kind")
        names.add(prompt["name"])
        if prompt.get("check") is not None:
            check = dict(prompt["check"])
            if (check.keys() != {"type", "expected"}
                    or check["type"] not in _CHECK_TYPES
                    or not isinstance(check["expected"], str) or not check["expected"].strip()):
                raise ValueError(f"check requires a type from {sorted(_CHECK_TYPES)} and expected text")
            if check["type"] in {"integer_text", "integer_first_line"} and not re.fullmatch(r"[+-]?\d+", check["expected"]):
                raise ValueError(f"{check['type']} expects an integer string")
            prompt["check"] = check
        cfg["prompts"].append(prompt)
    return cfg


def sprint_repetition_metrics(token_ids: Sequence[int], *, config: Mapping[str, Any] | None = None) -> dict:
    """Measure generated tokens only; the caller excludes a terminal EOS.

    Duplicate fraction = (ngram occurrences - distinct ngrams) / occurrences.
    A tail loop is an exact suffix containing >=3 consecutive copies of a unit.
    These thresholds are transparent heuristics, not calibrated acceptance bars.
    """
    cfg = normalize_sprint_quality_config(config)
    tokens = [int(token) for token in token_ids]
    length = len(tokens)
    ngrams = {}
    for size in sorted({2, 3, 4, 8, cfg["repeat_ngram_size"]}):
        total = max(0, length - size + 1)
        distinct = len({tuple(tokens[start:start + size]) for start in range(total)})
        ngrams[str(size)] = {
            "occurrences": total, "distinct": distinct, "duplicates": total - distinct,
            "duplicate_fraction": (total - distinct) / total if total else None,
        }
    best = {"period": 0, "repeats": 0, "tokens": 0, "fraction": 0.0, "unit_token_ids": []}
    for period in range(1, min(cfg["max_tail_period"], length // cfg["min_tail_repeats"]) + 1):
        unit = tokens[-period:]
        repeats = 1
        while (repeats + 1) * period <= length:
            if tokens[-(repeats + 1) * period:-repeats * period] != unit:
                break
            repeats += 1
        coverage = period * repeats
        if repeats >= cfg["min_tail_repeats"] and coverage > best["tokens"]:
            best = {"period": period, "repeats": repeats, "tokens": coverage,
                    "fraction": coverage / length, "unit_token_ids": unit}
    sufficient = length >= cfg["min_evidence_tokens"]
    fraction = ngrams[str(cfg["repeat_ngram_size"])]["duplicate_fraction"]
    reasons = []
    if sufficient and fraction is not None and fraction >= cfg["repeat_fraction_threshold"]:
        reasons.append("duplicate_ngrams")
    # A long exact tail is meaningful even if the output ends before the general
    # ngram evidence minimum. Report the short-output limitation independently.
    if best["tokens"] >= cfg["min_tail_tokens"]:
        reasons.append("repeated_tail")
    return {"num_tokens": length, "sufficient_length": sufficient,
            "ngrams": ngrams, "tail_loop": best, "flagged": bool(reasons), "reasons": reasons}


def _render_prompt(
    probe: Mapping[str, Any], prompt_format: str, tokenizer, *, chat_template: str | None = None,
) -> str:
    if chat_template is not None and (not isinstance(chat_template, str) or not chat_template.strip()):
        raise ValueError("chat_template must be a nonempty string when supplied")
    text = probe["prompt"]
    if probe["kind"] == "prose":
        return text
    if prompt_format == "plain":
        return f"Question: {text}\nAnswer:"
    if prompt_format == "inst":
        return f"[INST] {text} [/INST]"
    if prompt_format == "roles":
        return f"user: {text}\nassistant:"
    if chat_template is None and not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer_chat was requested but the tokenizer has no chat template")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
        **({"chat_template": chat_template} if chat_template is not None else {}),
    )


@contextmanager
def _preserve_inference_state(model, device: torch.device):
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            model.eval()
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _probe_check(text: str, check: Mapping[str, Any] | None) -> dict | None:
    if check is None:
        return None
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    expected = check["expected"]
    whole_text = check["type"] in {"exact_text", "integer_text"}
    observed = text.strip() if whole_text else first_line
    if check["type"] in {"integer_text", "integer_first_line"}:
        match = re.fullmatch(r"([+-]?\d+)[^\S\r\n]*[.!]?", observed)
        correct = match is not None and int(match.group(1)) == int(expected)
    elif whole_text:
        # Normalize surrounding and horizontal whitespace while keeping internal
        # line boundaries: an extra line must not disappear from an exact check.
        def normalize(value):
            return "\n".join(" ".join(line.split()) for line in value.strip().splitlines())

        correct = normalize(observed) == normalize(expected)
    else:
        correct = " ".join(observed.split()) == " ".join(expected.split())
    scope = ("whole generated completion excluding terminal EOS; extra text/lines fail; "
             "no reasoning or general-ability claim" if whole_text else
             "first nonempty line only; later text ignored; no reasoning or general-ability claim")
    return {**check, "observed_first_line": first_line, "observed_text": text, "correct": correct, "scope": scope}


def _sample_token(logits: torch.Tensor, cfg: dict, generator: torch.Generator) -> torch.Tensor:
    scores = logits.float() / cfg["temperature"]
    if cfg["top_p"] < 1.0:
        sorted_scores, indices = scores.sort(descending=True)
        cumulative = sorted_scores.softmax(-1).cumsum(-1)
        remove = cumulative > cfg["top_p"]
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
        selected = torch.multinomial(sorted_scores.softmax(-1), 1, generator=generator)
        return indices[selected]
    return torch.multinomial(scores.softmax(-1), 1, generator=generator)


def evaluate_sprint_generation(
    model, tokenizer, *, config: Mapping[str, Any] | None = None,
    context_factory: Callable[[], Any] | None = None,
    chat_template: str | None = None,
) -> dict[str, Any]:
    """Evaluate an eager Sprint model without touching its weights or optimizer.

    Pass ``engine.model`` (not its compiled wrapper) and a context factory for
    the desired attention/autocast settings. Every decode recomputes the full
    prefix with ``use_cache=False``; this supports both SDPA and native KDA and
    cannot consume or mutate a caller's cache. Prompt overflow is recorded as
    skipped rather than silently truncating a format or changing its meaning.
    A supplied ``chat_template`` binds tokenizer_chat probes to that exact
    template without changing the tokenizer. Rendering errors propagate; no
    alternate template is tried. Without it, explicit tokenizer_chat requests
    use the tokenizer's current template for experimental direct evaluation.

    Sampling uses a local per-output generator. Python/NumPy/Torch RNG and every
    submodule's train/eval mode are restored, including when inference raises.
    No penalties, ngram blocking, forced tokens or inherited GenerationConfig
    processors are applied. This function never declares a checkpoint usable.
    """
    cfg = normalize_sprint_quality_config(config)
    device = model.get_input_embeddings().weight.device
    context_length = int(model.config.max_position_embeddings)
    if context_length < 2:
        raise ValueError("generation diagnostics require at least two context positions")
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    eos_ids = {int(value) for value in eos} if isinstance(eos, (list, tuple, set)) else ({int(eos)} if eos is not None else set())
    records = []
    started = time.perf_counter()
    with _preserve_inference_state(model, device):
        for probe_index, probe in enumerate(cfg["prompts"]):
            prompt = _render_prompt(probe, cfg["prompt_format"], tokenizer, chat_template=chat_template)
            prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
            for mode in cfg["modes"]:
                seed = (cfg["seed"] + 2 * probe_index + int(mode == "sample")) % 2**32
                record = {"name": probe["name"], "kind": probe["kind"], "mode": mode,
                          "prompt_format": "plain_completion" if probe["kind"] == "prose" else cfg["prompt_format"],
                          "prompt": prompt, "prompt_token_ids": prompt_ids,
                          "settings": {"seed": seed, "max_new_tokens": cfg["max_new_tokens"],
                                       "temperature": cfg["temperature"] if mode == "sample" else None,
                                       "top_p": cfg["top_p"] if mode == "sample" else None,
                                       "top_k": 0, "repetition_penalty": 1.0,
                                       "no_repeat_ngram_size": 0, "use_cache": False,
                                       "add_special_tokens": False, "eos_token_ids": sorted(eos_ids)}}
                new_ids: list[int] = []
                top1, entropy = [], []
                finite = True
                allowed = min(cfg["max_new_tokens"], max(0, context_length - len(prompt_ids)))
                minimum = min(cfg["max_new_tokens"], cfg["min_evidence_tokens"])
                if not prompt_ids or len(prompt_ids) > cfg["max_prompt_tokens"] or allowed < minimum:
                    stop = ("empty_prompt" if not prompt_ids else "prompt_too_long"
                            if len(prompt_ids) > cfg["max_prompt_tokens"] or allowed == 0
                            else "insufficient_context")
                    record["status"] = "skipped"
                else:
                    stop = "context_limit" if allowed < cfg["max_new_tokens"] else "max_new_tokens"
                    record["status"] = "complete"
                    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with context_factory() if context_factory is not None else nullcontext():
                        autocast = torch.is_autocast_enabled(device.type)
                        record["settings"]["autocast_enabled"] = autocast
                        record["settings"]["autocast_dtype"] = str(torch.get_autocast_dtype(device.type)) if autocast else None
                        for _ in range(allowed):
                            output = model(input_ids=ids, use_cache=False, logits_to_keep=1, return_logits=True)
                            logits = output.logits[0, -1].float()
                            if not bool(torch.isfinite(logits).all()):
                                finite, stop, record["status"] = False, "nonfinite_logits", "failed"
                                break
                            probabilities = logits.softmax(-1)
                            top1.append(float(probabilities.max()))
                            entropy.append(float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum()))
                            token = logits.argmax().reshape(1) if mode == "greedy" else _sample_token(logits, cfg, generator)
                            token_id = int(token.item())
                            new_ids.append(token_id)
                            if token_id in eos_ids:
                                stop = "eos"
                                break
                            ids = torch.cat((ids, token.reshape(1, 1)), dim=1)
                scored_ids = new_ids[:-1] if new_ids and new_ids[-1] in eos_ids else new_ids
                text = tokenizer.decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                checked_text = tokenizer.decode(scored_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                record.update({"generated_token_ids": new_ids, "generated_text": text,
                               "available_new_tokens": allowed,
                               "stop_reason": stop, "raw_logits_finite": finite,
                               "raw_top1_probability_mean": sum(top1) / len(top1) if top1 else None,
                               "raw_entropy_mean": sum(entropy) / len(entropy) if entropy else None,
                               "repetition": sprint_repetition_metrics(scored_ids, config=cfg),
                               "check": (_probe_check(checked_text, probe.get("check"))
                                         if record["status"] == "complete" else None)})
                records.append(record)
    flagged = sum(record["repetition"]["flagged"] for record in records)
    nonfinite = sum(not record["raw_logits_finite"] for record in records)
    eligible = sum(record["repetition"]["sufficient_length"] for record in records)
    skipped = sum(record["status"] == "skipped" for record in records)
    checks = [record["check"] for record in records if record["check"] is not None]
    status = ("nonfinite_logits" if nonfinite else "repetition_detected" if flagged
              else "insufficient_length" if not eligible else "no_repetition_detected")
    return {"schema": QUALITY_SCHEMA, "config": cfg, "context_length": context_length,
            "device": str(device), "parameter_dtype": str(model.get_input_embeddings().weight.dtype),
            "elapsed_seconds": time.perf_counter() - started, "outputs": records,
            "summary": {"status": status, "outputs": len(records), "repetition_flagged_outputs": flagged,
                        "sufficient_length_outputs": eligible, "short_outputs": len(records) - eligible,
                        "skipped_outputs": skipped, "nonfinite_outputs": nonfinite,
                        "exact_answer_checks": len(checks), "exact_answer_passes": sum(check["correct"] for check in checks)},
            "limitations": _LIMITATION}
