"""Native AntiDoom mining and FTPO-LoRA worker.

The prompt mix is deliberately prompt-only.  This worker first samples the
selected checkpoint, mines the token that starts each verified repetition,
then performs Liquid AI-style Final Token Preference Optimization on a LoRA
adapter.  The source checkpoint is never modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
import traceback
from collections import Counter, deque
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import orjson
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .. import db
from ..artifacts import adapter_dir, run_dir, write_json
from ..models.loading import load_base
from ..train.antidoom import (
    FTPOPair,
    find_inner_repetition,
    ftpo_objective,
    mine_ftpo_pairs,
    normalize_prompt_row,
    render_prompt_ids,
)
from ..train.loop import attention_backend_context, make_optimizer

ANTIDOOM_RUN_KIND = "antidoom_ftpo"
ANTIDOOM_FORMAT = "yylab_antidoom_ftpo_v1"
DEFAULT_REASONING_SUFFIX = (
    "\n\nThink through the problem step by step, then respond with your final answer "
    'as "Answer: <your answer>".'
)
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
)
QUALITATIVE_CASES = (
    {
        "id": "copy",
        "messages": [
            {"role": "user", "content": "Reply with exactly the single word: blue."}
        ],
        "expect": {"kind": "exact", "value": "blue"},
    },
    {
        "id": "arithmetic",
        "messages": [
            {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
        ],
        "expect": {"kind": "exact", "value": "4"},
    },
    {
        "id": "antonym",
        "messages": [
            {
                "role": "user",
                "content": "Complete this with one word: The opposite of hot is",
            }
        ],
        "expect": {"kind": "exact", "value": "cold"},
    },
    {
        "id": "sentence",
        "messages": [
            {
                "role": "user",
                "content": "Write one grammatical sentence about a red lantern.",
            }
        ],
    },
    {
        "id": "knowledge",
        "messages": [
            {
                "role": "user",
                "content": "What planet do humans live on? Answer in one short sentence.",
            }
        ],
        "expect": {"kind": "contains", "value": "earth"},
    },
    {
        "id": "code",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python function named add that returns the sum of a and b."
                ),
            }
        ],
        "expect": {"kind": "contains", "value": "def add"},
    },
    {
        "id": "addition_story",
        "messages": [
            {
                "role": "user",
                "content": (
                    "A box has three red balls and two blue balls. How many balls are "
                    "there? Respond with only the numeral."
                ),
            }
        ],
        "expect": {"kind": "exact", "value": "5"},
    },
    {
        "id": "comparison",
        "messages": [
            {
                "role": "user",
                "content": "Which is larger, 9 or 4? Respond with only the larger numeral.",
            }
        ],
        "expect": {"kind": "exact", "value": "9"},
    },
    {
        "id": "grammar",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Correct the grammar and return only the corrected sentence: "
                    "The cat are sleeping."
                ),
            }
        ],
        "expect": {"kind": "exact", "value": "The cat is sleeping."},
    },
    {
        "id": "exact_copy",
        "messages": [
            {
                "role": "user",
                "content": "Repeat exactly and nothing else: cobalt fox 27",
            }
        ],
        "expect": {"kind": "exact", "value": "cobalt fox 27"},
    },
    {
        "id": "json",
        "messages": [
            {
                "role": "user",
                "content": 'Return exactly this JSON and nothing else: {"ok":true}',
            }
        ],
        "expect": {"kind": "json", "value": {"ok": True}},
    },
    {
        "id": "sequence",
        "messages": [
            {
                "role": "user",
                "content": (
                    "List the first five positive even numbers once each, "
                    "comma-separated, and stop."
                ),
            }
        ],
        "expect": {"kind": "exact", "value": "2, 4, 6, 8, 10"},
    },
    {
        "id": "farmer",
        "messages": [
            {
                "role": "user",
                "content": (
                    "A farmer has 17 sheep and all but 9 leave. How many remain? "
                    "Respond with only the numeral."
                ),
            }
        ],
        "expect": {"kind": "exact", "value": "9"},
    },
    {
        "id": "legitimate_repetition",
        "messages": [
            {"role": "user", "content": "Output exactly: A A A A"}
        ],
        "expect": {"kind": "exact", "value": "A A A A"},
    },
    {
        "id": "doom_stress",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Continue with one new sentence and do not repeat the existing "
                    "phrase: Wait, let me reconsider. Wait, let me reconsider."
                ),
            }
        ],
    },
)


def log(message: str) -> None:
    print(message, flush=True)


def _append_metric(rd: Path, metric: dict[str, object]) -> None:
    with (rd / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metric, ensure_ascii=True, allow_nan=False) + "\n")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2))
    temporary.replace(path)


def _write_progress(
    rd: Path,
    *,
    stage: str,
    step: int,
    total: int,
    value: float,
    label: str,
    message: str,
) -> None:
    write_json(
        rd / "progress.json",
        {
            "stage": stage,
            "step": int(step),
            "max_steps": int(total),
            "value": round(float(value), 8),
            "metric_label": label,
            "loss": round(float(value), 8),
            "msg": message,
        },
    )


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    required = ("checkpoint_id", "checkpoint_ref", "adapter_id", "prompt_jsonl")
    missing = [name for name in required if not str(cfg.get(name) or "").strip()]
    if missing:
        raise ValueError(f"AntiDoom config is missing: {', '.join(missing)}")
    source = Path(str(cfg["prompt_jsonl"])).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"AntiDoom prompt JSONL does not exist: {source}")
    cfg["prompt_jsonl"] = str(source)
    cfg.setdefault("prompt_field", "conversations")
    cfg.setdefault("reasoning_suffix", DEFAULT_REASONING_SUFFIX)
    cfg.setdefault("target_pairs", 8_000)
    cfg.setdefault("max_generated_prompts", 60_000)
    cfg.setdefault("generation_batch_size", 256)
    cfg.setdefault("max_prompt_tokens", 512)
    cfg.setdefault("max_new_tokens", 128)
    cfg.setdefault("candidate_pool", 128)
    cfg.setdefault("max_chosen", 20)
    cfg.setdefault("min_p", 0.01)
    cfg.setdefault("sample_interval", 32)
    cfg.setdefault("eval_prompts", 512)
    cfg.setdefault("nll_eval_prompts", 256)
    cfg.setdefault("nll_batch_size", 8)
    cfg.setdefault("eval_modulus", 10)
    cfg.setdefault("eval_bucket", 0)
    cfg.setdefault("max_train_examples", math.floor(int(cfg["target_pairs"]) * 0.7))
    cfg.setdefault("rejected_regularisation_strength", 0.3)
    cfg.setdefault("chosen_regularisation_strength", 0.5)
    cfg.setdefault("rank", 64)
    cfg.setdefault("alpha", 64)
    cfg.setdefault("dropout", 0.0)
    cfg.setdefault("target_modules", list(TARGET_MODULES))
    cfg.setdefault("micro_batch", 64)
    cfg.setdefault("grad_accum", 1)
    cfg.setdefault("epochs", 1)
    cfg.setdefault("learning_rate", 1.5e-5)
    cfg.setdefault("weight_decay", 0.01)
    cfg.setdefault("warmup_ratio", 0.1)
    cfg.setdefault("max_grad_norm", 2.5)
    cfg.setdefault("early_stopping_chosen_win", 0.4)
    cfg.setdefault("early_stopping_window", 5)
    cfg.setdefault("min_training_steps", 5)
    cfg.setdefault("log_every", 5)
    cfg.setdefault("qualitative_max_new_tokens", 128)
    cfg.setdefault("qualitative_temperature", 0.7)
    cfg.setdefault("qualitative_top_p", 0.9)
    cfg.setdefault("seed", 3407)
    positive = (
        "target_pairs",
        "max_generated_prompts",
        "generation_batch_size",
        "max_prompt_tokens",
        "max_new_tokens",
        "candidate_pool",
        "max_chosen",
        "eval_prompts",
        "nll_eval_prompts",
        "nll_batch_size",
        "eval_modulus",
        "max_train_examples",
        "rank",
        "alpha",
        "micro_batch",
        "grad_accum",
        "epochs",
        "early_stopping_window",
        "log_every",
        "qualitative_max_new_tokens",
    )
    for name in positive:
        if int(cfg[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= int(cfg["eval_bucket"]) < int(cfg["eval_modulus"]):
        raise ValueError("eval_bucket must lie within eval_modulus")
    if int(cfg["candidate_pool"]) < int(cfg["max_chosen"]):
        raise ValueError("candidate_pool cannot be smaller than max_chosen")
    if not 0 < float(cfg["min_p"]) <= 1:
        raise ValueError("min_p must lie within (0, 1]")
    if not 0 < float(cfg["early_stopping_chosen_win"]) <= 1:
        raise ValueError("chosen_win early-stop threshold must lie within (0, 1]")
    if float(cfg["qualitative_temperature"]) <= 0:
        raise ValueError("qualitative_temperature must be positive")
    if not 0 < float(cfg["qualitative_top_p"]) <= 1:
        raise ValueError("qualitative_top_p must lie within (0, 1]")
    return cfg


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=8 * 2**20) as stream:
        while chunk := stream.read(8 * 2**20):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_payload_manifest(reference: str) -> dict[str, Any]:
    path = Path(reference)
    if not path.exists():
        return {"kind": "model_reference", "reference": reference}
    if path.is_file():
        return {
            "kind": "local_file",
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    files = []
    for payload in sorted(item for item in path.rglob("*") if item.is_file()):
        files.append(
            {
                "path": payload.relative_to(path).as_posix(),
                "size": payload.stat().st_size,
                "sha256": _sha256_file(payload),
            }
        )
    return {
        "kind": "local_directory",
        "path": str(path.resolve()),
        "files": files,
    }


def _provenance_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    prompt_path = Path(str(cfg["prompt_jsonl"]))
    canonical_config = orjson.dumps(cfg, option=orjson.OPT_SORT_KEYS)
    return {
        "format": ANTIDOOM_FORMAT,
        "config_sha256": hashlib.sha256(canonical_config).hexdigest(),
        "prompt_source": {
            "path": str(prompt_path),
            "size": prompt_path.stat().st_size,
            "sha256": _sha256_file(prompt_path),
        },
        "checkpoint": _checkpoint_payload_manifest(str(cfg["checkpoint_ref"])),
    }


def _verify_or_write_provenance(rd: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    path = rd / "source_manifest.json"
    current = _provenance_manifest(cfg)
    if path.exists():
        recorded = orjson.loads(path.read_bytes())
        if recorded != current:
            raise RuntimeError(
                "AntiDoom source/config drifted since this run began; refusing unsafe resume"
            )
    else:
        _write_json_atomic(path, current)
    return current


def _source_key(row: dict[str, Any], raw_line: bytes) -> str:
    for name in ("id", "source_sample_id", "sample_id"):
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return hashlib.sha256(raw_line).hexdigest()


def _is_eval_row(key: str, cfg: dict[str, Any]) -> bool:
    digest = hashlib.sha256(key.encode("utf-8", errors="replace")).digest()
    bucket = int.from_bytes(digest[:8], "big") % int(cfg["eval_modulus"])
    return bucket == int(cfg["eval_bucket"])


def _render_prompt(
    row: dict[str, Any],
    tokenizer: Any,
    cfg: dict[str, Any],
) -> list[int]:
    messages = normalize_prompt_row(row, prompt_field=str(cfg["prompt_field"]))
    suffix = str(cfg.get("reasoning_suffix") or "")
    if suffix:
        messages = [dict(message) for message in messages]
        messages[-1]["content"] += suffix
    ids = render_prompt_ids(
        {"messages": messages},
        tokenizer,
        prompt_field="messages",
    )
    if not ids:
        raise ValueError("AntiDoom chat template rendered an empty prompt")
    return ids


def _iter_prompt_rows(path: Path) -> Iterator[tuple[dict[str, Any], bytes]]:
    with path.open("rb", buffering=4 * 2**20) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                value = orjson.loads(raw_line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: prompt row must be an object")
            yield value, raw_line


def _prompt_metadata(row: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "source_key": key,
        "source_dataset": str(row.get("source") or row.get("source_dataset") or ""),
        "source_config": str(row.get("source_config") or ""),
        "source_split": str(row.get("source_split") or ""),
    }


def _collect_eval_prompts(
    cfg: dict[str, Any],
    tokenizer: Any,
) -> tuple[list[list[int]], dict[str, int]]:
    prompts: list[list[int]] = []
    stats = Counter()
    for row, raw_line in _iter_prompt_rows(Path(cfg["prompt_jsonl"])):
        key = _source_key(row, raw_line)
        if not _is_eval_row(key, cfg):
            continue
        stats["eval_rows_seen"] += 1
        try:
            ids = _render_prompt(row, tokenizer, cfg)
        except (KeyError, TypeError, ValueError):
            stats["unrenderable"] += 1
            continue
        if len(ids) > int(cfg["max_prompt_tokens"]):
            stats["overlength"] += 1
            continue
        prompts.append(ids)
        if len(prompts) >= int(cfg["eval_prompts"]):
            break
    if not prompts:
        raise RuntimeError("AntiDoom held-out partition produced no eligible prompts")
    stats["eligible"] = len(prompts)
    return prompts, dict(stats)


def _left_pad(
    rows: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    width = max(len(row) for row in rows)
    input_ids = torch.full(
        (len(rows), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        values = torch.tensor(row, dtype=torch.long, device=device)
        input_ids[index, -len(row) :] = values
        attention_mask[index, -len(row) :] = 1
    return input_ids, attention_mask


def _model_device(model: nn.Module) -> torch.device:
    embeddings = model.get_input_embeddings()
    return embeddings.weight.device


def _generate_batch(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[Sequence[int]],
    cfg: dict[str, Any],
    *,
    output_scores: bool,
) -> tuple[Tensor, Tensor, Any]:
    device = _model_device(model)
    input_ids, attention_mask = _left_pad(
        prompts,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    with torch.inference_mode(), attention_backend_context(
        model,
        device,
        enabled=device.type == "cuda",
    ):
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(cfg["max_new_tokens"]),
            do_sample=False,
            use_cache=True,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            return_dict_in_generate=True,
            output_scores=output_scores,
        )
    return input_ids, attention_mask, output


def _continuation_ids(output: Any, input_width: int, tokenizer: Any) -> list[list[int]]:
    sequences = output.sequences.detach().cpu()
    continuations: list[list[int]] = []
    stop_ids = {int(tokenizer.eos_token_id), int(tokenizer.pad_token_id)}
    for sequence in sequences:
        values: list[int] = []
        for value in sequence[input_width:].tolist():
            token_id = int(value)
            if token_id in stop_ids:
                break
            values.append(token_id)
        continuations.append(values)
    return continuations


def _longest_token_run(token_ids: Sequence[int]) -> int:
    longest = 0
    current = 0
    previous: int | None = None
    for value in token_ids:
        token_id = int(value)
        current = current + 1 if token_id == previous else 1
        longest = max(longest, current)
        previous = token_id
    return longest


def _repeated_ngram_fraction(token_ids: Sequence[int], width: int) -> float:
    if width < 1 or len(token_ids) < width:
        return 0.0
    ngrams = [tuple(token_ids[index : index + width]) for index in range(len(token_ids) - width + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def _expectation_pass(text: str, expectation: dict[str, Any] | None) -> bool | None:
    if not expectation:
        return None
    kind = str(expectation.get("kind") or "")
    value = expectation.get("value")
    stripped = text.strip()
    if kind == "exact":
        return stripped.casefold() == str(value).strip().casefold()
    if kind == "contains":
        return str(value).casefold() in stripped.casefold()
    if kind == "json":
        try:
            return json.loads(stripped) == value
        except (TypeError, ValueError):
            return False
    raise ValueError(f"unsupported qualitative expectation kind: {kind!r}")


def _decoded_diagnostics(
    continuation: Sequence[int],
    tokenizer: Any,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    text = tokenizer.decode(
        list(continuation),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    hit = find_inner_repetition(text, sample_interval=int(cfg["sample_interval"]))
    return {
        "text": text,
        "token_ids": [int(value) for value in continuation],
        "token_count": len(continuation),
        "distinct_token_ratio": len(set(continuation)) / max(1, len(continuation)),
        "repeated_bigram_fraction": _repeated_ngram_fraction(continuation, 2),
        "repeated_trigram_fraction": _repeated_ngram_fraction(continuation, 3),
        "repeated_4gram_fraction": _repeated_ngram_fraction(continuation, 4),
        "longest_identical_token_run": _longest_token_run(continuation),
        "replacement_characters": text.count("\ufffd"),
        "empty": not bool(text.strip()),
        "loop": hit is not None,
        "repeat": (
            {
                "period": hit.period,
                "repeats": hit.repeats,
                "snippet": hit.snippet,
                "start_character": hit.start,
                "end_character": hit.end,
                "character_coverage": (hit.end - hit.start) / max(1, len(text)),
            }
            if hit is not None
            else None
        ),
    }


def evaluate_causal_nll(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[Sequence[int]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    selected = prompts[: int(cfg["nll_eval_prompts"])]
    batch_size = int(cfg["nll_batch_size"])
    device = _model_device(model)
    loss_sum = 0.0
    token_count = 0
    started = time.perf_counter()
    for offset in range(0, len(selected), batch_size):
        input_ids, attention_mask = _left_pad(
            selected[offset : offset + batch_size],
            pad_token_id=int(tokenizer.pad_token_id),
            device=device,
        )
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        with torch.inference_mode(), attention_backend_context(
            model,
            device,
            enabled=device.type == "cuda",
        ):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            ).logits
        valid = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
        if not valid.any():
            continue
        chosen_logits = logits[:, :-1, :][valid].float()
        targets = input_ids[:, 1:][valid]
        loss_sum += float(F.cross_entropy(chosen_logits, targets, reduction="sum"))
        token_count += int(targets.numel())
        del logits, chosen_logits, targets
    if token_count < 1:
        raise RuntimeError("AntiDoom held-out NLL probe produced no target tokens")
    nll = loss_sum / token_count
    elapsed = time.perf_counter() - started
    return {
        "prompts": len(selected),
        "tokens": token_count,
        "nll": nll,
        "perplexity": math.exp(min(nll, 50.0)),
        "elapsed_s": round(elapsed, 3),
    }


def evaluate_qualitative(
    model: nn.Module,
    tokenizer: Any,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    prompts = [
        render_prompt_ids({"messages": case["messages"]}, tokenizer, prompt_field="messages")
        for case in QUALITATIVE_CASES
    ]
    device = _model_device(model)
    input_ids, attention_mask = _left_pad(
        prompts,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    evaluations: dict[str, list[dict[str, Any]]] = {}
    modes = (("greedy", False), ("sampled", True))
    for mode_index, (mode, sampled) in enumerate(modes):
        seed = int(cfg["seed"]) + 10_000 + mode_index
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(cfg["qualitative_max_new_tokens"]),
            "do_sample": sampled,
            "use_cache": True,
            "pad_token_id": int(tokenizer.pad_token_id),
            "eos_token_id": int(tokenizer.eos_token_id),
            "return_dict_in_generate": True,
        }
        if sampled:
            generation_kwargs.update(
                temperature=float(cfg["qualitative_temperature"]),
                top_p=float(cfg["qualitative_top_p"]),
            )
        with torch.inference_mode(), attention_backend_context(
            model,
            device,
            enabled=device.type == "cuda",
        ):
            output = model.generate(**generation_kwargs)
        continuations = _continuation_ids(output, input_ids.shape[1], tokenizer)
        rows = []
        for case, continuation in zip(QUALITATIVE_CASES, continuations, strict=True):
            diagnostics = _decoded_diagnostics(continuation, tokenizer, cfg)
            expectation = case.get("expect")
            rows.append(
                {
                    "id": str(case["id"]),
                    "prompt": str(case["messages"][-1]["content"]),
                    "expectation": expectation,
                    "expectation_pass": _expectation_pass(
                        str(diagnostics["text"]),
                        expectation if isinstance(expectation, dict) else None,
                    ),
                    **diagnostics,
                }
            )
        evaluations[mode] = rows
    scores = {}
    for mode, rows in evaluations.items():
        scored = [row for row in rows if row["expectation_pass"] is not None]
        scores[mode] = {
            "passed": sum(bool(row["expectation_pass"]) for row in scored),
            "scored": len(scored),
            "pass_rate": (
                sum(bool(row["expectation_pass"]) for row in scored) / max(1, len(scored))
            ),
        }
    return {
        "greedy": evaluations["greedy"],
        "sampled": evaluations["sampled"],
        "scores": scores,
        "sample_seed": int(cfg["seed"]) + 10_001,
        "sample_temperature": float(cfg["qualitative_temperature"]),
        "sample_top_p": float(cfg["qualitative_top_p"]),
    }


def classify_diagnostic_result(
    baseline: dict[str, Any],
    post: dict[str, Any],
    baseline_nll: dict[str, Any],
    post_nll: dict[str, Any],
    baseline_qualitative: dict[str, Any],
    post_qualitative: dict[str, Any],
) -> dict[str, Any]:
    loop_delta = float(post["loop_rate"]) - float(baseline["loop_rate"])
    repeated_delta = float(post["mean_repeated_4gram_fraction"]) - float(
        baseline["mean_repeated_4gram_fraction"]
    )
    empty_delta = float(post["empty_rate"]) - float(baseline["empty_rate"])
    nll_delta = float(post_nll["nll"]) - float(baseline_nll["nll"])
    before_score = baseline_qualitative["scores"]["greedy"]
    after_score = post_qualitative["scores"]["greedy"]
    pass_delta = int(after_score["passed"]) - int(before_score["passed"])

    if empty_delta > 0.05:
        label = "output_collapse"
        explanation = "loop avoidance came from substantially more empty outputs"
    elif nll_delta > 0.05:
        label = "collateral_regression"
        explanation = "held-out prompt likelihood regressed beyond the preservation budget"
    elif loop_delta > 0.01 or repeated_delta > 0.02:
        label = "repetition_regression"
        explanation = "held-out repetition became measurably worse"
    elif loop_delta < -0.02 and pass_delta > 0:
        label = "latent_capability_unlocked"
        explanation = "loops fell and exact communication improved while likelihood stayed stable"
    elif loop_delta < -0.02:
        label = "repetition_suppression_only"
        explanation = "loops fell without evidence that task communication improved"
    elif pass_delta > 0:
        label = "capability_signal_without_loop_improvement"
        explanation = "exact communication improved, but loop behavior did not"
    else:
        label = "inconclusive_or_no_effect"
        explanation = "neither loop behavior nor exact communication changed materially"

    recommended = label == "latent_capability_unlocked"
    return {
        "label": label,
        "explanation": explanation,
        "recommended_for_inference": recommended,
        "loop_rate_delta": loop_delta,
        "repeated_4gram_fraction_delta": repeated_delta,
        "empty_rate_delta": empty_delta,
        "prompt_nll_delta": nll_delta,
        "greedy_exact_pass_delta": pass_delta,
        "greedy_exact_before": int(before_score["passed"]),
        "greedy_exact_after": int(after_score["passed"]),
    }


def evaluate_loop_rate(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[Sequence[int]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    batch_size = int(cfg["generation_batch_size"])
    loops = 0
    outputs = 0
    generated_tokens = 0
    examples: list[dict[str, Any]] = []
    lengths: list[int] = []
    distinct_ratios: list[float] = []
    repeated_4gram_fractions: list[float] = []
    identical_runs: list[int] = []
    decoded_outputs: list[str] = []
    empty_outputs = 0
    replacement_characters = 0
    started = time.perf_counter()
    for offset in range(0, len(prompts), batch_size):
        selected = prompts[offset : offset + batch_size]
        inputs, _mask, result = _generate_batch(
            model,
            tokenizer,
            selected,
            cfg,
            output_scores=False,
        )
        continuations = _continuation_ids(result, inputs.shape[1], tokenizer)
        for continuation in continuations:
            diagnostics = _decoded_diagnostics(continuation, tokenizer, cfg)
            text = str(diagnostics["text"])
            hit = diagnostics["repeat"]
            outputs += 1
            generated_tokens += len(continuation)
            lengths.append(len(continuation))
            distinct_ratios.append(float(diagnostics["distinct_token_ratio"]))
            repeated_4gram_fractions.append(float(diagnostics["repeated_4gram_fraction"]))
            identical_runs.append(int(diagnostics["longest_identical_token_run"]))
            decoded_outputs.append(text)
            empty_outputs += int(bool(diagnostics["empty"]))
            replacement_characters += int(diagnostics["replacement_characters"])
            if bool(diagnostics["loop"]):
                loops += 1
                if len(examples) < 8:
                    examples.append(
                        {
                            "text": text[:512],
                            **dict(hit or {}),
                        }
                    )
    elapsed = time.perf_counter() - started
    return {
        "prompts": outputs,
        "loops": loops,
        "loop_rate": loops / max(1, outputs),
        "generated_tokens": generated_tokens,
        "mean_generated_tokens": sum(lengths) / max(1, len(lengths)),
        "empty_outputs": empty_outputs,
        "empty_rate": empty_outputs / max(1, outputs),
        "mean_distinct_token_ratio": sum(distinct_ratios) / max(1, len(distinct_ratios)),
        "mean_repeated_4gram_fraction": sum(repeated_4gram_fractions)
        / max(1, len(repeated_4gram_fractions)),
        "max_identical_token_run": max(identical_runs, default=0),
        "max_length_hits": sum(
            length >= int(cfg["max_new_tokens"]) for length in lengths
        ),
        "max_length_hit_rate": sum(
            length >= int(cfg["max_new_tokens"]) for length in lengths
        )
        / max(1, outputs),
        "unique_output_rate": len(set(decoded_outputs)) / max(1, outputs),
        "replacement_characters": replacement_characters,
        "elapsed_s": round(elapsed, 3),
        "tokens_per_second": round(generated_tokens / max(elapsed, 1e-9), 2),
        "examples": examples,
    }


def _write_pairs(path: Path, pairs: Sequence[FTPOPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair.to_dict(), ensure_ascii=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _pair_from_dict(value: dict[str, Any]) -> FTPOPair:
    from ..train.antidoom import RepeatHit

    return FTPOPair(
        batch_index=int(value["batch_index"]),
        sequence_index=int(value["sequence_index"]),
        return_index=int(value["return_index"]),
        prompt_ids=tuple(int(item) for item in value["prompt_ids"]),
        chosen_ids=tuple(int(item) for item in value["chosen_ids"]),
        rejected_token_id=int(value["rejected_token_id"]),
        source_prompt_ids=tuple(int(item) for item in value["source_prompt_ids"]),
        generated_ids=tuple(int(item) for item in value["generated_ids"]),
        rejected_generation_index=int(value["rejected_generation_index"]),
        repetition=RepeatHit(**value["repetition"]),
        chosen_surfaces=tuple(str(item) for item in value["chosen_surfaces"]),
        rejected_surface=str(value["rejected_surface"]),
        generation_text=str(value["generation_text"]),
        metadata=dict(value.get("metadata") or {}),
    )


def _pair_identity(pair: FTPOPair) -> str:
    source_key = str(pair.metadata.get("source_key") or "")
    if source_key:
        return source_key
    payload = orjson.dumps(
        {
            "prompt_ids": pair.source_prompt_ids,
            "generated_ids": pair.generated_ids,
            "rejected_generation_index": pair.rejected_generation_index,
        },
        option=orjson.OPT_SORT_KEYS,
    )
    return hashlib.sha256(payload).hexdigest()


def _read_pairs(path: Path) -> list[FTPOPair]:
    if not path.exists():
        return []
    pairs: list[FTPOPair] = []
    seen: set[str] = set()
    with path.open("rb+") as stream:
        line_number = 0
        last_good_offset = 0
        while line := stream.readline():
            line_number += 1
            if not line.strip():
                last_good_offset = stream.tell()
                continue
            try:
                pair = _pair_from_dict(orjson.loads(line))
            except (KeyError, TypeError, ValueError) as exc:
                if not line.endswith(b"\n"):
                    stream.seek(last_good_offset)
                    stream.truncate()
                    break
                raise ValueError(f"{path}:{line_number}: invalid FTPO pair") from exc
            identity = _pair_identity(pair)
            if identity not in seen:
                pairs.append(pair)
                seen.add(identity)
            last_good_offset = stream.tell()
    return pairs


def mine_pairs(
    model: nn.Module,
    tokenizer: Any,
    cfg: dict[str, Any],
    rd: Path,
) -> tuple[list[FTPOPair], dict[str, Any]]:
    pair_path = rd / "ftpo_pairs.jsonl"
    pairs = _read_pairs(pair_path)
    state_path = rd / "mining_state.json"
    state = (
        orjson.loads(state_path.read_bytes())
        if state_path.exists()
        else {"source_rows_scanned": 0, "generated_prompts": 0}
    )
    source_rows_scanned = int(state.get("source_rows_scanned", 0))
    generated_prompts = int(state.get("generated_prompts", 0))
    stats = Counter(
        source_rows_scanned=source_rows_scanned,
        generated_prompts=generated_prompts,
        pairs=len(pairs),
    )
    pending_ids: list[list[int]] = []
    pending_metadata: list[dict[str, Any]] = []
    existing_pair_ids = {_pair_identity(pair) for pair in pairs}
    started = time.perf_counter()

    def flush() -> None:
        nonlocal pairs, generated_prompts
        if not pending_ids:
            return
        inputs, attention_mask, result = _generate_batch(
            model,
            tokenizer,
            pending_ids,
            cfg,
            output_scores=True,
        )
        mined = mine_ftpo_pairs(
            inputs,
            attention_mask,
            result,
            tokenizer,
            source_metadata=pending_metadata,
            max_chosen=int(cfg["max_chosen"]),
            candidate_pool=int(cfg["candidate_pool"]),
            min_p=float(cfg["min_p"]),
            sample_interval=int(cfg["sample_interval"]),
        )
        mined = [pair for pair in mined if _pair_identity(pair) not in existing_pair_ids]
        remaining = max(0, int(cfg["target_pairs"]) - len(pairs))
        mined = mined[:remaining]
        _write_pairs(pair_path, mined)
        pairs.extend(mined)
        existing_pair_ids.update(_pair_identity(pair) for pair in mined)
        generated_prompts += len(pending_ids)
        stats["generated_prompts"] = generated_prompts
        stats["pairs"] = len(pairs)
        stats["generated_tokens"] += len(pending_ids) * int(cfg["max_new_tokens"])
        pending_ids.clear()
        pending_metadata.clear()
        _write_json_atomic(
            state_path,
            {
                "source_rows_scanned": stats["source_rows_scanned"],
                "generated_prompts": generated_prompts,
                "pairs": len(pairs),
            },
        )
        elapsed = time.perf_counter() - started
        metric = {
            "stage": "antidoom pair mining",
            "pairs": len(pairs),
            "target_pairs": int(cfg["target_pairs"]),
            "generated_prompts": generated_prompts,
            "pair_yield": len(pairs) / max(1, generated_prompts),
            "elapsed_s": round(elapsed, 3),
        }
        _append_metric(rd, metric)
        _write_progress(
            rd,
            stage="antidoom pair mining",
            step=len(pairs),
            total=int(cfg["target_pairs"]),
            value=metric["pair_yield"],
            label="pair yield",
            message=(
                f"{generated_prompts:,} prompts sampled; "
                f"{len(pairs) / max(1, generated_prompts):.1%} loop-pair yield"
            ),
        )
        log(
            f"mining: pairs={len(pairs)}/{int(cfg['target_pairs'])} "
            f"prompts={generated_prompts} yield={metric['pair_yield']:.3f}"
        )

    for row_index, (row, raw_line) in enumerate(
        _iter_prompt_rows(Path(cfg["prompt_jsonl"]))
    ):
        if row_index < source_rows_scanned:
            continue
        stats["source_rows_scanned"] = row_index + 1
        key = _source_key(row, raw_line)
        if _is_eval_row(key, cfg):
            stats["held_out"] += 1
            continue
        try:
            ids = _render_prompt(row, tokenizer, cfg)
        except (KeyError, TypeError, ValueError):
            stats["unrenderable"] += 1
            continue
        if len(ids) > int(cfg["max_prompt_tokens"]):
            stats["overlength"] += 1
            continue
        pending_ids.append(ids)
        pending_metadata.append(_prompt_metadata(row, key))
        if len(pending_ids) >= int(cfg["generation_batch_size"]):
            flush()
        if len(pairs) >= int(cfg["target_pairs"]):
            break
        if generated_prompts + len(pending_ids) >= int(cfg["max_generated_prompts"]):
            flush()
            break
    flush()
    elapsed = time.perf_counter() - started
    summary = {
        **{key: int(value) for key, value in stats.items()},
        "pairs": len(pairs),
        "pair_yield": len(pairs) / max(1, generated_prompts),
        "elapsed_s": round(elapsed, 3),
        "pair_path": str(pair_path),
    }
    if len(pairs) < min(100, int(cfg["target_pairs"])):
        raise RuntimeError(
            f"AntiDoom mining produced only {len(pairs)} usable pairs; refusing to train"
        )
    return pairs, summary


def _weighted_without_replacement(
    pairs: Sequence[FTPOPair],
    count: int,
    strength: float,
    seed: int,
) -> list[FTPOPair]:
    if count >= len(pairs):
        return list(pairs)
    frequencies = Counter(pair.rejected_token_id for pair in pairs)
    ordered = sorted(frequencies.values())
    median = float(ordered[len(ordered) // 2])
    rng = random.Random(seed)
    weighted: list[tuple[float, int]] = []
    for index, pair in enumerate(pairs):
        frequency = frequencies[pair.rejected_token_id]
        weight = 1.0 if frequency <= median else (median / frequency) ** strength
        key = rng.random() ** (1.0 / max(weight, 1e-12))
        weighted.append((key, index))
    return [pairs[index] for _key, index in sorted(weighted, reverse=True)[:count]]


def _regularize_chosen(
    pairs: Sequence[FTPOPair],
    strength: float,
) -> list[FTPOPair]:
    if strength <= 0 or not pairs:
        return list(pairs)
    counts = Counter(token_id for pair in pairs for token_id in set(pair.chosen_ids))
    ordered = sorted(counts.values())
    percentile_index = int((len(ordered) - 1) * 0.95)
    reference = max(50, ordered[percentile_index])
    targets = {
        token_id: (
            count
            if count <= reference
            else max(1, int(round(count * (reference / count) ** strength)))
        )
        for token_id, count in counts.items()
    }
    current = Counter(counts)
    mutable = [list(dict.fromkeys(pair.chosen_ids)) for pair in pairs]
    for token_id in sorted(current, key=lambda value: current[value] - targets[value], reverse=True):
        for index in sorted(range(len(pairs)), key=lambda item: (-len(mutable[item]), item)):
            if current[token_id] <= targets[token_id]:
                break
            if token_id in mutable[index] and len(mutable[index]) > 1:
                mutable[index].remove(token_id)
                current[token_id] -= 1
    for token_id in sorted(current, key=lambda value: current[value] - targets[value], reverse=True):
        for index in range(len(pairs)):
            if current[token_id] <= targets[token_id]:
                break
            if mutable[index] == [token_id]:
                mutable[index].clear()
                current[token_id] -= 1
    regularized: list[FTPOPair] = []
    for pair, chosen in zip(pairs, mutable, strict=True):
        if not chosen:
            continue
        surfaces = tuple(
            surface
            for token_id, surface in zip(pair.chosen_ids, pair.chosen_surfaces, strict=True)
            if token_id in chosen
        )
        regularized.append(
            replace(pair, chosen_ids=tuple(chosen), chosen_surfaces=surfaces)
        )
    return regularized


def regularize_pairs(
    pairs: Sequence[FTPOPair],
    cfg: dict[str, Any],
) -> tuple[list[FTPOPair], dict[str, Any]]:
    selected = _weighted_without_replacement(
        pairs,
        min(len(pairs), int(cfg["max_train_examples"])),
        float(cfg["rejected_regularisation_strength"]),
        int(cfg["seed"]),
    )
    selected = _regularize_chosen(
        selected,
        float(cfg["chosen_regularisation_strength"]),
    )
    rejected = Counter(pair.rejected_token_id for pair in selected)
    chosen = Counter(token_id for pair in selected for token_id in pair.chosen_ids)
    return selected, {
        "input_pairs": len(pairs),
        "training_pairs": len(selected),
        "unique_rejected_tokens": len(rejected),
        "unique_chosen_tokens": len(chosen),
        "top_rejected_token_ids": rejected.most_common(20),
        "top_chosen_token_ids": chosen.most_common(20),
    }


def _training_batch(
    pairs: Sequence[FTPOPair],
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Tensor]:
    prompt_width = max(len(pair.prompt_ids) for pair in pairs)
    chosen_width = max(len(pair.chosen_ids) for pair in pairs)
    input_ids = torch.full(
        (len(pairs), prompt_width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    chosen_ids = torch.zeros(
        (len(pairs), chosen_width),
        dtype=torch.long,
        device=device,
    )
    chosen_mask = torch.zeros_like(chosen_ids, dtype=torch.bool)
    rejected = torch.empty(len(pairs), dtype=torch.long, device=device)
    for index, pair in enumerate(pairs):
        prompt = torch.tensor(pair.prompt_ids, dtype=torch.long, device=device)
        choices = torch.tensor(pair.chosen_ids, dtype=torch.long, device=device)
        input_ids[index, -len(prompt) :] = prompt
        attention_mask[index, -len(prompt) :] = 1
        chosen_ids[index, : len(choices)] = choices
        chosen_mask[index, : len(choices)] = True
        rejected[index] = pair.rejected_token_id
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "chosen_ids": chosen_ids,
        "chosen_mask": chosen_mask,
        "rejected_token_ids": rejected,
    }


def train_ftpo(
    base_model: nn.Module,
    tokenizer: Any,
    pairs: Sequence[FTPOPair],
    cfg: dict[str, Any],
    rd: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    from peft import LoraConfig, get_peft_model
    from transformers import get_scheduler

    lora = LoraConfig(
        r=int(cfg["rank"]),
        lora_alpha=int(cfg["alpha"]),
        lora_dropout=float(cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg["target_modules"]),
        # This adapter intentionally changes the output projection without
        # changing the tied input embedding.  It must remain adapter-only and
        # must never be merged back into the tied base tensor.
        ensure_weight_tying=False,
    )
    model = get_peft_model(
        base_model,
        lora,
        autocast_adapter_dtype=False,
    )
    model.config.use_cache = False
    model.train()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_dtypes = Counter(
        str(parameter.dtype)
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if set(trainable_dtypes) != {"torch.bfloat16"}:
        raise RuntimeError(
            f"FTPO accelerator adapters must remain BF16, got {dict(trainable_dtypes)}"
        )
    log(
        f"FTPO LoRA: trainable={trainable:,}/{total:,}; rank={cfg['rank']}; "
        f"alpha={cfg['alpha']}; dtypes={dict(trainable_dtypes)}"
    )
    optimizer = make_optimizer(
        model,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        cuda=_model_device(model).type == "cuda",
        optimizer_state="cpu_offload",
        implementation="cpu_offload_adamw",
        master_weight_storage="fp32",
    )
    micro_batch = int(cfg["micro_batch"])
    grad_accum = int(cfg["grad_accum"])
    epochs = int(cfg["epochs"])
    micro_batches_per_epoch = math.ceil(len(pairs) / micro_batch)
    total_steps = math.ceil(micro_batches_per_epoch * epochs / grad_accum)
    warmup_steps = min(total_steps, math.ceil(total_steps * float(cfg["warmup_ratio"])))
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    rng = random.Random(int(cfg["seed"]))
    device = _model_device(model)
    rolling_wins: deque[float] = deque(maxlen=int(cfg["early_stopping_window"]))
    optimizer.zero_grad(set_to_none=True)
    step = 0
    last: dict[str, Any] = {}
    stopped_early = False
    started = time.perf_counter()
    for epoch in range(epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        micro_batches = [order[offset : offset + micro_batch] for offset in range(0, len(order), micro_batch)]
        for group_offset in range(0, len(micro_batches), grad_accum):
            group = micro_batches[group_offset : group_offset + grad_accum]
            optimizer.zero_grad(set_to_none=True)
            aggregate = Counter()
            for indices in group:
                selected = [pairs[index] for index in indices]
                batch = _training_batch(
                    selected,
                    pad_token_id=int(tokenizer.pad_token_id),
                    device=device,
                )
                with attention_backend_context(
                    model,
                    device,
                    enabled=device.type == "cuda",
                ):
                    policy_logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        position_ids=batch["position_ids"],
                        use_cache=False,
                        return_dict=True,
                    ).logits[:, -1, :]
                with (
                    model.disable_adapter(),
                    torch.inference_mode(),
                    attention_backend_context(
                        model,
                        device,
                        enabled=device.type == "cuda",
                    ),
                ):
                    reference_logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        position_ids=batch["position_ids"],
                        use_cache=False,
                        return_dict=True,
                    ).logits[:, -1, :]
                result = ftpo_objective(
                    policy_logits.float(),
                    reference_logits.float(),
                    batch["chosen_ids"],
                    batch["rejected_token_ids"],
                    batch["chosen_mask"],
                )
                (result.loss / len(group)).backward()
                weight = len(selected)
                for name in (
                    "loss",
                    "preference_loss",
                    "untargeted_mse",
                    "target_excess_mse",
                    "chosen_win",
                    "margin_win",
                    "mean_delta",
                    "median_delta",
                    "active_weight",
                ):
                    value = result.loss if name == "loss" else getattr(result, name)
                    aggregate[name] += float(value.detach()) * weight
                aggregate["examples"] += weight
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(cfg["max_grad_norm"]),
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("non-finite FTPO gradient norm")
            optimizer.step()
            scheduler.step()
            step += 1
            examples = max(1, int(aggregate["examples"]))
            last = {
                "stage": "antidoom FTPO",
                "step": step,
                "max_steps": total_steps,
                **{
                    name: float(aggregate[name]) / examples
                    for name in aggregate
                    if name != "examples"
                },
                "gradient_norm": float(gradient_norm),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "elapsed_s": round(time.perf_counter() - started, 3),
            }
            rolling_wins.append(float(last["chosen_win"]))
            if step == 1 or step % int(cfg["log_every"]) == 0 or step == total_steps:
                _append_metric(rd, last)
                rolling = sum(rolling_wins) / len(rolling_wins)
                _write_progress(
                    rd,
                    stage="antidoom FTPO",
                    step=step,
                    total=total_steps,
                    value=float(last["loss"]),
                    label="FTPO loss",
                    message=(
                        f"chosen_win={float(last['chosen_win']):.1%}; "
                        f"rolling={rolling:.1%}; margin={float(last['margin_win']):.1%}"
                    ),
                )
                log(
                    f"FTPO step {step}/{total_steps} loss={float(last['loss']):.6f} "
                    f"chosen_win={float(last['chosen_win']):.3f} rolling={rolling:.3f}"
                )
            rolling = sum(rolling_wins) / len(rolling_wins)
            if (
                step >= int(cfg["min_training_steps"])
                and len(rolling_wins) == rolling_wins.maxlen
                and rolling >= float(cfg["early_stopping_chosen_win"])
            ):
                stopped_early = True
                log(
                    f"FTPO early stop: rolling chosen_win={rolling:.3f} >= "
                    f"{float(cfg['early_stopping_chosen_win']):.3f}"
                )
                break
        if stopped_early:
            break
    if step < 1:
        raise RuntimeError("FTPO produced no optimizer steps")
    master_summary_fn = getattr(optimizer, "master_state_summary", None)
    master_summary = master_summary_fn() if callable(master_summary_fn) else None
    return model, {
        "steps": step,
        "planned_steps": total_steps,
        "stopped_early": stopped_early,
        "final": last,
        "trainable_parameters": trainable,
        "loaded_parameters": total,
        "trainable_dtypes": dict(trainable_dtypes),
        "optimizer_master": master_summary,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


def _commit_adapter(
    cfg: dict[str, Any],
    output: Path,
    metadata: dict[str, Any],
) -> None:
    with db.transaction() as connection:
        checkpoint = connection.execute(
            "SELECT quant_locked FROM checkpoints WHERE id=?",
            (cfg["checkpoint_id"],),
        ).fetchone()
        if checkpoint is None:
            raise ValueError("AntiDoom source checkpoint disappeared")
        locked = str(checkpoint["quant_locked"] or "")
        if locked and locked != "bf16":
            raise RuntimeError(f"source checkpoint is quant-locked to {locked!r}, not bf16")
        lock = connection.execute(
            "UPDATE checkpoints SET quant_locked='bf16' WHERE id=? "
            "AND (quant_locked='' OR quant_locked IS NULL OR quant_locked='bf16')",
            (cfg["checkpoint_id"],),
        )
        adapter = connection.execute(
            "UPDATE adapters SET status='trained', path=?, meta_json=? "
            "WHERE id=? AND status='training'",
            (str(output), json.dumps(metadata), cfg["adapter_id"]),
        )
        if lock.rowcount != 1 or adapter.rowcount != 1:
            raise RuntimeError("could not atomically register AntiDoom adapter")


def run_antidoom(cfg: dict[str, Any], rd: Path) -> dict[str, Any]:
    cfg = _validate_config(cfg)
    checkpoint = db.q1("SELECT ref FROM checkpoints WHERE id=?", (cfg["checkpoint_id"],))
    adapter = db.q1(
        "SELECT checkpoint_id,arch,quant,rank,status FROM adapters WHERE id=?",
        (cfg["adapter_id"],),
    )
    if not checkpoint or checkpoint["ref"] != cfg["checkpoint_ref"]:
        raise ValueError("AntiDoom run does not match its registered source checkpoint")
    if not adapter or not (
        adapter["checkpoint_id"] == cfg["checkpoint_id"]
        and adapter["arch"] == "lora"
        and adapter["quant"] == "bf16"
        and int(adapter["rank"]) == int(cfg["rank"])
        and adapter["status"] == "training"
    ):
        raise ValueError("AntiDoom run does not match its registered adapter")

    log("AntiDoom provenance: hashing prompt and checkpoint payloads")
    provenance = _verify_or_write_provenance(rd, cfg)

    torch.manual_seed(int(cfg["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg["seed"]))
    model, tokenizer, cuda, _quant, notes = load_base(
        str(cfg["checkpoint_ref"]),
        "bf16",
        restore_saved_qat=False,
    )
    if not cuda:
        raise RuntimeError("AntiDoom mining requires CUDA for this experiment")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_positions = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    required_mining_context = int(cfg["max_prompt_tokens"]) + int(cfg["max_new_tokens"])
    required_qualitative_context = max(
        len(
            render_prompt_ids(
                {"messages": case["messages"]},
                tokenizer,
                prompt_field="messages",
            )
        )
        for case in QUALITATIVE_CASES
    ) + int(cfg["qualitative_max_new_tokens"])
    required_context = max(required_mining_context, required_qualitative_context)
    if max_positions and required_context > max_positions:
        raise ValueError(
            f"AntiDoom requires {required_context} positions but the checkpoint supports "
            f"only {max_positions}"
        )
    model.config.use_cache = True
    model.eval()
    for note in notes:
        log(f"note: {note}")
    eval_prompts, eval_selection = _collect_eval_prompts(cfg, tokenizer)
    log(f"AntiDoom held-out probe: {len(eval_prompts)} prompts")
    baseline = evaluate_loop_rate(model, tokenizer, eval_prompts, cfg)
    baseline_nll = evaluate_causal_nll(model, tokenizer, eval_prompts, cfg)
    baseline_qualitative = evaluate_qualitative(model, tokenizer, cfg)
    write_json(rd / "baseline_eval.json", baseline)
    write_json(
        rd / "diagnostic_baseline.json",
        {
            "loop": baseline,
            "causal_nll": baseline_nll,
            "qualitative": baseline_qualitative,
        },
    )
    log(f"baseline loop rate={baseline['loop_rate']:.3%} ({baseline['loops']}/{baseline['prompts']})")
    log(
        f"baseline prompt NLL={baseline_nll['nll']:.4f} "
        f"PPL={baseline_nll['perplexity']:.2f}"
    )

    raw_pairs, mining = mine_pairs(model, tokenizer, cfg, rd)
    selected_pairs, regularization = regularize_pairs(raw_pairs, cfg)
    if not selected_pairs:
        raise RuntimeError("pair regularization removed every AntiDoom sample")
    write_json(rd / "regularization.json", regularization)
    log(
        f"regularized pairs: {len(selected_pairs):,}/{len(raw_pairs):,}; "
        f"rejected tokens={regularization['unique_rejected_tokens']:,}; "
        f"chosen tokens={regularization['unique_chosen_tokens']:,}"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model, training = train_ftpo(model, tokenizer, selected_pairs, cfg, rd)
    output = adapter_dir(str(cfg["adapter_id"]))
    staging = rd / "adapter_staging"
    if staging.exists() or output.exists():
        raise RuntimeError("AntiDoom publish target already exists; refusing to overwrite it")
    staging.mkdir(parents=True)
    model.save_pretrained(
        staging,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    tokenizer.save_pretrained(staging)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from peft import PeftModel

    reload_base, reload_tokenizer, reload_cuda, _reload_quant, reload_notes = load_base(
        str(cfg["checkpoint_ref"]),
        "bf16",
        restore_saved_qat=False,
    )
    if not reload_cuda:
        raise RuntimeError("AntiDoom adapter reload unexpectedly lost CUDA")
    model = PeftModel.from_pretrained(
        reload_base,
        staging,
        is_trainable=False,
        autocast_adapter_dtype=False,
    )
    if reload_tokenizer.pad_token_id is None:
        reload_tokenizer.pad_token = reload_tokenizer.eos_token
    reload_tokenizer.padding_side = "left"
    model.config.use_cache = True
    model.eval()
    post = evaluate_loop_rate(model, reload_tokenizer, eval_prompts, cfg)
    post_nll = evaluate_causal_nll(model, reload_tokenizer, eval_prompts, cfg)
    post_qualitative = evaluate_qualitative(model, reload_tokenizer, cfg)
    write_json(rd / "post_eval.json", post)
    write_json(
        rd / "diagnostic_post.json",
        {
            "loop": post,
            "causal_nll": post_nll,
            "qualitative": post_qualitative,
        },
    )
    log(f"post-FTPO loop rate={post['loop_rate']:.3%} ({post['loops']}/{post['prompts']})")
    log(
        f"post-FTPO prompt NLL={post_nll['nll']:.4f} "
        f"PPL={post_nll['perplexity']:.2f}"
    )
    diagnostic_verdict = classify_diagnostic_result(
        baseline,
        post,
        baseline_nll,
        post_nll,
        baseline_qualitative,
        post_qualitative,
    )
    log(
        f"AntiDoom diagnostic={diagnostic_verdict['label']}; "
        f"recommended={diagnostic_verdict['recommended_for_inference']}"
    )
    metadata = {
        "format": ANTIDOOM_FORMAT,
        "method": "Final Token Preference Optimization",
        "upstream": "https://github.com/Liquid4All/antidoom",
        "source_checkpoint_id": cfg["checkpoint_id"],
        "source_checkpoint_ref": cfg["checkpoint_ref"],
        "prompt_jsonl": cfg["prompt_jsonl"],
        "corpus_id": cfg.get("corpus_id"),
        "config": cfg,
        "provenance": provenance,
        "eval_selection": eval_selection,
        "baseline": baseline,
        "baseline_causal_nll": baseline_nll,
        "baseline_qualitative": baseline_qualitative,
        "mining": mining,
        "regularization": regularization,
        "training": training,
        "post": post,
        "post_causal_nll": post_nll,
        "post_qualitative": post_qualitative,
        "loop_rate_delta": post["loop_rate"] - baseline["loop_rate"],
        "causal_nll_delta": post_nll["nll"] - baseline_nll["nll"],
        "diagnostic_verdict": diagnostic_verdict,
        "adapter_only": True,
        "merge_safe": False,
        "tied_lm_head_policy": "output-head LoRA only; base embeddings remain tied and frozen",
        "serialization_reload_validated": True,
        "notes": notes + reload_notes,
    }
    write_json(staging / "yylab_antidoom.json", metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(output)
    _commit_adapter(cfg, output, metadata)
    return metadata


def execute_run(run_id: str) -> dict[str, Any]:
    run = db.q1("SELECT * FROM runs WHERE id=?", (run_id,))
    if not run:
        raise ValueError(f"run {run_id!r} not found")
    if run["kind"] != ANTIDOOM_RUN_KIND:
        raise ValueError(
            f"AntiDoom worker requires run kind {ANTIDOOM_RUN_KIND!r}, got {run['kind']!r}"
        )
    cfg = _validate_config(json.loads(run["config_json"] or "{}"))
    rd = run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "config.json", cfg)
    db.execute(
        "UPDATE runs SET status='running', started_at=?, log_path=? WHERE id=?",
        (db.now_iso(), str(rd / "train.log"), run_id),
    )
    log(f"yylab AntiDoom worker: run={run_id} source={cfg['checkpoint_id']}")
    return run_antidoom(cfg, rd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_id = str(args.run)
    try:
        summary = execute_run(run_id)
        db.execute(
            "UPDATE runs SET status='done', finished_at=?, metrics_json=? WHERE id=?",
            (db.now_iso(), json.dumps(summary), run_id),
        )
        log("done")
    except Exception as exc:  # noqa: BLE001
        error = f"{exc}\n{traceback.format_exc()}"
        log(error)
        db.execute(
            "UPDATE runs SET status='failed', finished_at=?, error=? WHERE id=?",
            (db.now_iso(), error[-4000:], run_id),
        )
        run = db.q1("SELECT config_json FROM runs WHERE id=?", (run_id,))
        try:
            cfg = json.loads((run or {}).get("config_json") or "{}")
        except (TypeError, ValueError):
            cfg = {}
        if cfg.get("adapter_id"):
            db.execute(
                "UPDATE adapters SET status='failed' WHERE id=? AND status='training'",
                (cfg["adapter_id"],),
            )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
