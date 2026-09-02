"""Deterministic yyLab quality benchmarks, matrices, reports, and comparisons."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .backend import YyLabBenchmarkBackend
from .models import BenchmarkCase, BenchmarkSuite, canonical_json, load_suite
from .scoring import (
    SCORER_VERSION,
    SEMANTIC_SCORER_VERSION,
    ScoreResult,
    score_response,
    score_response_semantic,
    visible_answer,
)

RUN_SCHEMA = "yylab.benchmark-run.v1"
RESULT_SCHEMA = "yylab.benchmark-result.v1"
SUMMARY_SCHEMA = "yylab.benchmark-summary.v2"
GENERATION_CONTRACT_VERSION = "yylab.hf-generation.v3"
BUNDLED_SUITE = Path(__file__).parent / "suites" / "molor-core-v1.json"
ProgressCallback = Callable[[str, int, int, float, str], None]
DEFAULT_THINKING_TOKENS = 1024
THINKING_TEMPERATURE = 0.6
THINKING_TOP_P = 0.95
THINKING_TOP_K = 20


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _seed(suite: BenchmarkSuite, case: BenchmarkCase, repetition: int) -> int:
    material = f"{suite.fingerprint}:{case.case_id}:{repetition}:{suite.generation['seed']}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def _artifact_lock(ref: str | None) -> dict[str, object] | None:
    if not ref:
        return None
    path = Path(ref).expanduser()
    if not path.exists():
        return {"ref": ref, "kind": "external-ref", "content_addressed": False}
    path = path.resolve()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    entries: list[dict[str, object]] = []
    for item in files:
        digest = hashlib.sha256()
        with item.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        size = item.stat().st_size
        sha = digest.hexdigest()
        entries.append({"path": relative, "size": size, "sha256": sha})
    return {
        "ref": ref,
        "resolved_path": str(path),
        "kind": "file" if path.is_file() else "directory",
        "content_addressed": True,
        "sha256": hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest(),
        "files": entries,
    }


def _stack_lock(cfg: Mapping[str, object]) -> dict[str, object]:
    from ..models.families import family_from_reference

    checkpoint = _artifact_lock(str(cfg["checkpoint_ref"]))
    adapter = _artifact_lock(str(cfg.get("adapter_path") or ""))
    content_addressed = bool(checkpoint and checkpoint.get("content_addressed")) and (
        adapter is None or bool(adapter.get("content_addressed"))
    )
    model_family = cfg.get("model_family") or family_from_reference(cfg.get("checkpoint_ref"))
    value = {
        "schema_version": "yylab.benchmark-stack-lock.v1",
        "checkpoint": checkpoint,
        "adapter": adapter,
        "quant": cfg.get("quant"),
        "adapter_arch": cfg.get("adapter_arch"),
        "model_family": model_family,
        "fully_content_addressed": content_addressed,
    }
    if model_family == "spalmer":
        value["spalmer_runtime"] = {
            "physical_expert_offload": bool(cfg.get("spalmer_expert_offload", True)),
            "expert_cache_size": cfg.get("spalmer_expert_cache_size"),
        }
    value["stack_fingerprint"] = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return value


def _implementation_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    benchmark_hashes = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("quality.py", "models.py", "scoring.py", "backend.py")
    }
    source_root = directory.parent
    benchmark_hashes.update(
        {
            relative: hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
            for relative in (
                "infer/engine.py",
                "models/loading.py",
                "models/spalmer.py",
                "mole/inject.py",
                "mole/layer.py",
            )
        }
    )
    return benchmark_hashes


def _gpu_snapshot() -> dict[str, object]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return {
            "name": str(name),
            "memory_used_gb": memory.used / 2**30,
            "memory_total_gb": memory.total / 2**30,
        }
    except Exception:  # noqa: BLE001
        return {}


def policies_for_config(cfg: Mapping[str, object]) -> list[str]:
    from ..models.families import family_from_reference

    policy = str(cfg.get("policy") or "base")
    declared_family = str(cfg.get("model_family") or "").strip().casefold()
    if declared_family not in {"", "hf", "spalmer", "spalmer_v1"}:
        raise ValueError(f"unknown benchmark model family: {declared_family}")
    model_family = (
        "spalmer"
        if declared_family in {"spalmer", "spalmer_v1"}
        else ("hf" if declared_family else family_from_reference(cfg.get("checkpoint_ref")))
    )
    if model_family == "spalmer":
        if policy not in {"base", "dynamic"}:
            raise ValueError("SPALMER quality benchmarks support base or native dynamic-residency policy")
        if cfg.get("adapter_path"):
            raise ValueError("SPALMER quality benchmarks do not accept yyLab adapters")
        return [policy]
    adapter_path = str(cfg.get("adapter_path") or "")
    adapter_arch = str(cfg.get("adapter_arch") or "")
    experts = int(cfg.get("num_experts") or 0)
    if policy == "matrix":
        if not adapter_path or adapter_arch != "mole" or experts < 1:
            raise ValueError("matrix policy requires a trained MoLE adapter")
        return ["base", *(f"expert:{index}" for index in range(experts)), "dynamic"]
    if policy in {"dynamic", "adapter"} or policy.startswith("expert:"):
        if not adapter_path:
            raise ValueError(f"policy {policy!r} requires a trained adapter")
    if policy == "dynamic" or policy.startswith("expert:"):
        if adapter_arch != "mole":
            raise ValueError(f"policy {policy!r} requires a MoLE adapter")
    if policy.startswith("expert:"):
        index = int(policy.split(":", 1)[1])
        if not 0 <= index < experts:
            raise ValueError(f"forced expert {index} is outside this adapter's {experts} slots")
    if policy not in {"base", "dynamic", "adapter"} and not policy.startswith("expert:"):
        raise ValueError(f"unknown quality benchmark policy: {policy}")
    return [policy]


def _request(suite: BenchmarkSuite, case: BenchmarkCase, repetition: int, cfg: Mapping) -> dict:
    generation = suite.generation
    prompt_mode = str(cfg.get("prompt_mode") or "chat").strip().casefold()
    if prompt_mode not in {"chat", "raw"}:
        raise ValueError(f"unknown benchmark prompt mode: {prompt_mode}")
    thinking_override = cfg.get("enable_thinking")
    enable_thinking = (
        bool(thinking_override) if thinking_override is not None else bool(generation["enable_thinking"])
    )
    if prompt_mode == "raw" and enable_thinking:
        raise ValueError("thinking requires chat prompt mode")
    return {
        "max_new_tokens": int(
            cfg.get("max_new_tokens") or case.max_new_tokens or generation["max_new_tokens"]
        ),
        "temperature": THINKING_TEMPERATURE if enable_thinking else float(generation["temperature"]),
        "top_p": THINKING_TOP_P if enable_thinking else float(generation["top_p"]),
        "top_k": THINKING_TOP_K if enable_thinking else int(generation["top_k"]),
        "prompt_mode": prompt_mode,
        "enable_thinking": enable_thinking,
        "max_thinking_tokens": (
            int(cfg.get("max_thinking_tokens") or DEFAULT_THINKING_TOKENS) if enable_thinking else 0
        ),
        "cache_mode": str(generation["cache_mode"]),
        "seed": _seed(suite, case, repetition),
    }


def _generation_telemetry(generated=None) -> dict[str, object]:
    if generated is None:
        return {
            "elapsed_seconds": 0.0,
            "time_to_first_token_seconds": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "thinking_tokens": None,
            "answer_tokens": None,
            "thinking_budget_exhausted": False,
            "tokens_per_second": None,
            "finish_reason": "error",
        }
    answer_tokens = getattr(generated, "answer_tokens", None)
    return {
        "elapsed_seconds": generated.elapsed_seconds,
        "time_to_first_token_seconds": None,
        "prompt_tokens": generated.prompt_tokens,
        "completion_tokens": generated.completion_tokens,
        "thinking_tokens": getattr(generated, "thinking_tokens", 0),
        "answer_tokens": generated.completion_tokens if answer_tokens is None else answer_tokens,
        "thinking_budget_exhausted": bool(getattr(generated, "thinking_budget_exhausted", False)),
        "tokens_per_second": generated.tokens_per_second,
        "finish_reason": generated.finish_reason,
    }


def _canonical_generation_request(request: Mapping[str, object]) -> dict[str, object]:
    """Normalize controls that have no effect for legacy comparison compatibility."""

    value = dict(request)
    value.setdefault("prompt_mode", "chat")
    if not bool(value.get("enable_thinking")):
        value.pop("max_thinking_tokens", None)
    return value


def _workload_contract(
    suite: BenchmarkSuite,
    cases: Sequence[BenchmarkCase],
    policies: Sequence[str],
    cfg: Mapping[str, object],
    *,
    profile: str,
    repetitions: int,
    categories: Sequence[str],
    limit: int | None,
) -> dict[str, object]:
    cells = []
    for policy in policies:
        for repetition in range(repetitions):
            for case in cases:
                cells.append(
                    {
                        "policy": policy,
                        "case_id": case.case_id,
                        "repetition": repetition,
                        "request": _request(suite, case, repetition, cfg),
                    }
                )
    thinking_enabled = any(
        bool(cell["request"].get("enable_thinking"))
        for cell in cells
        if isinstance(cell.get("request"), Mapping)
    )
    return {
        "suite_fingerprint": suite.fingerprint,
        "profile": profile,
        "categories": list(categories),
        "limit": limit,
        "case_ids": [case.case_id for case in cases],
        "policies": list(policies),
        "repetitions": repetitions,
        "generation": dict(suite.generation),
        "runtime_context_limit": "loaded_model.max_position_embeddings",
        "prompt_mode": str(cfg.get("prompt_mode") or "chat").strip().casefold(),
        "context_mode": cfg.get("context_mode") or "native",
        "rope_scaling": cfg.get("rope_scaling") or "none",
        "rope_factor": cfg.get("rope_factor"),
        "max_context_tokens": cfg.get("max_context_tokens"),
        "max_new_tokens_override": cfg.get("max_new_tokens"),
        "enable_thinking_override": cfg.get("enable_thinking"),
        "max_thinking_tokens_override": cfg.get("max_thinking_tokens") if thinking_enabled else None,
        "spalmer_expert_offload": (
            bool(cfg.get("spalmer_expert_offload", True))
            if str(cfg.get("model_family") or "").casefold() in {"spalmer", "spalmer_v1"}
            else None
        ),
        "spalmer_expert_cache_size": cfg.get("spalmer_expert_cache_size"),
        "cells": cells,
    }


def _generation_comparison_fingerprint(workload_contract: Mapping[str, object]) -> str:
    """Fingerprint comparable prompts/settings while intentionally ignoring policy.

    A base policy and a dynamic policy must differ in model stack and policy;
    those are the subjects being compared. Requests, cases, repetitions, and
    generation controls must still match exactly.
    """

    unique_cells: dict[tuple[str, int], dict[str, object]] = {}
    for raw in workload_contract.get("cells", []):
        if not isinstance(raw, Mapping):
            continue
        case_id = str(raw.get("case_id") or "")
        repetition = raw.get("repetition")
        request = raw.get("request")
        if not case_id or not isinstance(repetition, int) or not isinstance(request, Mapping):
            continue
        key = case_id, repetition
        candidate = {
            "case_id": case_id,
            "repetition": repetition,
            "request": _canonical_generation_request(request),
        }
        previous = unique_cells.get(key)
        if previous is not None and previous != candidate:
            raise ValueError(f"policy requests differ within workload cell {case_id}/{repetition}")
        unique_cells[key] = candidate
    fingerprint_value = {
        "suite_fingerprint": workload_contract.get("suite_fingerprint"),
        "profile": workload_contract.get("profile"),
        "categories": workload_contract.get("categories"),
        "limit": workload_contract.get("limit"),
        "case_ids": workload_contract.get("case_ids"),
        "repetitions": workload_contract.get("repetitions"),
        "cells": [unique_cells[key] for key in sorted(unique_cells)],
    }
    if workload_contract.get("context_mode") not in {None, "native"}:
        fingerprint_value["context_extension"] = {
            "mode": workload_contract.get("context_mode"),
            "rope_scaling": workload_contract.get("rope_scaling"),
            "rope_factor": workload_contract.get("rope_factor"),
            "max_context_tokens": workload_contract.get("max_context_tokens"),
        }
    if workload_contract.get("spalmer_expert_offload") is not None:
        fingerprint_value["spalmer_expert_placement"] = {
            "physical_expert_offload": workload_contract.get("spalmer_expert_offload"),
            "expert_cache_size": workload_contract.get("spalmer_expert_cache_size"),
        }
    return _fingerprint(fingerprint_value)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _semantic_fields(result: ScoreResult) -> dict[str, object]:
    return {
        "semantic_score": result.score,
        "semantic_passed": result.passed,
        "semantic_extracted": result.extracted,
        "semantic_details": result.details,
    }


def _dimensions(
    canonical: Mapping[str, object],
    semantic: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    return {
        "canonical": {
            "score": canonical.get("score"),
            "passed": canonical.get("passed"),
            "extracted": canonical.get("extracted"),
            "details": canonical.get("details", {}),
        },
        "semantic": {
            "score": semantic.get("semantic_score"),
            "passed": semantic.get("semantic_passed"),
            "extracted": semantic.get("semantic_extracted"),
            "details": semantic.get("semantic_details", {}),
        },
    }


def _cell_id(
    *,
    suite_fingerprint: str,
    evaluation_fingerprint: str,
    workload_fingerprint: str,
    stack_fingerprint: str,
    policy: str,
    case_id: str,
    repetition: int,
    request: Mapping[str, object],
) -> str:
    return _fingerprint(
        {
            "suite_fingerprint": suite_fingerprint,
            "evaluation_fingerprint": evaluation_fingerprint,
            "workload_fingerprint": workload_fingerprint,
            "stack_fingerprint": stack_fingerprint,
            "policy": policy,
            "case_id": case_id,
            "repetition": repetition,
            "request": dict(request),
        }
    )


def _latest_records(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Select one result per cell, preferring a later successful retry."""

    latest: dict[tuple[str, str, int], dict[str, object]] = {}
    for raw in records:
        policy = raw.get("policy")
        case_id = raw.get("case_id")
        repetition = raw.get("repetition")
        if not isinstance(policy, str) or not isinstance(case_id, str) or not isinstance(repetition, int):
            continue
        key = policy, case_id, repetition
        previous = latest.get(key)
        if previous is None or previous.get("error") is not None or raw.get("error") is None:
            latest[key] = dict(raw)
    return list(latest.values())


def _revalidate_records(
    records: Sequence[Mapping[str, object]],
    cases: Mapping[str, BenchmarkCase],
) -> list[dict[str, object]]:
    """Re-score stored completions so summaries never trust serialized scores."""

    validated: list[dict[str, object]] = []
    for raw in records:
        record = dict(raw)
        if record.get("error") is not None:
            validated.append(record)
            continue
        case = cases.get(str(record.get("case_id")))
        response = record.get("response")
        if case is None or not isinstance(response, str):
            record.update(
                score=None,
                passed=False,
                semantic_score=None,
                semantic_passed=False,
                error="successful result has no valid case/response",
            )
            validated.append(record)
            continue
        try:
            result = score_response(response, case.scorer)
            semantic = score_response_semantic(response, case.scorer, prompt=case.prompt)
        except Exception as exc:  # noqa: BLE001
            record.update(
                score=None,
                passed=False,
                semantic_score=None,
                semantic_passed=False,
                error=f"stored response could not be re-scored: {type(exc).__name__}: {exc}",
            )
            validated.append(record)
            continue
        stored = record.get("score")
        if (
            isinstance(stored, bool)
            or not isinstance(stored, (int, float))
            or not math.isclose(float(stored), result.score, abs_tol=1e-12)
        ):
            record.update(
                score=None,
                passed=False,
                semantic_score=None,
                semantic_passed=False,
                error="stored score failed deterministic revalidation",
            )
        else:
            canonical_data = result.to_dict()
            semantic_data = _semantic_fields(semantic)
            record.update(canonical_data)
            record.update(semantic_data)
            record["dimensions"] = _dimensions(canonical_data, semantic_data)
        validated.append(record)
    return validated


def _score_value(record: Mapping[str, object], score_field: str) -> float | None:
    score = record.get(score_field)
    if score_field == "semantic_score" and score is None:
        # Synthetic/legacy summaries predate the semantic dimension. Canonical
        # fallback preserves API compatibility; persisted responses are always
        # re-evaluated before a real benchmark summary is written.
        score = record.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    return float(score)


def _weighted_mean(
    records: Iterable[Mapping[str, object]],
    *,
    score_field: str = "score",
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for record in records:
        value = _score_value(record, score_field)
        value = value if value is not None else 0.0
        weight = record.get("case_weight", 1.0)
        weight = float(weight) if isinstance(weight, (int, float)) and not isinstance(weight, bool) else 1.0
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator else None


def _overall(category_scores: Mapping[str, float], suite: BenchmarkSuite) -> float:
    included = {name: suite.category_weights[name] for name in category_scores}
    denominator = sum(included.values())
    if denominator <= 0:
        return 0.0
    return sum(category_scores[name] * weight for name, weight in included.items()) / denominator


def _bootstrap_quality_ci(
    records: Sequence[Mapping[str, object]],
    suite: BenchmarkSuite,
    *,
    seed: int,
    score_field: str = "score",
    iterations: int = 2000,
) -> dict[str, object] | None:
    by_category: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        score = _score_value(record, score_field)
        if score is not None:
            by_category[str(record["category"])].append(record)
    if not by_category:
        return None
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        category_scores: dict[str, float] = {}
        for category, category_records in by_category.items():
            sample = [rng.choice(category_records) for _ in category_records]
            mean = _weighted_mean(sample, score_field=score_field)
            if mean is not None:
                category_scores[category] = mean
        samples.append(_overall(category_scores, suite) * 1000)
    lower = _percentile(samples, 0.025)
    upper = _percentile(samples, 0.975)
    if lower is None or upper is None:
        return None
    return {
        "level": 0.95,
        "lower": lower,
        "upper": upper,
        "method": "stratified-bootstrap",
        "iterations": iterations,
    }


def summarize_policy(
    suite: BenchmarkSuite,
    policy: str,
    records: Sequence[Mapping[str, object]],
    expected: int,
) -> dict[str, object]:
    selected = [record for record in records if record.get("policy") == policy]
    by_category: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in selected:
        by_category[str(record.get("category"))].append(record)
    categories: dict[str, dict[str, object]] = {}
    category_scores: dict[str, float] = {}
    semantic_category_scores: dict[str, float] = {}
    for category in suite.category_weights:
        items = by_category.get(category, [])
        if not items:
            continue
        mean = _weighted_mean(items, score_field="score")
        semantic_mean = _weighted_mean(items, score_field="semantic_score")
        if mean is not None:
            category_scores[category] = mean
        if semantic_mean is not None:
            semantic_category_scores[category] = semantic_mean
        categories[category] = {
            "label": suite.category_labels[category],
            "suite_weight": suite.category_weights[category],
            "attempted": len(items),
            "scored": sum(item.get("score") is not None for item in items),
            "errors": sum(bool(item.get("error")) for item in items),
            "passed": sum(item.get("passed") is True for item in items),
            "mean_score": mean,
            "score_percent": None if mean is None else mean * 100,
            "semantic_passed": sum(
                (
                    item.get("semantic_passed")
                    if item.get("semantic_score") is not None
                    else item.get("passed")
                )
                is True
                for item in items
            ),
            "semantic_mean_score": semantic_mean,
            "semantic_score_percent": None if semantic_mean is None else semantic_mean * 100,
        }
    successful = [item for item in selected if item.get("score") is not None]
    elapsed = [float(item["telemetry"]["elapsed_seconds"]) for item in successful]
    throughput = [
        float(item["telemetry"]["tokens_per_second"])
        for item in successful
        if item["telemetry"].get("tokens_per_second") is not None
    ]
    route_shares = [item.get("routing") for item in successful if isinstance(item.get("routing"), Mapping)]
    quality = _overall(category_scores, suite) * 1000
    semantic_quality = _overall(semantic_category_scores, suite) * 1000
    comparable = len(successful) == expected and not any(item.get("error") for item in selected)
    digest = hashlib.sha256(f"{suite.fingerprint}:{policy}".encode()).digest()
    return {
        "policy": policy,
        "quality_index": quality,
        "strict_quality_index": quality,
        "quality_percent": quality / 10,
        "strict_quality_percent": quality / 10,
        "quality_index_ci": (
            _bootstrap_quality_ci(
                successful,
                suite,
                seed=int.from_bytes(digest[:8], "big"),
            )
            if comparable
            else None
        ),
        "semantic_quality_index": semantic_quality,
        "semantic_quality_percent": semantic_quality / 10,
        "semantic_quality_index_ci": (
            _bootstrap_quality_ci(
                successful,
                suite,
                seed=int.from_bytes(digest[:8], "big") ^ 0x5E6A17C,
                score_field="semantic_score",
            )
            if comparable
            else None
        ),
        "semantic_strict_gap": semantic_quality - quality,
        "categories": categories,
        "coverage": {
            "expected": expected,
            "recorded": len(selected),
            "scored": len(successful),
            "errors": sum(bool(item.get("error")) for item in selected),
            "fraction": len(successful) / expected if expected else 0.0,
        },
        "performance": {
            "elapsed_seconds_total": sum(elapsed),
            "latency_seconds_median": _percentile(elapsed, 0.5),
            "latency_seconds_p95": _percentile(elapsed, 0.95),
            "ttft_seconds_median": None,
            "ttft_seconds_p95": None,
            "tokens_per_second_median": _percentile(throughput, 0.5),
            "tokens_per_second_p10": _percentile(throughput, 0.1),
            "completion_tokens_total": sum(
                int(item["telemetry"]["completion_tokens"]) for item in successful
            ),
            "thinking_tokens_total": sum(
                int(item["telemetry"].get("thinking_tokens") or 0) for item in successful
            ),
            "answer_tokens_total": sum(
                int(item["telemetry"].get("answer_tokens") or 0) for item in successful
            ),
            "thinking_budget_exhaustions": sum(
                bool(item.get("telemetry", {}).get("thinking_budget_exhausted")) for item in selected
            ),
        },
        "routing_samples": route_shares,
        "comparable": comparable,
    }


def _oracle_records(
    records: Sequence[Mapping[str, object]],
    *,
    score_field: str = "score",
    oracle_policy: str = "forced-oracle",
) -> list[dict[str, object]]:
    forced = [record for record in records if str(record.get("policy", "")).startswith("expert:")]
    cells: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for record in forced:
        cells[(str(record["case_id"]), int(record["repetition"]))].append(record)
    oracle: list[dict[str, object]] = []
    for items in cells.values():
        best = max(
            items,
            key=lambda item: (
                _score_value(item, score_field) if _score_value(item, score_field) is not None else -1.0,
                -int(str(item["policy"]).split(":", 1)[1]),
            ),
        )
        copied = dict(best)
        copied["oracle_policy"] = best["policy"]
        copied["policy"] = oracle_policy
        oracle.append(copied)
    return oracle


def summarize_results(
    suite: BenchmarkSuite,
    records: Sequence[Mapping[str, object]],
    policies: Sequence[str],
    *,
    profile: str,
    repetitions: int,
    case_ids: Sequence[str],
    workload_fingerprint: str | None = None,
    generation_comparison_fingerprint: str | None = None,
) -> dict[str, object]:
    records = _latest_records(records)
    expected = len(case_ids) * repetitions
    policy_summaries = {policy: summarize_policy(suite, policy, records, expected) for policy in policies}
    oracle_summary = None
    semantic_oracle_summary = None
    dynamic_regret = None
    semantic_dynamic_regret = None
    if "dynamic" in policy_summaries and any(policy.startswith("expert:") for policy in policies):
        oracle_records = _oracle_records(records)
        oracle_summary = summarize_policy(
            suite,
            "forced-oracle",
            oracle_records,
            expected,
        )
        dynamic_regret = float(oracle_summary["quality_index"]) - float(
            policy_summaries["dynamic"]["quality_index"]
        )
        semantic_oracle_records = _oracle_records(
            records,
            score_field="semantic_score",
            oracle_policy="semantic-forced-oracle",
        )
        semantic_oracle_summary = summarize_policy(
            suite,
            "semantic-forced-oracle",
            semantic_oracle_records,
            expected,
        )
        semantic_dynamic_regret = float(semantic_oracle_summary["semantic_quality_index"]) - float(
            policy_summaries["dynamic"]["semantic_quality_index"]
        )
    primary = "dynamic" if "dynamic" in policy_summaries else policies[-1]
    chosen = policy_summaries[primary]
    result_cells = [
        {
            "policy": item.get("policy"),
            "case_id": item.get("case_id"),
            "repetition": item.get("repetition"),
            "score": item.get("score"),
            "passed": item.get("passed"),
            "semantic_score": (
                item.get("semantic_score") if item.get("semantic_score") is not None else item.get("score")
            ),
            "semantic_passed": (
                item.get("semantic_passed") if item.get("semantic_score") is not None else item.get("passed")
            ),
            "error": item.get("error"),
            "response_sha256": (
                hashlib.sha256(str(item["response"]).encode("utf-8")).hexdigest()
                if isinstance(item.get("response"), str)
                else None
            ),
        }
        for item in records
    ]
    return {
        "schema_version": SUMMARY_SCHEMA,
        "created_at": _now(),
        "suite_id": suite.suite_id,
        "suite_version": suite.version,
        "suite_fingerprint": suite.fingerprint,
        "profile": profile,
        "case_ids": list(case_ids),
        "repetitions": repetitions,
        "workload_fingerprint": workload_fingerprint,
        "generation_comparison_fingerprint": generation_comparison_fingerprint,
        "primary_policy": primary,
        "quality_index": chosen["quality_index"],
        "strict_quality_index": chosen["strict_quality_index"],
        "quality_index_ci": chosen["quality_index_ci"],
        "quality_percent": chosen["quality_percent"],
        "strict_quality_percent": chosen["strict_quality_percent"],
        "semantic_quality_index": chosen["semantic_quality_index"],
        "semantic_quality_index_ci": chosen["semantic_quality_index_ci"],
        "semantic_quality_percent": chosen["semantic_quality_percent"],
        "semantic_strict_gap": chosen["semantic_strict_gap"],
        "coverage": chosen["coverage"],
        "performance": chosen["performance"],
        "categories": chosen["categories"],
        "comparable": chosen["comparable"],
        "policies": policy_summaries,
        "matrix": {
            "forced_oracle": oracle_summary,
            "dynamic_regret": dynamic_regret,
            "semantic_forced_oracle": semantic_oracle_summary,
            "semantic_dynamic_regret": semantic_dynamic_regret,
            "note": "Expert slots are anonymous; this matrix does not claim domain-route accuracy.",
        }
        if oracle_summary is not None
        else None,
        "result_cells": result_cells,
        "result_fingerprint": _fingerprint(result_cells),
    }


def compare_summaries(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> dict:
    if candidate.get("comparable") is False or baseline.get("comparable") is False:
        raise ValueError("baseline comparison requires complete, error-free benchmark runs")
    identity_fields = ("suite_fingerprint", "profile", "case_ids", "repetitions")
    mismatched = [field for field in identity_fields if candidate.get(field) != baseline.get(field)]
    candidate_generation = candidate.get("generation_comparison_fingerprint")
    baseline_generation = baseline.get("generation_comparison_fingerprint")
    if candidate_generation or baseline_generation:
        if candidate_generation != baseline_generation:
            mismatched.append("generation_comparison_fingerprint")
    elif candidate.get("workload_fingerprint") or baseline.get("workload_fingerprint"):
        if candidate.get("workload_fingerprint") != baseline.get("workload_fingerprint"):
            mismatched.append("workload_fingerprint")
    if mismatched:
        raise ValueError("baseline workload differs in: " + ", ".join(mismatched))
    candidate_generation_impl = candidate.get("generation_implementation_fingerprint")
    baseline_generation_impl = baseline.get("generation_implementation_fingerprint")
    if (
        candidate_generation_impl
        and baseline_generation_impl
        and candidate_generation_impl != baseline_generation_impl
    ):
        raise ValueError("baseline generation implementation differs from this run")
    candidate_semantic_fingerprint = candidate.get("semantic_evaluation_fingerprint")
    baseline_semantic_fingerprint = baseline.get("semantic_evaluation_fingerprint")
    if (
        candidate_semantic_fingerprint
        and baseline_semantic_fingerprint
        and candidate_semantic_fingerprint != baseline_semantic_fingerprint
    ):
        raise ValueError("baseline semantic evaluator differs from this run")
    candidate_cells = {
        (str(item["case_id"]), int(item["repetition"])): item
        for item in candidate.get("result_cells", [])
        if item.get("policy") == candidate.get("primary_policy")
    }
    baseline_cells = {
        (str(item["case_id"]), int(item["repetition"])): item
        for item in baseline.get("result_cells", [])
        if item.get("policy") == baseline.get("primary_policy")
    }
    shared = sorted(set(candidate_cells) & set(baseline_cells))
    expected = {
        (str(case_id), repetition)
        for case_id in candidate.get("case_ids", [])
        for repetition in range(int(candidate.get("repetitions") or 0))
    }
    if set(candidate_cells) != expected or set(baseline_cells) != expected:
        raise ValueError("baseline comparison requires every primary-policy workload cell")
    deltas: list[float] = []
    semantic_deltas: list[float] = []
    for key in shared:
        left = candidate_cells[key].get("score")
        right = baseline_cells[key].get("score")
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            deltas.append(float(left) - float(right))
        semantic_left = candidate_cells[key].get("semantic_score", left)
        semantic_right = baseline_cells[key].get("semantic_score", right)
        if isinstance(semantic_left, (int, float)) and isinstance(semantic_right, (int, float)):
            semantic_deltas.append(float(semantic_left) - float(semantic_right))
    category_deltas: dict[str, float] = {}
    semantic_category_deltas: dict[str, float] = {}
    candidate_categories = candidate.get("categories")
    baseline_categories = baseline.get("categories")
    if isinstance(candidate_categories, Mapping) and isinstance(baseline_categories, Mapping):
        for category in sorted(set(candidate_categories) & set(baseline_categories)):
            left = candidate_categories[category]
            right = baseline_categories[category]
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                continue
            left_score = left.get("mean_score")
            right_score = right.get("mean_score")
            if isinstance(left_score, (int, float)) and isinstance(right_score, (int, float)):
                category_deltas[str(category)] = (float(left_score) - float(right_score)) * 100
            semantic_left_score = left.get("semantic_mean_score", left_score)
            semantic_right_score = right.get("semantic_mean_score", right_score)
            if isinstance(semantic_left_score, (int, float)) and isinstance(
                semantic_right_score, (int, float)
            ):
                semantic_category_deltas[str(category)] = (
                    float(semantic_left_score) - float(semantic_right_score)
                ) * 100
    index_delta = float(candidate["quality_index"]) - float(baseline["quality_index"])
    semantic_index_delta = float(candidate.get("semantic_quality_index", candidate["quality_index"])) - float(
        baseline.get("semantic_quality_index", baseline["quality_index"])
    )
    return {
        "schema_version": "yylab.benchmark-comparison.v1",
        "created_at": _now(),
        "quality_index_delta": index_delta,
        "regression": index_delta <= -5.0,
        "category_score_percent_deltas": category_deltas,
        "semantic_quality_index_delta": semantic_index_delta,
        "semantic_regression": semantic_index_delta <= -5.0,
        "semantic_category_score_percent_deltas": semantic_category_deltas,
        "paired_cells": len(deltas),
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "mean_cell_delta": statistics.fmean(deltas) if deltas else None,
        "semantic_paired_cells": len(semantic_deltas),
        "semantic_wins": sum(delta > 0 for delta in semantic_deltas),
        "semantic_ties": sum(delta == 0 for delta in semantic_deltas),
        "semantic_losses": sum(delta < 0 for delta in semantic_deltas),
        "semantic_mean_cell_delta": (statistics.fmean(semantic_deltas) if semantic_deltas else None),
    }


def _write_scores(path: Path, summary: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "policy",
                "category",
                "label",
                "semantic_score_percent",
                "molor_strict_score_percent",
                "semantic_passed",
                "strict_passed",
                "attempted",
                "errors",
            ]
        )
        for policy, policy_summary in summary["policies"].items():
            for category, result in policy_summary["categories"].items():
                writer.writerow(
                    [
                        policy,
                        category,
                        result["label"],
                        result["semantic_score_percent"],
                        result["score_percent"],
                        result["semantic_passed"],
                        result["passed"],
                        result["attempted"],
                        result["errors"],
                    ]
                )


def _write_report(path: Path, cfg: Mapping, summary: Mapping[str, object]) -> None:
    lines = [
        f"# yyLab benchmark — {cfg.get('label') or 'run'}",
        "",
        f"**Semantic Quality Index: {float(summary['semantic_quality_index']):.1f} / 1000**",
        "",
        f"Molor strict Quality Index: {float(summary['quality_index']):.1f} / 1000",
        "",
        (
            "Semantic correctness gives full credit to a supported correct answer even when "
            "explanatory text violates the requested output shape."
        ),
        "",
        f"Suite: `{summary['suite_id']}` · profile `{summary['profile']}` · policy `{summary['primary_policy']}`",
        "",
        "## Policies",
        "",
        "| Policy | Semantic index | Molor strict | Scored | Errors | Median tok/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, item in summary["policies"].items():
        rate = item["performance"].get("tokens_per_second_median")
        lines.append(
            f"| {policy} | {float(item['semantic_quality_index']):.1f} | "
            f"{float(item['quality_index']):.1f} | {item['coverage']['scored']} | "
            f"{item['coverage']['errors']} | {rate if rate is not None else 'n/a'} |"
        )
    matrix = summary.get("matrix")
    if matrix:
        lines.extend(
            [
                "",
                "## Learned routing",
                "",
                (
                    "Semantic forced-expert oracle regret: "
                    f"{float(matrix['semantic_dynamic_regret']):+.1f} index points."
                ),
                "",
                f"Molor strict oracle regret: {float(matrix['dynamic_regret']):+.1f} index points.",
                "",
                str(matrix["note"]),
            ]
        )
    comparison = summary.get("comparison")
    if isinstance(comparison, Mapping):
        lines.extend(
            [
                "",
                "## Baseline comparison",
                "",
                (f"Semantic Quality Index delta: {float(comparison['semantic_quality_index_delta']):+.1f}"),
                "",
                f"Molor strict delta: {float(comparison['quality_index_delta']):+.1f}",
                "",
                (
                    f"Semantic paired cells: {comparison['semantic_paired_cells']} · "
                    f"wins {comparison['semantic_wins']} · ties {comparison['semantic_ties']} · "
                    f"losses {comparison['semantic_losses']}"
                ),
                "",
                (f"Semantic regression flag (≤ -5 points): {bool(comparison['semantic_regression'])}"),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_quality_benchmark(
    cfg: Mapping[str, object],
    output_dir: Path,
    *,
    progress: ProgressCallback | None = None,
    backend=None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_path = Path(str(cfg.get("suite_path") or BUNDLED_SUITE))
    suite = load_suite(suite_path)
    profile = str(cfg.get("profile") or "quick")
    categories = tuple(str(item) for item in cfg.get("categories", []) or [])
    limit = int(cfg["limit"]) if cfg.get("limit") else None
    repetitions = int(cfg.get("repetitions") or 1)
    if repetitions < 1:
        raise ValueError("benchmark repetitions must be positive")
    cases = suite.select(profile=profile, categories=categories, limit=limit)
    policies = policies_for_config(cfg)
    total = len(cases) * repetitions * len(policies)
    workload_contract = _workload_contract(
        suite,
        cases,
        policies,
        cfg,
        profile=profile,
        repetitions=repetitions,
        categories=categories,
        limit=limit,
    )
    workload_fingerprint = _fingerprint(workload_contract)
    generation_comparison_fingerprint = _generation_comparison_fingerprint(workload_contract)
    implementation = _implementation_hashes()
    evaluation_fingerprint = _fingerprint(
        {
            "suite_fingerprint": suite.fingerprint,
            "scorer_version": SCORER_VERSION,
            "implementation": implementation,
        }
    )
    semantic_evaluation_fingerprint = _fingerprint(
        {
            "suite_fingerprint": suite.fingerprint,
            "semantic_scorer_version": SEMANTIC_SCORER_VERSION,
            "scoring_sha256": implementation["scoring.py"],
        }
    )
    generation_implementation_fingerprint = _fingerprint(
        {
            "contract": GENERATION_CONTRACT_VERSION,
            "backend_sha256": implementation["backend.py"],
            "engine_sha256": implementation["infer/engine.py"],
            "loading_sha256": implementation["models/loading.py"],
            "mole_inject_sha256": implementation["mole/inject.py"],
            "mole_layer_sha256": implementation["mole/layer.py"],
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        }
    )
    baseline_summary = None
    baseline_path = str(cfg.get("baseline_summary_path") or "")
    if baseline_path:
        baseline_summary = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        if not isinstance(baseline_summary, dict):
            raise ValueError("baseline summary must be one JSON object")
        if baseline_summary.get("comparable") is False:
            raise ValueError("baseline comparison requires a complete, error-free baseline run")
        baseline_generation_fingerprint = baseline_summary.get("generation_comparison_fingerprint")
        comparable_fingerprint = (
            baseline_generation_fingerprint == generation_comparison_fingerprint
            if baseline_generation_fingerprint
            else baseline_summary.get("workload_fingerprint") == workload_fingerprint
        )
        if not comparable_fingerprint:
            raise ValueError(
                "baseline generation workload differs from this run; choose a baseline with "
                "the same profile, cases, repetitions, and generation settings"
            )
        baseline_generation_impl = baseline_summary.get("generation_implementation_fingerprint")
        if baseline_generation_impl and baseline_generation_impl != generation_implementation_fingerprint:
            raise ValueError("baseline generation implementation differs from this run")
        baseline_semantic_impl = baseline_summary.get("semantic_evaluation_fingerprint")
        if baseline_semantic_impl and baseline_semantic_impl != semantic_evaluation_fingerprint:
            raise ValueError("baseline semantic evaluator differs from this run")
    print("benchmark provenance: hashing checkpoint and adapter bytes", flush=True)
    if progress:
        progress(
            "benchmark provenance",
            0,
            total,
            0.0,
            "Hashing checkpoint and adapter bytes",
        )
    stack_lock = _stack_lock(cfg)
    manifest = {
        "schema_version": RUN_SCHEMA,
        "created_at": _now(),
        "label": str(cfg.get("label") or "yyLab benchmark"),
        "suite_id": suite.suite_id,
        "suite_version": suite.version,
        "suite_fingerprint": suite.fingerprint,
        "scorer_version": SCORER_VERSION,
        "semantic_scorer_version": SEMANTIC_SCORER_VERSION,
        "implementation_sha256": implementation,
        "evaluation_fingerprint": evaluation_fingerprint,
        "semantic_evaluation_fingerprint": semantic_evaluation_fingerprint,
        "generation_implementation_fingerprint": generation_implementation_fingerprint,
        "profile": profile,
        "case_ids": [case.case_id for case in cases],
        "repetitions": repetitions,
        "policies": policies,
        "workload_contract": workload_contract,
        "workload_fingerprint": workload_fingerprint,
        "generation_comparison_fingerprint": generation_comparison_fingerprint,
        "stack_fingerprint": stack_lock["stack_fingerprint"],
        "stack_fully_content_addressed": stack_lock["fully_content_addressed"],
        "config": dict(cfg),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu_before": _gpu_snapshot(),
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("torch", "transformers", "peft", "accelerate")
                if _package_exists(package)
            },
        },
    }
    run_path = output_dir / "run.json"
    if run_path.is_file():
        existing_manifest = json.loads(run_path.read_text(encoding="utf-8"))
        checks = {
            "suite_fingerprint": suite.fingerprint,
            "workload_fingerprint": workload_fingerprint,
            "stack_fingerprint": stack_lock["stack_fingerprint"],
            "evaluation_fingerprint": evaluation_fingerprint,
        }
        changed = [key for key, value in checks.items() if existing_manifest.get(key) != value]
        if changed:
            raise ValueError("cannot resume benchmark after drift in: " + ", ".join(changed))
        manifest = existing_manifest
    else:
        _atomic_json(run_path, manifest)
    _atomic_json(output_dir / "suite.snapshot.json", suite.raw)
    _atomic_json(output_dir / "stack.lock.json", stack_lock)
    results_path = output_dir / "results.jsonl"
    records = _read_jsonl(results_path)
    expected_cells = {
        (str(item["policy"]), str(item["case_id"]), int(item["repetition"])): item
        for item in workload_contract["cells"]
    }
    completed_cells: set[tuple[str, str, int]] = set()
    for item in _latest_records(records):
        policy_value = item.get("policy")
        case_value = item.get("case_id")
        repetition_value = item.get("repetition")
        if (
            not isinstance(policy_value, str)
            or not isinstance(case_value, str)
            or not isinstance(repetition_value, int)
        ):
            continue
        key = policy_value, case_value, repetition_value
        expected = expected_cells.get(key)
        if expected is None:
            continue
        expected_id = _cell_id(
            suite_fingerprint=suite.fingerprint,
            evaluation_fingerprint=evaluation_fingerprint,
            workload_fingerprint=workload_fingerprint,
            stack_fingerprint=str(stack_lock["stack_fingerprint"]),
            policy=key[0],
            case_id=key[1],
            repetition=key[2],
            request=expected["request"],
        )
        if (
            item.get("error") is None
            and item.get("request") == expected["request"]
            and item.get("cell_id") == expected_id
            and item.get("suite_fingerprint") == suite.fingerprint
            and item.get("evaluation_fingerprint") == evaluation_fingerprint
            and item.get("workload_fingerprint") == workload_fingerprint
            and item.get("stack_fingerprint") == stack_lock["stack_fingerprint"]
        ):
            completed_cells.add(key)
    owns_backend = backend is None
    if backend is None:
        backend_settings: dict[str, object] = {
            "attn_implementation": str(cfg.get("attn_implementation") or "sdpa"),
        }
        if str(cfg.get("model_family") or "").casefold() in {"spalmer", "spalmer_v1"}:
            backend_settings.update(
                spalmer_expert_offload=bool(cfg.get("spalmer_expert_offload", True)),
                spalmer_expert_cache_size=cfg.get("spalmer_expert_cache_size"),
            )
        if str(cfg.get("rope_scaling") or "none") != "none":
            backend_settings.update(
                rope_scaling=str(cfg["rope_scaling"]),
                rope_factor=float(cfg.get("rope_factor") or 4.0),
                max_context_tokens=int(cfg.get("max_context_tokens") or 131_072),
            )
        backend = YyLabBenchmarkBackend(
            checkpoint_ref=str(cfg["checkpoint_ref"]),
            quant=str(cfg.get("quant") or "bf16"),
            adapter_path=str(cfg.get("adapter_path") or "") or None,
            adapter_arch=str(cfg.get("adapter_arch") or ""),
            model_family=str(cfg.get("model_family") or "") or None,
            settings=backend_settings,
        )
    completed = len(completed_cells)
    continue_on_error = bool(cfg.get("continue_on_error", True))
    try:
        for policy in policies:
            pending = [
                (repetition, case)
                for repetition in range(repetitions)
                for case in cases
                if (policy, case.case_id, repetition) not in completed_cells
            ]
            if not pending:
                print(f"resume: policy={policy} is already complete", flush=True)
                continue
            notes = backend.configure_policy(policy)
            for note in notes or []:
                print(f"note: {note}", flush=True)
            print(f"benchmark policy={policy}: {len(cases) * repetitions} generations", flush=True)
            for repetition in range(repetitions):
                for case in cases:
                    request = _request(suite, case, repetition, cfg)
                    if (policy, case.case_id, repetition) in completed_cells:
                        print(
                            f"[{completed}/{total}] resume: {case.case_id} · {policy} already complete",
                            flush=True,
                        )
                        continue
                    error = None
                    response = None
                    generated = None
                    incomplete_thinking = False
                    try:
                        generated = backend.generate(
                            messages=case.messages(suite.default_system),
                            **request,
                        )
                        response = generated.text
                        if request["enable_thinking"] and not visible_answer(response):
                            incomplete_thinking = True
                            raise RuntimeError(
                                "thinking generation ended without a visible final answer after "
                                "yyLab reserved the full answer budget"
                            )
                        score = score_response(response, case.scorer)
                        score_data = score.to_dict()
                        semantic = score_response_semantic(
                            response,
                            case.scorer,
                            prompt=case.prompt,
                        )
                        semantic_data = _semantic_fields(semantic)
                        telemetry = _generation_telemetry(generated)
                        routing = generated.routing
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"
                        score_data = {"score": None, "passed": False, "extracted": None, "details": {}}
                        semantic_data = {
                            "semantic_score": None,
                            "semantic_passed": False,
                            "semantic_extracted": None,
                            "semantic_details": {},
                        }
                        telemetry = _generation_telemetry(generated)
                        routing = generated.routing if generated is not None else None
                    record = {
                        "schema_version": RESULT_SCHEMA,
                        "created_at": _now(),
                        "suite_fingerprint": suite.fingerprint,
                        "evaluation_fingerprint": evaluation_fingerprint,
                        "workload_fingerprint": workload_fingerprint,
                        "stack_fingerprint": stack_lock["stack_fingerprint"],
                        "policy": policy,
                        "case_id": case.case_id,
                        "category": case.category,
                        "case_weight": case.weight,
                        "repetition": repetition,
                        "request": request,
                        "cell_id": _cell_id(
                            suite_fingerprint=suite.fingerprint,
                            evaluation_fingerprint=evaluation_fingerprint,
                            workload_fingerprint=workload_fingerprint,
                            stack_fingerprint=str(stack_lock["stack_fingerprint"]),
                            policy=policy,
                            case_id=case.case_id,
                            repetition=repetition,
                            request=request,
                        ),
                        "response": response,
                        **score_data,
                        **semantic_data,
                        "dimensions": _dimensions(score_data, semantic_data),
                        "telemetry": telemetry,
                        "routing": routing,
                        "error": error,
                    }
                    records.append(record)
                    _append_jsonl(results_path, record)
                    completed += 1
                    current_scores = [
                        float(item["semantic_score"])
                        for item in records
                        if item["policy"] == policy and isinstance(item.get("semantic_score"), (int, float))
                    ]
                    running = statistics.fmean(current_scores) * 100 if current_scores else 0.0
                    message = (
                        f"{case.case_id} · {policy} · semantic {semantic_data['semantic_score']} · "
                        f"Molor strict {score_data['score']}"
                    )
                    print(f"[{completed}/{total}] {message}", flush=True)
                    if progress:
                        progress("quality benchmark", completed, total, running, message)
                    if error and (incomplete_thinking or not continue_on_error):
                        raise RuntimeError(error)
        current_records: list[dict[str, object]] = []
        for item in _latest_records(records):
            policy_value = item.get("policy")
            case_value = item.get("case_id")
            repetition_value = item.get("repetition")
            if (
                not isinstance(policy_value, str)
                or not isinstance(case_value, str)
                or not isinstance(repetition_value, int)
            ):
                continue
            expected = expected_cells.get((policy_value, case_value, repetition_value))
            if expected is None:
                continue
            expected_id = _cell_id(
                suite_fingerprint=suite.fingerprint,
                evaluation_fingerprint=evaluation_fingerprint,
                workload_fingerprint=workload_fingerprint,
                stack_fingerprint=str(stack_lock["stack_fingerprint"]),
                policy=policy_value,
                case_id=case_value,
                repetition=repetition_value,
                request=expected["request"],
            )
            if (
                item.get("request") == expected["request"]
                and item.get("cell_id") == expected_id
                and item.get("suite_fingerprint") == suite.fingerprint
                and item.get("evaluation_fingerprint") == evaluation_fingerprint
                and item.get("workload_fingerprint") == workload_fingerprint
                and item.get("stack_fingerprint") == stack_lock["stack_fingerprint"]
            ):
                current_records.append(item)
        current_records = _revalidate_records(
            current_records,
            {case.case_id: case for case in cases},
        )
        summary = summarize_results(
            suite,
            current_records,
            policies,
            profile=profile,
            repetitions=repetitions,
            case_ids=[case.case_id for case in cases],
            workload_fingerprint=workload_fingerprint,
            generation_comparison_fingerprint=generation_comparison_fingerprint,
        )
        summary.update(
            label=manifest["label"],
            evaluation_fingerprint=evaluation_fingerprint,
            semantic_scorer_version=SEMANTIC_SCORER_VERSION,
            semantic_evaluation_fingerprint=semantic_evaluation_fingerprint,
            generation_implementation_fingerprint=generation_implementation_fingerprint,
            stack_fingerprint=stack_lock["stack_fingerprint"],
            stack_fully_content_addressed=stack_lock["fully_content_addressed"],
            gpu_after=_gpu_snapshot(),
        )
        if baseline_summary is not None:
            comparison = compare_summaries(summary, baseline_summary)
            summary["comparison"] = comparison
            _atomic_json(output_dir / "comparison.json", comparison)
        _atomic_json(output_dir / "summary.json", summary)
        _write_scores(output_dir / "scores.csv", summary)
        _write_report(output_dir / "report.md", cfg, summary)
        if progress:
            progress(
                "quality benchmark",
                total,
                total,
                float(summary["semantic_quality_percent"]),
                f"Semantic Quality Index {float(summary['semantic_quality_index']):.1f}/1000",
            )
        return summary
    finally:
        if owns_backend:
            backend.close()


def rescore_quality_artifact(
    output_dir: Path,
    *,
    output_name: str = "semantic-summary.json",
) -> dict[str, object]:
    """Derive dual scores from saved responses without running generation again.

    The append-only ``results.jsonl`` and original summary stay untouched. The
    projection fingerprints both inputs so the derived evaluation is auditable.
    """

    output_dir = output_dir.expanduser().resolve()
    run_path = output_dir / "run.json"
    results_path = output_dir / "results.jsonl"
    suite_path = output_dir / "suite.snapshot.json"
    if not run_path.is_file() or not results_path.is_file() or not suite_path.is_file():
        raise FileNotFoundError("benchmark artifact needs run.json, results.jsonl, and suite.snapshot.json")
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("run.json must contain one JSON object")
    suite = load_suite(suite_path)
    if manifest.get("suite_fingerprint") != suite.fingerprint:
        raise ValueError("suite snapshot fingerprint differs from run.json")
    case_ids = [str(case_id) for case_id in manifest.get("case_ids", [])]
    repetitions = int(manifest.get("repetitions") or 1)
    policies = [str(policy) for policy in manifest.get("policies", [])]
    if not case_ids or not policies:
        raise ValueError("run.json does not declare benchmark cases and policies")
    selected_cases = {case.case_id: case for case in suite.cases if case.case_id in set(case_ids)}
    if set(selected_cases) != set(case_ids):
        raise ValueError("suite snapshot does not contain every run case")
    records = _revalidate_records(_latest_records(_read_jsonl(results_path)), selected_cases)
    workload_contract = manifest.get("workload_contract")
    generation_comparison_fingerprint = (
        _generation_comparison_fingerprint(workload_contract)
        if isinstance(workload_contract, Mapping)
        else None
    )
    summary = summarize_results(
        suite,
        records,
        policies,
        profile=str(manifest.get("profile") or "full"),
        repetitions=repetitions,
        case_ids=case_ids,
        workload_fingerprint=(
            str(manifest["workload_fingerprint"])
            if manifest.get("workload_fingerprint") is not None
            else None
        ),
        generation_comparison_fingerprint=generation_comparison_fingerprint,
    )
    scoring_sha256 = hashlib.sha256(Path(__file__).with_name("scoring.py").read_bytes()).hexdigest()
    semantic_evaluation_fingerprint = _fingerprint(
        {
            "suite_fingerprint": suite.fingerprint,
            "semantic_scorer_version": SEMANTIC_SCORER_VERSION,
            "scoring_sha256": scoring_sha256,
        }
    )
    original_summary_path = output_dir / "summary.json"
    original_summary: Mapping[str, object] = {}
    if original_summary_path.is_file():
        parsed = json.loads(original_summary_path.read_text(encoding="utf-8"))
        if isinstance(parsed, Mapping):
            original_summary = parsed
    summary.update(
        label=manifest.get("label") or original_summary.get("label") or output_dir.name,
        evaluation_fingerprint=manifest.get("evaluation_fingerprint"),
        generation_implementation_fingerprint=manifest.get("generation_implementation_fingerprint"),
        semantic_scorer_version=SEMANTIC_SCORER_VERSION,
        semantic_evaluation_fingerprint=semantic_evaluation_fingerprint,
        stack_fingerprint=manifest.get("stack_fingerprint"),
        stack_fully_content_addressed=manifest.get("stack_fully_content_addressed"),
        gpu_after=original_summary.get("gpu_after", {}),
        projection={
            "schema_version": "yylab.benchmark-semantic-projection.v1",
            "created_at": _now(),
            "source_run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
            "source_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
            "source_summary_sha256": (
                hashlib.sha256(original_summary_path.read_bytes()).hexdigest()
                if original_summary_path.is_file()
                else None
            ),
        },
    )
    _atomic_json(output_dir / output_name, summary)
    report_cfg = manifest.get("config") if isinstance(manifest.get("config"), Mapping) else {}
    _write_scores(output_dir / "semantic-scores.csv", summary)
    _write_report(output_dir / "semantic-report.md", report_cfg, summary)
    return summary


def _package_exists(package: str) -> bool:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


__all__ = [
    "BUNDLED_SUITE",
    "compare_summaries",
    "policies_for_config",
    "rescore_quality_artifact",
    "run_quality_benchmark",
    "summarize_policy",
    "summarize_results",
]
