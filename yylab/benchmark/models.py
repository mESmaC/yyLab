"""Versioned benchmark-suite models and strict JSON validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUITE_SCHEMA = "molor.benchmark-suite.v1"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SuiteError(ValueError):
    """A benchmark suite is malformed or internally inconsistent."""


def validate_json_value(value: object, label: str) -> None:
    """Reject non-finite or non-JSON values in suite evaluation metadata."""

    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a finite JSON value") from exc


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SuiteError(f"{label} must be an object")
    return dict(value)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise SuiteError(f"{label} must match {_ID.pattern}")
    return value


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SuiteError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise SuiteError(f"{label} must be a positive finite number")
    return result


def _positive_int(value: object, label: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise SuiteError(f"{label} must be an integer from 1 to {maximum}")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SuiteError(f"{label} must be an array of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise SuiteError(f"{label} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise SuiteError(f"{label} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    category: str
    prompt: str
    scorer: dict[str, Any]
    weight: float = 1.0
    system: str = ""
    max_new_tokens: int | None = None
    tags: tuple[str, ...] = ()
    acceptable_routes: tuple[str, ...] = ()
    reference: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def messages(self, default_system: str) -> list[dict[str, str]]:
        system = self.system.strip() or default_system.strip()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": self.prompt})
        return messages


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    path: Path
    raw: dict[str, Any]
    fingerprint: str
    suite_id: str
    version: str
    title: str
    description: str
    default_system: str
    generation: dict[str, Any]
    category_weights: dict[str, float]
    category_labels: dict[str, str]
    profiles: dict[str, tuple[str, ...]]
    cases: tuple[BenchmarkCase, ...]

    def select(
        self,
        *,
        profile: str = "full",
        categories: Sequence[str] = (),
        limit: int | None = None,
    ) -> tuple[BenchmarkCase, ...]:
        if profile not in self.profiles:
            available = ", ".join(sorted(self.profiles))
            raise SuiteError(f"unknown profile {profile!r}; choose one of: {available}")
        selected_ids = set(self.profiles[profile])
        category_filter = set(categories)
        unknown_categories = category_filter.difference(self.category_weights)
        if unknown_categories:
            raise SuiteError("unknown categories: " + ", ".join(sorted(unknown_categories)))
        selected = tuple(
            case
            for case in self.cases
            if case.case_id in selected_ids and (not category_filter or case.category in category_filter)
        )
        if limit is not None:
            if isinstance(limit, bool) or limit < 1:
                raise SuiteError("limit must be a positive integer")
            selected = selected[:limit]
        if not selected:
            raise SuiteError("the selected profile/category filters contain no cases")
        return selected


def load_suite(path: str | Path) -> BenchmarkSuite:
    suite_path = Path(path).expanduser().resolve()
    try:
        raw_value = json.loads(suite_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SuiteError(f"cannot read benchmark suite {suite_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SuiteError(f"invalid JSON in benchmark suite {suite_path}: {exc}") from exc
    raw = _object(raw_value, "suite")
    allowed = {
        "schema_version",
        "suite_id",
        "version",
        "title",
        "description",
        "default_system",
        "generation",
        "categories",
        "profiles",
        "cases",
    }
    unknown = set(raw).difference(allowed)
    if unknown:
        raise SuiteError("suite has unknown fields: " + ", ".join(sorted(unknown)))
    if raw.get("schema_version") != SUITE_SCHEMA:
        raise SuiteError(f"schema_version must be {SUITE_SCHEMA!r}")
    suite_id = _identifier(raw.get("suite_id"), "suite_id")
    version = str(raw.get("version") or "").strip()
    title = str(raw.get("title") or "").strip()
    description = str(raw.get("description") or "").strip()
    if not version or not title or not description:
        raise SuiteError("version, title, and description are required")
    default_system = str(raw.get("default_system") or "")

    generation = _object(raw.get("generation"), "generation")
    generation_allowed = {
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "enable_thinking",
        "context_token_budget",
        "cache_mode",
        "seed",
    }
    generation_unknown = set(generation).difference(generation_allowed)
    if generation_unknown:
        raise SuiteError("generation has unknown fields: " + ", ".join(sorted(generation_unknown)))
    generation["max_new_tokens"] = _positive_int(
        generation.get("max_new_tokens", 96), "generation.max_new_tokens", maximum=8192
    )
    temperature = generation.get("temperature", 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise SuiteError("generation.temperature must be a finite number")
    if not math.isfinite(float(temperature)) or not 0 <= float(temperature) <= 10:
        raise SuiteError("generation.temperature must be from 0 to 10")
    generation["temperature"] = float(temperature)
    top_p = generation.get("top_p", 1.0)
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise SuiteError("generation.top_p must be a finite number")
    if not math.isfinite(float(top_p)) or not 0 < float(top_p) <= 1:
        raise SuiteError("generation.top_p must be in (0, 1]")
    generation["top_p"] = float(top_p)
    top_k = generation.get("top_k", 0)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise SuiteError("generation.top_k must be a non-negative integer")
    generation["top_k"] = top_k
    if not isinstance(generation.get("enable_thinking", False), bool):
        raise SuiteError("generation.enable_thinking must be boolean")
    generation.setdefault("enable_thinking", False)
    context_token_budget = generation.get("context_token_budget")
    if context_token_budget is not None:
        context_token_budget = _positive_int(
            context_token_budget,
            "generation.context_token_budget",
            maximum=2_000_000,
        )
    generation["context_token_budget"] = context_token_budget
    cache_mode = generation.get("cache_mode", "dynamic")
    if cache_mode not in {"dynamic", "offloaded", "disabled"}:
        raise SuiteError("generation.cache_mode must be 'dynamic', 'offloaded', or 'disabled'")
    generation["cache_mode"] = cache_mode
    seed = generation.get("seed", 8419)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SuiteError("generation.seed must be an integer")
    generation["seed"] = seed

    categories_raw = _object(raw.get("categories"), "categories")
    if not categories_raw:
        raise SuiteError("categories must not be empty")
    category_weights: dict[str, float] = {}
    category_labels: dict[str, str] = {}
    for raw_name, raw_category in categories_raw.items():
        name = _identifier(raw_name, "category name")
        category = _object(raw_category, f"categories.{name}")
        unknown_category = set(category).difference({"label", "weight"})
        if unknown_category:
            raise SuiteError(f"categories.{name} has unknown fields: " + ", ".join(sorted(unknown_category)))
        label = str(category.get("label") or "").strip()
        if not label:
            raise SuiteError(f"categories.{name}.label is required")
        category_labels[name] = label
        category_weights[name] = _positive(category.get("weight"), f"categories.{name}.weight")
    total_weight = sum(category_weights.values())
    category_weights = {name: weight / total_weight for name, weight in category_weights.items()}

    raw_cases = raw.get("cases")
    if isinstance(raw_cases, (str, bytes)) or not isinstance(raw_cases, Sequence) or not raw_cases:
        raise SuiteError("cases must be a non-empty array")
    cases: list[BenchmarkCase] = []
    case_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = _object(raw_case, f"cases[{index}]")
        allowed_case = {
            "id",
            "category",
            "prompt",
            "system",
            "scorer",
            "weight",
            "max_new_tokens",
            "tags",
            "acceptable_routes",
            "reference",
            "metadata",
        }
        unknown_case = set(case).difference(allowed_case)
        if unknown_case:
            raise SuiteError(f"cases[{index}] has unknown fields: " + ", ".join(sorted(unknown_case)))
        case_id = _identifier(case.get("id"), f"cases[{index}].id")
        if case_id in case_ids:
            raise SuiteError(f"duplicate case id {case_id!r}")
        case_ids.add(case_id)
        category = _identifier(case.get("category"), f"cases[{index}].category")
        if category not in category_weights:
            raise SuiteError(f"case {case_id!r} uses unknown category {category!r}")
        prompt = str(case.get("prompt") or "")
        if not prompt.strip():
            raise SuiteError(f"case {case_id!r} has an empty prompt")
        scorer = _object(case.get("scorer"), f"cases[{index}].scorer")
        if not isinstance(scorer.get("type"), str):
            raise SuiteError(f"case {case_id!r} scorer.type is required")
        try:
            from .scoring import ScoreSpecError, validate_score_spec

            validate_score_spec(scorer)
        except (ScoreSpecError, TypeError, ValueError) as exc:
            raise SuiteError(f"case {case_id!r} has an invalid scorer: {exc}") from exc
        max_new_tokens = case.get("max_new_tokens")
        if max_new_tokens is not None:
            max_new_tokens = _positive_int(max_new_tokens, f"cases[{index}].max_new_tokens", maximum=8192)
        reference = case.get("reference")
        metadata = _object(case.get("metadata", {}), f"cases[{index}].metadata")
        try:
            validate_json_value(reference, f"cases[{index}].reference")
            validate_json_value(metadata, f"cases[{index}].metadata")
        except (TypeError, ValueError) as exc:
            raise SuiteError(f"case {case_id!r} has invalid evaluation data: {exc}") from exc
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                category=category,
                prompt=prompt,
                scorer=scorer,
                weight=_positive(case.get("weight", 1.0), f"cases[{index}].weight"),
                system=str(case.get("system") or ""),
                max_new_tokens=max_new_tokens,
                tags=_strings(case.get("tags", []), f"cases[{index}].tags"),
                acceptable_routes=_strings(
                    case.get("acceptable_routes", []),
                    f"cases[{index}].acceptable_routes",
                ),
                reference=reference,
                metadata=metadata,
            )
        )

    profiles_raw = _object(raw.get("profiles"), "profiles")
    profiles: dict[str, tuple[str, ...]] = {"full": tuple(case.case_id for case in cases)}
    for raw_name, raw_members in profiles_raw.items():
        name = _identifier(raw_name, "profile name")
        if name == "full":
            raise SuiteError("the full profile is generated automatically")
        members = _strings(raw_members, f"profiles.{name}")
        missing = set(members).difference(case_ids)
        if missing:
            raise SuiteError(f"profiles.{name} references unknown cases: " + ", ".join(sorted(missing)))
        if not members:
            raise SuiteError(f"profiles.{name} must not be empty")
        profiles[name] = members

    return BenchmarkSuite(
        path=suite_path,
        raw=raw,
        fingerprint=fingerprint(raw),
        suite_id=suite_id,
        version=version,
        title=title,
        description=description,
        default_system=default_system,
        generation=generation,
        category_weights=category_weights,
        category_labels=category_labels,
        profiles=profiles,
        cases=tuple(cases),
    )


__all__ = [
    "BenchmarkCase",
    "BenchmarkSuite",
    "SUITE_SCHEMA",
    "SuiteError",
    "canonical_json",
    "fingerprint",
    "load_suite",
]
