"""Deterministic, inspectable scorers for the bundled benchmark suite."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

SCORER_VERSION = "molor.deterministic-scorers.v1.2"
SEMANTIC_SCORER_VERSION = "yylab.semantic-scorers.v2"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ScoreResult:
    score: float
    passed: bool
    extracted: object | None
    details: dict[str, object]

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("score must be finite and from zero to one")

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "passed": self.passed,
            "extracted": self.extracted,
            "details": self.details,
        }


class ScoreSpecError(ValueError):
    """The scorer declaration is invalid."""


_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNFINISHED_THINK = re.compile(r"<think>.*\Z", re.IGNORECASE | re.DOTALL)
_NUMBER = re.compile(
    r"(?<![\w.])[-+]?\d+\s*/\s*\d+"
    r"|(?<![\w.])[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?%?"
)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScoreSpecError(f"{label} must be an object")
    return dict(value)


def _sequence(value: object, label: str) -> list[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScoreSpecError(f"{label} must be an array")
    return list(value)


def _strings(value: object, label: str, *, nonempty: bool = True) -> list[str]:
    items = _sequence(value, label)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ScoreSpecError(f"{label} must contain non-empty strings")
    if nonempty and not items:
        raise ScoreSpecError(f"{label} must not be empty")
    return [str(item) for item in items]


def visible_answer(text: str) -> str:
    """Remove complete Qwen reasoning blocks before deterministic answer parsing."""

    without_complete_blocks = _THINK.sub("", text)
    return _UNFINISHED_THINK.sub("", without_complete_blocks).strip()


def normalize_text(
    text: str,
    *,
    casefold: bool = True,
    strip_punctuation: bool = False,
) -> str:
    value = unicodedata.normalize("NFKC", visible_answer(text))
    value = " ".join(value.split())
    if strip_punctuation:
        value = "".join(
            character for character in value if not unicodedata.category(character).startswith("P")
        )
        value = " ".join(value.split())
    return value.casefold() if casefold else value


def _result(score: float, extracted: object | None, **details: object) -> ScoreResult:
    bounded = min(1.0, max(0.0, float(score)))
    return ScoreResult(
        score=bounded,
        passed=math.isclose(bounded, 1.0, abs_tol=1e-12),
        extracted=extracted,
        details=dict(details),
    )


def _score_exact(response: str, spec: Mapping[str, object]) -> ScoreResult:
    raw_answers = spec.get("answers")
    if raw_answers is None and "answer" in spec:
        raw_answers = [spec["answer"]]
    answers = _strings(raw_answers, "exact.answers")
    casefold = bool(spec.get("casefold", True))
    strip_punctuation = bool(spec.get("strip_punctuation", True))
    extracted = normalize_text(
        response,
        casefold=casefold,
        strip_punctuation=strip_punctuation,
    )
    expected = [
        normalize_text(answer, casefold=casefold, strip_punctuation=strip_punctuation) for answer in answers
    ]
    return _result(float(extracted in expected), extracted, expected=expected)


def _extract_choice(response: str, labels: Sequence[str]) -> str | None:
    answer = visible_answer(response).strip()
    label_set = {label.upper() for label in labels}
    patterns = (
        r"(?:final\s+answer|answer|choice)\s*(?:is|:|-)?\s*[\[(]?([A-Za-z0-9]+)[\])]?\s*[.!]?\s*$",
        r"^\s*[\[(]?([A-Za-z0-9]+)[\])]?\s*[.!]?\s*$",
        r"(?:^|\n)\s*[\[(]?([A-Za-z0-9]+)[\])]?\s*(?:[.)]|$)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, answer, flags=re.IGNORECASE | re.MULTILINE)
        for raw in reversed(matches):
            candidate = str(raw).upper()
            if candidate in label_set:
                return candidate
    return None


def _score_choice(response: str, spec: Mapping[str, object]) -> ScoreResult:
    expected = str(spec.get("answer") or "").strip().upper()
    if not expected:
        raise ScoreSpecError("choice.answer is required")
    raw_labels = spec.get("labels", ["A", "B", "C", "D"])
    labels = [label.upper() for label in _strings(raw_labels, "choice.labels")]
    if expected not in labels:
        raise ScoreSpecError("choice.answer must occur in choice.labels")
    extracted = _extract_choice(response, labels)
    return _result(float(extracted == expected), extracted, expected=expected)


def _parse_number(raw: str, *, percent_mode: str) -> float:
    value = raw.strip().replace(",", "")
    percent = value.endswith("%")
    if percent:
        if percent_mode == "reject":
            raise ValueError("percentage notation is not allowed")
        value = value[:-1]
    if "/" in value:
        parsed = float(Fraction(value.replace(" ", "")))
    else:
        parsed = float(value)
    if not percent:
        return parsed
    return parsed / 100.0 if percent_mode == "fraction" else parsed


def _extract_number(response: str, *, percent_mode: str) -> tuple[float | None, str | None]:
    answer = visible_answer(response)
    preferred = re.findall(
        r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*\$?\s*(" + _NUMBER.pattern + r")",
        answer,
        flags=re.IGNORECASE,
    )
    candidates = preferred or _NUMBER.findall(answer)
    for raw in reversed(candidates):
        try:
            return _parse_number(raw, percent_mode=percent_mode), raw
        except (ValueError, ZeroDivisionError):
            continue
    return None, None


def _score_numeric(response: str, spec: Mapping[str, object]) -> ScoreResult:
    raw_expected = spec.get("answer")
    if isinstance(raw_expected, bool) or not isinstance(raw_expected, (int, float)):
        raise ScoreSpecError("numeric.answer must be a number")
    expected = float(raw_expected)
    atol = float(spec.get("atol", 1e-9))
    rtol = float(spec.get("rtol", 1e-9))
    if not all(math.isfinite(value) and value >= 0 for value in (atol, rtol)):
        raise ScoreSpecError("numeric tolerances must be non-negative finite numbers")
    percent_mode = str(spec.get("percent_mode", "reject")).strip().lower()
    if percent_mode not in {"reject", "fraction", "number"}:
        raise ScoreSpecError("numeric.percent_mode must be reject, fraction, or number")
    extracted, raw = _extract_number(response, percent_mode=percent_mode)
    passed = extracted is not None and math.isclose(extracted, expected, abs_tol=atol, rel_tol=rtol)
    return _result(
        float(passed),
        extracted,
        expected=expected,
        raw=raw,
        atol=atol,
        rtol=rtol,
        percent_mode=percent_mode,
    )


def _score_contains(response: str, spec: Mapping[str, object]) -> ScoreResult:
    casefold = bool(spec.get("casefold", True))
    text = normalize_text(response, casefold=casefold, strip_punctuation=False)
    required = _strings(spec.get("all", []), "contains.all", nonempty=False)
    any_terms = _strings(spec.get("any", []), "contains.any", nonempty=False)
    forbidden = _strings(spec.get("none", []), "contains.none", nonempty=False)
    if not required and not any_terms and not forbidden:
        raise ScoreSpecError("contains needs at least one of all, any, or none")

    def normalize(value: str) -> str:
        return normalize_text(value, casefold=casefold, strip_punctuation=False)

    required_hits = {term: normalize(term) in text for term in required}
    any_hits = {term: normalize(term) in text for term in any_terms}
    forbidden_hits = {term: normalize(term) in text for term in forbidden}
    checks: list[bool] = list(required_hits.values())
    if any_terms:
        checks.append(any(any_hits.values()))
    if forbidden:
        checks.append(not any(forbidden_hits.values()))
    score = sum(checks) / len(checks) if checks else 0.0
    return _result(
        score,
        text,
        required_hits=required_hits,
        any_hits=any_hits,
        forbidden_hits=forbidden_hits,
    )


def _extract_json(response: str) -> object | None:
    answer = visible_answer(response).strip()
    candidates = [answer]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)
    candidates = fenced + candidates
    first_object, last_object = answer.find("{"), answer.rfind("}")
    if 0 <= first_object < last_object:
        candidates.append(answer[first_object : last_object + 1])
    first_array, last_array = answer.find("["), answer.rfind("]")
    if 0 <= first_array < last_array:
        candidates.append(answer[first_array : last_array + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_subset(actual: object, expected: object) -> tuple[int, int]:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return 0, max(1, len(expected))
        hit = 0
        total = 0
        for key, expected_value in expected.items():
            child_hit, child_total = _json_subset(actual.get(key), expected_value)
            hit += child_hit
            total += child_total
        return hit, max(total, 1)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return 0, max(1, len(expected))
        pairs = [_json_subset(a, e) for a, e in zip(actual, expected, strict=True)]
        return sum(hit for hit, _ in pairs), max(1, sum(total for _, total in pairs))
    return (1, 1) if actual == expected else (0, 1)


def _score_json(response: str, spec: Mapping[str, object]) -> ScoreResult:
    expected = spec.get("expected")
    if expected is None:
        raise ScoreSpecError("json.expected is required")
    answer = visible_answer(response).strip()
    allow_fence = bool(spec.get("allow_fence", True))
    minified = bool(spec.get("minified", False))
    fenced = bool(re.search(r"```", answer))
    if allow_fence:
        extracted = _extract_json(response)
    else:
        try:
            extracted = json.loads(answer)
        except json.JSONDecodeError:
            extracted = None
    if extracted is None:
        return _result(
            0.0,
            None,
            expected=expected,
            valid_json=False,
            allow_fence=allow_fence,
            fenced=fenced,
            minified=minified,
        )
    hit, total = _json_subset(extracted, expected)
    exact = bool(spec.get("exact", True))
    score = float(extracted == expected) if exact else hit / total
    forbidden_keys = _strings(spec.get("forbidden_keys", []), "json.forbidden_keys", nonempty=False)
    forbidden_present = sorted(
        key for key in forbidden_keys if isinstance(extracted, Mapping) and key in extracted
    )
    if forbidden_present:
        score = 0.0
    canonical_minified = json.dumps(extracted, ensure_ascii=False, separators=(",", ":"))
    minified_match = answer == canonical_minified
    if fenced and not allow_fence:
        score = 0.0
    if minified and not minified_match:
        score = 0.0
    return _result(
        score,
        extracted,
        expected=expected,
        valid_json=True,
        exact=exact,
        allow_fence=allow_fence,
        fenced=fenced,
        minified=minified,
        minified_match=minified_match,
        forbidden_present=forbidden_present,
    )


def _score_regex(response: str, spec: Mapping[str, object]) -> ScoreResult:
    patterns = _strings(spec.get("all", []), "regex.all", nonempty=False)
    forbidden = _strings(spec.get("none", []), "regex.none", nonempty=False)
    if not patterns and not forbidden:
        raise ScoreSpecError("regex needs all or none patterns")
    flags = re.IGNORECASE | re.MULTILINE if bool(spec.get("ignore_case", True)) else re.MULTILINE
    answer = visible_answer(response)
    hits = {pattern: bool(re.search(pattern, answer, flags=flags)) for pattern in patterns}
    forbidden_hits = {pattern: bool(re.search(pattern, answer, flags=flags)) for pattern in forbidden}
    checks = list(hits.values()) + [not value for value in forbidden_hits.values()]
    return _result(
        sum(checks) / len(checks),
        answer,
        pattern_hits=hits,
        forbidden_hits=forbidden_hits,
    )


def _normalize_command(value: str) -> str:
    text = normalize_text(value, casefold=False, strip_punctuation=False)
    text = re.sub(r"\s*([|;&])\s*", r" \1 ", text)
    return " ".join(text.split())


def _score_command(response: str, spec: Mapping[str, object]) -> ScoreResult:
    commands = _strings(spec.get("answers"), "command.answers")
    answer = visible_answer(response).strip()
    only_command = bool(spec.get("only_command", True))
    invalid_wrapper = only_command and (
        "\n" in answer or "\r" in answer or "`" in answer or answer.lower().startswith("command:")
    )
    extracted = _normalize_command(answer)
    expected = [_normalize_command(command) for command in commands]
    return _result(
        float(not invalid_wrapper and extracted in expected),
        extracted,
        expected=expected,
        only_command=only_command,
        invalid_wrapper=invalid_wrapper,
    )


def _score_composite(response: str, spec: Mapping[str, object]) -> ScoreResult:
    raw_components = _sequence(spec.get("components"), "composite.components")
    if not raw_components:
        raise ScoreSpecError("composite.components must not be empty")
    results: list[dict[str, object]] = []
    weighted = 0.0
    total_weight = 0.0
    for index, raw_component in enumerate(raw_components):
        component = _mapping(raw_component, f"composite.components[{index}]")
        weight = component.pop("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            raise ScoreSpecError(f"composite.components[{index}].weight must be positive")
        result = score_response(response, component)
        weighted += result.score * float(weight)
        total_weight += float(weight)
        results.append({"weight": float(weight), **result.to_dict()})
    score = weighted / total_weight
    return _result(score, visible_answer(response), components=results)


def score_response(response: str, raw_spec: Mapping[str, object]) -> ScoreResult:
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    spec = _mapping(raw_spec, "scorer")
    scorer_type = str(spec.pop("type", "")).strip().lower()
    scorers = {
        "choice": _score_choice,
        "command": _score_command,
        "composite": _score_composite,
        "contains": _score_contains,
        "exact": _score_exact,
        "json": _score_json,
        "numeric": _score_numeric,
        "regex": _score_regex,
    }
    scorer = scorers.get(scorer_type)
    if scorer is None:
        raise ScoreSpecError(
            f"unsupported scorer type {scorer_type!r}; choose one of: " + ", ".join(sorted(scorers))
        )
    result = scorer(response, spec)
    return ScoreResult(
        score=result.score,
        passed=result.passed,
        extracted=result.extracted,
        details={"scorer_type": scorer_type, **result.details},
    )


_SEMANTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _semantic_result(
    score: float,
    extracted: object | None,
    strict: ScoreResult,
    **details: object,
) -> ScoreResult:
    return _result(
        score,
        extracted,
        strict_score=strict.score,
        strict_passed=strict.passed,
        **details,
    )


def _normalized_answer_occurs(
    response: str,
    expected: str,
    *,
    casefold: bool,
    strip_punctuation: bool,
    collapse_separator_spacing: bool = False,
) -> bool:
    boundary_marker = "␞"
    response_text = visible_answer(response)
    if strip_punctuation:
        # Keep sentence boundaries available for the negation check after
        # punctuation normalization. The marker is a symbol, not punctuation,
        # so normalize_text preserves it without affecting in-clause matches.
        response_text = re.sub(
            r"[!?;]+|\.(?=\s|$)|\n+",
            f" {boundary_marker} ",
            response_text,
        )
    text = normalize_text(response_text, casefold=casefold, strip_punctuation=strip_punctuation)
    answer = normalize_text(expected, casefold=casefold, strip_punctuation=strip_punctuation)
    if collapse_separator_spacing:
        text = re.sub(r"\s*([,|;:/])\s*", r"\1", text)
        answer = re.sub(r"\s*([,|;:/])\s*", r"\1", answer)
    if not answer:
        return False
    start_boundary = r"(?<!\w)" if answer[0].isalnum() else ""
    end_boundary = r"(?!\w)" if answer[-1].isalnum() else ""
    pattern = re.compile(start_boundary + re.escape(answer) + end_boundary)
    for match in pattern.finditer(text):
        before = text[max(0, match.start() - 48) : match.start()]
        # Negation applies within its current clause. A negative statement in
        # the preceding sentence ("the alert did not fire. The server was not
        # down") must not invalidate the expected phrase in the next one.
        before_clause = before.rsplit(boundary_marker, 1)[-1]
        if not strip_punctuation:
            before_clause = re.split(r"[.!?;\n]", before_clause)[-1]
        after = text[match.end() : match.end() + 40]
        denied_before = re.search(
            r"\b(?:not|never|no|isnt|isn't)\b(?:\W+\w+){0,4}\W*$",
            before_clause,
        )
        denied_after = re.match(
            r"\W*(?:is\s+)?(?:false|incorrect|wrong|untrue|not\s+correct)\b",
            after,
        )
        if denied_before is None and denied_after is None:
            return True
    return False


def _prompt_options(prompt: str | None, labels: Sequence[str]) -> dict[str, str]:
    if not prompt:
        return {}
    label_set = {label.upper() for label in labels}
    options: dict[str, str] = {}
    for line in prompt.splitlines():
        match = re.match(r"^\s*([A-Za-z0-9]+)\s*[.) :]\s+(.+?)\s*$", line)
        if match and match.group(1).upper() in label_set:
            options[match.group(1).upper()] = match.group(2)
    return options


def _content_tokens(value: str) -> set[str]:
    normalized = normalize_text(value, casefold=True, strip_punctuation=True)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if token not in _SEMANTIC_STOPWORDS and len(token) > 1
    }


def _semantic_choice(
    response: str,
    spec: Mapping[str, object],
    strict: ScoreResult,
    prompt: str | None,
) -> ScoreResult:
    expected = str(spec.get("answer") or "").strip().upper()
    labels = [label.upper() for label in _strings(spec.get("labels", ["A", "B", "C", "D"]), "choice.labels")]
    answer = visible_answer(response).strip()
    # Do not mistake a verbatim prompt echo containing several option lines for
    # an answer merely because the first English article is "A".
    echoed_labels = {
        match.upper()
        for match in re.findall(r"(?:^|\n)\s*([A-Za-z0-9]+)\s*[.) :]\s+", answer)
        if match.upper() in labels
    }
    options = _prompt_options(prompt, labels)
    explicit = _extract_choice(response, labels) if len(echoed_labels) < 2 else None
    if explicit is not None and explicit != expected:
        return _semantic_result(
            0.0,
            explicit,
            strict,
            expected=expected,
            matched_options=[],
            prompt_echo=[],
            mode="explicit-choice-contradiction",
        )
    phrase_matches: set[str] = set()
    loose_matches: set[str] = set()
    response_tokens = _content_tokens(response)
    for label, option in options.items():
        tokens = _content_tokens(option)
        if _normalized_answer_occurs(
            response,
            option,
            casefold=True,
            strip_punctuation=True,
        ):
            phrase_matches.add(label)
        elif (
            tokens
            and tokens.issubset(response_tokens)
            and all(
                _normalized_answer_occurs(
                    response,
                    token,
                    casefold=True,
                    strip_punctuation=True,
                )
                for token in tokens
            )
        ):
            loose_matches.add(label)
    # A complete option phrase is stronger evidence than unrelated keywords
    # spread across an explanation or a restated premise.
    matched_options = phrase_matches or loose_matches
    extracted = None
    if len(echoed_labels) < 2:
        strong = re.match(r"^\s*[\[(]?([A-Za-z0-9]+)[\]).:-]\s*", answer)
        if strong and strong.group(1).upper() in labels:
            extracted = strong.group(1).upper()
        else:
            relaxed = re.match(r"^\s*([A-Za-z0-9]+)\s+", answer)
            if relaxed and relaxed.group(1).upper() in labels:
                candidate = relaxed.group(1).upper()
                # A leading uppercase B/C/D is an answer label followed by an
                # explanation. "A" remains guarded because it is also a common
                # English article ("A service ...").
                if not options or candidate in matched_options or candidate != "A":
                    extracted = candidate
    if extracted is None and len(matched_options) == 1:
        extracted = next(iter(matched_options))
    return _semantic_result(
        float(extracted == expected),
        extracted,
        strict,
        expected=expected,
        matched_options=sorted(matched_options),
        prompt_echo=sorted(echoed_labels) if len(echoed_labels) >= 2 else [],
        mode="choice-content",
    )


def _semantic_numeric(response: str, spec: Mapping[str, object], strict: ScoreResult) -> ScoreResult:
    expected = float(spec["answer"])
    atol = float(spec.get("atol", 1e-9))
    rtol = float(spec.get("rtol", 1e-9))
    percent_mode = str(spec.get("percent_mode", "reject")).strip().lower()
    answer = visible_answer(response)
    number_pattern = r"(" + _NUMBER.pattern + r")"

    def parsed(raw_values: Sequence[str]) -> list[float]:
        values: list[float] = []
        for raw in raw_values:
            try:
                values.append(_parse_number(raw, percent_mode=percent_mode))
            except (ValueError, ZeroDivisionError):
                continue
        return values

    def conclude(values: Sequence[float], mode: str) -> ScoreResult:
        candidate = values[-1] if values else None
        passed = candidate is not None and math.isclose(
            candidate,
            expected,
            abs_tol=atol,
            rel_tol=rtol,
        )
        return _semantic_result(
            float(passed),
            candidate,
            strict,
            expected=expected,
            candidates=list(values),
            mode=mode,
        )

    explicit = re.findall(
        r"(?:\*{1,2}|_{1,2})?\s*(?:final\s+answer|answer)"
        r"(?:\*{1,2}|_{1,2})?\s*(?:is|:|=)?"
        r"(?:\*{1,2}|_{1,2})?\s*\$?\s*" + number_pattern,
        answer,
        flags=re.IGNORECASE,
    )
    explicit_values = parsed(explicit)
    if explicit_values:
        return conclude(explicit_values, "explicit-final")

    boxed_values = parsed(
        re.findall(
            r"\\boxed\{\s*\$?\s*" + number_pattern + r"\s*\}",
            answer,
            flags=re.IGNORECASE,
        )
    )
    if boxed_values:
        return conclude(boxed_values, "boxed-conclusion")

    stripped = answer.rstrip()
    if re.search(r"(?:[+*/=-]|(?<!\*)-)\s*$", stripped):
        return _semantic_result(
            0.0,
            None,
            strict,
            expected=expected,
            candidates=[],
            mode="unfinished-expression",
        )

    terminal_equation = re.findall(
        r"=\s*\$?\s*" + number_pattern + r"\s*(?:[A-Za-z%]+)?\s*[.!]?\s*(?:\$\$|\\\]|```)?\s*$",
        answer,
        flags=re.IGNORECASE,
    )
    terminal_values = parsed(terminal_equation)
    if terminal_values:
        return conclude(terminal_values, "terminal-equation")

    conclusions = re.findall(
        r"(?:yields?|equals?|equivalent\s+to|which\s+is|therefore|thus|hence|"
        r"result(?:\s+is)?|value\s+is)"
        r"\s*(?:is|:|=)?\s*\$?\s*" + number_pattern,
        answer,
        flags=re.IGNORECASE,
    )
    conclusion_values = parsed(conclusions)
    if conclusion_values:
        return conclude(conclusion_values, "supported-conclusion")

    candidates = parsed(_NUMBER.findall(answer))
    distinct: list[float] = []
    for value in candidates:
        if not any(math.isclose(value, other, abs_tol=atol, rel_tol=rtol) for other in distinct):
            distinct.append(value)
    if len(distinct) != 1:
        return _semantic_result(
            0.0,
            None,
            strict,
            expected=expected,
            candidates=candidates,
            mode="ambiguous-numbers" if distinct else "no-number",
        )
    try:
        candidate = distinct[0]
    except IndexError:
        candidate = None
    passed = candidate is not None and math.isclose(candidate, expected, abs_tol=atol, rel_tol=rtol)
    return _semantic_result(
        float(passed),
        candidate,
        strict,
        expected=expected,
        candidates=candidates,
        mode="sole-candidate",
    )


def _semantic_command(response: str, spec: Mapping[str, object], strict: ScoreResult) -> ScoreResult:
    expected = {_normalize_command(command) for command in _strings(spec.get("answers"), "command.answers")}
    answer = visible_answer(response).strip()
    candidates = [answer]
    candidates.extend(re.findall(r"```(?:\w+)?\s*(.*?)```", answer, flags=re.DOTALL))
    candidates.extend(line for line in answer.splitlines() if line.strip())
    normalized: set[str] = set()
    for candidate in candidates:
        candidate = re.sub(r"^\s*(?:command|answer)\s*:\s*", "", candidate, flags=re.IGNORECASE)
        candidate = candidate.strip().strip("`").strip()
        if candidate:
            normalized.add(_normalize_command(candidate))
    matches = sorted(expected & normalized)
    return _semantic_result(
        float(bool(matches)),
        matches[0] if matches else strict.extracted,
        strict,
        expected=sorted(expected),
        mode="command-contained",
    )


def _semantic_json(response: str, spec: Mapping[str, object], strict: ScoreResult) -> ScoreResult:
    expected = spec.get("expected")
    extracted = _extract_json(response)
    if extracted is None:
        return _semantic_result(0.0, None, strict, expected=expected, valid_json=False, mode="json-content")
    hit, total = _json_subset(extracted, expected)
    forbidden_keys = _strings(spec.get("forbidden_keys", []), "json.forbidden_keys", nonempty=False)
    forbidden_present = sorted(
        key for key in forbidden_keys if isinstance(extracted, Mapping) and key in extracted
    )
    score = 0.0 if forbidden_present else hit / total
    return _semantic_result(
        score,
        extracted,
        strict,
        expected=expected,
        valid_json=True,
        forbidden_present=forbidden_present,
        mode="json-content",
    )


def score_response_semantic(
    response: str,
    raw_spec: Mapping[str, object],
    *,
    prompt: str | None = None,
) -> ScoreResult:
    """Score answer content independently from strict output-shape compliance.

    ``score_response`` remains the immutable Molor scorer. This companion score
    gives semantic credit when the correct value is unambiguously present in
    explanatory prose, code fences, or non-minified JSON. Both are retained so
    a benchmark can distinguish knowledge from instruction-contract behavior.
    """

    spec = _mapping(raw_spec, "scorer")
    scorer_type = str(spec.pop("type", "")).strip().lower()
    strict = score_response(response, raw_spec)
    # Numeric canonical scoring historically accepts the last number, which
    # can be an intermediate value or a truncation boundary. Re-evaluate it
    # semantically even when the compatibility scorer passed.
    if strict.passed and scorer_type != "numeric":
        return _semantic_result(1.0, strict.extracted, strict, mode="strict-pass")
    if scorer_type == "exact":
        raw_answers = spec.get("answers")
        if raw_answers is None and "answer" in spec:
            raw_answers = [spec["answer"]]
        answers = _strings(raw_answers, "exact.answers")
        casefold = bool(spec.get("casefold", True))
        strip_punctuation = bool(spec.get("strip_punctuation", True))
        if prompt:
            normalized_response = normalize_text(response, casefold=True, strip_punctuation=True)
            normalized_prompt = normalize_text(prompt, casefold=True, strip_punctuation=True)
            if (
                len(normalized_response) >= 20
                and normalized_response != ""
                and normalized_response in normalized_prompt
            ):
                return _semantic_result(
                    0.0,
                    strict.extracted,
                    strict,
                    expected=answers,
                    mode="prompt-echo",
                )
        explicit_final = re.findall(
            r"(?:final\s+answer|answer|result)\s*(?:is|:|=)\s*(.+?)(?:\n|$)",
            visible_answer(response),
            flags=re.IGNORECASE,
        )
        if explicit_final and not any(
            _normalized_answer_occurs(
                explicit_final[-1],
                answer,
                casefold=casefold,
                strip_punctuation=strip_punctuation,
            )
            for answer in answers
        ):
            return _semantic_result(
                0.0,
                explicit_final[-1].strip(),
                strict,
                expected=answers,
                mode="explicit-final-contradiction",
            )
        matches = [
            answer
            for answer in answers
            if _normalized_answer_occurs(
                response,
                answer,
                casefold=casefold,
                strip_punctuation=strip_punctuation,
                collapse_separator_spacing=True,
            )
        ]
        return _semantic_result(
            float(bool(matches)),
            matches[0] if matches else strict.extracted,
            strict,
            expected=answers,
            mode="exact-contained",
        )
    if scorer_type == "choice":
        return _semantic_choice(response, spec, strict, prompt)
    if scorer_type == "numeric":
        return _semantic_numeric(response, spec, strict)
    if scorer_type == "command":
        return _semantic_command(response, spec, strict)
    if scorer_type == "json":
        return _semantic_json(response, spec, strict)
    if scorer_type == "composite":
        raw_components = _sequence(spec.get("components"), "composite.components")
        weighted = 0.0
        total_weight = 0.0
        components: list[dict[str, object]] = []
        for index, raw_component in enumerate(raw_components):
            component = _mapping(raw_component, f"composite.components[{index}]")
            weight = float(component.pop("weight", 1.0))
            result = score_response_semantic(response, component, prompt=prompt)
            weighted += result.score * weight
            total_weight += weight
            components.append({"weight": weight, **result.to_dict()})
        return _semantic_result(
            weighted / total_weight,
            visible_answer(response),
            strict,
            components=components,
            mode="composite-semantic",
        )
    # contains and regex already score content rather than presentation.
    return _semantic_result(strict.score, strict.extracted, strict, mode="strict-equivalent")


def validate_score_spec(raw_spec: Mapping[str, object]) -> None:
    """Validate a scorer declaration without launching an external verifier."""

    spec = _mapping(raw_spec, "scorer")
    scorer_type = str(spec.get("type", "")).strip().lower()
    if scorer_type == "external":
        unknown = set(spec).difference({"type", "verifier"})
        if unknown:
            raise ScoreSpecError("external scorer has unknown fields: " + ", ".join(sorted(unknown)))
        verifier = spec.get("verifier")
        if not isinstance(verifier, str) or _IDENTIFIER.fullmatch(verifier) is None:
            raise ScoreSpecError("external.verifier must be a benchmark identifier")
        return
    score_response("", spec)


__all__ = [
    "SEMANTIC_SCORER_VERSION",
    "SCORER_VERSION",
    "ScoreResult",
    "ScoreSpecError",
    "normalize_text",
    "score_response",
    "score_response_semantic",
    "validate_score_spec",
    "visible_answer",
]
