from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from .quality import QualityReport, score_quality

FORMAT_NAME = "yylab.stage1_corpus_ranker"
FORMAT_VERSION = 1
WEIGHTS_NAME = "ranker.safetensors"
METADATA_NAME = "ranker.json"

MIN_TRAINING_EXAMPLES = 8
MIN_CLASS_EXAMPLES = 2
OTHER_CATEGORY = "__other__"

_FEATURE_SAMPLE_CHARS = 20_000
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_URL_RE = re.compile(r"https?://[^\s)]+|www\.[^\s)]+", re.IGNORECASE)
_SYMBOL_TOKEN_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_BULLET_RE = re.compile(r"^\s*[-*•▪‣]\s+")
_BOILERPLATE_RES = (
    re.compile(r"(?:we|this site) (?:uses|use) cookies|cookie (?:policy|consent)", re.IGNORECASE),
    re.compile(r"subscribe to (?:our|this)|sign up for (?:our|free)", re.IGNORECASE),
    re.compile(r"all rights reserved|©\s*\d{4}", re.IGNORECASE),
    re.compile(r"lorem ipsum", re.IGNORECASE),
    re.compile(r"skip to (?:content|main|navigation)|toggle navigation", re.IGNORECASE),
)
_EN_STOPWORDS = frozenset(
    "the be to of and a in that have it for not on with as at this but by from they we or an will one "
    "all would there their what so if about who which when can into your good could other than now also "
    "after use how our work first well because any these is are was were been has had did does doing".split()
)

CONTINUOUS_FEATURES = (
    "log_chars",
    "log_words",
    "avg_word_length",
    "alphabetic_word_fraction",
    "unique_word_fraction",
    "stopword_fraction",
    "symbol_token_fraction",
    "url_word_fraction",
    "log_nonempty_lines",
    "line_punctuation_fraction",
    "bullet_line_fraction",
    "duplicate_line_char_fraction",
    "duplicate_paragraph_char_fraction",
    "top_bigram_token_fraction",
    "top_trigram_token_fraction",
    "uppercase_letter_fraction",
    "digit_char_fraction",
    "whitespace_char_fraction",
    "boilerplate_hit_fraction",
    "quality_score",
    "quality_reason_count",
)


class RankerError(ValueError):
    """Base error for invalid ranker data or artifacts."""


class RankerTrainingError(RankerError):
    """Raised when labels cannot support a useful binary model."""


class RankerPersistenceError(RankerError):
    """Raised when a saved ranker is incomplete, incompatible, or corrupt."""


@dataclass(frozen=True, slots=True)
class LabeledDocument:
    text: str
    label: bool | int | str
    domain: str = "other"
    lang: str = "unknown"
    quality_report: QualityReport | Mapping[str, Any] | float | None = None
    group: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    strategy: str


@dataclass(frozen=True, slots=True)
class _PreparedDocument:
    text: str
    label: int
    domain: str
    lang: str
    continuous: tuple[float, ...]
    group: str | None


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _quality_values(
    text: str,
    quality_report: QualityReport | Mapping[str, Any] | float | None,
) -> tuple[float, int]:
    if quality_report is None:
        report = score_quality(text)
        return float(report.score), len(report.reasons)
    if isinstance(quality_report, (int, float)) and not isinstance(quality_report, bool):
        score = float(quality_report)
        reasons: Sequence[Any] = ()
    elif isinstance(quality_report, Mapping):
        raw_score = quality_report.get("score", quality_report.get("quality", 1.0))
        score = float(1.0 if raw_score is None else raw_score)
        raw_reasons = quality_report.get("reasons", quality_report.get("quarantine_reasons", ()))
        reasons = raw_reasons if isinstance(raw_reasons, Sequence) and not isinstance(raw_reasons, str) else ()
    else:
        try:
            score = float(quality_report.score)
            reasons = quality_report.reasons
        except (AttributeError, TypeError, ValueError) as exc:
            raise RankerError("quality_report must be a number, mapping, or QualityReport") from exc
    if not math.isfinite(score):
        raise RankerError("quality score must be finite")
    return min(max(score, 0.0), 1.0), len(reasons)


def featurize(
    text: str,
    quality_report: QualityReport | Mapping[str, Any] | float | None = None,
) -> dict[str, float]:
    """Return the fixed, deterministic Stage-1 continuous feature schema."""
    if not isinstance(text, str):
        raise RankerError("document text must be a string")

    sample = text[:_FEATURE_SAMPLE_CHARS]
    words = [match.group(0).lower().replace("’", "'") for match in _TOKEN_RE.finditer(sample)]
    raw_tokens = sample.split()
    lines = [line.strip() for line in sample.splitlines() if line.strip()]
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", sample) if paragraph.strip()]

    n_words = len(words)
    n_chars = len(sample)
    avg_word_length = _safe_fraction(sum(len(word) for word in words), n_words)
    alpha_words = sum(1 for token in raw_tokens if any(char.isalpha() for char in token))
    unique_word_fraction = _safe_fraction(len(set(words)), n_words)
    stopword_fraction = _safe_fraction(sum(word in _EN_STOPWORDS for word in words), n_words)
    symbol_token_fraction = _safe_fraction(
        sum(bool(_SYMBOL_TOKEN_RE.fullmatch(token)) for token in raw_tokens), len(raw_tokens)
    )
    url_word_fraction = _safe_fraction(len(_URL_RE.findall(sample)), n_words)

    line_punctuation_fraction = _safe_fraction(
        sum(line[-1] in ".!?…\"')]}" for line in lines), len(lines)
    )
    bullet_line_fraction = _safe_fraction(sum(bool(_BULLET_RE.match(line)) for line in lines), len(lines))

    line_counts = Counter(lines)
    duplicate_line_chars = sum(len(line) * (count - 1) for line, count in line_counts.items() if count > 1)
    total_line_chars = sum(len(line) for line in lines)
    duplicate_line_fraction = _safe_fraction(duplicate_line_chars, total_line_chars)

    paragraph_counts = Counter(paragraphs)
    duplicate_paragraph_chars = sum(
        len(paragraph) * (count - 1)
        for paragraph, count in paragraph_counts.items()
        if count > 1
    )
    total_paragraph_chars = sum(len(paragraph) for paragraph in paragraphs)
    duplicate_paragraph_fraction = _safe_fraction(duplicate_paragraph_chars, total_paragraph_chars)

    bigrams = Counter(zip(words, words[1:]))
    trigrams = Counter(zip(words, words[1:], words[2:]))
    top_bigram_fraction = _safe_fraction(2 * max(bigrams.values(), default=0), n_words)
    top_trigram_fraction = _safe_fraction(3 * max(trigrams.values(), default=0), n_words)

    letters = [char for char in sample if char.isalpha()]
    uppercase_fraction = _safe_fraction(sum(char.isupper() for char in letters), len(letters))
    digit_fraction = _safe_fraction(sum(char.isdigit() for char in sample), n_chars)
    whitespace_fraction = _safe_fraction(sum(char.isspace() for char in sample), n_chars)
    boilerplate_hits = sum(bool(pattern.search(sample)) for pattern in _BOILERPLATE_RES)
    quality_score, quality_reason_count = _quality_values(text, quality_report)

    values = {
        "log_chars": math.log1p(n_chars),
        "log_words": math.log1p(n_words),
        "avg_word_length": avg_word_length,
        "alphabetic_word_fraction": _safe_fraction(alpha_words, len(raw_tokens)),
        "unique_word_fraction": unique_word_fraction,
        "stopword_fraction": stopword_fraction,
        "symbol_token_fraction": symbol_token_fraction,
        "url_word_fraction": url_word_fraction,
        "log_nonempty_lines": math.log1p(len(lines)),
        "line_punctuation_fraction": line_punctuation_fraction,
        "bullet_line_fraction": bullet_line_fraction,
        "duplicate_line_char_fraction": duplicate_line_fraction,
        "duplicate_paragraph_char_fraction": duplicate_paragraph_fraction,
        "top_bigram_token_fraction": top_bigram_fraction,
        "top_trigram_token_fraction": top_trigram_fraction,
        "uppercase_letter_fraction": uppercase_fraction,
        "digit_char_fraction": digit_fraction,
        "whitespace_char_fraction": whitespace_fraction,
        "boilerplate_hit_fraction": boilerplate_hits / len(_BOILERPLATE_RES),
        "quality_score": quality_score,
        "quality_reason_count": float(quality_reason_count),
    }
    return {name: float(values[name]) for name in CONTINUOUS_FEATURES}


def _normalize_label(label: bool | int | str) -> int:
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, np.integer)) and int(label) in {0, 1}:
        return int(label)
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized in {"good", "approved", "approve", "accepted", "positive", "1"}:
            return 1
        if normalized in {"bad", "rejected", "reject", "negative", "0"}:
            return 0
    raise RankerTrainingError(f"unsupported binary label: {label!r}")


def _normalize_category(value: Any, *, default: str) -> str:
    normalized = str(value or default).strip().lower()
    return normalized or default


def _coerce_example(example: LabeledDocument | Mapping[str, Any], index: int) -> _PreparedDocument:
    if isinstance(example, LabeledDocument):
        text = example.text
        label = example.label
        domain = example.domain
        lang = example.lang
        quality_report = example.quality_report
        group = example.group
    elif isinstance(example, Mapping):
        text = example.get("text")
        if "label" in example:
            label = example["label"]
        elif "verdict" in example:
            label = example["verdict"]
        else:
            raise RankerTrainingError(f"example {index} has no label or verdict")
        domain = example.get("domain", "other")
        lang = example.get("lang", example.get("language", "unknown"))
        # Only an explicit, complete report may override recomputation. Historical
        # annotations inconsistently stored the scalar score/reasons; consuming those
        # partial fields would teach the model which ingest version produced a row.
        quality_report = example.get("quality_report")
        group = example.get("group", example.get("corpus_id"))
    else:
        raise RankerTrainingError(f"example {index} must be a mapping or LabeledDocument")

    if not isinstance(text, str) or not text.strip():
        raise RankerTrainingError(f"example {index} has empty or non-string text")
    features = featurize(text, quality_report)
    return _PreparedDocument(
        text=text,
        label=_normalize_label(label),
        domain=_normalize_category(domain, default="other"),
        lang=_normalize_category(lang, default="unknown"),
        continuous=tuple(features[name] for name in CONTINUOUS_FEATURES),
        group=None if group is None else str(group),
    )


def _validate_training_labels(labels: np.ndarray, minimum_examples: int) -> None:
    if len(labels) < minimum_examples:
        raise RankerTrainingError(
            f"at least {minimum_examples} labeled documents are required; got {len(labels)}"
        )
    counts = np.bincount(labels, minlength=2)
    if not counts[0] or not counts[1]:
        observed = "good" if counts[1] else "bad"
        raise RankerTrainingError(
            f"cannot train a binary ranker from a single class ({observed}); collect both good and bad labels"
        )
    if min(counts) < MIN_CLASS_EXAMPLES:
        raise RankerTrainingError(
            f"at least {MIN_CLASS_EXAMPLES} labels from each class are required; "
            f"got bad={counts[0]} and good={counts[1]}"
        )


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8", errors="surrogatepass")).hexdigest()


def _stratified_indices(labels: np.ndarray, validation_fraction: float, seed: int) -> ValidationSplit:
    validation: list[int] = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label).tolist()
        indices.sort(key=lambda index: _stable_key(seed, f"{label}:{index}"))
        count = min(max(int(round(len(indices) * validation_fraction)), 1), len(indices) - 1)
        validation.extend(indices[:count])
    validation_set = set(validation)
    train = tuple(index for index in range(len(labels)) if index not in validation_set)
    return ValidationSplit(train, tuple(sorted(validation)), "stratified")


def _group_token(value: Any) -> str:
    return f"{type(value).__qualname__}:{value!r}"


def _split_cost(
    positive: int,
    negative: int,
    total: int,
    targets: tuple[float, float, float],
) -> float:
    target_positive, target_negative, target_total = targets
    return (
        ((positive - target_positive) / max(target_positive, 1.0)) ** 2
        + ((negative - target_negative) / max(target_negative, 1.0)) ** 2
        + 0.25 * ((total - target_total) / max(target_total, 1.0)) ** 2
    )


def _grouped_indices(
    labels: np.ndarray,
    groups: Sequence[Any],
    validation_fraction: float,
    seed: int,
) -> ValidationSplit | None:
    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        token = _group_token(group)
        members.setdefault(token, []).append(index)
    if len(members) < 2:
        return None

    ordered = sorted(members, key=lambda token: (_stable_key(seed, token), token))
    stats = [
        (
            sum(int(labels[index]) for index in members[token]),
            sum(1 - int(labels[index]) for index in members[token]),
            len(members[token]),
        )
        for token in ordered
    ]
    total_positive = int(labels.sum())
    total_negative = len(labels) - total_positive
    targets = (
        total_positive * validation_fraction,
        total_negative * validation_fraction,
        len(labels) * validation_fraction,
    )

    def valid(positive: int, negative: int) -> bool:
        return (
            positive > 0
            and negative > 0
            and positive < total_positive
            and negative < total_negative
        )

    best_selection: tuple[int, ...] | None = None
    best_rank: tuple[float, float, int] | None = None
    n_groups = len(ordered)
    if n_groups <= 16:
        for mask in range(1, (1 << n_groups) - 1):
            selected = tuple(index for index in range(n_groups) if mask & (1 << index))
            positive = sum(stats[index][0] for index in selected)
            negative = sum(stats[index][1] for index in selected)
            if not valid(positive, negative):
                continue
            total = positive + negative
            rank = (
                _split_cost(positive, negative, total, targets),
                abs(total - targets[2]),
                mask,
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_selection = selected
    else:
        selected: list[int] = []
        remaining = set(range(n_groups))
        positive = negative = total = 0
        while remaining:
            candidates: list[tuple[float, str, int, int, int, int]] = []
            for index in remaining:
                next_positive = positive + stats[index][0]
                next_negative = negative + stats[index][1]
                if next_positive >= total_positive or next_negative >= total_negative:
                    continue
                next_total = total + stats[index][2]
                missing_class_penalty = 10.0 * int(next_positive == 0 or next_negative == 0)
                candidates.append(
                    (
                        _split_cost(next_positive, next_negative, next_total, targets)
                        + missing_class_penalty,
                        ordered[index],
                        index,
                        next_positive,
                        next_negative,
                        next_total,
                    )
                )
            if not candidates:
                break
            _, _, index, positive, negative, total = min(candidates)
            selected.append(index)
            remaining.remove(index)
            if valid(positive, negative):
                rank = (
                    _split_cost(positive, negative, total, targets),
                    abs(total - targets[2]),
                    len(selected),
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_selection = tuple(selected)
                if total >= targets[2] and rank[0] <= 0.1:
                    break

    if best_selection is None:
        return None
    validation_groups = {ordered[index] for index in best_selection}
    validation = tuple(
        index for index, group in enumerate(groups) if _group_token(group) in validation_groups
    )
    validation_set = set(validation)
    train = tuple(index for index in range(len(labels)) if index not in validation_set)
    return ValidationSplit(train, validation, "stratified_group")


def deterministic_validation_split(
    labels: Sequence[bool | int | str],
    *,
    groups: Sequence[Any] | None = None,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> ValidationSplit:
    """Create a repeatable stratified holdout, preserving groups when feasible."""
    if not 0.0 < validation_fraction < 1.0:
        raise RankerTrainingError("validation_fraction must be strictly between 0 and 1")
    normalized = np.asarray([_normalize_label(label) for label in labels], dtype=np.int64)
    _validate_training_labels(normalized, 4)
    if groups is not None:
        if len(groups) != len(normalized):
            raise RankerTrainingError("groups and labels must have the same length")
        grouped = _grouped_indices(normalized, groups, validation_fraction, seed)
        if grouped is not None:
            return grouped
        raise RankerTrainingError(
            "no grouped holdout can keep both classes in both train and validation; "
            "collect labels from more groups"
        )
    return _stratified_indices(normalized, validation_fraction, seed)


def _category_schema(documents: Sequence[_PreparedDocument]) -> dict[str, tuple[str, ...]]:
    domains = sorted({document.domain for document in documents} - {OTHER_CATEGORY})
    languages = sorted({document.lang for document in documents} - {OTHER_CATEGORY})
    return {
        "domain": tuple([*domains, OTHER_CATEGORY]),
        "lang": tuple([*languages, OTHER_CATEGORY]),
    }


def _encoded_feature_names(categories: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    return (
        *CONTINUOUS_FEATURES,
        *(f"domain={category}" for category in categories["domain"]),
        *(f"lang={category}" for category in categories["lang"]),
    )


def _continuous_matrix(documents: Sequence[_PreparedDocument]) -> np.ndarray:
    return np.asarray([document.continuous for document in documents], dtype=np.float64)


def _design_matrix(
    documents: Sequence[_PreparedDocument],
    categories: Mapping[str, Sequence[str]],
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    continuous = (_continuous_matrix(documents) - mean) / scale
    domain_index = {category: index for index, category in enumerate(categories["domain"])}
    lang_index = {category: index for index, category in enumerate(categories["lang"])}
    domain = np.zeros((len(documents), len(domain_index)), dtype=np.float64)
    language = np.zeros((len(documents), len(lang_index)), dtype=np.float64)
    for row, document in enumerate(documents):
        domain[row, domain_index.get(document.domain, domain_index[OTHER_CATEGORY])] = 1.0
        language[row, lang_index.get(document.lang, lang_index[OTHER_CATEGORY])] = 1.0
    return np.concatenate((continuous, domain, language), axis=1)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _objective(matrix: np.ndarray, labels: np.ndarray, parameters: np.ndarray, l2: float) -> float:
    logits = matrix @ parameters
    data_loss = (np.logaddexp(0.0, logits).sum() - np.dot(labels, logits)) / len(labels)
    return float(data_loss + 0.5 * l2 * np.dot(parameters[1:], parameters[1:]))


def _fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    augmented = np.column_stack((np.ones(len(matrix), dtype=np.float64), matrix))
    parameters = np.zeros(augmented.shape[1], dtype=np.float64)
    positive_rate = float(np.clip(labels.mean(), 1e-6, 1.0 - 1e-6))
    parameters[0] = math.log(positive_rate / (1.0 - positive_rate))
    converged = False
    objective = _objective(augmented, labels, parameters, l2)

    for iteration in range(1, max_iter + 1):
        probabilities = _sigmoid(augmented @ parameters)
        gradient = augmented.T @ (probabilities - labels) / len(labels)
        gradient[1:] += l2 * parameters[1:]
        curvature = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
        hessian = augmented.T @ (augmented * curvature[:, None]) / len(labels)
        hessian[1:, 1:] += l2 * np.eye(augmented.shape[1] - 1)
        hessian.flat[:: hessian.shape[0] + 1] += 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        step_size = 1.0
        candidate = parameters - step
        candidate_objective = _objective(augmented, labels, candidate, l2)
        while candidate_objective > objective and step_size > 1e-6:
            step_size *= 0.5
            candidate = parameters - step_size * step
            candidate_objective = _objective(augmented, labels, candidate, l2)
        if not np.isfinite(candidate_objective):
            raise RankerTrainingError("logistic regression optimization produced non-finite values")

        parameter_change = float(np.max(np.abs(candidate - parameters)))
        objective_change = abs(objective - candidate_objective)
        parameters = candidate
        objective = candidate_objective
        if parameter_change <= tolerance or objective_change <= tolerance * max(1.0, objective):
            converged = True
            break

    if not converged:
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm > max(1e-4, tolerance * 100):
            raise RankerTrainingError(
                f"logistic regression did not converge in {max_iter} iterations "
                f"(gradient={gradient_norm:.3g})"
            )
    return parameters, {
        "converged": converged,
        "iterations": iteration,
        "objective": objective,
    }


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = np.argsort(probabilities, kind="mergesort")
    ranks = np.empty(len(probabilities), dtype=np.float64)
    sorted_probabilities = probabilities[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_probabilities[end] == sorted_probabilities[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = int(labels.sum())
    if not positive:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        positives = int(sorted_labels[start:end].sum())
        true_positive += positives
        false_positive += (end - start) - positives
        recall = true_positive / positive
        precision = _safe_fraction(true_positive, true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(area)


def _expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    total = max(len(labels), 1)
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        upper = probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1]
        mask = (probabilities >= edges[index]) & upper
        count = int(mask.sum())
        if not count:
            continue
        confidence = float(probabilities[mask].mean())
        observed = float(labels[mask].mean())
        error += count / total * abs(confidence - observed)
    return float(error)


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities >= 0.5
    true_positive = int(np.sum(predictions & (labels == 1)))
    false_positive = int(np.sum(predictions & (labels == 0)))
    false_negative = int(np.sum(~predictions & (labels == 1)))
    true_negative = int(np.sum(~predictions & (labels == 0)))
    precision = _safe_fraction(true_positive, true_positive + false_positive)
    recall = _safe_fraction(true_positive, true_positive + false_negative)
    specificity = _safe_fraction(true_negative, true_negative + false_positive)
    bad_precision = _safe_fraction(true_negative, true_negative + false_negative)
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return {
        "count": len(labels),
        "positive_rate": float(labels.mean()),
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "calibration_error": _expected_calibration_error(labels, probabilities),
        "accuracy": float(np.mean(predictions == labels)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "f1": _safe_fraction(2 * precision * recall, precision + recall),
        "bad_precision": bad_precision,
        "bad_recall": specificity,
        "bad_f1": _safe_fraction(2 * bad_precision * specificity, bad_precision + specificity),
        "roc_auc": _roc_auc(labels, probabilities),
        "good_pr_auc": _average_precision(labels, probabilities),
        "bad_pr_auc": _average_precision(1 - labels, 1.0 - probabilities),
    }


def _fit_parameters(
    documents: Sequence[_PreparedDocument],
    *,
    l2: float,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, dict[str, tuple[str, ...]], dict[str, Any]]:
    continuous = _continuous_matrix(documents)
    mean = continuous.mean(axis=0)
    scale = continuous.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    categories = _category_schema(documents)
    matrix = _design_matrix(documents, categories, mean, scale)
    labels = np.asarray([document.label for document in documents], dtype=np.float64)
    parameters, optimizer = _fit_logistic(
        matrix,
        labels,
        l2=l2,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    return parameters[1:], float(parameters[0]), mean, scale, categories, optimizer


def _model_hash(
    coefficients: np.ndarray,
    intercept: float,
    mean: np.ndarray,
    scale: np.ndarray,
    categories: Mapping[str, Sequence[str]],
) -> str:
    schema = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "continuous_features": list(CONTINUOUS_FEATURES),
        "categories": {key: list(categories[key]) for key in ("domain", "lang")},
    }
    digest = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name, array in (
        ("coefficients", coefficients),
        ("intercept", np.asarray([intercept], dtype=np.float64)),
        ("mean", mean),
        ("scale", scale),
    ):
        normalized = np.ascontiguousarray(array, dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(str(normalized.shape).encode("ascii"))
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def _training_data_hash(documents: Sequence[_PreparedDocument]) -> str:
    records: list[bytes] = []
    for document in documents:
        payload = json.dumps(
            {
                "text": document.text,
                "label": document.label,
                "domain": document.domain,
                "lang": document.lang,
                "group": document.group,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="surrogatepass")
        records.append(hashlib.sha256(payload).digest())
    digest = hashlib.sha256()
    for record in sorted(records):
        digest.update(record)
    return digest.hexdigest()


@dataclass(slots=True)
class Stage1Ranker:
    coefficients: np.ndarray
    intercept: float
    continuous_mean: np.ndarray
    continuous_scale: np.ndarray
    categories: dict[str, tuple[str, ...]]
    metrics: dict[str, Any] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    model_hash: str = ""

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=np.float64)
        self.continuous_mean = np.asarray(self.continuous_mean, dtype=np.float64)
        self.continuous_scale = np.asarray(self.continuous_scale, dtype=np.float64)
        self.categories = {
            "domain": tuple(self.categories["domain"]),
            "lang": tuple(self.categories["lang"]),
        }
        expected = len(_encoded_feature_names(self.categories))
        if self.coefficients.shape != (expected,):
            raise RankerPersistenceError(
                f"coefficient shape {self.coefficients.shape} does not match {expected} encoded features"
            )
        if self.continuous_mean.shape != (len(CONTINUOUS_FEATURES),):
            raise RankerPersistenceError("continuous feature mean has the wrong shape")
        if self.continuous_scale.shape != (len(CONTINUOUS_FEATURES),):
            raise RankerPersistenceError("continuous feature scale has the wrong shape")
        if not np.all(np.isfinite(self.coefficients)) or not math.isfinite(self.intercept):
            raise RankerPersistenceError("model weights must be finite")
        if not np.all(np.isfinite(self.continuous_mean)):
            raise RankerPersistenceError("scaler means must be finite")
        if not np.all(np.isfinite(self.continuous_scale)) or np.any(self.continuous_scale <= 0):
            raise RankerPersistenceError("scaler scales must be finite and positive")
        for name in ("domain", "lang"):
            if OTHER_CATEGORY not in self.categories[name]:
                raise RankerPersistenceError(f"{name} categories must include {OTHER_CATEGORY!r}")
            if len(set(self.categories[name])) != len(self.categories[name]):
                raise RankerPersistenceError(f"{name} categories must be unique")

        computed_hash = _model_hash(
            self.coefficients,
            self.intercept,
            self.continuous_mean,
            self.continuous_scale,
            self.categories,
        )
        if self.model_hash and self.model_hash != computed_hash:
            raise RankerPersistenceError("model hash does not match weights and schema")
        self.model_hash = computed_hash

    @property
    def model_version(self) -> str:
        return f"stage1-v{FORMAT_VERSION}-{self.model_hash[:12]}"

    @property
    def feature_names(self) -> tuple[str, ...]:
        return _encoded_feature_names(self.categories)

    def _document_vector(
        self,
        text: str,
        *,
        domain: str,
        lang: str,
        quality_report: QualityReport | Mapping[str, Any] | float | None,
    ) -> np.ndarray:
        continuous_values = featurize(text, quality_report)
        continuous = np.asarray(
            [continuous_values[name] for name in CONTINUOUS_FEATURES], dtype=np.float64
        )
        standardized = (continuous - self.continuous_mean) / self.continuous_scale
        domain_values = np.zeros(len(self.categories["domain"]), dtype=np.float64)
        lang_values = np.zeros(len(self.categories["lang"]), dtype=np.float64)
        normalized_domain = _normalize_category(domain, default="other")
        normalized_lang = _normalize_category(lang, default="unknown")
        domain_index = {value: index for index, value in enumerate(self.categories["domain"])}
        lang_index = {value: index for index, value in enumerate(self.categories["lang"])}
        domain_values[domain_index.get(normalized_domain, domain_index[OTHER_CATEGORY])] = 1.0
        lang_values[lang_index.get(normalized_lang, lang_index[OTHER_CATEGORY])] = 1.0
        return np.concatenate((standardized, domain_values, lang_values))

    def score(
        self,
        text: str,
        *,
        domain: str = "other",
        lang: str = "unknown",
        quality_report: QualityReport | Mapping[str, Any] | float | None = None,
    ) -> float:
        vector = self._document_vector(
            text,
            domain=domain,
            lang=lang,
            quality_report=quality_report,
        )
        logit = self.intercept + float(np.dot(vector, self.coefficients))
        return float(_sigmoid(np.asarray([logit]))[0])

    def uncertainty(
        self,
        text: str,
        *,
        domain: str = "other",
        lang: str = "unknown",
        quality_report: QualityReport | Mapping[str, Any] | float | None = None,
    ) -> float:
        probability = self.score(
            text,
            domain=domain,
            lang=lang,
            quality_report=quality_report,
        )
        return 1.0 - abs(2.0 * probability - 1.0)

    @staticmethod
    def _doc_quality(document: Mapping[str, Any]) -> Any:
        # A scalar shard score is not a complete report and was absent for many
        # historical rows. Recompute unless a caller deliberately provides the full
        # feature input under ``quality_report``.
        return document.get("quality_report")

    def score_doc(self, document: Mapping[str, Any]) -> dict[str, float | str]:
        text = document.get("text")
        if not isinstance(text, str):
            raise RankerError("document text must be a string")
        score = self.score(
            text,
            domain=str(document.get("domain", "other")),
            lang=str(document.get("lang", document.get("language", "unknown"))),
            quality_report=self._doc_quality(document),
        )
        return {
            "rank_score": score,
            "uncertainty": 1.0 - abs(2.0 * score - 1.0),
            "model_version": self.model_version,
        }

    def score_docs(self, documents: Sequence[Mapping[str, Any]]) -> list[dict[str, float | str]]:
        return [self.score_doc(document) for document in documents]

    def predict(
        self,
        text: str,
        *,
        threshold: float = 0.5,
        domain: str = "other",
        lang: str = "unknown",
        quality_report: QualityReport | Mapping[str, Any] | float | None = None,
    ) -> bool:
        if not 0.0 <= threshold <= 1.0:
            raise RankerError("threshold must be between 0 and 1")
        return self.score(
            text,
            domain=domain,
            lang=lang,
            quality_report=quality_report,
        ) >= threshold

    def save(self, model_dir: str | Path) -> tuple[Path, Path]:
        directory = Path(model_dir)
        directory.mkdir(parents=True, exist_ok=True)
        weights_path = directory / WEIGHTS_NAME
        metadata_path = directory / METADATA_NAME
        weights_tmp = directory / f".{WEIGHTS_NAME}.tmp"
        metadata_tmp = directory / f".{METADATA_NAME}.tmp"

        tensors = {
            "coefficients": np.ascontiguousarray(self.coefficients, dtype=np.float64),
            "intercept": np.asarray([self.intercept], dtype=np.float64),
            "format_version": np.asarray([FORMAT_VERSION], dtype=np.int64),
        }
        save_file(
            tensors,
            str(weights_tmp),
            metadata={
                "format": FORMAT_NAME,
                "format_version": str(FORMAT_VERSION),
                "model_hash": self.model_hash,
            },
        )
        metadata = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "model_version": self.model_version,
            "model_hash": self.model_hash,
            "weights_file": WEIGHTS_NAME,
            "feature_schema": {
                "continuous": list(CONTINUOUS_FEATURES),
                "categorical": ["domain", "lang"],
                "encoded": list(self.feature_names),
            },
            "scaler": {
                "mean": self.continuous_mean.tolist(),
                "scale": self.continuous_scale.tolist(),
            },
            "categories": {key: list(self.categories[key]) for key in ("domain", "lang")},
            "metrics": self.metrics,
            "training_config": self.training_config,
        }
        metadata_tmp.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(weights_tmp, weights_path)
        os.replace(metadata_tmp, metadata_path)
        return weights_path, metadata_path

    @classmethod
    def load(cls, model_dir: str | Path) -> Stage1Ranker:
        directory = Path(model_dir)
        metadata_path = directory / METADATA_NAME
        weights_path = directory / WEIGHTS_NAME
        if not metadata_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"ranker artifact pair not found in {directory}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RankerPersistenceError(f"cannot read ranker metadata: {exc}") from exc
        if metadata.get("format") != FORMAT_NAME:
            raise RankerPersistenceError(f"unsupported ranker format: {metadata.get('format')!r}")
        if metadata.get("format_version") != FORMAT_VERSION:
            raise RankerPersistenceError(
                f"unsupported ranker format version: {metadata.get('format_version')!r}"
            )
        schema = metadata.get("feature_schema", {})
        if schema.get("continuous") != list(CONTINUOUS_FEATURES):
            raise RankerPersistenceError("saved continuous feature schema is incompatible")
        categories = metadata.get("categories")
        if not isinstance(categories, dict) or not all(name in categories for name in ("domain", "lang")):
            raise RankerPersistenceError("saved categorical schema is missing")
        expected_encoded = list(
            _encoded_feature_names({name: tuple(categories[name]) for name in ("domain", "lang")})
        )
        if schema.get("encoded") != expected_encoded:
            raise RankerPersistenceError("saved encoded feature schema is incompatible")
        scaler = metadata.get("scaler", {})
        try:
            tensors = load_file(str(weights_path))
        except Exception as exc:  # noqa: BLE001
            raise RankerPersistenceError(f"cannot read ranker weights: {exc}") from exc
        required_tensors = {"coefficients", "intercept", "format_version"}
        if set(tensors) != required_tensors:
            raise RankerPersistenceError(
                f"saved weights must contain exactly {sorted(required_tensors)}"
            )
        version_tensor = np.asarray(tensors["format_version"])
        if version_tensor.shape != (1,) or int(version_tensor[0]) != FORMAT_VERSION:
            raise RankerPersistenceError("saved weight version is incompatible")
        intercept_tensor = np.asarray(tensors["intercept"])
        if intercept_tensor.shape != (1,):
            raise RankerPersistenceError("saved intercept has the wrong shape")
        try:
            return cls(
                coefficients=np.asarray(tensors["coefficients"], dtype=np.float64),
                intercept=float(intercept_tensor[0]),
                continuous_mean=np.asarray(scaler["mean"], dtype=np.float64),
                continuous_scale=np.asarray(scaler["scale"], dtype=np.float64),
                categories={name: tuple(categories[name]) for name in ("domain", "lang")},
                metrics=dict(metadata.get("metrics", {})),
                training_config=dict(metadata.get("training_config", {})),
                model_hash=str(metadata.get("model_hash", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, RankerPersistenceError):
                raise
            raise RankerPersistenceError(f"invalid ranker metadata: {exc}") from exc


def train_ranker(
    examples: Iterable[LabeledDocument | Mapping[str, Any]],
    *,
    l2: float = 0.1,
    validation_fraction: float = 0.2,
    seed: int = 0,
    max_iter: int = 100,
    tolerance: float = 1e-8,
    minimum_examples: int = MIN_TRAINING_EXAMPLES,
) -> Stage1Ranker:
    """Train a regularized, standardized logistic ranker from human verdicts."""
    if l2 <= 0 or not math.isfinite(l2):
        raise RankerTrainingError("l2 must be finite and greater than zero")
    if max_iter <= 0:
        raise RankerTrainingError("max_iter must be greater than zero")
    if tolerance <= 0 or not math.isfinite(tolerance):
        raise RankerTrainingError("tolerance must be finite and greater than zero")
    if minimum_examples < 4:
        raise RankerTrainingError("minimum_examples must be at least 4")
    if validation_fraction < 0 or validation_fraction >= 1:
        raise RankerTrainingError("validation_fraction must be at least 0 and less than 1")

    documents = [_coerce_example(example, index) for index, example in enumerate(examples)]
    labels = np.asarray([document.label for document in documents], dtype=np.int64)
    _validate_training_labels(labels, minimum_examples)

    metrics: dict[str, Any] = {}
    groups = [document.group for document in documents]
    if any(group is not None for group in groups) and not all(group is not None for group in groups):
        raise RankerTrainingError(
            "grouped validation requires a group/corpus_id on every labeled document"
        )
    usable_groups: Sequence[str] | None = groups if groups and groups[0] is not None else None
    if validation_fraction > 0 and min(np.bincount(labels, minlength=2)) >= 3:
        try:
            split = deterministic_validation_split(
                labels.tolist(),
                groups=usable_groups,
                validation_fraction=validation_fraction,
                seed=seed,
            )
        except RankerTrainingError as exc:
            if usable_groups is None:
                raise
            metrics["holdout"] = {
                "available": False,
                "strategy": "stratified_group",
                "reason": str(exc),
            }
        else:
            train_documents = [documents[index] for index in split.train_indices]
            validation_documents = [documents[index] for index in split.validation_indices]
            train_labels = np.asarray([document.label for document in train_documents], dtype=np.int64)
            validation_labels = np.asarray(
                [document.label for document in validation_documents], dtype=np.int64
            )
            provisional = _fit_parameters(
                train_documents,
                l2=l2,
                max_iter=max_iter,
                tolerance=tolerance,
            )
            coefficients, intercept, mean, scale, categories, optimizer = provisional
            train_matrix = _design_matrix(train_documents, categories, mean, scale)
            validation_matrix = _design_matrix(validation_documents, categories, mean, scale)
            metrics["holdout"] = {
                "available": True,
                "strategy": split.strategy,
                "seed": seed,
                "validation_fraction": validation_fraction,
                "train": _binary_metrics(
                    train_labels, _sigmoid(intercept + train_matrix @ coefficients)
                ),
                "validation": _binary_metrics(
                    validation_labels, _sigmoid(intercept + validation_matrix @ coefficients)
                ),
                "optimizer": optimizer,
            }
    else:
        metrics["holdout"] = {
            "available": False,
            "reason": "validation disabled or fewer than three labels in one class",
        }

    coefficients, intercept, mean, scale, categories, optimizer = _fit_parameters(
        documents,
        l2=l2,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    matrix = _design_matrix(documents, categories, mean, scale)
    metrics["fit_all"] = _binary_metrics(labels, _sigmoid(intercept + matrix @ coefficients))
    metrics["fit_all"]["optimizer"] = optimizer
    metrics["fit_all"]["note"] = "final deployment model refit on all labeled documents"

    training_config = {
        "algorithm": "standardized_logistic_regression",
        "l2": l2,
        "max_iter": max_iter,
        "tolerance": tolerance,
        "minimum_examples": minimum_examples,
        "training_examples": len(documents),
        "bad_examples": int(np.sum(labels == 0)),
        "good_examples": int(np.sum(labels == 1)),
        "training_data_hash": _training_data_hash(documents),
        "validation_fraction": validation_fraction,
        "seed": seed,
    }
    return Stage1Ranker(
        coefficients=coefficients,
        intercept=intercept,
        continuous_mean=mean,
        continuous_scale=scale,
        categories={key: tuple(categories[key]) for key in ("domain", "lang")},
        metrics=metrics,
        training_config=training_config,
    )


def grouped_cross_validation(
    examples: Iterable[LabeledDocument | Mapping[str, Any]],
    *,
    l2: float = 0.1,
    max_iter: int = 100,
    tolerance: float = 1e-8,
    minimum_examples: int = MIN_TRAINING_EXAMPLES,
) -> dict[str, Any]:
    """Return leave-one-group-out predictions without leaking a corpus across a fold."""
    documents = [_coerce_example(example, index) for index, example in enumerate(examples)]
    labels = np.asarray([document.label for document in documents], dtype=np.int64)
    _validate_training_labels(labels, minimum_examples)
    if l2 <= 0 or not math.isfinite(l2):
        raise RankerTrainingError("l2 must be finite and greater than zero")
    if max_iter <= 0:
        raise RankerTrainingError("max_iter must be greater than zero")
    if tolerance <= 0 or not math.isfinite(tolerance):
        raise RankerTrainingError("tolerance must be finite and greater than zero")
    if any(document.group is None for document in documents):
        raise RankerTrainingError("grouped cross-validation requires a group/corpus_id on every document")

    group_tokens = sorted({_group_token(document.group) for document in documents})
    if len(group_tokens) < 2:
        raise RankerTrainingError("grouped cross-validation requires at least two groups")
    probabilities = np.empty(len(documents), dtype=np.float64)
    folds: list[dict[str, Any]] = []
    for token in group_tokens:
        validation_indices = [
            index for index, document in enumerate(documents) if _group_token(document.group) == token
        ]
        validation_set = set(validation_indices)
        train_documents = [
            document for index, document in enumerate(documents) if index not in validation_set
        ]
        train_labels = np.asarray([document.label for document in train_documents], dtype=np.int64)
        _validate_training_labels(train_labels, minimum_examples)
        coefficients, intercept, mean, scale, categories, optimizer = _fit_parameters(
            train_documents,
            l2=l2,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        validation_documents = [documents[index] for index in validation_indices]
        validation_matrix = _design_matrix(validation_documents, categories, mean, scale)
        fold_probabilities = _sigmoid(intercept + validation_matrix @ coefficients)
        probabilities[validation_indices] = fold_probabilities
        fold_labels = labels[validation_indices]
        folds.append(
            {
                "group": documents[validation_indices[0]].group,
                "count": len(validation_indices),
                "good": int(fold_labels.sum()),
                "bad": int(len(fold_labels) - fold_labels.sum()),
                "metrics": _binary_metrics(fold_labels, fold_probabilities),
                "optimizer": optimizer,
            }
        )

    return {
        "available": True,
        "strategy": "leave_one_group_out",
        "groups": len(group_tokens),
        "l2": l2,
        "overall": _binary_metrics(labels, probabilities),
        "folds": folds,
    }


def load_ranker_if_available(model_dir: str | Path) -> Stage1Ranker | None:
    """Load an artifact pair, returning None only when neither artifact exists."""
    directory = Path(model_dir)
    weights_exists = (directory / WEIGHTS_NAME).exists()
    metadata_exists = (directory / METADATA_NAME).exists()
    if not weights_exists and not metadata_exists:
        return None
    if not weights_exists or not metadata_exists:
        raise RankerPersistenceError(f"incomplete ranker artifact pair in {directory}")
    return Stage1Ranker.load(directory)


__all__ = [
    "CONTINUOUS_FEATURES",
    "FORMAT_VERSION",
    "LabeledDocument",
    "RankerError",
    "RankerPersistenceError",
    "RankerTrainingError",
    "Stage1Ranker",
    "ValidationSplit",
    "deterministic_validation_split",
    "featurize",
    "grouped_cross_validation",
    "load_ranker_if_available",
    "train_ranker",
]
