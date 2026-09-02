"""Strict offline preference data and Direct Preference Optimization primitives.

yyLab does not currently have a reward model or an online rollout service.  This
module therefore implements the useful, auditable subset of preference RL:
same-prompt ``chosen``/``rejected`` pairs optimized with DPO against the frozen
base route of a PEFT policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..ingest.records import infer_source_split, normalize_chat_messages

PREFERENCE_RUN_KIND = "preference_dpo"
PREFERENCE_OBJECTIVE = "offline_dpo"
PREFERENCE_DATA_FORMATS = frozenset({"auto", "pairs", "traces"})
_PROMPT_ROLES = frozenset({"system", "user", "assistant", "tool"})
_DATA_EXTENSIONS = frozenset({".jsonl", ".ndjson", ".json", ".parquet", ".pq"})
_CHOSEN_FIELDS = ("chosen", "preferred", "chosen_response", "positive")
_REJECTED_FIELDS = ("rejected", "dispreferred", "rejected_response", "negative")
_PROMPT_FIELDS = ("prompt", "input", "instruction")
_TRACE_FIELDS = (
    "messages",
    "conversations",
    "trace",
    "messages_json",
    "conversations_json",
)
_RESPONSE_FIELDS = ("response", "completion", "output", "answer")
_GROUP_FIELDS = ("pair_id", "preference_id", "group_id", "prompt_id")
_LABEL_FIELDS = (
    "preference_label",
    "verdict",
    "label",
    "preference",
    "is_chosen",
    "is_preferred",
    "preferred",
    "accepted",
)
_SCORE_FIELDS = ("reward", "score", "preference_score", "quality_score")
_PARQUET_COLUMNS = frozenset(
    {
        *_CHOSEN_FIELDS,
        *_REJECTED_FIELDS,
        *_PROMPT_FIELDS,
        *_TRACE_FIELDS,
        *_RESPONSE_FIELDS,
        *_GROUP_FIELDS,
        *_LABEL_FIELDS,
        *_SCORE_FIELDS,
        "id",
        "trace_id",
        "metadata",
        "preference",
        "quality_scores",
        "rank",
        "row_json",
        "split",
        "source_split",
        "partition",
    }
)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"unsupported JSON constant {value!r}")


def _strict_json_loads(line: str) -> object:
    return json.loads(
        line,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
    )


@dataclass(frozen=True, slots=True)
class PreferenceMessage:
    role: str
    content: str
    reasoning_content: str | None = None
    tool_calls: tuple[Mapping[str, object], ...] = ()
    name: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in _PROMPT_ROLES:
            raise ValueError(f"unsupported prompt role {self.role!r}")
        if not isinstance(self.content, str):
            raise ValueError("message content must be a string")
        for field_name, value in (
            ("content", self.content),
            ("reasoning_content", self.reasoning_content),
        ):
            if value is not None and any(
                0xD800 <= ord(character) <= 0xDFFF for character in value
            ):
                raise ValueError(f"message {field_name} contains an invalid Unicode surrogate")
        if self.reasoning_content is not None and not self.reasoning_content.strip():
            raise ValueError("reasoning_content must be non-empty when supplied")
        if not self.content.strip() and not self.reasoning_content and not self.tool_calls:
            raise ValueError("message needs content, reasoning_content, or tool_calls")
        if self.role == "tool" and not self.content.strip():
            raise ValueError("tool message content must be non-empty")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("tool_calls are supported only on assistant messages")
        for field_name, value in (("name", self.name), ("tool_call_id", self.tool_call_id)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be non-empty when supplied")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {"role": self.role, "content": self.content}
        if self.reasoning_content is not None:
            value["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            value["tool_calls"] = [dict(call) for call in self.tool_calls]
        if self.name is not None:
            value["name"] = self.name
        if self.tool_call_id is not None:
            value["tool_call_id"] = self.tool_call_id
        return value


def _parse_message(value: object, *, location: str) -> PreferenceMessage:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a role/content message object")
    raw_tool_calls = value.get("tool_calls")
    if raw_tool_calls is not None and (
        not isinstance(raw_tool_calls, list)
        or any(not isinstance(call, Mapping) for call in raw_tool_calls)
    ):
        raise ValueError(f"{location}.tool_calls must be an array of objects")
    raw_reasoning = value.get("reasoning_content")
    if raw_reasoning is not None and not isinstance(raw_reasoning, str):
        raise ValueError(f"{location}.reasoning_content must be a string")
    normalized = normalize_chat_messages([dict(value)])
    if len(normalized) != 1:
        raise ValueError(f"{location} is not a valid structured message")
    message = normalized[0]
    tool_calls = message.get("tool_calls")
    if tool_calls is None:
        tool_calls = []
    if not isinstance(tool_calls, list):
        raise ValueError(f"{location}.tool_calls must be an array")
    return PreferenceMessage(
        role=str(message["role"]),
        content=str(message.get("content") or ""),
        reasoning_content=(
            str(message["reasoning_content"])
            if message.get("reasoning_content") is not None
            else None
        ),
        tool_calls=tuple(dict(call) for call in tool_calls if isinstance(call, Mapping)),
        name=str(message["name"]) if message.get("name") is not None else None,
        tool_call_id=(
            str(message["tool_call_id"])
            if message.get("tool_call_id") is not None
            else None
        ),
    )


def _json_message_value(value: object, *, location: str) -> object:
    """Decode Parquet/JSONL columns that store structured messages as JSON text."""

    if not isinstance(value, str) or not value.lstrip().startswith(("[", "{")):
        return value
    try:
        decoded = _strict_json_loads(value)
    except (ValueError, json.JSONDecodeError, RecursionError):
        return value
    if isinstance(decoded, Mapping):
        if "messages" in decoded:
            return decoded["messages"]
        if any(field in decoded for field in ("role", "from", "speaker")):
            return decoded
        return value
    if isinstance(decoded, list) and decoded and all(
        isinstance(item, Mapping) for item in decoded
    ):
        return decoded
    return value


def _parse_messages(value: object, *, location: str) -> tuple[PreferenceMessage, ...]:
    value = _json_message_value(value, location=location)
    if isinstance(value, Mapping) and "messages" in value:
        value = value["messages"]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty message array")
    return tuple(
        _parse_message(raw, location=f"{location}[{index}]")
        for index, raw in enumerate(value)
    )


def _validate_prompt_order(messages: Sequence[PreferenceMessage]) -> None:
    if not messages or messages[-1].role not in {"user", "tool"}:
        raise ValueError("prompt messages must end with a user or tool message")
    if any(message.role == "system" for message in messages[1:]):
        raise ValueError("system may appear only as the first prompt message")
    start = 1 if messages[0].role == "system" else 0
    if start == len(messages) or messages[start].role != "user":
        raise ValueError("prompt messages must begin with a user message after system")
    previous = "system" if start else ""
    for message in messages[start:]:
        if message.role == "user" and previous not in {"", "system", "assistant"}:
            raise ValueError("user messages must follow system or assistant turns")
        if message.role == "assistant" and previous not in {"user", "tool"}:
            raise ValueError("assistant messages must follow user or tool turns")
        if message.role == "tool" and previous not in {"assistant", "tool"}:
            raise ValueError("tool messages must follow an assistant tool call")
        previous = message.role


def _parse_prompt(value: object) -> tuple[PreferenceMessage, ...]:
    value = _json_message_value(value, location="prompt")
    if isinstance(value, str):
        messages = (PreferenceMessage("user", value),)
    elif isinstance(value, list):
        messages = _parse_messages(value, location="prompt")
    else:
        raise ValueError("prompt must be a string or an array of role/content messages")
    _validate_prompt_order(messages)
    return messages


def _response_message(
    value: object,
    *,
    prompt: tuple[PreferenceMessage, ...],
    location: str,
) -> PreferenceMessage:
    value = _json_message_value(value, location=location)
    if isinstance(value, str):
        return PreferenceMessage("assistant", value)
    if isinstance(value, Mapping):
        if "messages" in value:
            value = value["messages"]
        else:
            response = _parse_message(value, location=location)
            if response.role != "assistant":
                raise ValueError(f"{location} response must have assistant role")
            return response
    if isinstance(value, Mapping):
        response = _parse_message(value, location=location)
        if response.role != "assistant":
            raise ValueError(f"{location} response must have assistant role")
        return response
    branch = _parse_messages(value, location=location)
    if len(branch) == 1 and branch[0].role == "assistant":
        return branch[0]
    if branch[:-1] != prompt or branch[-1].role != "assistant":
        raise ValueError(
            f"{location} must be one assistant response or the exact prompt plus one "
            "assistant response; divergent tool trajectories are not simple DPO pairs"
        )
    return branch[-1]


def _response_text(message: PreferenceMessage) -> str:
    parts = [message.reasoning_content or "", message.content]
    if message.tool_calls:
        parts.append(json.dumps(message.to_dict()["tool_calls"], sort_keys=True))
    return "\n".join(part for part in parts if part)


@dataclass(frozen=True, slots=True)
class PreferenceExample:
    prompt: tuple[PreferenceMessage, ...]
    chosen: PreferenceMessage
    rejected: PreferenceMessage
    example_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_prompt_order(self.prompt)
        for name, value in (("chosen", self.chosen), ("rejected", self.rejected)):
            if not isinstance(value, PreferenceMessage) or value.role != "assistant":
                raise ValueError(f"{name} must be an assistant response")
        if self.chosen == self.rejected:
            raise ValueError("chosen and rejected must differ")
        if self.example_id is not None and (
            not isinstance(self.example_id, str) or not self.example_id.strip()
        ):
            raise ValueError("id must be a non-empty string when supplied")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")

    @classmethod
    def from_dict(cls, value: object) -> PreferenceExample:
        if not isinstance(value, Mapping):
            raise ValueError("preference row must be an object")
        allowed = {"prompt", "chosen", "rejected", "id", "metadata"}
        required = {"chosen", "rejected"}
        if not required.issubset(value) or not set(value).issubset(allowed):
            raise ValueError(
                "preference row needs chosen/rejected and only optional prompt/id/metadata"
            )
        example_id = value.get("id")
        metadata = value.get("metadata", {})
        if example_id is not None and not isinstance(example_id, str):
            raise ValueError("id must be a string")
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        raw_prompt = value.get("prompt")
        if raw_prompt is None:
            chosen_branch = _parse_messages(value["chosen"], location="chosen")
            rejected_branch = _parse_messages(value["rejected"], location="rejected")
            if (
                len(chosen_branch) < 2
                or chosen_branch[:-1] != rejected_branch[:-1]
                or chosen_branch[-1].role != "assistant"
                or rejected_branch[-1].role != "assistant"
            ):
                raise ValueError(
                    "message-array branches without prompt must share an identical prompt "
                    "and diverge only at their final assistant response"
                )
            prompt = chosen_branch[:-1]
            _validate_prompt_order(prompt)
            chosen = chosen_branch[-1]
            rejected = rejected_branch[-1]
        else:
            prompt = _parse_prompt(raw_prompt)
            chosen = _response_message(value["chosen"], prompt=prompt, location="chosen")
            rejected = _response_message(value["rejected"], prompt=prompt, location="rejected")
        return cls(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            example_id=example_id,
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class PreferenceDataset:
    examples: tuple[PreferenceExample, ...]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("preference dataset contains no canonical pairs")


@dataclass(frozen=True, slots=True)
class _SourceRow:
    value: Mapping[str, object]
    path: Path
    row_number: int

    @property
    def location(self) -> str:
        return f"{self.path}:{self.row_number}"


@dataclass(frozen=True, slots=True)
class _TraceCandidate:
    prompt: tuple[PreferenceMessage, ...]
    response: PreferenceMessage
    group: str
    source_split: str
    label: str | None
    score: float | None
    rank: float | None
    example_id: str | None
    metadata: Mapping[str, object]
    location: str


def _source_files(path: str | Path) -> tuple[Path, ...]:
    source = Path(path).expanduser().resolve()
    if source.is_file():
        if source.suffix.casefold() not in _DATA_EXTENSIONS:
            raise ValueError(
                f"unsupported preference extension {source.suffix!r}; use JSONL, JSON, or Parquet"
            )
        return (source,)
    if source.is_dir():
        files = tuple(
            sorted(
                (
                    candidate.resolve()
                    for candidate in source.rglob("*")
                    if candidate.is_file()
                    and candidate.suffix.casefold() in _DATA_EXTENSIONS
                ),
                key=lambda candidate: str(candidate).casefold(),
            )
        )
        if not files:
            raise ValueError(f"preference directory contains no JSONL/JSON/Parquet shards: {source}")
        return files
    raise FileNotFoundError(f"preference dataset not found: {source}")


def _iter_json_rows(path: Path) -> Iterator[_SourceRow]:
    if path.suffix.casefold() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank JSONL records are not allowed")
                try:
                    value = _strict_json_loads(line)
                except (ValueError, json.JSONDecodeError, RecursionError) as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number}: preference row must be an object")
                yield _SourceRow(dict(value), path, line_number)
        return
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_constant,
            )
    except (ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if isinstance(value, Mapping) and isinstance(value.get("data"), list):
        value = value["data"]
    rows = value if isinstance(value, list) else [value]
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:row {row_number}: preference row must be an object")
        yield _SourceRow(dict(row), path, row_number)


def _iter_parquet_rows(path: Path) -> Iterator[_SourceRow]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise RuntimeError("Parquet preference data requires pyarrow") from exc
    try:
        parquet = pq.ParquetFile(str(path))
        projected = [
            name for name in parquet.schema_arrow.names if name in _PARQUET_COLUMNS
        ]
        # Unknown schemas still receive a useful row-level diagnostic. Known
        # trace schemas avoid materializing large token/logprob/event columns.
        columns = projected or parquet.schema_arrow.names
        row_number = 0
        for batch in parquet.iter_batches(batch_size=2_048, columns=columns):
            for row in batch.to_pylist():
                row_number += 1
                if not isinstance(row, Mapping):
                    raise ValueError(f"{path}:row {row_number}: preference row must be an object")
                yield _SourceRow(dict(row), path, row_number)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{path}: could not read Parquet preference shard: {exc}") from exc


def _iter_source_rows(files: Sequence[Path]) -> Iterator[_SourceRow]:
    for path in files:
        if path.suffix.casefold() in {".parquet", ".pq"}:
            yield from _iter_parquet_rows(path)
        else:
            yield from _iter_json_rows(path)


def _selected_field(
    value: Mapping[str, object],
    names: Sequence[str],
    *,
    label: str,
) -> tuple[str | None, object]:
    present = [name for name in names if name in value and value[name] is not None]
    if len(present) > 1:
        raise ValueError(f"ambiguous {label} fields: {', '.join(present)}")
    return (present[0], value[present[0]]) if present else (None, None)


def _canonical_example(example: PreferenceExample) -> str:
    value = {
        "prompt": [message.to_dict() for message in example.prompt],
        "chosen": example.chosen.to_dict(),
        "rejected": example.rejected.to_dict(),
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_prompt(prompt: Sequence[PreferenceMessage]) -> str:
    return json.dumps(
        [message.to_dict() for message in prompt],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_message(message: PreferenceMessage) -> str:
    return json.dumps(
        message.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pair_row(value: Mapping[str, object], *, source_path: Path) -> PreferenceExample | None:
    nested = value.get("preference")
    nested_pair = nested if isinstance(nested, Mapping) else {}
    chosen_name, chosen = _selected_field(value, _CHOSEN_FIELDS, label="chosen response")
    rejected_name, rejected = _selected_field(
        value, _REJECTED_FIELDS, label="rejected response"
    )
    if chosen_name is None:
        chosen_name, chosen = _selected_field(
            nested_pair, _CHOSEN_FIELDS, label="nested chosen response"
        )
    if rejected_name is None:
        rejected_name, rejected = _selected_field(
            nested_pair, _REJECTED_FIELDS, label="nested rejected response"
        )
    if chosen_name is None and rejected_name is None:
        return None
    if rejected_name is None and chosen_name == "preferred" and isinstance(chosen, bool):
        return None
    if chosen_name is None or rejected_name is None:
        raise ValueError("pair row must supply both chosen and rejected responses")
    _prompt_name, prompt = _selected_field(value, _PROMPT_FIELDS, label="prompt")
    if prompt is None and "messages" in value:
        prompt = value["messages"]
    example_id = value.get("id")
    if example_id is None:
        example_id = value.get("pair_id")
    if example_id is not None and not isinstance(example_id, (str, int)):
        raise ValueError("id must be a string or integer")
    metadata = value.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")
    split = infer_source_split(
        source_path,
        value.get("split", value.get("source_split", value.get("partition"))),
    )
    normalized_metadata = dict(metadata)
    if split != "unspecified":
        normalized_metadata["source_split"] = split
    canonical: dict[str, object] = {
        "chosen": chosen,
        "rejected": rejected,
        "metadata": normalized_metadata,
    }
    if prompt is not None:
        canonical["prompt"] = prompt
    if example_id is not None:
        canonical["id"] = str(example_id)
    return PreferenceExample.from_dict(canonical)


def _nested_pair_row(
    value: Mapping[str, object],
    *,
    location: str,
) -> Mapping[str, object]:
    """Unwrap common archival ``row_json`` payloads only when they are pairs."""

    raw = value.get("row_json")
    if raw is None:
        return value
    direct_chosen = any(field in value for field in _CHOSEN_FIELDS)
    direct_rejected = any(field in value for field in _REJECTED_FIELDS)
    if direct_chosen and direct_rejected:
        return value
    if not isinstance(raw, str):
        raise ValueError("row_json must be a JSON object string")
    try:
        decoded = _strict_json_loads(raw)
    except (ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"invalid row_json payload: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("row_json must decode to an object")
    chosen_present = any(field in decoded for field in _CHOSEN_FIELDS)
    rejected_present = any(field in decoded for field in _REJECTED_FIELDS)
    nested = decoded.get("preference")
    if isinstance(nested, Mapping):
        chosen_present = chosen_present or any(field in nested for field in _CHOSEN_FIELDS)
        rejected_present = rejected_present or any(field in nested for field in _REJECTED_FIELDS)
    if not (chosen_present and rejected_present):
        raise ValueError(
            "row_json is a unary/raw trace with no explicit chosen/rejected comparison; "
            "use chat SFT unless the source includes a preference signal"
        )
    result = dict(decoded)
    for field_name in ("id", "pair_id", "metadata", "split", "source_split", "partition"):
        if field_name not in result and field_name in value:
            result[field_name] = value[field_name]
    if "metadata" not in result:
        result["metadata"] = {"row_json_source": location}
    return result


def _preference_label(value: object) -> str | None:
    if isinstance(value, bool):
        return "chosen" if value else "rejected"
    if isinstance(value, int) and value in {0, 1}:
        return "chosen" if value == 1 else "rejected"
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"chosen", "preferred", "winner", "positive", "accepted", "good"}:
        return "chosen"
    if normalized in {"rejected", "dispreferred", "loser", "negative", "bad"}:
        return "rejected"
    return None


def _finite_number(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _trace_candidate(
    value: Mapping[str, object],
    *,
    location: str,
    source_path: Path,
) -> _TraceCandidate:
    trace_name, trace = _selected_field(value, _TRACE_FIELDS, label="trace")
    if trace_name is not None:
        branch = _parse_messages(trace, location=trace_name)
        if len(branch) < 2 or branch[-1].role != "assistant":
            raise ValueError("normalized trace must end with an assistant response")
        prompt = branch[:-1]
        _validate_prompt_order(prompt)
        response = branch[-1]
    else:
        _prompt_name, raw_prompt = _selected_field(value, _PROMPT_FIELDS, label="prompt")
        _response_name, raw_response = _selected_field(
            value, _RESPONSE_FIELDS, label="response"
        )
        if raw_prompt is None or raw_response is None:
            raise ValueError(
                "trace row needs messages/conversations/trace, or prompt plus response/completion/output"
            )
        prompt = _parse_prompt(raw_prompt)
        response = _response_message(raw_response, prompt=prompt, location="response")
    _group_name, raw_group = _selected_field(value, _GROUP_FIELDS, label="trace group")
    group = str(raw_group).strip() if raw_group is not None else _canonical_prompt(prompt)
    if not group:
        raise ValueError("trace group id must be non-empty")
    _label_name, raw_label = _selected_field(value, _LABEL_FIELDS, label="preference label")
    label = _preference_label(raw_label)
    if raw_label is not None and label is None:
        raise ValueError(f"unrecognized preference label {raw_label!r}")
    _score_name, raw_score = _selected_field(value, _SCORE_FIELDS, label="preference score")
    quality_scores = value.get("quality_scores")
    if isinstance(quality_scores, str) and quality_scores.lstrip().startswith("{"):
        try:
            decoded_quality_scores = _strict_json_loads(quality_scores)
        except (ValueError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(f"quality_scores is not valid JSON: {exc}") from exc
        if not isinstance(decoded_quality_scores, Mapping):
            raise ValueError("quality_scores JSON must decode to an object")
        quality_scores = decoded_quality_scores
    if raw_score is None and isinstance(quality_scores, Mapping):
        raw_score = quality_scores.get("overall")
    score = _finite_number(raw_score, label="preference score")
    rank = _finite_number(value.get("rank"), label="rank")
    if label is None and score is None and rank is None:
        raise ValueError(
            "unlabelled unary trace: add chosen/rejected label, reward/score, or rank; "
            "yyLab will not invent a rejected response"
        )
    example_id = value.get("id") or value.get("trace_id")
    if example_id is not None and not isinstance(example_id, (str, int)):
        raise ValueError("trace id must be a string or integer")
    metadata = value.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")
    source_split = infer_source_split(
        source_path,
        value.get("split", value.get("source_split", value.get("partition"))),
    )
    return _TraceCandidate(
        prompt=prompt,
        response=response,
        group=group,
        source_split=source_split,
        label=label,
        score=score,
        rank=rank,
        example_id=str(example_id) if example_id is not None else None,
        metadata=dict(metadata),
        location=location,
    )


def _trace_group_examples(
    group: str,
    candidates: Sequence[_TraceCandidate],
) -> list[PreferenceExample]:
    prompts = {_canonical_prompt(candidate.prompt) for candidate in candidates}
    if len(prompts) != 1:
        locations = ", ".join(candidate.location for candidate in candidates[:3])
        raise ValueError(f"trace group {group!r} contains different prompts ({locations})")
    source_splits = {candidate.source_split for candidate in candidates}
    if len(source_splits) != 1:
        raise ValueError(f"trace group {group!r} crosses source splits {sorted(source_splits)}")
    chosen = sorted(
        (candidate for candidate in candidates if candidate.label == "chosen"),
        key=lambda candidate: _canonical_message(candidate.response),
    )
    rejected = sorted(
        (candidate for candidate in candidates if candidate.label == "rejected"),
        key=lambda candidate: _canonical_message(candidate.response),
    )
    unlabeled = [candidate for candidate in candidates if candidate.label is None]
    pairs: list[tuple[_TraceCandidate, _TraceCandidate]] = []
    if chosen or rejected:
        if unlabeled:
            raise ValueError(
                f"trace group {group!r} mixes labelled and score-only candidates; label every candidate"
            )
        if not chosen or not rejected:
            present = "chosen" if chosen else "rejected"
            raise ValueError(
                f"unpaired trace group {group!r} has only {present} candidates; both labels are required"
            )
        pairs.extend((winner, loser) for winner in chosen for loser in rejected)
    else:
        if len(candidates) < 2:
            raise ValueError(
                f"unpaired trace group {group!r} contains one scored candidate; a comparison needs two"
            )
        if all(candidate.score is not None for candidate in candidates):
            low = min(float(candidate.score) for candidate in candidates)
            high = max(float(candidate.score) for candidate in candidates)
            if low == high:
                raise ValueError(f"trace group {group!r} has tied scores and no preference")
            winners = sorted(
                (candidate for candidate in candidates if float(candidate.score) == high),
                key=lambda candidate: _canonical_message(candidate.response),
            )
            losers = sorted(
                (candidate for candidate in candidates if float(candidate.score) == low),
                key=lambda candidate: _canonical_message(candidate.response),
            )
            pairs.extend((winner, loser) for winner in winners for loser in losers)
        elif all(candidate.rank is not None for candidate in candidates):
            high = max(float(candidate.rank) for candidate in candidates)
            low = min(float(candidate.rank) for candidate in candidates)
            if low == high:
                raise ValueError(f"trace group {group!r} has tied ranks and no preference")
            winners = sorted(
                (candidate for candidate in candidates if float(candidate.rank) == low),
                key=lambda candidate: _canonical_message(candidate.response),
            )
            losers = sorted(
                (candidate for candidate in candidates if float(candidate.rank) == high),
                key=lambda candidate: _canonical_message(candidate.response),
            )
            pairs.extend((winner, loser) for winner in winners for loser in losers)
        else:
            raise ValueError(
                f"trace group {group!r} must consistently use labels, scores, or ranks"
            )
    return [
        PreferenceExample(
            prompt=winner.prompt,
            chosen=winner.response,
            rejected=loser.response,
            metadata={
                "trace_group": group,
                "source_split": winner.source_split,
                "chosen_id": winner.example_id,
                "rejected_id": loser.example_id,
                "chosen_source": winner.location,
                "rejected_source": loser.location,
            },
        )
        for winner, loser in pairs
    ]


def load_preference_dataset(
    path: str | Path,
    *,
    data_format: str = "auto",
) -> PreferenceDataset:
    """Stream JSON/Parquet shards and normalize them into exact DPO pairs.

    ``pairs`` accepts one chosen/rejected comparison per row. ``traces`` accepts
    labelled or scored candidate traces grouped by an explicit group id or an
    exactly identical normalized prompt. ``auto`` accepts both in one dataset.
    """

    if data_format not in PREFERENCE_DATA_FORMATS:
        raise ValueError(f"data_format must be one of {sorted(PREFERENCE_DATA_FORMATS)}")
    files = _source_files(path)
    direct: list[PreferenceExample] = []
    trace_groups: dict[str, list[_TraceCandidate]] = defaultdict(list)
    source_rows = 0
    pair_rows = 0
    trace_rows = 0
    file_rows: dict[Path, int] = defaultdict(int)
    for source_row in _iter_source_rows(files):
        source_rows += 1
        file_rows[source_row.path] += 1
        try:
            row_value = _nested_pair_row(source_row.value, location=source_row.location)
            example = (
                _pair_row(row_value, source_path=source_row.path)
                if data_format != "traces"
                else None
            )
            if example is not None:
                direct.append(example)
                pair_rows += 1
                continue
            if data_format == "pairs":
                raise ValueError("row does not contain chosen and rejected responses")
            candidate = _trace_candidate(
                row_value,
                location=source_row.location,
                source_path=source_row.path,
            )
            trace_groups[f"{candidate.source_split}\0{candidate.group}"].append(candidate)
            trace_rows += 1
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f"{source_row.location}: invalid preference row: {exc}") from exc
    if source_rows == 0:
        raise ValueError(f"{Path(path).expanduser().resolve()} contains no preference rows")
    normalized = list(direct)
    for split_group, candidates in trace_groups.items():
        _split, group = split_group.split("\0", 1)
        try:
            normalized.extend(_trace_group_examples(group, candidates))
        except ValueError as exc:
            raise ValueError(f"invalid normalized traces: {exc}") from exc
    seen_ids: dict[str, str] = {}
    seen_pairs: set[str] = set()
    examples: list[PreferenceExample] = []
    duplicate_pairs = 0
    for example in normalized:
        canonical = _canonical_example(example)
        if example.example_id is not None:
            previous = seen_ids.get(example.example_id)
            if previous is not None and previous != canonical:
                raise ValueError(
                    f"duplicate preference id {example.example_id!r} refers to different pairs"
                )
            seen_ids[example.example_id] = canonical
        if canonical in seen_pairs:
            duplicate_pairs += 1
            continue
        seen_pairs.add(canonical)
        examples.append(example)
    if not examples:
        raise ValueError("preference source produced no unique canonical pairs")
    pair_hashes = sorted(
        hashlib.sha256(canonical.encode("utf-8")).hexdigest() for canonical in seen_pairs
    )
    canonical_checksum = hashlib.sha256("\n".join(pair_hashes).encode("ascii")).hexdigest()
    formats = sorted({path.suffix.casefold().lstrip(".") for path in files})
    provenance: dict[str, object] = {
        "requested_format": data_format,
        "detected_formats": formats,
        "source_files": [
            {
                "path": str(source),
                "bytes": source.stat().st_size,
                "modified_ns": source.stat().st_mtime_ns,
                "rows": file_rows[source],
            }
            for source in files
        ],
        "source_rows": source_rows,
        "pair_rows": pair_rows,
        "trace_rows": trace_rows,
        "trace_groups": len(trace_groups),
        "normalized_pairs": len(normalized),
        "unique_pairs": len(examples),
        "duplicate_pairs": duplicate_pairs,
        "split_counts": dict(
            sorted(
                {
                    split: sum(
                        str(example.metadata.get("source_split") or "unspecified") == split
                        for example in examples
                    )
                    for split in {
                        str(example.metadata.get("source_split") or "unspecified")
                        for example in examples
                    }
                }.items()
            )
        ),
        "canonical_sha256": canonical_checksum,
    }
    return PreferenceDataset(tuple(examples), provenance)


def load_preference_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Compatibility loader for strict pair-row JSONL data."""

    return list(load_preference_dataset(path, data_format="pairs").examples)


def preference_dataset_stats(examples: Sequence[PreferenceExample]) -> dict[str, int | float]:
    if not examples:
        raise ValueError("preference statistics require at least one pair")
    chosen_chars = sum(len(_response_text(example.chosen)) for example in examples)
    rejected_chars = sum(len(_response_text(example.rejected)) for example in examples)
    return {
        "pairs": len(examples),
        "multi_turn_prompts": sum(len(example.prompt) > 1 for example in examples),
        "chosen_chars": chosen_chars,
        "rejected_chars": rejected_chars,
        "mean_chosen_chars": round(chosen_chars / len(examples), 2),
        "mean_rejected_chars": round(rejected_chars / len(examples), 2),
    }


@dataclass(frozen=True, slots=True)
class DPOConfig:
    checkpoint_id: str
    checkpoint_ref: str
    adapter_id: str
    adapter_name: str
    dataset_path: str
    dataset_format: str = "auto"
    dataset_checksum: str = ""
    quant: str = "bf16"
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    beta: float = 0.1
    label_smoothing: float = 0.0
    max_length: int = 1024
    epochs: int = 1
    max_steps: int | None = None
    micro_batch: int = 1
    grad_accum: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    seed: int = 42
    log_every: int = 5
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_id",
            "checkpoint_ref",
            "adapter_id",
            "adapter_name",
            "dataset_path",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} is required")
        if self.dataset_format not in PREFERENCE_DATA_FORMATS:
            raise ValueError(
                f"dataset_format must be one of {sorted(PREFERENCE_DATA_FORMATS)}"
            )
        if self.dataset_checksum and (
            not isinstance(self.dataset_checksum, str)
            or len(self.dataset_checksum) != 64
            or any(character not in "0123456789abcdef" for character in self.dataset_checksum)
        ):
            raise ValueError("dataset_checksum must be a lowercase SHA-256 hex digest")
        if self.quant not in {"bf16", "fp16", "fp32", "nf4", "int8"}:
            raise ValueError("quant must be bf16, fp16, fp32, nf4, or int8")
        for name in ("rank", "max_length", "epochs", "micro_batch", "grad_accum", "log_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_length < 8:
            raise ValueError("max_length must be at least 8")
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps < 1):
            raise ValueError("max_steps must be a positive integer or null")
        for name in ("alpha", "beta", "learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")
        if (
            isinstance(self.label_smoothing, bool)
            or not isinstance(self.label_smoothing, (int, float))
            or not math.isfinite(self.label_smoothing)
            or not 0 <= self.label_smoothing < 0.5
        ):
            raise ValueError("label_smoothing must be in [0, 0.5)")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be non-negative")
        if (
            isinstance(self.warmup_ratio, bool)
            or not isinstance(self.warmup_ratio, (int, float))
            or not math.isfinite(self.warmup_ratio)
            or not 0 <= self.warmup_ratio <= 1
        ):
            raise ValueError("warmup_ratio must be in [0, 1]")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DPOConfig:
        raw = dict(value)
        objective = raw.pop("objective", PREFERENCE_OBJECTIVE)
        if objective != PREFERENCE_OBJECTIVE:
            raise ValueError(f"unsupported preference objective {objective!r}")
        max_steps = raw.get("max_steps")
        if max_steps in (None, 0, "", False):
            raw["max_steps"] = None
        return cls(**raw)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": PREFERENCE_OBJECTIVE,
            **{item.name: getattr(self, item.name) for item in fields(self)},
        }


@dataclass(frozen=True, slots=True)
class EncodedResponse:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    target_tokens: int
    prompt_tokens_truncated: int = 0
    target_tokens_truncated: int = 0

    def __post_init__(self) -> None:
        if not self.input_ids or len(self.input_ids) != len(self.labels):
            raise ValueError("encoded response IDs and labels need equal non-zero length")
        if self.labels[0] != -100 or not any(label != -100 for label in self.labels[1:]):
            raise ValueError("encoded response needs a shifted assistant target")


@dataclass(frozen=True, slots=True)
class EncodedPreferencePair:
    chosen: EncodedResponse
    rejected: EncodedResponse


class PreferenceCollator:
    """Render exact completed chats and mask their exact prompt-only prefix.

    This intentionally does not assume that ``add_generation_prompt=True`` is
    a prefix of a stored assistant answer.  Qwen thinking templates can add a
    ``<think>`` prefill only on the generation path.  The assistant header is
    therefore part of each scored completion; identical header tokens cancel
    from the chosen/rejected ratio, while response-dependent thinking markers
    remain trainable.
    """

    def __init__(self, tokenizer: Any, *, max_length: int) -> None:
        if max_length < 8:
            raise ValueError("max_length must be at least 8")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("preference tokenization requires a pad or EOS token")
        self.pad_token_id = int(pad_token_id)

    @staticmethod
    def _flat_ids(value: object) -> list[int]:
        if isinstance(value, Mapping):
            if "input_ids" not in value:
                raise ValueError("chat template mapping did not contain input_ids")
            value = value["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
            raise ValueError("chat template must return one non-empty flat token list")
        return list(value)

    def _render(self, messages: Sequence[PreferenceMessage], *, generating: bool) -> list[int]:
        value = self.tokenizer.apply_chat_template(
            [message.to_dict() for message in messages],
            tokenize=True,
            add_generation_prompt=generating,
        )
        return self._flat_ids(value)

    def _encode_response(
        self,
        prompt: tuple[PreferenceMessage, ...],
        response: PreferenceMessage,
    ) -> EncodedResponse:
        prompt_ids = self._render(prompt, generating=False)
        completed = (*prompt, response)
        full_ids = self._render(completed, generating=False)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("chat template rewrote the prompt when the assistant turn was appended")
        original_prompt_tokens = len(prompt_ids)
        crop = max(0, len(full_ids) - self.max_length)
        prompt_truncated = min(crop, original_prompt_tokens)
        target_truncated = max(0, crop - original_prompt_tokens)
        if crop:
            full_ids = full_ids[crop:]
        prompt_tokens = max(0, original_prompt_tokens - crop)
        labels = [-100] * prompt_tokens + full_ids[prompt_tokens:]
        if labels:
            labels[0] = -100
        target_tokens = sum(label != -100 for label in labels[1:])
        if not target_tokens:
            raise ValueError("max_length removed every shifted assistant target")
        return EncodedResponse(
            tuple(full_ids),
            tuple(labels),
            prompt_tokens,
            target_tokens,
            prompt_truncated,
            target_truncated,
        )

    def encode(self, example: PreferenceExample) -> EncodedPreferencePair:
        return EncodedPreferencePair(
            chosen=self._encode_response(example.prompt, example.chosen),
            rejected=self._encode_response(example.prompt, example.rejected),
        )

    def collate(self, pairs: Sequence[EncodedPreferencePair]) -> dict[str, Tensor]:
        if not pairs:
            raise ValueError("cannot collate an empty preference batch")
        rows = [response for pair in pairs for response in (pair.chosen, pair.rejected)]
        width = max(len(row.input_ids) for row in rows)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for row in rows:
            padding = width - len(row.input_ids)
            input_ids.append([*row.input_ids, *([self.pad_token_id] * padding)])
            labels.append([*row.labels, *([-100] * padding)])
            attention_mask.append([1] * len(row.input_ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def sequence_log_probabilities(model: nn.Module, batch: Mapping[str, Tensor]) -> Tensor:
    """Return summed assistant-token log probabilities for every response row."""

    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise RuntimeError("policy forward did not return rank-3 token logits")
    shifted_labels = batch["labels"][:, 1:]
    shifted_logits = logits[:, :-1].float()
    mask = shifted_labels.ne(-100)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every preference response needs at least one shifted target")
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    )
    return (token_logps.squeeze(-1) * mask).sum(dim=-1)


@dataclass(frozen=True, slots=True)
class DPOMetrics:
    loss: Tensor
    preference_accuracy: Tensor
    reward_margin: Tensor
    chosen_reward: Tensor
    rejected_reward: Tensor


def dpo_objective(
    policy_logps: Tensor,
    reference_logps: Tensor,
    *,
    beta: float,
    label_smoothing: float = 0.0,
) -> DPOMetrics:
    """Compute the reference-relative pairwise DPO objective for interleaved rows."""

    if policy_logps.ndim != 1 or policy_logps.shape != reference_logps.shape:
        raise ValueError("policy and reference log probabilities must be equal rank-1 tensors")
    if policy_logps.numel() < 2 or policy_logps.numel() % 2:
        raise ValueError("DPO log probabilities must contain chosen/rejected row pairs")
    if (
        not math.isfinite(float(beta))
        or beta <= 0
        or not math.isfinite(float(label_smoothing))
        or not 0 <= label_smoothing < 0.5
    ):
        raise ValueError("beta must be positive and label_smoothing must be in [0, 0.5)")
    policy_ratio = policy_logps[0::2] - policy_logps[1::2]
    reference_ratio = reference_logps[0::2] - reference_logps[1::2]
    logits = float(beta) * (policy_ratio - reference_ratio)
    losses = -(
        (1.0 - label_smoothing) * F.logsigmoid(logits)
        + label_smoothing * F.logsigmoid(-logits)
    )
    chosen_reward = float(beta) * (policy_logps[0::2] - reference_logps[0::2])
    rejected_reward = float(beta) * (policy_logps[1::2] - reference_logps[1::2])
    margin = chosen_reward - rejected_reward
    return DPOMetrics(
        loss=losses.mean(),
        preference_accuracy=margin.gt(0).float().mean(),
        reward_margin=margin.mean(),
        chosen_reward=chosen_reward.mean(),
        rejected_reward=rejected_reward.mean(),
    )


def optimizer_steps(pair_count: int, config: DPOConfig) -> int:
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    micro_steps = math.ceil(pair_count / config.micro_batch) * config.epochs
    steps = math.ceil(micro_steps / config.grad_accum)
    return min(steps, config.max_steps) if config.max_steps is not None else steps
