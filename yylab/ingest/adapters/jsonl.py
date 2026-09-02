from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import orjson

from ..records import Doc, infer_source_split, normalize_chat_messages
from .base import Adapter

FULL_DOCUMENT_FIELDS = (
    "text",
    "content",
    "body",
    "document",
    "article",
)
PROMPT_FIELDS = (
    "prompt",
    "instruction",
    "question",
    "query",
    "request",
    "input",
)
RESPONSE_FIELDS = (
    "output",
    "completion",
    "answer",
    "response",
    "reply",
)
TEXT_FIELDS = (
    *FULL_DOCUMENT_FIELDS,
    *PROMPT_FIELDS,
    *RESPONSE_FIELDS,
    "message",
)
MESSAGE_FIELDS = ("messages", "conversations", "messages_json", "conversations_json")
_DOMAINS = {"code", "math", "language", "science", "web", "news", "conversation", "other"}
_NATIVE_CDAS_EXPORT_FIELDS = (
    "schema_version",
    "sequence_id",
    "document_id",
    "source_id",
    "asset_id",
    "logical_path",
    "locator",
    "format",
    "content_sha256",
    "mixture_group",
    "selection_rank",
    "sample_weight",
    "estimated_tokens",
    "target_tokens",
    "sequence_boundary",
)
_ADDITIONAL_METADATA_FIELDS = (
    "sample_id",
    "language",
    "lang",
    "format_name",
    "path",
    "source_path",
)


def _format_tool_calls(calls) -> list[str]:
    lines: list[str] = []
    if not isinstance(calls, list):
        return lines
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = fn.get("name") or call.get("name") or "tool"
        args = fn.get("arguments") if isinstance(fn, dict) else None
        if args is None:
            args = call.get("arguments")
        if isinstance(args, (dict, list)):
            args_s = json.dumps(args, ensure_ascii=False)
        else:
            args_s = str(args) if args else ""
        lines.append(f"tool_call {name}: {args_s}" if args_s else f"tool_call {name}")
    return lines


def format_messages(msgs: list) -> str | None:
    parts: list[str] = []
    for m in normalize_chat_messages(msgs):
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "tool":
            name = m.get("name") or "tool"
            if content.strip():
                parts.append(f"tool_result {name}: {content}")
            continue
        body: list[str] = []
        reasoning = m.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            body.append(f"reasoning: {reasoning.strip()}")
        if content.strip():
            body.append(content.strip())
        body.extend(_format_tool_calls(m.get("tool_calls")))
        if body:
            joined = "\n".join(body)
            parts.append(f"{role}: {joined}" if role else joined)
    return "\n\n".join(parts) if parts else None


def _string_field(rec: dict, field: str) -> str:
    for key in (field, f"{field}_text"):
        value = rec.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _full_document_field(rec: dict, field: str) -> str:
    """Return source document text without silently trimming its boundaries."""

    for key in (field, f"{field}_text"):
        value = rec.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _paired_messages(rec: dict) -> list[dict]:
    if any(_string_field(rec, field) for field in FULL_DOCUMENT_FIELDS):
        return []
    instruction = _string_field(rec, "instruction")
    prompt = next(
        (
            value
            for field in ("instruction", "question", "prompt", "query", "request")
            if (value := _string_field(rec, field))
        ),
        "",
    )
    if not prompt:
        prompt = _string_field(rec, "input")
    extra_input = _string_field(rec, "input") if instruction else ""
    if extra_input:
        prompt = f"{prompt}\n\n{extra_input}"
    response = next(
        (
            value
            for field in RESPONSE_FIELDS
            if (value := _string_field(rec, field))
        ),
        "",
    )
    if not prompt or not response:
        return []
    messages: list[dict] = []
    system = _string_field(rec, "system")
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(
        (
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        )
    )
    return normalize_chat_messages(messages)


def _record_messages(rec: dict) -> list[dict]:
    def decode(value: object) -> list:
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value.lstrip().startswith("["):
            try:
                parsed = orjson.loads(value)
            except orjson.JSONDecodeError:
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    return []
            return parsed if isinstance(parsed, list) else []
        return []

    for field in MESSAGE_FIELDS:
        messages = normalize_chat_messages(decode(rec.get(field)))
        if messages:
            return messages
    return _paired_messages(rec)


def _record_text(rec: dict, messages: list[dict] | None = None) -> str | None:
    msgs = _record_messages(rec) if messages is None else messages
    if msgs:
        rendered = format_messages(msgs)
        if rendered:
            return rendered
    # Unstructured full-document fields are authoritative. Paired trace fields
    # were converted to structured messages above and must never collapse to
    # only a prompt or only an answer.
    for field in FULL_DOCUMENT_FIELDS:
        if value := _full_document_field(rec, field):
            return value
    instruction = _string_field(rec, "instruction")
    if instruction:
        extra_input = _string_field(rec, "input")
        return f"{instruction}\n\n{extra_input}" if extra_input else instruction
    for field in TEXT_FIELDS:
        if value := _string_field(rec, field):
            return value
    return None


def _record_split(rec: dict, path: Path) -> str:
    explicit = rec.get("split")
    if explicit is None:
        explicit = rec.get("source_split")
    if explicit is None:
        explicit = rec.get("partition")
    provenance = rec.get("provenance")
    if isinstance(provenance, dict) and explicit is None:
        explicit = provenance.get("split") or provenance.get("partition")
    for field in ("logical_path", "path", "source_path"):
        origin = rec.get(field)
        if isinstance(origin, str) and origin.strip():
            return infer_source_split(origin, explicit)
    if isinstance(provenance, dict):
        origin = (
            provenance.get("logical_path")
            or provenance.get("path")
            or provenance.get("source_path")
        )
        if isinstance(origin, str) and origin.strip():
            return infer_source_split(origin, explicit)
    return infer_source_split(path, explicit)


def _record_domain(rec: dict, messages: list[dict] | None = None) -> str:
    if messages:
        return "conversation"
    direct = rec.get("domain")
    if isinstance(direct, str) and direct.casefold() in _DOMAINS:
        return direct.casefold()
    labels = rec.get("labels")
    if isinstance(labels, dict):
        content_type = labels.get("content_type")
        if isinstance(content_type, str):
            value = content_type.casefold()
            if value.startswith("code"):
                return "code"
            if value in {"conversation", "chat", "trace"}:
                return "conversation"
            if value in {"natural", "natural-text", "prose"}:
                return "language"
        domains = labels.get("domains")
        if isinstance(domains, list):
            joined = " ".join(str(value).casefold() for value in domains)
            if "machine-learning" in joined or "science" in joined:
                return "science"
            if "code" in joined:
                return "code"
    return "other"


def _metadata_value(value: object) -> object | None:
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return None


def _record_metadata(rec: dict, path: Path, messages: list[dict]) -> dict:
    meta: dict = {
        "split": _record_split(rec, path),
        "representation": "chat_messages" if messages else "text_field",
    }
    if messages:
        meta["messages"] = messages
    top_level_lineage: dict = {}
    for field in (*_NATIVE_CDAS_EXPORT_FIELDS, *_ADDITIONAL_METADATA_FIELDS):
        if (value := _metadata_value(rec.get(field))) is not None:
            meta[field] = value
            if field in _NATIVE_CDAS_EXPORT_FIELDS:
                top_level_lineage[field] = value

    for field in ("labels", "metrics"):
        value = rec.get(field)
        if isinstance(value, dict):
            meta[field] = dict(value)

    provenance_value = rec.get("provenance")
    provenance = dict(provenance_value) if isinstance(provenance_value, dict) else {}
    for field, value in top_level_lineage.items():
        provenance.setdefault(field, value)
    if provenance:
        meta["provenance"] = provenance
    return meta


def _locator_value(value: object) -> str:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return ""
    normalized = str(value).strip()
    if not normalized:
        return ""
    return (
        normalized.replace("%", "%25")
        .replace(";", "%3B")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace("\t", "%09")[:512]
    )


def _record_locator(rec: dict, record_index: int, prefix: str) -> str:
    parts = [f"{prefix}={record_index}"]
    for field, label in (
        ("document_id", "document_id"),
        ("sequence_id", "sequence_id"),
        ("sample_id", "sample"),
    ):
        if value := _locator_value(rec.get(field)):
            parts.append(f"{label}={value}")
    if value := _locator_value(rec.get("locator")):
        parts.append(f"native_locator={value}")
    return ";".join(parts)


def _record_format_name(rec: dict, fallback: str) -> str:
    for field in ("format", "format_name"):
        value = rec.get(field)
        if isinstance(value, str) and value.strip():
            normalized = value.strip().casefold().removeprefix(".")
            if normalized:
                return normalized[:128]
    return fallback


def _unsupported_detail(rec: dict) -> str:
    keys = sorted(str(key)[:128] for key in rec)[:32]
    return f"no supported text or message fields; keys={keys!r}"


def _loads_record(raw: bytes) -> object:
    """Use orjson on the hot path while retaining stdlib's permissive fallback."""

    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError:
        # The old adapter used ``json.loads``. Its permissive handling of values
        # such as NaN is worth preserving for uncommon, pre-existing datasets.
        return json.loads(raw.decode("utf-8", errors="ignore"))


def _report_unsupported_type(on_error, locator: str, value: object) -> None:
    if on_error is not None:
        on_error(locator, "unsupported_record_type", type(value).__name__)


def _iter_records(path: Path, on_error=None) -> Iterator[dict]:
    """Stream line-oriented JSON without materializing the entire source file.

    Pretty-printed objects and top-level arrays still use the compatible
    whole-document path. Normal JSONL/NDJSON, including a single compact object,
    stays bounded to one physical line.
    """

    with path.open("rb") as fh:
        first = b""
        for raw_line in fh:
            if raw_line.strip():
                first = raw_line
                break
        if not first:
            return

        if first.startswith(b"\xef\xbb\xbf"):
            first = first[3:]

        stripped = first.lstrip()
        if not stripped.startswith(b"["):
            decode_failed = False
            try:
                first_record = _loads_record(first)
            except (json.JSONDecodeError, UnicodeDecodeError):
                first_record = None
                decode_failed = True
            if isinstance(first_record, dict):
                yield first_record
                for line_number, raw_line in enumerate(fh, start=2):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        record = _loads_record(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        if on_error is not None:
                            on_error(
                                f"line={line_number}",
                                "json_decode_error",
                                f"{type(error).__name__}: {error}",
                            )
                        continue
                    if isinstance(record, dict):
                        yield record
                    else:
                        _report_unsupported_type(
                            on_error,
                            f"line={line_number}",
                            record,
                        )
                return
            if (
                path.suffix.casefold() in {".jsonl", ".ndjson"}
                and stripped.strip() != b"{"
            ):
                if on_error is not None:
                    if decode_failed:
                        on_error(
                            "line=1",
                            "json_decode_error",
                            "first line is not a valid JSON object",
                        )
                    else:
                        _report_unsupported_type(on_error, "line=1", first_record)
                for line_number, raw_line in enumerate(fh, start=2):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        record = _loads_record(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        if on_error is not None:
                            on_error(
                                f"line={line_number}",
                                "json_decode_error",
                                f"{type(error).__name__}: {error}",
                            )
                        continue
                    if isinstance(record, dict):
                        yield record
                    else:
                        _report_unsupported_type(
                            on_error,
                            f"line={line_number}",
                            record,
                        )
                return

    # A top-level array or pretty-printed object cannot be decoded one line at
    # a time. This compatibility path is deliberately the only unbounded case.
    raw = path.read_bytes().strip()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    if not raw:
        return
    decode_failed = False
    try:
        parsed = _loads_record(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        parsed = None
        decode_failed = True
    if isinstance(parsed, list):
        for item_index, record in enumerate(parsed, start=1):
            if isinstance(record, dict):
                yield record
            else:
                _report_unsupported_type(on_error, f"item={item_index}", record)
        return
    if isinstance(parsed, dict):
        yield parsed
        return
    if not decode_failed:
        _report_unsupported_type(on_error, "$document", parsed)
        return

    # Match the previous adapter's recovery behavior for mixed/broken JSONL:
    # malformed physical lines are skipped while valid object lines survive.
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            record = _loads_record(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if on_error is not None:
                on_error(
                    f"line={line_number}",
                    "json_decode_error",
                    f"{type(error).__name__}: {error}",
                )
            continue
        if isinstance(record, dict):
            yield record
        else:
            _report_unsupported_type(on_error, f"line={line_number}", record)


class JsonlAdapter(Adapter):
    name = "jsonl"
    extensions = {".jsonl", ".ndjson", ".json"}
    default_domain = "other"

    def extract(self, path: Path) -> Iterator[Doc]:
        self._reset_issues()
        for record_index, rec in enumerate(
            _iter_records(path, self._record_issue),
            start=1,
        ):
            messages = _record_messages(rec)
            text = _record_text(rec, messages)
            locator = _record_locator(rec, record_index, "record")
            if not text:
                self._record_issue(
                    locator,
                    "unsupported_record_schema",
                    _unsupported_detail(rec),
                )
                continue
            yield Doc(
                text=text,
                source=str(path),
                domain=_record_domain(rec, messages),
                meta=_record_metadata(rec, path, messages),
                locator=locator,
                format_name=_record_format_name(rec, "jsonl"),
            )
