from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_NL_RE = re.compile(r"\n{3,}")
TRAIL_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
SOURCE_SPLITS = frozenset({"train", "validation", "test", "unspecified"})
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "dev": "validation",
    "eval": "validation",
    "evaluation": "validation",
    "test": "test",
    "testing": "test",
    "unspecified": "unspecified",
    "unknown": "unspecified",
}
_ROLE_ALIASES = {
    "human": "user",
    "prompt": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "function": "tool",
}


@dataclass(slots=True)
class Doc:
    text: str
    source: str = ""
    domain: str = "other"
    meta: dict = field(default_factory=dict)
    locator: str = ""
    format_name: str = ""


def normalize_text(s: str) -> str | None:
    if not s:
        return None
    # Preserve semantic compatibility characters in stored/training text.  In
    # particular, NFKC folds mathematical superscripts/subscripts into ordinary
    # digits (``2⁴`` -> ``24``), which silently changes training targets.  CDAS
    # keeps source normalization minimal; NFC gives yyLab canonical composition
    # without compatibility folding.  More aggressive normalization belongs only
    # in duplicate fingerprints, never in model-visible content.
    s = s.removeprefix("\ufeff")
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = CTRL_RE.sub("", s)
    s = TRAIL_WS_RE.sub("", s)
    s = MULTI_NL_RE.sub("\n\n", s)
    s = s.strip()
    return s or None


def canonical_source_split(value: object) -> str | None:
    """Return one canonical source split, or ``None`` when it is not recognizable."""

    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")
    return _SPLIT_ALIASES.get(normalized)


def infer_source_split(source: str | Path = "", explicit: object = None) -> str:
    """Resolve record metadata first, then infer a split from path components."""

    resolved = canonical_source_split(explicit)
    if resolved is not None:
        return resolved
    path = Path(str(source or ""))
    for token in re.findall(r"[a-z0-9]+", path.stem.casefold()):
        resolved = _SPLIT_ALIASES.get(token)
        if resolved is not None and resolved != "unspecified":
            return resolved
    # Directory inference is intentionally stricter than filename inference.
    # Dataset roots and temporary test directories often contain words such as
    # ``test`` inside a longer name; only a directory whose whole name denotes a
    # split is authoritative.
    for candidate in reversed(path.parent.parts):
        resolved = canonical_source_split(candidate)
        if resolved is not None and resolved != "unspecified":
            return resolved
    return "unspecified"


def _message_content(value: object) -> str:
    if isinstance(value, str):
        return normalize_text(value) or ""
    if isinstance(value, list):
        parts: list[str] = []
        for segment in value:
            if not isinstance(segment, dict):
                continue
            text = segment.get("text")
            if isinstance(text, str):
                normalized = normalize_text(text)
                if normalized:
                    parts.append(normalized)
        return "\n".join(parts)
    return ""


def normalize_chat_messages(value: object) -> list[dict[str, Any]]:
    """Normalize common chat-record variants into tokenizer-ready JSON messages.

    The normalized payload deliberately remains structured. Flattened ``text``
    is still produced for deduplication, review, and causal-LM training, while
    chat SFT can consume exact roles, tool calls, and turn boundaries.
    """

    if not isinstance(value, list):
        return []
    messages: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        raw_role = raw.get("role")
        if not isinstance(raw_role, str) or not raw_role.strip():
            raw_role = raw.get("from") or raw.get("speaker")
        if not isinstance(raw_role, str) or not raw_role.strip():
            continue
        role = raw_role.strip().casefold().replace("-", "_").replace(" ", "_")
        role = _ROLE_ALIASES.get(role, role)
        content_value = raw.get("content")
        if content_value is None:
            content_value = raw.get("value")
        if content_value is None:
            content_value = raw.get("text")
        message: dict[str, Any] = {
            "role": role,
            "content": _message_content(content_value),
        }
        reasoning = raw.get("reasoning_content")
        if isinstance(reasoning, str):
            normalized_reasoning = normalize_text(reasoning)
            if normalized_reasoning:
                message["reasoning_content"] = normalized_reasoning
        tool_calls = raw.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_calls = [dict(call) for call in tool_calls if isinstance(call, dict)]
            if normalized_calls:
                message["tool_calls"] = normalized_calls
        for field_name in ("name", "tool_call_id"):
            field_value = raw.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                message[field_name] = field_value.strip()
        if message["content"] or any(
            key in message for key in ("reasoning_content", "tool_calls")
        ):
            messages.append(message)
    return messages


def printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    printable = sum(1 for ch in s if ch.isprintable() or ch in "\n\t")
    return printable / len(s)


def doc_ok(text: str, min_chars: int = 40) -> bool:
    if len(text) < min_chars:
        return False
    if printable_ratio(text) < 0.85:
        return False
    return True
