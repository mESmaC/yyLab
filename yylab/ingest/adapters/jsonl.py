from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..records import Doc
from .base import Adapter

TEXT_FIELDS = (
    "text",
    "content",
    "body",
    "prompt",
    "output",
    "completion",
    "instruction",
    "question",
    "answer",
    "input",
    "message",
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
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(
                seg.get("text", "") for seg in content if isinstance(seg, dict) and isinstance(seg.get("text"), str)
            )
        content = content if isinstance(content, str) else ""
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


def _record_text(rec: dict) -> str | None:
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        msgs = rec.get("conversations")
    if isinstance(msgs, list) and msgs:
        rendered = format_messages(msgs)
        if rendered:
            return rendered
    for f in TEXT_FIELDS:
        v = rec.get(f)
        if isinstance(v, str) and v.strip():
            return v
    return None


class JsonlAdapter(Adapter):
    name = "jsonl"
    extensions = {".jsonl", ".ndjson", ".json"}
    default_domain = "other"

    def extract(self, path: Path) -> Iterator[Doc]:
        raw = self._read_text(path).strip()
        if not raw:
            return
        records: list[dict] = []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    records = [r for r in parsed if isinstance(r, dict)]
            except json.JSONDecodeError:
                records = []
        if not records:
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    records.append(rec)
        for rec in records:
            text = _record_text(rec)
            if text:
                yield Doc(text=text, source=str(path), domain=self.default_domain)
