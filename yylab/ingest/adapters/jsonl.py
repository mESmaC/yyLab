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


def _record_text(rec: dict) -> str | None:
    for f in TEXT_FIELDS:
        v = rec.get(f)
        if isinstance(v, str) and v.strip():
            return v
    msgs = rec.get("messages")
    if isinstance(msgs, list) and msgs:
        parts = []
        for m in msgs:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                role = m.get("role", "")
                parts.append(f"{role}: {m['content']}" if role else m["content"])
        if parts:
            return "\n".join(parts)
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
