from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_NL_RE = re.compile(r"\n{3,}")
TRAIL_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


@dataclass(slots=True)
class Doc:
    text: str
    source: str = ""
    domain: str = "other"
    meta: dict = field(default_factory=dict)


def normalize_text(s: str) -> str | None:
    if not s:
        return None
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = CTRL_RE.sub("", s)
    s = TRAIL_WS_RE.sub("", s)
    s = MULTI_NL_RE.sub("\n\n", s)
    s = s.strip()
    return s or None


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
