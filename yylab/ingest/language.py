from __future__ import annotations

import re

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

_EN_STOPWORDS = frozenset(
    "the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us is are was were been has had did does doing".split()
)
_WORD_RE = re.compile(r"[A-Za-z']+")

_SCRIPT_RANGES = (
    ("zh", (0x4E00, 0x9FFF)),
    ("ja", (0x3040, 0x30FF)),
    ("ko", (0xAC00, 0xD7AF)),
    ("ru", (0x0400, 0x04FF)),
    ("ar", (0x0600, 0x06FF)),
    ("he", (0x0590, 0x05FF)),
    ("th", (0x0E00, 0x0E7F)),
    ("hi", (0x0900, 0x097F)),
    ("el", (0x0370, 0x03FF)),
)


def _script_fast_path(sample: str) -> str | None:
    counts: dict[str, int] = {}
    letters = 0
    for ch in sample:
        cp = ord(ch)
        if (65 <= cp <= 90) or (97 <= cp <= 122) or (0xC0 <= cp <= 0x24F):
            letters += 1
            continue
        for lang, (lo, hi) in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                letters += 1
                break
    if letters < 20:
        return None
    for lang, n in counts.items():
        if n / letters > 0.2:
            return lang
    return None


def _english_fast_path(sample: str) -> bool:
    words = _WORD_RE.findall(sample[:4000].lower())
    if len(words) < 12:
        return False
    hits = sum(1 for w in words if w in _EN_STOPWORDS)
    return hits / len(words) >= 0.10


def detect_language(text: str) -> str:
    sample = text[:1500]
    script = _script_fast_path(sample)
    if script is not None:
        return script
    if _english_fast_path(sample):
        return "en"
    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"
