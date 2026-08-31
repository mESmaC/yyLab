from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .language import _EN_STOPWORDS, _WORD_RE

SCORE_CHARS = 20_000

_URL_RE = re.compile(r"https?://[^\s)]+|www\.[^\s)]+", re.IGNORECASE)
_SYMBOL_WORD_RE = re.compile(r"^[\W_]+$")
_BULLET_RE = re.compile(r"^\s*[-*•▪‣]\s+")
_BOILERPLATE = (
    ("cookie_notice", re.compile(r"(?:we|this site) (?:uses|use) cookies|cookie (?:policy|consent)|accept all cookies", re.IGNORECASE)),
    ("subscribe", re.compile(r"subscribe to (?:our|this) (?:newsletter|channel|feed)|sign up for (?:our|free)", re.IGNORECASE)),
    ("rights", re.compile(r"all rights reserved|© \d{4}", re.IGNORECASE)),
    ("lorem", re.compile(r"lorem ipsum", re.IGNORECASE)),
    ("nav_menu", re.compile(r"skip to (?:content|main|navigation)|toggle navigation|menu close", re.IGNORECASE)),
)


@dataclass(slots=True)
class QualityReport:
    score: float = 1.0
    reasons: list[str] = field(default_factory=list)

    def penalize(self, amount: float, reason: str) -> None:
        self.score = max(0.0, self.score - amount)
        self.reasons.append(reason)


def score_quality(text: str) -> QualityReport:
    rep = QualityReport()
    sample = text[:SCORE_CHARS]
    words = _WORD_RE.findall(sample.lower())
    n_words = len(words)
    if n_words < 20:
        rep.penalize(1.0, "too_short")
        return rep
    if n_words < 50:
        rep.penalize(0.5, "short")

    avg_len = sum(len(w) for w in words) / n_words
    if avg_len < 2.5 or avg_len > 16:
        rep.penalize(0.6, "word_length")

    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / n_words < 0.7:
        rep.penalize(0.4, "nonalphabetic_words")

    if n_words >= 30:
        stop_frac = sum(1 for w in words if w in _EN_STOPWORDS) / n_words
        if stop_frac < 0.04:
            rep.penalize(0.4, "stopword_desert")

    raw_tokens = sample.split()
    if raw_tokens:
        sym = sum(1 for t in raw_tokens if _SYMBOL_WORD_RE.match(t))
        if sym / len(raw_tokens) > 0.1:
            rep.penalize(0.3, "symbol_heavy")

    urls = len(_URL_RE.findall(sample))
    if urls / n_words > 0.1:
        rep.penalize(0.4, "url_heavy")

    lines = [ln for ln in sample.split("\n") if ln.strip()]
    if len(lines) >= 8:
        punct_end = sum(1 for ln in lines if ln.rstrip()[-1] in ".!?…\"')")
        if punct_end / len(lines) < 0.10:
            rep.penalize(0.2, "line_punctuation")
        bullets = sum(1 for ln in lines if _BULLET_RE.match(ln))
        if bullets / len(lines) > 0.6:
            rep.penalize(0.2, "bullet_list")
        line_counts = Counter(ln.strip() for ln in lines)
        dup_chars = sum(len(ln) * (c - 1) for ln, c in line_counts.items() if c > 1)
        total_chars = max(sum(len(ln) for ln in lines), 1)
        if dup_chars / total_chars > 0.3:
            rep.penalize(0.4, "duplicate_lines")

    paras = [p for p in sample.split("\n\n") if p.strip()]
    if len(paras) >= 4:
        para_counts = Counter(p.strip() for p in paras)
        dup_p = sum(len(p) * (c - 1) for p, c in para_counts.items() if c > 1)
        if dup_p / max(sum(len(p) for p in paras), 1) > 0.3:
            rep.penalize(0.3, "duplicate_paragraphs")

    if n_words >= 30:
        bigrams = Counter(zip(words, words[1:]))
        if bigrams:
            _, c2 = bigrams.most_common(1)[0]
            if c2 * 2 / n_words > 0.2:
                rep.penalize(0.35, "bigram_repetition")
        trigrams = Counter(zip(words, words[1:], words[2:]))
        if trigrams:
            _, c3 = trigrams.most_common(1)[0]
            if c3 * 3 / n_words > 0.18:
                rep.penalize(0.35, "trigram_repetition")

    letters = sum(1 for ch in sample if ch.isalpha())
    if letters:
        upper = sum(1 for ch in sample if ch.isupper())
        if upper / letters > 0.25:
            rep.penalize(0.3, "uppercase_heavy")

    bp_hits = sum(1 for _, rx in _BOILERPLATE if rx.search(sample[:4000]) or rx.search(sample[-2000:]))
    if bp_hits:
        rep.penalize(min(0.2 * bp_hits, 0.4), "boilerplate")

    return rep
