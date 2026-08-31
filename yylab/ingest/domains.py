from __future__ import annotations

import re

DOMAINS = ["code", "math", "language", "science", "web", "news", "conversation", "other"]

CODE_EXTS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".c", ".h", ".cpp",
    ".hpp", ".cc", ".hh", ".cxx", ".java", ".kt", ".kts", ".rs", ".go", ".rb", ".php",
    ".cs", ".swift", ".scala", ".sh", ".bash", ".ps1", ".psm1", ".sql", ".lua", ".pl",
    ".r", ".jl", ".dart", ".zig", ".ex", ".exs", ".hs", ".ml", ".clj", ".groovy", ".v",
    ".asm", ".s", ".cu", ".cuh", ".glsl", ".hlsl",
}

WEB_EXTS = {".css", ".xml", ".svg", ".vue"}

MATH_RE = re.compile(
    r"\\(?:frac|int|sum|prod|sqrt|begin|end|align|equation|text|mathrm|mathbf|partial|"
    r"infty|lim|nabla|times|cdot|alpha|beta|gamma|theta|lambda|sigma|omega|left|right|"
    r"mathbb|mathcal|textbf|operatorname)\b|\$\$.+?\$\$|\\\(.+?\\\)|\\\['?",
    re.DOTALL,
)
CODE_RE = re.compile(
    r"^\s*(?:def |class |import |from \w+ import|#include|function |const |let |var |"
    r"public |private |protected |fn |use |package |module |return |if \(|for \(|while \()"
    r"|;\s*$|\{\s*$|=>|\w+\.\w+\(",
    re.MULTILINE,
)
SCIENCE_RE = re.compile(
    r"\b(?:abstract|figure \d|table \d|hypothesis|experiment|et al\.|doi:|arxiv:|"
    r"p\s*[<>]\s*0\.\d+|confidence interval|randomized|peer[- ]review)\b",
    re.IGNORECASE,
)
NEWS_RE = re.compile(
    r"\b(?:reuters|associated press|staff reporter|by [A-Z][a-z]+ [A-Z][a-z]+,?\s+(?:staff|correspondent)|"
    r"according to (?:officials|police|sources|reports)|"
    r"(?:announced|reportedly|allegedly) (?:on|that)|press release|"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s+\w+ \d{1,2},? \d{4})\b",
    re.IGNORECASE,
)
CONVERSATION_RE = re.compile(
    r"^\s*(?:\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*)?[A-Za-z0-9_\-]{1,24}\s*:\s+\S",
    re.MULTILINE,
)


def domain_for_ext(path_suffix: str) -> str | None:
    ext = path_suffix.lower()
    if ext in CODE_EXTS:
        return "code"
    if ext in WEB_EXTS:
        return "web"
    return None


TRUSTED_HINTS = {"code", "web", "math", "science"}


def _indent_ratio(lines: list[str]) -> float:
    if not lines:
        return 0.0
    indented = sum(1 for ln in lines if ln.startswith(("    ", "\t")) and ln.strip())
    return indented / len(lines)


def classify_content(text: str, hint: str = "other") -> str:
    if hint in TRUSTED_HINTS:
        return hint
    sample = text[:8000]
    lines_list = sample.split("\n")
    lines = max(len(lines_list), 1)
    math_hits = len(MATH_RE.findall(sample))
    code_hits = len(CODE_RE.findall(sample))
    sci_hits = len(SCIENCE_RE.findall(sample))
    if math_hits >= 4 and math_hits / lines > 0.15:
        return "math"
    if code_hits >= 5 and (code_hits / lines > 0.2 or _indent_ratio(lines_list) > 0.3):
        return "code"
    conv_lines = len(CONVERSATION_RE.findall(sample))
    if conv_lines >= 8 and conv_lines / lines > 0.4:
        return "conversation"
    if sci_hits >= 3:
        return "science"
    if NEWS_RE.search(sample):
        return "news"
    return "language"
