from __future__ import annotations

import re

DOMAINS = ["code", "math", "language", "science", "web", "other"]

CODE_EXTS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".c", ".h", ".cpp",
    ".hpp", ".cc", ".hh", ".cxx", ".java", ".kt", ".kts", ".rs", ".go", ".rb", ".php",
    ".cs", ".swift", ".scala", ".sh", ".bash", ".ps1", ".psm1", ".sql", ".lua", ".pl",
    ".r", ".jl", ".dart", ".zig", ".ex", ".exs", ".hs", ".ml", ".clj", ".groovy", ".v",
    ".asm", ".s", ".cu", ".cuh", ".glsl", ".hlsl",
}

WEB_EXTS = {".html", ".htm", ".css", ".xml", ".svg", ".vue"}

MATH_RE = re.compile(
    r"\\(?:frac|int|sum|prod|sqrt|begin|end|align|equation|text|mathrm|mathbf|partial|"
    r"infty|lim|nabla|times|cdot|alpha|beta|gamma|theta|lambda|sigma|omega)\b|\$\$.+?\$\$"
)
CODE_RE = re.compile(
    r"^\s*(?:def |class |import |from \w+ import|#include|function |const |let |var |"
    r"public |private |fn |use |package |module )|;\s*$|\{\s*$",
    re.MULTILINE,
)
SCIENCE_RE = re.compile(
    r"\b(?:abstract|figure \d|table \d|hypothesis|experiment|et al\.|doi:|arxiv:)\b", re.IGNORECASE
)


def domain_for_ext(path_suffix: str) -> str | None:
    ext = path_suffix.lower()
    if ext in CODE_EXTS:
        return "code"
    if ext in WEB_EXTS:
        return "web"
    return None


TRUSTED_HINTS = {"code", "web", "math", "science"}


def classify_content(text: str, hint: str = "other") -> str:
    if hint in TRUSTED_HINTS:
        return hint
    sample = text[:8000]
    lines = max(sample.count("\n") + 1, 1)
    math_hits = len(MATH_RE.findall(sample))
    code_hits = len(CODE_RE.findall(sample))
    sci_hits = len(SCIENCE_RE.findall(sample))
    if math_hits >= 4 and math_hits / lines > 0.15:
        return "math"
    if code_hits >= 5 and code_hits / lines > 0.2:
        return "code"
    if sci_hits >= 3:
        return "science"
    return "language"
