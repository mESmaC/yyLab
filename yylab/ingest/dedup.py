from __future__ import annotations

import ast
import hashlib
import io
import json
import keyword
import math
import re
import sys
import tokenize
from functools import lru_cache
from pathlib import Path

import numpy as np
import xxhash

NUM_PERM = 128
SHINGLE_K = 5
MAX_SHINGLE_WORDS = 12000
SIGNATURE_CHUNK = 1_024
DEFAULT_THRESHOLD = 0.8
TRAINING_HASH_NAMESPACE = "training_sha256_v3"
# datasketch.MinHashLSH(threshold=.8, num_perm=128) selects 9 bands of
# 13 rows and intentionally leaves the final 11 permutation values unused.
_DEFAULT_BAND_RANGES = tuple((start, start + 13) for start in range(0, 117, 13))

_MERSENNE_PRIME = np.uint64((1 << 61) - 1)
_rng = np.random.RandomState(17)
_PERM_A = _rng.randint(1, int(_MERSENNE_PRIME), size=NUM_PERM, dtype=np.uint64)
_PERM_B = _rng.randint(0, int(_MERSENNE_PRIME), size=NUM_PERM, dtype=np.uint64)
_WS_RE = re.compile(r"\s+")
_WORD_OR_PUNCTUATION = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*|[^\w\s]", re.UNICODE)
_GENERIC_CODE_TOKEN = re.compile(
    r"(?:'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"
    r"|(?:\b\d+(?:\.\d+)?\b)"
    r"|(?:[A-Za-z_$][\w$]*)"
    r"|(?:==|!=|<=|>=|=>|->|::|&&|\|\||<<|>>|\+\+|--)"
    r"|[^\s]"
)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def training_hash_key(digest: str) -> str:
    """Namespace exact training hashes away from legacy normalized hashes.

    Historical yyLab releases stored raw and case-folded hashes in the same
    SQLite column.  A prefixed key is therefore required before an exact hash
    can safely participate in global deduplication or review rejection.
    """

    return f"{TRAINING_HASH_NAMESPACE}:{digest}"


def norm_sha256_hex(text: str) -> str:
    norm = _WS_RE.sub(" ", text.casefold()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class _IdentifierAnonymizer(ast.NodeTransformer):
    """Preserve Python syntax while ignoring project-specific identifier names."""

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        return ast.copy_location(ast.Name(id="_id", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:  # noqa: N802
        return ast.copy_location(
            ast.arg(arg="_arg", annotation=self.visit(node.annotation) if node.annotation else None),
            node,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        node.name = "_function"
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:  # noqa: N802
        node.name = "_function"
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:  # noqa: N802
        node.name = "_class"
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:  # noqa: N802
        node.attr = "_attribute"
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        value = node.value
        if value is None or value is Ellipsis or isinstance(value, bool):
            return node
        node.value = f"<{type(value).__name__}>"
        return node


def _normalize_code(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _generic_code_structure(text: str) -> str:
    structural: list[str] = []
    for token in _GENERIC_CODE_TOKEN.findall(text):
        if token[0:1] in {"'", '"'}:
            structural.append("<string>")
        elif token[0].isdigit():
            structural.append("<number>")
        elif token.isidentifier() or token.startswith("$"):
            structural.append(token if keyword.iskeyword(token) else "<identifier>")
        else:
            structural.append(token)
    return " ".join(structural)


def _python_structure(text: str) -> tuple[str, str, list[str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        location = f"{error.lineno or 0}:{error.offset or 0}"
        return _generic_code_structure(text), "invalid", [f"python_parse_error:{location}"]
    anonymized = _IdentifierAnonymizer().visit(tree)
    ast.fix_missing_locations(anonymized)
    return ast.dump(anonymized, include_attributes=False), "valid", []


def _python_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in stream:
            if token.type in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.COMMENT,
            }:
                continue
            if token.type == tokenize.NAME:
                tokens.append(token.string if keyword.iskeyword(token.string) else "<id>")
            elif token.type == tokenize.NUMBER:
                tokens.append("<number>")
            elif token.type == tokenize.STRING:
                tokens.append("<string>")
            else:
                tokens.append(token.string)
    except (IndentationError, tokenize.TokenError):
        return _GENERIC_CODE_TOKEN.findall(text)
    return tokens


def dedup_fingerprints(
    text: str,
    *,
    mode: str,
    logical_path: str,
    language: str = "",
) -> dict[str, object]:
    """Return independent exact, normalized, and structural CDAS-style evidence.

    Compatibility folding is deliberately absent from the natural-text identity,
    so even analysis evidence keeps mathematical superscripts/subscripts distinct.
    Only the independent content hash is eligible for automatic suppression.
    """

    reasons: list[str] = []
    parse_status = "not_applicable"
    if mode == "structured":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            normalized = text
            structure = text
            parse_status = "invalid"
            reasons.append(f"json_parse_error:{error.lineno}:{error.colno}")
        else:
            normalized = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            structure = normalized
            parse_status = "valid"
    elif mode == "code":
        normalized = _normalize_code(text)
        if Path(logical_path).suffix.casefold() == ".py" or language.casefold() == "python":
            structure, parse_status, parse_reasons = _python_structure(text)
            reasons.extend(parse_reasons)
        else:
            structure = _generic_code_structure(text)
            parse_status = "not_checked"
    else:
        normalized = " ".join(text.casefold().split())
        structure = " ".join(_WORD_OR_PUNCTUATION.findall(normalized))
    if not text.strip():
        reasons.append("empty_content")
    controls = sum(
        1 for character in text if ord(character) < 32 and character not in "\n\r\t\f\b"
    )
    if controls:
        reasons.append("control_characters")
    tokens = (
        _python_tokens(normalized)
        if mode == "code"
        and (Path(logical_path).suffix.casefold() == ".py" or language.casefold() == "python")
        else _GENERIC_CODE_TOKEN.findall(normalized)
        if mode == "code"
        else _WORD_OR_PUNCTUATION.findall(normalized)
    )
    return {
        "content_sha256": sha256_hex(text),
        "normalized_sha256": sha256_hex(normalized),
        "structural_sha256": sha256_hex(structure),
        "parse_status": parse_status,
        "token_count": len(tokens),
        "reason_codes": sorted(set(reasons)),
    }


def _shingles(text: str) -> set[bytes]:
    words = text.lower().split()
    if len(words) >= SHINGLE_K:
        words = words[:MAX_SHINGLE_WORDS]
        return {" ".join(words[i : i + SHINGLE_K]).encode("utf-8") for i in range(len(words) - SHINGLE_K + 1)}
    chars = text.lower()
    k = 8
    if len(chars) < k:
        return {chars.encode("utf-8")} if chars else set()
    return {chars[i : i + k].encode("utf-8") for i in range(len(chars) - k + 1)}


def _shingle_hashes(text: str) -> np.ndarray | None:
    words = text.lower().split()
    if len(words) >= SHINGLE_K:
        words = words[:MAX_SHINGLE_WORDS]
        n = len(words) - SHINGLE_K + 1
        hs = np.empty(n, dtype=np.uint64)
        for i in range(n):
            hs[i] = xxhash.xxh64(" ".join(words[i : i + SHINGLE_K]).encode("utf-8")).intdigest()
        return hs
    chars = text.lower()
    k = 8
    if len(chars) < k:
        if not chars:
            return None
        return np.array([xxhash.xxh64(chars.encode("utf-8")).intdigest()], dtype=np.uint64)
    n = len(chars) - k + 1
    hs = np.empty(n, dtype=np.uint64)
    enc = chars.encode("utf-8")
    for i in range(n):
        hs[i] = xxhash.xxh64(enc[i : i + k]).intdigest()
    return hs


def _signature(hs: np.ndarray) -> np.ndarray:
    # Materializing NUM_PERM x every shingle peaks above 20 MiB for one long
    # document once NumPy temporaries are included. Minima are associative, so
    # bounded chunks produce byte-identical signatures with much lower pressure.
    minima = np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, len(hs), SIGNATURE_CHUNK):
        chunk = hs[start : start + SIGNATURE_CHUNK]
        vals = (_PERM_A[:, None] * chunk[None, :] + _PERM_B[:, None]) % _MERSENNE_PRIME
        np.minimum(minima, vals.min(axis=1), out=minima)
    return minima


@lru_cache(maxsize=16)
def _band_ranges_for_threshold(threshold: float) -> tuple[tuple[int, int], ...]:
    """Return datasketch-compatible LSH bands without retaining its index.

    The overwhelmingly common 0.8 lane avoids constructing datasketch objects
    entirely. The compatibility fallback is cached for callers that explicitly
    request a different threshold.
    """

    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be a finite number in (0, 1)")
    if threshold == DEFAULT_THRESHOLD:
        return _DEFAULT_BAND_RANGES
    from datasketch import MinHashLSH

    probe = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    return tuple((int(start), int(end)) for start, end in probe.hashranges)


def _band_fingerprints(
    signature: np.ndarray,
    band_ranges: tuple[tuple[int, int], ...] = _DEFAULT_BAND_RANGES,
) -> tuple[int, ...]:
    """Compress each MinHash band to a band-qualified 128-bit fingerprint.

    Storing the complete 13xuint64 band is collision-free but expensive in a
    Python hash table. XXH3-128 reduces every band to one Python integer. The
    only behavioral difference from datasketch band equality is the negligible
    possibility of a 128-bit collision (roughly 1.3e-20 even after one billion
    documents across nine bands under the birthday bound).
    """

    if signature.shape != (NUM_PERM,) or signature.dtype != np.uint64:
        raise ValueError(f"signature must be a uint64 vector of length {NUM_PERM}")
    return tuple(
        (band_index << 128)
        | xxhash.xxh3_128_intdigest(memoryview(signature[start:end]))
        for band_index, (start, end) in enumerate(band_ranges)
    )


class MinHashDedup:
    """Boolean MinHash LSH index with no per-document keys or candidate sets."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self._band_ranges = _band_ranges_for_threshold(float(threshold))
        self._seen_bands: set[int] = set()
        self._n = 0

    @property
    def indexed_documents(self) -> int:
        return self._n

    @property
    def band_fingerprint_count(self) -> int:
        return len(self._seen_bands)

    @property
    def estimated_index_bytes(self) -> int:
        """Shallow-exact bytes for the set table and its immutable integers."""

        return sys.getsizeof(self._seen_bands) + sum(
            sys.getsizeof(fingerprint) for fingerprint in self._seen_bands
        )

    def _minhash(self, text: str) -> np.ndarray | None:
        hs = _shingle_hashes(text)
        if hs is None or len(hs) == 0:
            return None
        return _signature(hs)

    def _is_duplicate_signature(self, signature: np.ndarray) -> bool:
        fingerprints = _band_fingerprints(signature, self._band_ranges)
        if any(fingerprint in self._seen_bands for fingerprint in fingerprints):
            return True
        self._seen_bands.update(fingerprints)
        self._n += 1
        return False

    def is_duplicate(self, text: str) -> bool:
        signature = self._minhash(text)
        if signature is None:
            return False
        return self._is_duplicate_signature(signature)
