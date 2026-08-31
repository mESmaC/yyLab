from __future__ import annotations

import hashlib
import re

import numpy as np
import xxhash
from datasketch import MinHash, MinHashLSH

NUM_PERM = 128
SHINGLE_K = 5
MAX_SHINGLE_WORDS = 12000

_MERSENNE_PRIME = np.uint64((1 << 61) - 1)
_rng = np.random.RandomState(17)
_PERM_A = _rng.randint(1, int(_MERSENNE_PRIME), size=NUM_PERM, dtype=np.uint64)
_PERM_B = _rng.randint(0, int(_MERSENNE_PRIME), size=NUM_PERM, dtype=np.uint64)
_WS_RE = re.compile(r"\s+")


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm_sha256_hex(text: str) -> str:
    norm = _WS_RE.sub(" ", text.casefold()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


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
    vals = (_PERM_A[:, None] * hs[None, :] + _PERM_B[:, None]) % _MERSENNE_PRIME
    return vals.min(axis=1).astype(np.uint64)


class MinHashDedup:
    def __init__(self, threshold: float = 0.8) -> None:
        self.lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
        self._n = 0

    def _minhash(self, text: str) -> MinHash | None:
        hs = _shingle_hashes(text)
        if hs is None or len(hs) == 0:
            return None
        m = MinHash(num_perm=NUM_PERM)
        m.hashvalues = _signature(hs)
        return m

    def is_duplicate(self, text: str) -> bool:
        m = self._minhash(text)
        if m is None:
            return False
        key = f"d{self._n}"
        try:
            if self.lsh.query(m):
                return True
            self.lsh.insert(key, m)
            self._n += 1
        except ValueError:
            return True
        return False
