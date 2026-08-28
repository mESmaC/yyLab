from __future__ import annotations

import hashlib

from datasketch import MinHash, MinHashLSH

NUM_PERM = 128
SHINGLE_K = 5
MAX_SHINGLE_WORDS = 12000


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


class MinHashDedup:
    def __init__(self, threshold: float = 0.8) -> None:
        self.lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
        self._n = 0

    def is_duplicate(self, text: str) -> bool:
        sh = _shingles(text)
        if not sh:
            return False
        m = MinHash(num_perm=NUM_PERM)
        for s in sh:
            m.update(s)
        key = f"d{self._n}"
        try:
            if self.lsh.query(m):
                return True
            self.lsh.insert(key, m)
            self._n += 1
        except ValueError:
            return True
        return False
