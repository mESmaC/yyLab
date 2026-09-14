"""Conservative, opt-in filtering of complete Sprint quota pool rows.

The version fixes both criteria: global exact token-row SHA256 equality and
at least 90% redundant overlapping token 8-gram occurrences in rows of at
least 256 tokens. This is not semantic deduplication or a text quality score.
"""
from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "sprint-quota-row-filter-v1"
SAMPLE_LIMIT = 8


def normalize_row_filter(value: Any) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) - {"enabled", "version"}:
        raise ValueError("quota_row_filter must contain only enabled and version")
    if not isinstance(value.get("enabled"), bool):
        raise ValueError("quota_row_filter.enabled must be a boolean")
    if not value["enabled"]:
        return None
    if value.get("version", VERSION) != VERSION:
        raise ValueError("unsupported Sprint quota row filter version")
    return {"enabled": True, "version": VERSION}


def redundant_eightgram_fraction(row: np.ndarray) -> float:
    """Exact token-tuple equality, counting each 8-gram after its first copy."""
    row = np.ascontiguousarray(row)
    if row.ndim != 1 or row.dtype.kind != "u":
        raise ValueError("row filtering requires one-dimensional unsigned token IDs")
    count = len(row) - 7
    if count <= 0:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(row, 8)
    keys = windows.view(np.dtype((np.void, row.dtype.itemsize * 8))).reshape(-1)
    unique = len(np.unique(keys))
    return (count - unique) / count


class QuotaRowFilter:
    """Stateful CPU filter; its disk-backed digest index spans selected sources.

    Rows must arrive in configured source order, in original packed row order.
    Only retained rows claim an exact-hash key. Low-information rejection takes
    precedence over exact-duplicate rejection, so reason counters are exclusive.
    The caller owns the new SQLite file and may remove it after close().
    """

    def __init__(self, state_path: str | Path, policy: Mapping | None = None):
        self.policy = normalize_row_filter({"enabled": True, "version": VERSION} if policy is None else policy)
        if self.policy is None:
            raise ValueError("QuotaRowFilter requires an enabled policy")
        state_path = Path(state_path)
        if state_path.exists():
            raise ValueError("row filter state path must be new")
        self.connection = sqlite3.connect(state_path)
        # Derived preparation state is disposable; completed pools are separately
        # fsynced and checksummed before publication. Bound the SQLite page cache.
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA cache_size=-16384")
        self.connection.execute("CREATE TABLE seen (digest BLOB PRIMARY KEY, source TEXT NOT NULL, row_number INTEGER NOT NULL) WITHOUT ROWID")
        self.receipts: dict[str, dict] = {}
        self.row_contract: tuple[str, int] | None = None

    def source_receipt(self, source: str) -> dict:
        if source not in self.receipts:
            self.receipts[source] = {
                "policy": dict(self.policy),
                "input_rows": 0, "retained_rows": 0, "rejected_rows": 0,
                "input_token_slots": 0, "retained_token_slots": 0, "rejected_token_slots": 0,
                "reason_counts": {"low_information_8gram": 0, "exact_duplicate_row": 0},
                "reason_token_slots": {"low_information_8gram": 0, "exact_duplicate_row": 0},
                "rejected_samples": [],
                "sample_limit": SAMPLE_LIMIT,
                "exact_equality": "SHA256 of original unsigned little-endian token-row bytes; cryptographic collision resistance assumed",
                "reason_precedence": "low_information_8gram before exact_duplicate_row; only retained rows own exact hashes",
                "original_row_scope": "zero-based complete row offset in the unfiltered source pool",
            }
        return self.receipts[source]

    def filter_batch(self, ids: np.ndarray, *, source: str, row_offset: int) -> np.ndarray:
        ids = np.asarray(ids)
        if ids.ndim != 2 or ids.dtype.str not in {"<u2", "<u4"} or ids.shape[1] < 1:
            raise ValueError("row filter expects complete little-endian uint16 or uint32 rows")
        contract = (ids.dtype.str, ids.shape[1])
        if self.row_contract is not None and self.row_contract != contract:
            raise ValueError("row filter token dtype and sequence length must remain constant")
        self.row_contract = contract
        receipt = self.source_receipt(source)
        if row_offset != receipt["input_rows"]:
            raise ValueError("row filter requires contiguous original row offsets per source")
        keep = np.ones(len(ids), dtype=bool)
        length = ids.shape[1]
        with self.connection:
            for index, row in enumerate(ids):
                original_row = row_offset + index
                digest = hashlib.sha256(row.tobytes()).digest()
                fraction = redundant_eightgram_fraction(row) if length >= 256 else None
                reason = "low_information_8gram" if fraction is not None and fraction >= 0.9 else None
                duplicate = None
                if reason is None:
                    cursor = self.connection.execute(
                        "INSERT OR IGNORE INTO seen VALUES (?, ?, ?)", (digest, source, original_row))
                    if cursor.rowcount == 0:
                        reason = "exact_duplicate_row"
                        duplicate = self.connection.execute(
                            "SELECT source, row_number FROM seen WHERE digest=?", (digest,)).fetchone()
                if reason is not None:
                    keep[index] = False
                    receipt["reason_counts"][reason] += 1
                    receipt["reason_token_slots"][reason] += length
                    if len(receipt["rejected_samples"]) < SAMPLE_LIMIT:
                        sample = {"original_pool_row": original_row, "sha256": digest.hex(), "reason": reason}
                        if reason == "low_information_8gram":
                            sample["redundant_8gram_fraction"] = fraction
                        if duplicate is not None:
                            sample["duplicate_of"] = {"source": duplicate[0], "original_pool_row": duplicate[1]}
                        receipt["rejected_samples"].append(sample)
        retained = int(keep.sum())
        receipt["input_rows"] += len(ids)
        receipt["retained_rows"] += retained
        receipt["rejected_rows"] += len(ids) - retained
        receipt["input_token_slots"] += ids.size
        receipt["retained_token_slots"] += retained * length
        receipt["rejected_token_slots"] += (len(ids) - retained) * length
        return keep

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
