"""Per-corpus cache of prepared documents for quick re-import.

Extraction and analysis are the expensive part of ingestion; ingest *policy*
(hygiene thresholds, language/quality filters, asset selection profiles,
deduplication) is cheap.  The cache stores every prepared document of every
extracted asset, keyed by the asset's content hash and the adapter that
produced it, so a re-import with different policies only re-runs the policy
stage.  Assets whose content, adapter, or analysis parameters changed are
simply re-extracted and the cache entry replaced.

Layout: ``corpora/<id>/prepared.sqlite3`` with per-chunk zstd JSONL payloads
(serialized by the worker that prepared them) plus optional per-tokenizer
document token counts.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapters.base import Adapter
from .catalog import AssetObservation
from .prepare import ChunkResult, PreparedDoc, decode_payload, encode_payload

CACHE_SCHEMA_VERSION = 2
CACHE_FILE = "prepared.sqlite3"
_COMMIT_EVERY = 64


@dataclass(frozen=True, slots=True)
class CachedAsset:
    asset_id: str
    logical_path: str
    raw_sha256: str
    byte_size: int
    modified_ns: int
    adapter_name: str
    adapter_version: int
    chunk_count: int
    doc_count: int
    records_seen: int
    lines_seen: int
    analysis_key: str = ""


class PreparedCache:
    def __init__(self, path: Path, *, analysis_key: dict[str, Any]) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        # Incremental vacuum only takes effect on a fresh file; it lets
        # ``retain_only`` give space back without a full rewrite.
        self.connection.execute("PRAGMA auto_vacuum=INCREMENTAL")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS assets(
              asset_id TEXT PRIMARY KEY, logical_path TEXT NOT NULL,
              raw_sha256 TEXT NOT NULL, byte_size INTEGER NOT NULL,
              modified_ns INTEGER NOT NULL, adapter_name TEXT NOT NULL,
              adapter_version INTEGER NOT NULL, chunk_count INTEGER NOT NULL,
              doc_count INTEGER NOT NULL, records_seen INTEGER NOT NULL,
              lines_seen INTEGER NOT NULL, created_at TEXT NOT NULL,
              analysis_key TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS chunks(
              asset_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
              doc_count INTEGER NOT NULL, records_seen INTEGER NOT NULL,
              lines_seen INTEGER NOT NULL, issues_json TEXT NOT NULL,
              payload BLOB NOT NULL,
              PRIMARY KEY(asset_id, chunk_index)
            );
            CREATE TABLE IF NOT EXISTS chunk_tokens(
              asset_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
              tokenizer_key TEXT NOT NULL, counts BLOB NOT NULL,
              PRIMARY KEY(asset_id, chunk_index, tokenizer_key)
            );
            """
        )
        self.stats: dict[str, int] = {
            "assets_reused": 0,
            "assets_stored": 0,
            "assets_invalidated": 0,
            "chunks_reused": 0,
            "chunks_stored": 0,
            "chunks_augmented": 0,
            "token_chunks_reused": 0,
            "assets_pruned": 0,
        }
        self._pending_writes = 0
        self.closed = False
        # The analysis key is stored per asset so a run that changes it (for
        # example a different domain override) only replaces the assets it
        # actually re-extracts; a failed run leaves a self-describing mix.
        self.analysis_key = json.dumps(analysis_key, sort_keys=True, separators=(",", ":"))
        stored = dict(self.connection.execute("SELECT key, value FROM meta"))
        if stored.get("cache_schema_version") != str(CACHE_SCHEMA_VERSION):
            if stored:
                self.stats["assets_invalidated"] += int(
                    self.connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
                )
            self.connection.executescript(
                "DELETE FROM assets; DELETE FROM chunks; DELETE FROM chunk_tokens;"
            )
        self.connection.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            (
                ("analysis_key", self.analysis_key),
                ("cache_schema_version", str(CACHE_SCHEMA_VERSION)),
            ),
        )
        self.connection.commit()

    # -- lookups -----------------------------------------------------------

    def lookup(self, asset_id: str) -> CachedAsset | None:
        row = self.connection.execute(
            "SELECT asset_id, logical_path, raw_sha256, byte_size, modified_ns, adapter_name,"
            " adapter_version, chunk_count, doc_count, records_seen, lines_seen, analysis_key"
            " FROM assets WHERE asset_id=?",
            (asset_id,),
        ).fetchone()
        if row is None:
            return None
        return CachedAsset(
            asset_id=str(row[0]),
            logical_path=str(row[1]),
            raw_sha256=str(row[2]),
            byte_size=int(row[3]),
            modified_ns=int(row[4]),
            adapter_name=str(row[5]),
            adapter_version=int(row[6]),
            chunk_count=int(row[7]),
            doc_count=int(row[8]),
            records_seen=int(row[9]),
            lines_seen=int(row[10]),
            analysis_key=str(row[11] or ""),
        )

    def matches(self, cached: CachedAsset, observation: AssetObservation, adapter: Adapter) -> bool:
        """Content hash, adapter identity/version, and chunk completeness must agree.

        The ``assets`` row is only written after every chunk of an asset was
        prepared without error, so its presence already implies completeness;
        the chunk count is re-checked so a damaged cache degrades to a miss.
        """

        if not (
            observation.raw_sha256
            and cached.raw_sha256 == observation.raw_sha256
            and cached.adapter_name == adapter.name
            and cached.adapter_version == int(getattr(adapter, "version", 1))
            and cached.analysis_key == self.analysis_key
        ):
            return False
        stored = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE asset_id=?", (cached.asset_id,)
            ).fetchone()[0]
        )
        return stored == cached.chunk_count and cached.chunk_count > 0

    def load_chunk(self, asset_id: str, chunk_index: int) -> ChunkResult:
        row = self.connection.execute(
            "SELECT records_seen, lines_seen, issues_json, payload FROM chunks"
            " WHERE asset_id=? AND chunk_index=?",
            (asset_id, chunk_index),
        ).fetchone()
        if row is None:
            raise LookupError(f"prepared cache is missing chunk {chunk_index} of {asset_id}")
        issues = [
            (str(locator), str(code), str(detail)) for locator, code, detail in json.loads(row[2])
        ]
        self.stats["chunks_reused"] += 1
        docs = decode_payload(bytes(row[3]))
        return ChunkResult(
            docs=docs,
            issues=issues,
            records_seen=int(row[0]),
            lines_seen=int(row[1]),
            approx_bytes=sum(doc.approx_bytes() for doc in docs),
        )

    def load_tokens(self, asset_id: str, chunk_index: int, keys: list[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        placeholders = ",".join("?" * len(keys))
        rows = self.connection.execute(
            "SELECT tokenizer_key, counts FROM chunk_tokens"
            f" WHERE asset_id=? AND chunk_index=? AND tokenizer_key IN ({placeholders})",
            (asset_id, chunk_index, *keys),
        ).fetchall()
        out = {str(key): np.frombuffer(bytes(blob), dtype=np.int32).copy() for key, blob in rows}
        if out:
            self.stats["token_chunks_reused"] += 1
        return out

    # -- writes ------------------------------------------------------------

    def _maybe_commit(self) -> None:
        self._pending_writes += 1
        if self._pending_writes >= _COMMIT_EVERY:
            self.commit()

    def store_chunk(self, asset_id: str, chunk_index: int, result: ChunkResult) -> None:
        if result.payload is None:
            raise ValueError("chunk result carries no cache payload")
        self.connection.execute(
            "INSERT OR REPLACE INTO chunks(asset_id, chunk_index, doc_count, records_seen,"
            " lines_seen, issues_json, payload) VALUES(?,?,?,?,?,?,?)",
            (
                asset_id,
                int(chunk_index),
                len(result.docs),
                int(result.records_seen),
                int(result.lines_seen),
                json.dumps(
                    [[locator, code, detail] for locator, code, detail in result.issues],
                    ensure_ascii=False,
                ),
                sqlite3.Binary(result.payload),
            ),
        )
        self.stats["chunks_stored"] += 1
        self._maybe_commit()

    def replace_chunk_docs(self, asset_id: str, chunk_index: int, docs: list[PreparedDoc]) -> None:
        """Write back augmented documents (e.g. MinHash bands computed later)."""

        self.connection.execute(
            "UPDATE chunks SET payload=?, doc_count=? WHERE asset_id=? AND chunk_index=?",
            (sqlite3.Binary(encode_payload(docs)), len(docs), asset_id, int(chunk_index)),
        )
        self.stats["chunks_augmented"] += 1
        self._maybe_commit()

    def store_asset(
        self,
        observation: AssetObservation,
        adapter: Adapter,
        *,
        chunk_count: int,
        doc_count: int,
        records_seen: int,
        lines_seen: int,
    ) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO assets(asset_id, logical_path, raw_sha256, byte_size,"
            " modified_ns, adapter_name, adapter_version, chunk_count, doc_count, records_seen,"
            " lines_seen, created_at, analysis_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                observation.asset_id,
                observation.logical_path,
                observation.raw_sha256,
                int(observation.byte_size),
                int(observation.modified_ns),
                adapter.name,
                int(getattr(adapter, "version", 1)),
                int(chunk_count),
                int(doc_count),
                int(records_seen),
                int(lines_seen),
                dt.datetime.now().isoformat(timespec="seconds"),
                self.analysis_key,
            ),
        )
        for table in ("chunks", "chunk_tokens"):
            self.connection.execute(
                f"DELETE FROM {table} WHERE asset_id=? AND chunk_index>=?",
                (observation.asset_id, int(chunk_count)),
            )
        self.stats["assets_stored"] += 1
        self._maybe_commit()

    def store_tokens(self, asset_id: str, chunk_index: int, key: str, counts: np.ndarray) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO chunk_tokens(asset_id, chunk_index, tokenizer_key, counts)"
            " VALUES(?,?,?,?)",
            (asset_id, int(chunk_index), key, sqlite3.Binary(np.asarray(counts, dtype=np.int32).tobytes())),
        )
        self._maybe_commit()

    def invalidate_asset(self, asset_id: str) -> None:
        for table in ("assets", "chunks", "chunk_tokens"):
            self.connection.execute(f"DELETE FROM {table} WHERE asset_id=?", (asset_id,))
        self.stats["assets_invalidated"] += 1
        self._maybe_commit()

    def retain_only(self, asset_ids: Iterable[str]) -> int:
        """Drop cached assets that were not seen by a completed walk of the source.

        Also removes chunk/token rows whose asset row was never written (a run
        interrupted between chunks) and returns freed pages to the file.
        """

        self.connection.execute("CREATE TEMP TABLE IF NOT EXISTS live(asset_id TEXT PRIMARY KEY)")
        self.connection.execute("DELETE FROM live")
        self.connection.executemany(
            "INSERT OR IGNORE INTO live(asset_id) VALUES(?)", ((asset_id,) for asset_id in set(asset_ids))
        )
        removed = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM assets WHERE asset_id NOT IN (SELECT asset_id FROM live)"
            ).fetchone()[0]
        )
        for table in ("assets", "chunks", "chunk_tokens"):
            self.connection.execute(
                f"DELETE FROM {table} WHERE asset_id NOT IN (SELECT asset_id FROM live)"
            )
        for table in ("chunks", "chunk_tokens"):
            self.connection.execute(
                f"DELETE FROM {table} WHERE asset_id NOT IN (SELECT asset_id FROM assets)"
            )
        self.connection.execute("DROP TABLE live")
        self.connection.commit()
        self._pending_writes = 0
        try:
            self.connection.execute("PRAGMA incremental_vacuum")
        except sqlite3.DatabaseError:
            pass
        self.stats["assets_pruned"] += removed
        return removed

    def commit(self) -> None:
        if not self.closed:
            self.connection.commit()
            self._pending_writes = 0

    def summary(self) -> dict[str, Any]:
        assets = int(self.connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0])
        docs = int(self.connection.execute("SELECT COALESCE(SUM(doc_count),0) FROM assets").fetchone()[0])
        return {"enabled": True, "assets": assets, "documents": docs, **self.stats}

    def close(self) -> None:
        if self.closed:
            return
        self.commit()
        self.connection.close()
        self.closed = True


def cache_size_bytes(corpus_directory: Path) -> int:
    total = 0
    for suffix in ("", "-wal", "-shm"):
        candidate = corpus_directory / f"{CACHE_FILE}{suffix}"
        if candidate.exists():
            total += candidate.stat().st_size
    return total


def clear_cache(corpus_directory: Path) -> bool:
    removed = False
    for suffix in ("", "-wal", "-shm"):
        candidate = corpus_directory / f"{CACHE_FILE}{suffix}"
        if candidate.exists():
            candidate.unlink()
            removed = True
    return removed
