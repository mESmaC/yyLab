from __future__ import annotations

import io
import os
import re
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from sqlite3 import Connection
from typing import Any

import numpy as np
import orjson
import zstandard as zstd

from .. import db
from ..artifacts import (
    corpus_dir,
    next_corpus_generation,
    publish_corpus_generation,
    read_corpus_layout,
    reclaim_retired_generations,
    write_json,
)
from ..config import get_settings
from ..jobs import ProgressFn
from .adapters.base import Adapter, ChunkSpec
from .adapters.loader import discover_adapters, dispatch, walk_files
from .cache import CACHE_FILE, PreparedCache
from .catalog import (
    SELECTION_PROFILES,
    AssetObservation,
    CorpusCatalog,
    asset_policy_reasons,
    digest_file,
    profile_quarantine_reasons,
)
from .dedup import MinHashDedup, training_hash_key
from .language import detect_language
from .parallel import (
    WINDOW_PER_WORKER,
    PrepareTask,
    _Deferred,
    make_executor,
    ordered_results,
    resolve_worker_count,
)
from .prepare import (
    LANG_SKIP_DOMAINS,
    QUALITY_SKIP_DOMAINS,
    TRAINING_NORMALIZATION,
    AnalysisParams,
    ChunkResult,
    PreparedDoc,
    needs_augmentation,
    rebase_locator,
)
from .quality import score_quality
from .tokenize_stats import SPEC_KEYS, TokenCounter

SHARD_DOCS = 20_000
# Amortize SQLite WAL commits and primary-key probes across a bounded text
# batch. At typical corpus row sizes this remains only a few MiB, while cutting
# transaction count eightfold for multi-million-row datasets.
GLOBAL_BATCH = 8_192
GLOBAL_BATCH_BYTES = 64 * 2**20
TOK_BATCH = 2_048
SIDE_HASH_BATCH = 8_192
PROGRESS_ROWS = 4_096
#: Serial runs extract each asset whole (the historical path).  Parallel runs
#: split large assets so a handful of big files can occupy every worker.
CHUNK_WHEN_SERIAL = False

__all__ = [
    "IngestConfig",
    "ingest_in_progress",
    "reimport_corpus",
    "run_ingest",
    "startup_housekeeping",
]

# One ingest or re-import at a time per process: global-dedup registration is
# a check-then-register sequence, and re-import defers registration until its
# output is complete, so two concurrent runs could both keep the same text.
_INGEST_LOCK = threading.Lock()


def ingest_in_progress() -> bool:
    """True while this process is running an ingest or re-import."""

    if _INGEST_LOCK.acquire(blocking=False):
        _INGEST_LOCK.release()
        return False
    return True


@contextmanager
def _exclusive_ingest():
    if not _INGEST_LOCK.acquire(blocking=False):
        raise RuntimeError("another ingest or re-import is already running")
    try:
        yield
    finally:
        _INGEST_LOCK.release()


@dataclass
class IngestConfig:
    name: str
    source_dir: str
    domain_override: str | None = None
    near_dedup: bool = True
    global_dedup: bool = True
    language_filter: bool = True
    quality_filter: bool = True
    quality_threshold: float = 0.35
    min_chars: int = 40
    max_file_mb: float = 300.0
    tokenizer_keys: list[str] = field(default_factory=lambda: list(SPEC_KEYS))
    selection_profile: str = "dataset"
    include_path_regex: str = ""
    exclude_path_regex: str = ""
    #: ``None`` resolves from ``YYLAB_INGEST_WORKERS`` / settings / CPU count.
    workers: int | None = None
    #: Keep prepared documents for quick re-import with different policies.
    prepared_cache: bool = True

    def analysis_params(self) -> AnalysisParams:
        return AnalysisParams(
            domain_override=self.domain_override or None,
            near_dedup=bool(self.near_dedup),
            min_chars=int(self.min_chars),
        )

    def ingest_config_stats(self, *, workers: int) -> dict[str, Any]:
        return {
            "near_dedup": bool(self.near_dedup),
            "global_dedup": bool(self.global_dedup),
            "language_filter": bool(self.language_filter),
            "quality_filter": bool(self.quality_filter),
            "quality_threshold": float(self.quality_threshold),
            "min_chars": int(self.min_chars),
            "tokenizer_keys": list(self.tokenizer_keys),
            "global_hash_storage": "training_sha256_v3_namespaced",
            "training_normalization": TRAINING_NORMALIZATION,
            "selection_profile": self.selection_profile,
            "include_path_regex": self.include_path_regex,
            "exclude_path_regex": self.exclude_path_regex,
            "max_file_mb": float(self.max_file_mb),
            "domain_override": self.domain_override or None,
            "workers": int(workers),
            "prepared_cache": bool(self.prepared_cache),
        }

    @classmethod
    def from_ingest_config(
        cls,
        *,
        name: str,
        source_dir: str,
        ingest_config: dict[str, Any] | None,
    ) -> IngestConfig:
        """Rebuild the policy settings recorded in ``stats.json`` for re-import."""

        stored = dict(ingest_config or {})
        defaults = cls(name=name, source_dir=source_dir)
        tokenizer_keys = stored.get("tokenizer_keys")
        return cls(
            name=name,
            source_dir=source_dir,
            domain_override=(stored.get("domain_override") or None),
            near_dedup=bool(stored.get("near_dedup", defaults.near_dedup)),
            global_dedup=bool(stored.get("global_dedup", defaults.global_dedup)),
            language_filter=bool(stored.get("language_filter", defaults.language_filter)),
            quality_filter=bool(stored.get("quality_filter", defaults.quality_filter)),
            quality_threshold=float(stored.get("quality_threshold", defaults.quality_threshold)),
            min_chars=int(stored.get("min_chars", defaults.min_chars)),
            max_file_mb=float(stored.get("max_file_mb", defaults.max_file_mb)),
            tokenizer_keys=list(tokenizer_keys) if isinstance(tokenizer_keys, list) else [],
            selection_profile=str(stored.get("selection_profile") or defaults.selection_profile),
            include_path_regex=str(stored.get("include_path_regex") or ""),
            exclude_path_regex=str(stored.get("exclude_path_regex") or ""),
            workers=None,
            prepared_cache=bool(stored.get("prepared_cache", True)),
        )


class _ShardWriter:
    def __init__(self, data_dir: Path, start_index: int = 0) -> None:
        self.data_dir = data_dir
        self.shards: list[str] = []
        self.index = start_index
        self.count = 0
        self.finished = False
        self._fh = None
        self._comp = None
        self._open()

    def _open(self) -> None:
        name = f"shard-{self.index:05d}.jsonl.zst"
        self.shards.append(name)
        self._fh = (self.data_dir / name).open("wb")
        self._comp = zstd.ZstdCompressor(level=3).stream_writer(self._fh)

    def write(self, line: dict) -> None:
        self._comp.write(orjson.dumps(line) + b"\n")
        self.count += 1
        if self.count >= SHARD_DOCS:
            self._close()
            self.index += 1
            self.count = 0
            self._open()

    def _close(self) -> None:
        if self._comp is not None:
            self._comp.close()
            self._comp = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        if self.count == 0 and self.index > 0:
            # A boundary-sized shard opens the next frame eagerly. Close that
            # empty frame before unlinking it (required on Windows).
            self._close()
            (self.data_dir / self.shards.pop()).unlink(missing_ok=True)
        else:
            self._close()


class _LazyShardWriter:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._writer: _ShardWriter | None = None

    def write(self, line: dict) -> None:
        if self._writer is None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._writer = _ShardWriter(self.data_dir)
        self._writer.write(line)

    def finish(self) -> list[str]:
        if self._writer is None:
            return []
        self._writer.finish()
        return self._writer.shards


def _iter_shards(data_dir: Path, shards: list[str]):
    dctx = zstd.ZstdDecompressor()
    for name in shards:
        with (data_dir / name).open("rb") as fh:
            with dctx.stream_reader(fh) as reader:
                # BufferedReader implements line iteration in C and avoids the
                # repeated immutable-byte concatenation/splitting of the old
                # decompression loop. Memory remains bounded by one JSONL row.
                with io.BufferedReader(reader, buffer_size=1 << 20) as buffered:
                    for line in buffered:
                        if line.strip():
                            yield orjson.loads(line)


def _existing_hashes(con: Connection, table: str, hashes: list[str]) -> set[str]:
    if table not in {"doc_hashes", "rejected_hashes"}:
        raise ValueError(f"unsupported hash table: {table}")
    existing: set[str] = set()
    for i in range(0, len(hashes), 500):
        chunk = hashes[i : i + 500]
        ph = ",".join("?" * len(chunk))
        rows = con.execute(f"SELECT hash FROM {table} WHERE hash IN ({ph})", tuple(chunk))
        existing.update(row[0] for row in rows)
    return existing


def _external_document_hashes(
    con: Connection,
    hashes: list[str],
    corpus_id: str,
) -> set[str]:
    existing: set[str] = set()
    for index in range(0, len(hashes), 500):
        chunk = hashes[index : index + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = con.execute(
            f"SELECT hash,corpus_id FROM doc_hashes WHERE hash IN ({placeholders})",
            tuple(chunk),
        )
        existing.update(hash_value for hash_value, owner in rows if owner != corpus_id)
    return existing


def _verified_legacy_rejections(con: Connection, raw_hashes: list[str]) -> set[str]:
    """Recover old rejection decisions only when the review label proves identity.

    ``rejected_hashes`` historically mixed exact and case-folded hashes.  The
    annotation hash was always calculated from the actual reviewed shard text,
    so it is safe evidence for upgrading a legacy rejection to the namespaced
    exact-hash lane.
    """

    verified: set[str] = set()
    for index in range(0, len(raw_hashes), 500):
        chunk = raw_hashes[index : index + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = con.execute(
            f"SELECT hash FROM review_annotations "
            f"WHERE verdict='bad' AND hash IN ({placeholders})",
            tuple(chunk),
        )
        verified.update(training_hash_key(str(row[0])) for row in rows)
    if verified:
        now = db.now_iso()
        con.executemany(
            "INSERT OR IGNORE INTO rejected_hashes(hash,added_at,source) VALUES(?,?,?)",
            ((hash_key, now, "migration:review_annotation") for hash_key in verified),
        )
    return verified


def _global_existing(hashes: list[str]) -> set[str]:
    with db.transaction() as con:
        return _existing_hashes(con, "doc_hashes", hashes)


def _rejected_existing(hashes: list[str]) -> set[str]:
    with db.transaction() as con:
        try:
            return _existing_hashes(con, "rejected_hashes", hashes)
        except Exception:  # noqa: BLE001
            return set()


class _DirectHashSink:
    """Register kept hashes in the global table as they are decided (fresh ingest)."""

    def register(self, con: Connection, rows: list[tuple[str, str]]) -> None:
        con.executemany("INSERT OR IGNORE INTO doc_hashes(hash, corpus_id) VALUES(?,?)", rows)

    def close(self) -> None:
        return None


class _StagedHashSink:
    """Collect kept hashes in a run-local database; committed by ``reimport_corpus``.

    The corpus keeps its previous ownership of every hash until the new output
    has been produced completely, so a failed re-import changes nothing.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("CREATE TABLE IF NOT EXISTS hashes(hash TEXT PRIMARY KEY)")
        self.connection.commit()

    def register(self, con: Connection, rows: list[tuple[str, str]]) -> None:
        _ = con
        self.connection.executemany(
            "INSERT OR IGNORE INTO hashes(hash) VALUES(?)",
            ((hash_value,) for hash_value, _owner in rows),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


@dataclass(slots=True)
class _PendingDocument:
    prepared: PreparedDoc
    line: dict[str, Any]
    asset: AssetObservation
    locator: str
    language: str
    reasons: list[str]
    chunk_key: tuple[str, int]
    ordinal: int


@dataclass(slots=True)
class _AssetPlan:
    index: int
    path: Path
    adapter: Adapter | None
    kind: str
    asset: AssetObservation | None = None
    error: str = ""
    reasons: list[str] = field(default_factory=list)
    file_size: int = 0
    tasks: list[PrepareTask] = field(default_factory=list)
    chunk_count: int = 0
    estimated_rows: int | None = None
    # Per-asset state filled while its chunks are consumed in order.
    rows_seen: int = 0
    sample_ids: set[str] = field(default_factory=set)
    record_base: int = 0
    line_base: int = 0
    docs_total: int = 0
    records_total: int = 0
    lines_total: int = 0
    cache_ok: bool = True
    #: A stale cache entry exists; it is replaced only once extraction starts,
    #: so a run that fails earlier leaves untouched entries intact.
    replace_cache: bool = False


class _TokenTracker:
    """Batch tokenizer counting with per-chunk count arrays for the cache.

    A chunk's arrays are persisted once the chunk has been fully consumed and
    every one of its documents that may still be tokenized (kept documents
    waiting in ``pending`` or in the tokenizer batch) has been resolved.
    """

    def __init__(self, keys: list[str], cache: PreparedCache | None) -> None:
        self.keys = list(keys)
        self.cache = cache
        self.arrays: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self.outstanding: dict[tuple[str, int], int] = {}
        self.finished_chunks: set[tuple[str, int]] = set()

    @property
    def active(self) -> bool:
        return bool(self.keys) and self.cache is not None

    def prepare_chunk(self, chunk_key: tuple[str, int], doc_count: int, *, cached: bool) -> None:
        if not self.active:
            return
        arrays: dict[str, np.ndarray] = {}
        if cached:
            arrays = self.cache.load_tokens(chunk_key[0], chunk_key[1], self.keys)
        for key in self.keys:
            existing = arrays.get(key)
            if existing is None or len(existing) != doc_count:
                arrays[key] = np.full(doc_count, -1, dtype=np.int32)
        self.arrays[chunk_key] = arrays
        self.outstanding[chunk_key] = 0

    def cached_counts(self, chunk_key: tuple[str, int], ordinal: int) -> dict[str, int] | None:
        arrays = self.arrays.get(chunk_key)
        if not arrays:
            return None
        counts: dict[str, int] = {}
        for key in self.keys:
            value = int(arrays[key][ordinal])
            if value < 0:
                return None
            counts[key] = value
        return counts

    def note_pending(self, chunk_key: tuple[str, int]) -> None:
        if chunk_key in self.outstanding:
            self.outstanding[chunk_key] += 1

    def release(self, chunk_key: tuple[str, int]) -> None:
        if chunk_key in self.outstanding:
            self.outstanding[chunk_key] -= 1
            self._maybe_store(chunk_key)

    def record(self, chunk_key: tuple[str, int], ordinal: int, counts: dict[str, int]) -> None:
        arrays = self.arrays.get(chunk_key)
        if arrays is not None:
            for key, value in counts.items():
                if key in arrays:
                    arrays[key][ordinal] = int(value)
        self.release(chunk_key)

    def finish_chunk(self, chunk_key: tuple[str, int]) -> None:
        if chunk_key not in self.arrays:
            return
        self.finished_chunks.add(chunk_key)
        self._maybe_store(chunk_key)

    def _maybe_store(self, chunk_key: tuple[str, int]) -> None:
        if chunk_key not in self.finished_chunks or self.outstanding.get(chunk_key, 0) > 0:
            return
        arrays = self.arrays.pop(chunk_key)
        self.outstanding.pop(chunk_key, None)
        self.finished_chunks.discard(chunk_key)
        if self.cache is not None:
            for key, counts in arrays.items():
                if (counts >= 0).any():
                    self.cache.store_tokens(chunk_key[0], chunk_key[1], key, counts)

    def drop_chunk(self, chunk_key: tuple[str, int]) -> None:
        self.arrays.pop(chunk_key, None)
        self.outstanding.pop(chunk_key, None)
        self.finished_chunks.discard(chunk_key)


class _IngestRun:
    """Ordered coordinator: plans assets, farms preparation out, decides policy."""

    def __init__(
        self,
        cfg: IngestConfig,
        cid: str,
        *,
        out_dir: Path,
        cache_path: Path | None,
        hash_sink: _DirectHashSink | _StagedHashSink,
    ) -> None:
        if cfg.selection_profile not in SELECTION_PROFILES:
            raise ValueError(
                f"selection_profile must be one of: {', '.join(SELECTION_PROFILES)}"
            )
        self.cfg = cfg
        self.cid = cid
        self.out_dir = out_dir
        self.cache_path = cache_path
        self.hash_sink = hash_sink
        self.include_path = (
            re.compile(cfg.include_path_regex, re.IGNORECASE) if cfg.include_path_regex else None
        )
        self.exclude_path = (
            re.compile(cfg.exclude_path_regex, re.IGNORECASE) if cfg.exclude_path_regex else None
        )
        settings = get_settings()
        self.settings = settings
        self.workers = resolve_worker_count(cfg.workers, settings.get("ingest_workers"))
        self.params = cfg.analysis_params()
        self.data_dir = out_dir / "data"
        self.stats: dict[str, Any] = {
            "files_found": 0,
            "files_processed": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "records_failed": 0,
            "assets_policy_skipped": 0,
            "assets_size_skipped": 0,
            "docs_policy_skipped": 0,
            "docs_seen": 0,
            "docs_written": 0,
            "exact_dupes_local": 0,
            "exact_dupes_global": 0,
            "near_dupes": 0,
            "near_duplicate_candidates": 0,
            "split_leakage_dupes": 0,
            "duplicate_sample_ids": 0,
            "filtered": 0,
            "non_english_docs": 0,
            "quarantined_docs": 0,
            "rejected_known_bad": 0,
            "langs": {},
            "quarantine_reasons": {},
            "chars_written": 0,
            "bytes_on_disk": 0,
            "domains": {},
            "splits": {},
            "sources": {},
            "totals": {"tokens": {}},
            "tokenizer_failures": {},
            "shards": [],
            "nonen_shards": [],
            "quarantine_shards": [],
            "errors": [],
            "ingest_config": cfg.ingest_config_stats(workers=self.workers),
        }
        self.local_hash_splits: dict[str, set[str]] = {}
        self.pending: list[_PendingDocument] = []
        self.pending_bytes = 0
        self.writer: _ShardWriter | None = None
        self.nonen_writer = _LazyShardWriter(out_dir / "data_nonen")
        self.quarantine_writer = _LazyShardWriter(out_dir / "data_quarantine")
        self.mh = MinHashDedup() if cfg.near_dedup else None
        self.token_counter: TokenCounter | None = None
        self.token_keys: list[str] = []
        self.tokens: _TokenTracker | None = None
        self.batch_texts: list[str] = []
        self.batch_domains: list[str] = []
        self.batch_splits: list[str] = []
        self.batch_refs: list[tuple[tuple[str, int], int]] = []
        self.catalog: CorpusCatalog | None = None
        self.cache: PreparedCache | None = None
        self.files: list[Path] = []
        self.adapters: list[Adapter] = []
        self.seen_asset_ids: set[str] = set()
        self.progress: ProgressFn = lambda *args: None
        self.started = 0.0
        self.finalized = False

    # -- stats helpers -----------------------------------------------------

    def _count_reason(self, reason: str) -> None:
        reasons = self.stats["quarantine_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1

    def _record_error(self, path: Path, message: str) -> None:
        if len(self.stats["errors"]) < 50:
            self.stats["errors"].append(f"{path.name}: {message[:200]}")

    def _apply_token_counts(self, counts: dict[str, int], dom: str, split: str) -> None:
        stats = self.stats
        for key, n in counts.items():
            stats["totals"]["tokens"][key] = stats["totals"]["tokens"].get(key, 0) + n
            stats["domains"][dom]["tokens"][key] = stats["domains"][dom]["tokens"].get(key, 0) + n
            split_tokens = stats["splits"][split]["tokens"]
            split_tokens[key] = split_tokens.get(key, 0) + n
            split_domain_tokens = stats["splits"][split]["domains"][dom].setdefault("tokens", {})
            split_domain_tokens[key] = split_domain_tokens.get(key, 0) + n

    def tok_flush(self) -> None:
        if self.token_counter is None or not self.batch_texts:
            return
        counts = self.token_counter.count(self.batch_texts)
        per_doc = [dict() for _ in self.batch_texts]
        for key, values in counts.items():
            for index, n in enumerate(values):
                per_doc[index][key] = n
        for index, (dom, split) in enumerate(zip(self.batch_domains, self.batch_splits, strict=True)):
            self._apply_token_counts(per_doc[index], dom, split)
            if self.tokens is not None:
                chunk_key, ordinal = self.batch_refs[index]
                self.tokens.record(chunk_key, ordinal, per_doc[index])
        self.batch_texts.clear()
        self.batch_domains.clear()
        self.batch_splits.clear()
        self.batch_refs.clear()

    def queue_token_stats(self, line: dict, chunk_key: tuple[str, int], ordinal: int) -> None:
        if self.token_counter is None:
            return
        dom = line["domain"]
        split = str(line.get("split") or "unspecified")
        if self.tokens is not None:
            cached = self.tokens.cached_counts(chunk_key, ordinal)
            if cached is not None:
                self._apply_token_counts(cached, dom, split)
                self.tokens.release(chunk_key)
                return
        self.batch_texts.append(line["text"])
        self.batch_domains.append(dom)
        self.batch_splits.append(split)
        self.batch_refs.append((chunk_key, ordinal))
        if len(self.batch_texts) >= TOK_BATCH:
            self.tok_flush()

    # -- pending batch (global dedup / rejection / registration) -------------

    def flush_pending(self) -> None:
        if not self.pending:
            return
        cfg = self.cfg
        stats = self.stats
        catalog = self.catalog
        pending = self.pending
        exact_hashes = [training_hash_key(item.prepared.training_sha256) for item in pending]
        raw_hashes = [item.prepared.training_sha256 for item in pending]
        kept_rows: list[tuple[str, str]] = []
        token_lines: list[tuple[dict, tuple[str, int], int]] = []
        # One connection and transaction handles every membership chunk plus
        # registration. The old helpers opened a new SQLite connection for each
        # 500 hashes and then another for inserts.
        with db.transaction() as con:
            blocked = (
                _external_document_hashes(con, exact_hashes, self.cid)
                if cfg.global_dedup
                else set()
            )
            try:
                rejected = _existing_hashes(con, "rejected_hashes", exact_hashes)
            except Exception:  # noqa: BLE001
                rejected = set()
            rejected.update(_verified_legacy_rejections(con, raw_hashes))
            for item in pending:
                line = item.line
                prepared = item.prepared
                storage_hash = training_hash_key(prepared.training_sha256)
                if storage_hash in rejected:
                    stats["rejected_known_bad"] += 1
                    if self.tokens is not None:
                        self.tokens.release(item.chunk_key)
                    self._catalog_document(
                        item.asset,
                        item.locator,
                        prepared,
                        training_text=line["text"],
                        language=item.language,
                        selection="rejected_known_bad",
                        reasons=[*item.reasons, "known_bad_hash"],
                    )
                    continue
                if storage_hash in blocked:
                    stats["exact_dupes_global"] += 1
                    if self.tokens is not None:
                        self.tokens.release(item.chunk_key)
                    self._catalog_document(
                        item.asset,
                        item.locator,
                        prepared,
                        training_text=line["text"],
                        language=item.language,
                        selection="duplicate_global",
                        reasons=[*item.reasons, "duplicate:global_exact"],
                    )
                    continue
                self.writer.write(line)
                kept_rows.append((storage_hash, self.cid))
                dom = line["domain"]
                text_len = len(line["text"])
                d = stats["domains"].setdefault(dom, {"docs": 0, "chars": 0, "tokens": {}})
                d["docs"] += 1
                d["chars"] += text_len
                split = str(line.get("split") or "unspecified")
                split_stats = stats["splits"].setdefault(
                    split,
                    {"docs": 0, "chars": 0, "tokens": {}, "domains": {}},
                )
                split_stats["docs"] += 1
                split_stats["chars"] += text_len
                split_domain = split_stats["domains"].setdefault(dom, {"docs": 0, "chars": 0})
                split_domain["docs"] += 1
                split_domain["chars"] += text_len
                stats["chars_written"] += text_len
                stats["docs_written"] += 1
                source_stats = stats["sources"].setdefault(
                    prepared.logical_path,
                    {"docs": 0, "chars": 0, "domain": dom, "split": split},
                )
                source_stats["docs"] += 1
                source_stats["chars"] += text_len
                token_lines.append((line, item.chunk_key, item.ordinal))
                self._catalog_document(
                    item.asset,
                    item.locator,
                    prepared,
                    training_text=line["text"],
                    language=item.language,
                    selection="kept",
                    reasons=item.reasons,
                )
            if cfg.global_dedup and kept_rows:
                self.hash_sink.register(con, kept_rows)
        _ = catalog
        # Tokenizers can use their own worker threads and may take seconds for
        # large batches. Never hold yyLab's global SQLite writer lock over them.
        for line, chunk_key, ordinal in token_lines:
            self.queue_token_stats(line, chunk_key, ordinal)
        self.pending = []
        self.pending_bytes = 0
        if self.cache is not None:
            self.cache.commit()

    def _catalog_document(
        self,
        asset: AssetObservation,
        locator: str,
        prepared: PreparedDoc,
        *,
        training_text: str | None,
        language: str,
        selection: str,
        reasons: list[str],
    ) -> None:
        if self.catalog is None:
            return
        self.catalog.record_document(
            asset=asset,
            locator=locator,
            training_text=training_text,
            domain=prepared.domain,
            language=language,
            selection=selection,
            reasons=reasons,
            roles=prepared.roles,
            signals=prepared.signals,
            metadata=prepared.metadata,
            source_sha256=prepared.source_sha256,
            source_utf8_bytes=prepared.source_utf8_bytes,
            training_sha256=prepared.training_sha256 if training_text is not None else "",
            signals_json=prepared.signals_json or None,
            roles_json=prepared.roles_json or None,
            metadata_json=prepared.metadata_json or None,
        )

    # -- planning ----------------------------------------------------------

    def _path_reasons(self, logical_path: str) -> list[str]:
        reasons = asset_policy_reasons(self.cfg.selection_profile, logical_path)
        if self.include_path is not None and self.include_path.search(logical_path) is None:
            reasons.append("cdas:path_not_included")
        if self.exclude_path is not None and self.exclude_path.search(logical_path) is not None:
            reasons.append("cdas:path_excluded")
        return reasons

    def _plan_asset(self, index: int, path: Path) -> _AssetPlan:
        cfg = self.cfg
        adapter = dispatch(path, self.adapters)
        try:
            file_size = path.stat().st_size
        except OSError as error:
            return _AssetPlan(index=index, path=path, adapter=adapter, kind="failed", error=str(error))
        size_limit = float(cfg.max_file_mb) * 1024 * 1024
        oversized = size_limit > 0 and file_size > size_limit
        adapter_name = adapter.name if adapter is not None else ""
        try:
            asset = self.catalog.describe_asset(path, adapter_name=adapter_name, hash_contents=False)
        except (OSError, ValueError) as error:
            return _AssetPlan(index=index, path=path, adapter=adapter, kind="failed", error=str(error))
        plan = _AssetPlan(index=index, path=path, adapter=adapter, kind="extract", asset=asset, file_size=file_size)
        if oversized:
            plan.kind = "oversized"
            plan.reasons = [f"asset_too_large:{cfg.max_file_mb:g}MiB"]
            return plan
        path_reasons = self._path_reasons(asset.logical_path)
        hash_task = PrepareTask(
            path=str(path),
            adapter_name="",
            chunk=None,
            asset_logical_path=asset.logical_path,
            asset_format_name=asset.format_name,
            params=self.params,
            hash_file=True,
        )
        if adapter is None:
            plan.kind = "no_adapter"
            plan.tasks = [hash_task]
            return plan
        if path_reasons:
            plan.kind = "policy_excluded"
            plan.reasons = path_reasons
            plan.tasks = [hash_task]
            return plan
        try:
            plan.estimated_rows = adapter.estimated_records(path)
        except (OSError, ValueError):
            plan.estimated_rows = None
        if self.cache is not None:
            cached = self.cache.lookup(asset.asset_id)
            if cached is not None and cached.byte_size == file_size:
                try:
                    asset = replace(asset, raw_sha256=digest_file(path))
                except OSError as error:
                    return _AssetPlan(index=index, path=path, adapter=adapter, kind="failed", error=str(error))
                plan.asset = asset
                if self.cache.matches(cached, asset, adapter):
                    plan.kind = "cached"
                    plan.chunk_count = cached.chunk_count
                    return plan
                plan.replace_cache = True
            elif cached is not None:
                plan.replace_cache = True
        chunks: list[ChunkSpec] | None = None
        if self.workers > 1 or CHUNK_WHEN_SERIAL:
            try:
                chunks = adapter.plan_chunks(path)
            except (OSError, ValueError):
                chunks = None
        build_payload = self.cache is not None
        if chunks and len(chunks) > 1:
            if not asset.raw_sha256:
                try:
                    asset = replace(asset, raw_sha256=digest_file(path))
                except OSError as error:
                    return _AssetPlan(index=index, path=path, adapter=adapter, kind="failed", error=str(error))
                plan.asset = asset
            plan.tasks = [
                PrepareTask(
                    path=str(path),
                    adapter_name=adapter.name,
                    chunk=chunk,
                    asset_logical_path=asset.logical_path,
                    asset_format_name=asset.format_name,
                    params=self.params,
                    hash_file=False,
                    build_payload=build_payload,
                )
                for chunk in chunks
            ]
        else:
            plan.tasks = [
                PrepareTask(
                    path=str(path),
                    adapter_name=adapter.name,
                    chunk=None,
                    asset_logical_path=asset.logical_path,
                    asset_format_name=asset.format_name,
                    params=self.params,
                    hash_file=not asset.raw_sha256,
                    build_payload=build_payload,
                )
            ]
        plan.chunk_count = len(plan.tasks)
        return plan

    def _cached_chunk(self, plan: _AssetPlan, chunk_index: int, executor) -> Any:
        asset_id = plan.asset.asset_id

        def load() -> ChunkResult:
            result = self.cache.load_chunk(asset_id, chunk_index)
            if needs_augmentation(result.docs, self.params):
                result.docs = executor.submit_augment(result.docs, self.params).result()
                self.cache.replace_chunk_docs(asset_id, chunk_index, result.docs)
            return result

        if self.workers > 1:
            # Decode now so an augment task (if any) overlaps with other work.
            # Materialized results report ``done()`` and their size, so the
            # ordered window's byte budget bounds how many are decoded ahead.
            result = self.cache.load_chunk(asset_id, chunk_index)
            if needs_augmentation(result.docs, self.params):
                future = executor.submit_augment(result.docs, self.params)

                def finish() -> ChunkResult:
                    result.docs = future.result()
                    result.approx_bytes = sum(doc.approx_bytes() for doc in result.docs)
                    self.cache.replace_chunk_docs(asset_id, chunk_index, result.docs)
                    return result

                return _Deferred(finish, upstream=future)
            return _Deferred.completed(result)
        return _Deferred(load)

    def _submissions(self, executor):
        for index, path in enumerate(self.files):
            plan = self._plan_asset(index, path)
            if plan.asset is not None:
                self.seen_asset_ids.add(plan.asset.asset_id)
            if plan.kind == "cached":
                for chunk_index in range(plan.chunk_count):
                    yield (plan, chunk_index), self._cached_chunk(plan, chunk_index, executor)
            elif plan.tasks:
                for chunk_index, task in enumerate(plan.tasks):
                    yield (plan, chunk_index), executor.submit_prepare(task)
            else:
                yield (plan, 0), _Deferred(lambda: None)

    # -- consumption -------------------------------------------------------

    def _report_progress(self, plan: _AssetPlan, *, final: bool) -> None:
        total = max(len(self.files), 1)
        stage_rate = ""
        elapsed = time.perf_counter() - self.started
        if elapsed > 0 and self.stats["docs_seen"]:
            stage_rate = f" · {self.stats['docs_seen'] / elapsed:,.0f} docs/s"
        workers = f" · {self.workers} worker{'s' if self.workers != 1 else ''}"
        if final:
            self.progress("extract", (plan.index + 1) / total, f"{plan.path.name}{workers}{stage_rate}")
            return
        estimated = plan.estimated_rows
        within = min(plan.rows_seen / estimated, 0.999) if estimated else 0.0
        self.progress(
            "extract",
            (plan.index + within) / total,
            f"{plan.path.name} · {plan.rows_seen:,}"
            + (f"/{estimated:,}" if estimated else "")
            + f" rows · {self.stats['docs_written']:,} kept{workers}{stage_rate}",
        )

    def _consume(self, plan: _AssetPlan, chunk_index: int, result: ChunkResult | None) -> None:
        stats = self.stats
        catalog = self.catalog
        kind = plan.kind
        if kind == "failed":
            stats["files_failed"] += 1
            self._record_error(plan.path, plan.error)
            return
        asset = plan.asset
        if kind in {"no_adapter", "policy_excluded"}:
            assert result is not None
            if result.hash_failed:
                stats["files_failed"] += 1
                self._record_error(plan.path, result.error)
                return
            asset = replace(asset, raw_sha256=result.raw_sha256)
            catalog.record_asset(asset)
            stats["files_skipped"] += 1
            if kind == "no_adapter":
                catalog.finish_asset(asset, status="inventory_only", reasons=["adapter_unavailable"])
                return
            stats["assets_policy_skipped"] += 1
            for reason in sorted(set(plan.reasons)):
                self._count_reason(reason)
            catalog.finish_asset(asset, status="excluded_by_policy", reasons=plan.reasons)
            return
        if kind == "oversized":
            catalog.record_asset(asset)
            stats["files_skipped"] += 1
            stats["assets_size_skipped"] += 1
            reason = plan.reasons[0]
            self._count_reason(reason)
            catalog.record_issue(
                asset.asset_id,
                "$",
                "asset_too_large",
                f"{plan.file_size} bytes exceeds {self.cfg.max_file_mb:g} MiB",
            )
            catalog.finish_asset(asset, status="excluded_by_limit", reasons=plan.reasons)
            return

        assert result is not None
        cached = kind == "cached"
        if chunk_index == 0:
            if result.hash_failed:
                stats["files_failed"] += 1
                self._record_error(plan.path, result.error)
                plan.kind = "failed"
                return
            if result.raw_sha256 and not asset.raw_sha256:
                asset = replace(asset, raw_sha256=result.raw_sha256)
                plan.asset = asset
            catalog.record_asset(asset)
            if plan.replace_cache and self.cache is not None:
                self.cache.invalidate_asset(asset.asset_id)
                plan.replace_cache = False
        elif plan.kind == "failed":
            return
        chunk_key = (asset.asset_id, chunk_index)
        multi = plan.chunk_count > 1
        if self.tokens is not None:
            self.tokens.prepare_chunk(chunk_key, len(result.docs), cached=cached)
        for ordinal, prepared in enumerate(result.docs):
            locator = prepared.locator
            if multi and locator:
                locator = rebase_locator(locator, plan.record_base, plan.line_base)
            self._decide(plan, asset, prepared, locator, chunk_key, ordinal)
        for locator, reason_code, detail in result.issues:
            stats["records_failed"] += 1
            if multi:
                locator = rebase_locator(locator, plan.record_base, plan.line_base)
            catalog.record_issue(asset.asset_id, locator, reason_code, detail)
        plan.record_base += result.records_seen
        plan.line_base += result.lines_seen
        plan.docs_total += len(result.docs)
        plan.records_total += result.records_seen
        plan.lines_total += result.lines_seen
        if self.tokens is not None:
            self.tokens.finish_chunk(chunk_key)
        # A record whose preparation raised never became a document, but the
        # adapter did yield it; keep docs_seen/progress equal to the serial run.
        dropped = max(int(result.docs_yielded) - len(result.docs), 0)
        if dropped:
            stats["docs_seen"] += dropped
            plan.rows_seen += dropped
        if result.error:
            detail = f"{result.error_type}: {result.error}" if result.error_type else result.error
            if self.cache is not None and not cached:
                self.cache.invalidate_asset(asset.asset_id)
            stats["files_failed"] += 1
            catalog.record_issue(asset.asset_id, "$", "adapter_exception", detail)
            catalog.finish_asset(asset, status="extraction_failed", reasons=["adapter_exception"])
            self._record_error(plan.path, result.error)
            raise RuntimeError(f"asset extraction failed for {asset.logical_path}: {result.error}")
        if self.cache is not None and not cached and result.payload is not None and plan.cache_ok:
            self.cache.store_chunk(asset.asset_id, chunk_index, result)
        elif self.cache is not None and not cached and result.payload is None:
            plan.cache_ok = False
        if chunk_index + 1 >= plan.chunk_count:
            catalog.finish_asset(asset, status="normalized")
            stats["files_processed"] += 1
            if self.cache is not None and not cached:
                if plan.cache_ok:
                    self.cache.store_asset(
                        asset,
                        plan.adapter,
                        chunk_count=plan.chunk_count,
                        doc_count=plan.docs_total,
                        records_seen=plan.records_total,
                        lines_seen=plan.lines_total,
                    )
                else:
                    self.cache.invalidate_asset(asset.asset_id)
            elif cached:
                self.cache.stats["assets_reused"] += 1
            self._report_progress(plan, final=True)

    def _decide(
        self,
        plan: _AssetPlan,
        asset: AssetObservation,
        prepared: PreparedDoc,
        locator: str,
        chunk_key: tuple[str, int],
        ordinal: int,
    ) -> None:
        cfg = self.cfg
        stats = self.stats
        catalog = self.catalog
        stats["docs_seen"] += 1
        plan.rows_seen += 1
        if plan.rows_seen % PROGRESS_ROWS == 0:
            self._report_progress(plan, final=False)
        if not locator:
            locator = f"record={plan.rows_seen}"
        sample_id = prepared.sample_id
        if sample_id:
            if sample_id in plan.sample_ids:
                stats["duplicate_sample_ids"] += 1
                catalog.record_issue(asset.asset_id, locator, "duplicate_sample_id", sample_id)
            else:
                plan.sample_ids.add(sample_id)
        text = prepared.text
        if not prepared.hygiene_ok(cfg.min_chars):
            stats["filtered"] += 1
            self._catalog_document(
                asset,
                locator,
                prepared,
                training_text=text,
                language="",
                selection="filtered",
                reasons=["minimum_document_hygiene"],
            )
            return
        line = prepared.shard_line()
        domain = prepared.domain
        split = prepared.split
        lang = ""
        if domain not in LANG_SKIP_DOMAINS:
            lang = prepared.lang if prepared.lang is not None else detect_language(text)
            line["lang"] = lang
            stats["langs"][lang] = stats["langs"].get(lang, 0) + 1
            if cfg.language_filter and lang not in {"en", "unknown"}:
                stats["non_english_docs"] += 1
                self.nonen_writer.write(line)
                self._catalog_document(
                    asset,
                    locator,
                    prepared,
                    training_text=text,
                    language=lang,
                    selection="non_english",
                    reasons=[f"language:{lang}"],
                )
                return

        quarantine_reasons = profile_quarantine_reasons(
            cfg.selection_profile,
            prepared.signals,
            prepared.roles,
        )
        if prepared.content_hash_mismatch:
            quarantine_reasons.append("cdas:content_hash_mismatch")
        record_path_reasons = (
            asset_policy_reasons(cfg.selection_profile, prepared.logical_path)
            if prepared.logical_path != asset.logical_path
            else []
        )
        if record_path_reasons:
            stats["docs_policy_skipped"] += 1
            quarantine_reasons.extend(record_path_reasons)
        if domain not in QUALITY_SKIP_DOMAINS:
            if prepared.quality_computed:
                score, quality_reasons = prepared.quality, prepared.quality_reasons
            else:
                report = score_quality(text)
                score, quality_reasons = report.score, list(report.reasons)
            line["quality"] = round(float(score), 3)
            if cfg.quality_filter and score < cfg.quality_threshold:
                quarantine_reasons.extend(quality_reasons)
        quarantine_reasons = sorted(set(quarantine_reasons))
        if quarantine_reasons:
            stats["quarantined_docs"] += 1
            for reason in quarantine_reasons:
                self._count_reason(reason)
            line["quarantine_reasons"] = quarantine_reasons
            self.quarantine_writer.write(line)
            self._catalog_document(
                asset,
                locator,
                prepared,
                training_text=text,
                language=lang,
                selection="quarantined",
                reasons=quarantine_reasons,
            )
            return

        h = prepared.training_sha256
        previous_splits = self.local_hash_splits.get(h, set())
        candidate_reasons: list[str] = []
        if split in previous_splits:
            stats["exact_dupes_local"] += 1
            self._catalog_document(
                asset,
                locator,
                prepared,
                training_text=text,
                language=lang,
                selection="duplicate_local",
                reasons=["duplicate:local_exact"],
            )
            return
        if previous_splits:
            stats["split_leakage_dupes"] += 1
            previous = ",".join(sorted(previous_splits))
            candidate_reasons.append(f"split_leakage:{previous}:{split}")
        if self.mh is not None:
            bands = prepared.minhash_bands if prepared.minhash_computed else self.mh.fingerprints(text)
            if self.mh.is_duplicate_bands(bands):
                stats["near_duplicate_candidates"] += 1
                candidate_reasons.append("candidate:minhash_lsh")
        self.local_hash_splits.setdefault(h, set()).add(split)
        if self.tokens is not None:
            self.tokens.note_pending(chunk_key)
        self.pending.append(
            _PendingDocument(
                prepared=prepared,
                line=line,
                asset=asset,
                locator=locator,
                language=lang,
                reasons=candidate_reasons,
                chunk_key=chunk_key,
                ordinal=ordinal,
            )
        )
        self.pending_bytes += int(prepared.signals.get("utf8_bytes") or 0) + len(text) * 2 + 2048
        if len(self.pending) >= GLOBAL_BATCH or self.pending_bytes >= GLOBAL_BATCH_BYTES:
            self.flush_pending()

    # -- lifecycle ---------------------------------------------------------

    def _finalize(self, error: str = "") -> None:
        if self.finalized:
            return
        self.finalized = True
        stats = self.stats
        if self.writer is not None:
            self.writer.finish()
            stats["shards"] = self.writer.shards
        stats["nonen_shards"] = self.nonen_writer.finish()
        stats["quarantine_shards"] = self.quarantine_writer.finish()
        stats["bytes_on_disk"] = sum(
            (self.data_dir / s).stat().st_size for s in stats["shards"] if (self.data_dir / s).exists()
        )
        elapsed = time.perf_counter() - self.started if self.started else 0.0
        stats["timing"] = {
            "seconds": round(elapsed, 3),
            "docs_per_second": round(stats["docs_seen"] / elapsed, 1) if elapsed > 0 else 0.0,
            "workers": self.workers,
        }
        if self.cache is not None and not self.cache.closed:
            try:
                stats["prepared_cache"] = self.cache.summary()
            finally:
                self.cache.close()
        elif self.cache is None:
            stats["prepared_cache"] = {"enabled": False}
        if self.catalog is not None and not self.catalog.closed:
            stats["cdas"] = self.catalog.summary()
            write_json(
                self.out_dir / "ingest_manifest.json",
                {
                    "schema_version": 1,
                    "kind": "yylab_cdas_compatible_corpus",
                    "corpus_id": self.cid,
                    "name": self.cfg.name,
                    "source_root": str(Path(self.cfg.source_dir).resolve()),
                    "ingest_config": stats["ingest_config"],
                    "catalog": stats["cdas"],
                    "shards": list(stats["shards"]),
                    "totals": {
                        "documents": stats["docs_written"],
                        "characters": stats["chars_written"],
                        "tokens": stats["totals"]["tokens"],
                    },
                    "errors": list(stats["errors"]),
                },
            )
            self.catalog.close()
        self.hash_sink.close()
        if error:
            stats["error"] = error
        write_json(self.out_dir / "stats.json", stats)

    def execute(self, progress: ProgressFn) -> dict[str, Any]:
        self.progress = progress
        self.started = time.perf_counter()
        cfg = self.cfg
        stats = self.stats
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.writer = _ShardWriter(self.data_dir)
        executor = None
        try:
            self.adapters = discover_adapters()
            root = Path(cfg.source_dir)
            if not root.is_dir():
                raise FileNotFoundError(f"source folder not found: {cfg.source_dir}")
            self.catalog = CorpusCatalog(
                self.out_dir / "inventory.sqlite3",
                source_root=root,
                source_name=cfg.name,
            )
            if cfg.prepared_cache and self.cache_path is not None:
                self.cache = PreparedCache(self.cache_path, analysis_key=self.params.analysis_key())
            # Discovery inventories every visible corpus asset. Size limits are an
            # explicit catalog decision below, never a silent pre-scan omission.
            self.files = list(walk_files(root))
            stats["files_found"] = len(self.files)
            progress("scan", 0.0, f"{len(self.files)} files found")

            if cfg.tokenizer_keys:
                enabled = self.settings.get("tokenizers_enabled", {})
                keys = [key for key in cfg.tokenizer_keys if enabled.get(key, True)]
                candidate = TokenCounter(keys, hf_token=self.settings.get("hf_token") or None)
                available = candidate.available_specs()
                if available:
                    self.token_counter = candidate
                    self.token_keys = [
                        spec.key for spec in available if isinstance(getattr(spec, "key", None), str)
                    ]
                    self.tokens = _TokenTracker(self.token_keys, self.cache)
                stats["tokenizer_failures"] = dict(candidate.failures)

            executor = make_executor(self.workers, self.adapters)
            window = WINDOW_PER_WORKER * self.workers + 2
            for (plan, chunk_index), result in ordered_results(self._submissions(executor), window):
                self._consume(plan, chunk_index, result)
            self.flush_pending()
            self.writer.finish()

            if self.token_counter is not None:
                self.tok_flush()
                stats["tokenizer_failures"] = dict(self.token_counter.failures)
                progress("tokenize", 1.0, "done")
            if self.cache is not None:
                # The walk completed, so anything not seen has left the source.
                self.cache.retain_only(self.seen_asset_ids)
            self._finalize()
            progress("done", 1.0, "complete")
        except BaseException as e:
            if executor is not None:
                executor.close(abort=True)
                executor = None
            self._finalize(error=str(e)[:500])
            raise
        finally:
            if executor is not None:
                executor.close()
        return stats


def run_ingest(progress: ProgressFn, cfg: IngestConfig) -> str:
    if cfg.selection_profile not in SELECTION_PROFILES:
        raise ValueError(
            f"selection_profile must be one of: {', '.join(SELECTION_PROFILES)}"
        )
    with _exclusive_ingest():
        cid = db.new_id("cor")
        cdir = corpus_dir(cid)
        run = _IngestRun(
            cfg,
            cid,
            out_dir=cdir,
            cache_path=cdir / CACHE_FILE,
            hash_sink=_DirectHashSink(),
        )
        (cdir / "data").mkdir(parents=True, exist_ok=True)
        now = db.now_iso()
        db.execute(
            "INSERT INTO corpora(id, name, created_at, updated_at, status, source_path) VALUES(?,?,?,?,?,?)",
            (cid, cfg.name, now, now, "processing", cfg.source_dir),
        )
        try:
            stats = run.execute(progress)
        except BaseException as e:
            db.execute(
                "UPDATE corpora SET status=?, stats_json=?, error=?, updated_at=? WHERE id=?",
                ("failed", orjson.dumps(run.stats).decode(), str(e)[:500], db.now_iso(), cid),
            )
            raise
        db.execute(
            "UPDATE corpora SET status=?, stats_json=?, error=?, updated_at=? WHERE id=?",
            ("ready", orjson.dumps(stats).decode(), "", db.now_iso(), cid),
        )
    return cid


def _commit_reimport_hashes(cid: str, generation_dir: Path, global_dedup: bool) -> None:
    """Make the corpus own exactly the hashes staged by a published generation.

    The staged file is removed only after the registry commit, so a crash in
    between leaves it in place for :func:`reconcile_pending_reimport`.
    """

    staged = generation_dir / "hashes.sqlite3"
    if global_dedup:
        if staged.exists():
            db.replace_corpus_hashes(cid, staged)
    else:
        db.execute("DELETE FROM doc_hashes WHERE corpus_id=?", (cid,))
    staged.unlink(missing_ok=True)


def reconcile_pending_reimport(cid: str) -> bool:
    """Finish a re-import that was published but never committed its hashes.

    Returns ``True`` when a pending commit was found and applied.  The live
    generation's staged ``hashes.sqlite3`` is the signal: it survives a crash
    or a failed registry write between publish and commit.
    """

    cdir = corpus_dir(cid)
    layout = read_corpus_layout(cdir)
    if not layout["generation"]:
        return False
    generation_dir = cdir / layout["generation"]
    staged = generation_dir / "hashes.sqlite3"
    stats_path = generation_dir / "stats.json"
    if not staged.exists() or not stats_path.exists():
        return False
    stats = orjson.loads(stats_path.read_bytes())
    global_dedup = bool((stats.get("ingest_config") or {}).get("global_dedup", True))
    _commit_reimport_hashes(cid, generation_dir, global_dedup)
    stats.pop("pending_hash_commit", None)
    stats.pop("reimport", None)
    write_json(stats_path, stats)
    write_json(cdir / "stats.json", stats)
    db.execute(
        "UPDATE corpora SET status=?, stats_json=?, error=?, updated_at=? WHERE id=?",
        ("ready", orjson.dumps(stats).decode(), "", db.now_iso(), cid),
    )
    return True


def startup_housekeeping() -> dict[str, Any]:
    """Repair state left by an app that died mid-run; call before serving the UI.

    Ingest jobs and the ingest lock are in-process, so at start-up no run can
    be live: corpora still marked ``processing`` failed with the app, and a
    published-but-uncommitted re-import is completed here.
    """

    report: dict[str, Any] = {"reconciled": [], "marked_failed": 0, "reclaimed": {}}
    for row in db.q("SELECT id FROM corpora WHERE COALESCE(kind,'ingested')='ingested'"):
        try:
            if reconcile_pending_reimport(str(row["id"])):
                report["reconciled"].append(str(row["id"]))
        except Exception:  # noqa: BLE001 - housekeeping must not block startup
            continue
    report["marked_failed"] = db.execute(
        "UPDATE corpora SET status='failed', error=?, updated_at=? WHERE status='processing'",
        ("ingest interrupted by app shutdown", db.now_iso()),
    )
    try:
        from ..artifacts import reclaim_all_retired_generations

        report["reclaimed"] = reclaim_all_retired_generations()
    except Exception:  # noqa: BLE001
        pass
    return report


def reimport_corpus(progress: ProgressFn, cid: str, cfg: IngestConfig) -> str:
    """Re-run ingest policy for an existing corpus in place.

    Unchanged assets come from the corpus's prepared-document cache; new,
    changed, or newly supported assets are extracted fresh.  The new output is
    written as a complete new generation beside the live one and published by
    an atomic pointer update, so the corpus stays ``ready`` (and usable) during
    the run, keeps its id (reviews, verdicts, and training references survive),
    and is untouched if the run fails.
    """

    if (corpus_dir(cid) / "gccf-source.json").exists():
        raise ValueError("Import a new GCCF receipt to update a corpus that reads original sources")
    if cfg.selection_profile not in SELECTION_PROFILES:
        raise ValueError(
            f"selection_profile must be one of: {', '.join(SELECTION_PROFILES)}"
        )
    with _exclusive_ingest():
        row = db.q1("SELECT * FROM corpora WHERE id=?", (cid,))
        if row is None:
            raise ValueError(f"corpus {cid} does not exist")
        if (row.get("kind") or "ingested") != "ingested":
            raise ValueError("only ingested corpora can be re-imported")
        if row["status"] == "processing":
            raise ValueError("corpus is already being processed")
        source_dir = str(row.get("source_path") or "")
        if not source_dir:
            raise ValueError("corpus has no recorded source folder")
        cfg = replace(cfg, name=str(row["name"]), source_dir=source_dir)
        cdir = corpus_dir(cid)
        if reconcile_pending_reimport(cid):
            row = db.q1("SELECT * FROM corpora WHERE id=?", (cid,))
        previous_stats = orjson.loads(row.get("stats_json") or "{}") or {}
        # A leftover marker means an earlier run died with the app; the lock
        # proves nothing is running now, so it is safe to start over.
        previous_stats.pop("reimport", None)
        previous_stats.pop("pending_hash_commit", None)
        started = db.now_iso()
        db.execute(
            "UPDATE corpora SET stats_json=?, error=?, updated_at=? WHERE id=?",
            (
                orjson.dumps({**previous_stats, "reimport": {"started_at": started}}).decode(),
                "",
                started,
                cid,
            ),
        )
        reclaim_retired_generations(cid)
        generation = next_corpus_generation(cid)
        out_dir = cdir / generation
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        run = _IngestRun(
            cfg,
            cid,
            out_dir=out_dir,
            cache_path=cdir / CACHE_FILE,
            hash_sink=_StagedHashSink(out_dir / "hashes.sqlite3"),
        )
        published = False
        try:
            stats = run.execute(progress)
            history = list(previous_stats.get("reimport_history") or [])
            history.append(
                {
                    "at": started,
                    "generation": read_corpus_layout(cdir)["generation"] or "root",
                    "docs_written": previous_stats.get("docs_written"),
                    "ingest_config": previous_stats.get("ingest_config"),
                }
            )
            stats["reimport_history"] = history[-20:]
            stats["generation"] = generation
            write_json(out_dir / "stats.json", stats)
            # Publish (the step that can fail for environmental reasons) before
            # the irreversible registry swap; the staged hashes stay on disk
            # until the swap has committed.
            publish_corpus_generation(cid, generation, stats)
            published = True
            _commit_reimport_hashes(cid, out_dir, cfg.global_dedup)
            db.execute(
                "UPDATE corpora SET status=?, stats_json=?, error=?, updated_at=? WHERE id=?",
                ("ready", orjson.dumps(stats).decode(), "", db.now_iso(), cid),
            )
        except BaseException as e:
            if not published:
                shutil.rmtree(out_dir, ignore_errors=True)
                db.execute(
                    "UPDATE corpora SET stats_json=?, error=?, updated_at=? WHERE id=?",
                    (
                        orjson.dumps(previous_stats).decode(),
                        f"re-import failed: {str(e)[:480]}",
                        db.now_iso(),
                        cid,
                    ),
                )
            else:
                # The new generation is live; only the registry commit is
                # outstanding.  Leave hashes.sqlite3 in place so the next
                # re-import or app start completes it.
                pending = {**stats, "pending_hash_commit": generation}
                write_json(out_dir / "stats.json", pending)
                db.execute(
                    "UPDATE corpora SET status=?, stats_json=?, error=?, updated_at=? WHERE id=?",
                    (
                        "ready",
                        orjson.dumps(pending).decode(),
                        f"re-import published but the hash commit failed: {str(e)[:400]}",
                        db.now_iso(),
                        cid,
                    ),
                )
            raise
        if os.name == "nt":
            # The rename probe only proves "not in use" on Windows; elsewhere
            # retired outputs wait for app start or the next re-import.
            reclaim_retired_generations(cid)
    return cid
