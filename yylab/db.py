from __future__ import annotations

import datetime as dt
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from .config import PATHS

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS corpora(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  source_path TEXT DEFAULT '',
  notes TEXT DEFAULT '',
  stats_json TEXT DEFAULT '{}',
  error TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS doc_hashes(
  hash TEXT PRIMARY KEY,
  corpus_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_hashes_corpus ON doc_hashes(corpus_id);
CREATE TABLE IF NOT EXISTS checkpoints(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'base',
  source TEXT NOT NULL,
  ref TEXT NOT NULL,
  arch TEXT DEFAULT '',
  params_b REAL DEFAULT 0,
  hidden_size INTEGER DEFAULT 0,
  num_layers INTEGER DEFAULT 0,
  quant_locked TEXT DEFAULT '',
  path TEXT DEFAULT '',
  meta_json TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS adapters(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  checkpoint_id TEXT NOT NULL REFERENCES checkpoints(id),
  arch TEXT NOT NULL DEFAULT 'mole',
  quant TEXT NOT NULL,
  num_experts INTEGER DEFAULT 1,
  top_k INTEGER DEFAULT 1,
  rank INTEGER DEFAULT 16,
  target_modules_json TEXT DEFAULT '[]',
  domains_json TEXT DEFAULT '[]',
  corpora_json TEXT DEFAULT '[]',
  path TEXT DEFAULT '',
  status TEXT DEFAULT 'untrained',
  meta_json TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS routers(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  adapter_id TEXT NOT NULL REFERENCES adapters(id),
  top_k INTEGER NOT NULL DEFAULT 2,
  path TEXT DEFAULT '',
  status TEXT DEFAULT 'untrained',
  meta_json TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS tokenizers(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  kind TEXT NOT NULL DEFAULT 'byte_bpe',
  vocab_size INTEGER DEFAULT 0,
  path TEXT DEFAULT '',
  corpora_json TEXT DEFAULT '[]',
  splits_json TEXT DEFAULT '[]',
  config_json TEXT DEFAULT '{}',
  stats_json TEXT DEFAULT '{}',
  error TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS runs(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  config_json TEXT DEFAULT '{}',
  metrics_json TEXT DEFAULT '{}',
  log_path TEXT DEFAULT '',
  error TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS reviews(
  id TEXT PRIMARY KEY,
  corpus_id TEXT NOT NULL REFERENCES corpora(id),
  created_at TEXT NOT NULL,
  updated_at TEXT DEFAULT '',
  status TEXT NOT NULL DEFAULT 'completed',
  docs_reviewed INTEGER DEFAULT 0,
  docs_good INTEGER DEFAULT 0,
  docs_bad INTEGER DEFAULT 0,
  verdict TEXT DEFAULT '',
  approved_corpus_id TEXT DEFAULT '',
  rejected_corpus_id TEXT DEFAULT '',
  annotation_path TEXT DEFAULT '',
  sampling_strategy TEXT DEFAULT 'uniform',
  model_version TEXT DEFAULT '',
  notes TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS rejected_hashes(
  hash TEXT PRIMARY KEY,
  added_at TEXT NOT NULL,
  source TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS review_annotations(
  hash TEXT PRIMARY KEY,
  verdict TEXT NOT NULL,
  corpus_id TEXT NOT NULL,
  review_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  text TEXT NOT NULL DEFAULT '',
  domain TEXT DEFAULT '',
  lang TEXT DEFAULT '',
  quality REAL,
  source TEXT DEFAULT '',
  bucket TEXT DEFAULT '',
  sampling_strategy TEXT DEFAULT 'uniform',
  sampling_component TEXT DEFAULT '',
  sampling_propensity REAL,
  model_version TEXT DEFAULT '',
  rank_score REAL,
  uncertainty REAL,
  model_scores_json TEXT DEFAULT '{}',
  metadata_json TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_review_annotations_corpus ON review_annotations(corpus_id);
CREATE TABLE IF NOT EXISTS review_swipes(
  id TEXT PRIMARY KEY,
  review_id TEXT NOT NULL,
  hash TEXT NOT NULL,
  corpus_id TEXT NOT NULL,
  verdict TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  text TEXT NOT NULL,
  domain TEXT DEFAULT '',
  lang TEXT DEFAULT '',
  quality REAL,
  source TEXT DEFAULT '',
  bucket TEXT DEFAULT '',
  sampling_strategy TEXT DEFAULT 'uniform',
  sampling_component TEXT DEFAULT '',
  sampling_propensity REAL,
  model_version TEXT DEFAULT '',
  rank_score REAL,
  uncertainty REAL,
  model_scores_json TEXT DEFAULT '{}',
  metadata_json TEXT DEFAULT '{}',
  is_new_annotation INTEGER NOT NULL DEFAULT 0,
  UNIQUE(review_id, hash)
);
CREATE INDEX IF NOT EXISTS idx_review_swipes_review ON review_swipes(review_id);
CREATE INDEX IF NOT EXISTS idx_review_swipes_hash ON review_swipes(hash);
"""

_REVIEW_COLUMNS = {
    "updated_at": "TEXT DEFAULT ''",
    "status": "TEXT NOT NULL DEFAULT 'completed'",
    "rejected_corpus_id": "TEXT DEFAULT ''",
    "annotation_path": "TEXT DEFAULT ''",
    "sampling_strategy": "TEXT DEFAULT 'uniform'",
    "model_version": "TEXT DEFAULT ''",
}

_ANNOTATION_COLUMNS = {
    "text": "TEXT NOT NULL DEFAULT ''",
    "domain": "TEXT DEFAULT ''",
    "lang": "TEXT DEFAULT ''",
    "quality": "REAL",
    "source": "TEXT DEFAULT ''",
    "bucket": "TEXT DEFAULT ''",
    "sampling_strategy": "TEXT DEFAULT 'uniform'",
    "sampling_component": "TEXT DEFAULT ''",
    "sampling_propensity": "REAL",
    "model_version": "TEXT DEFAULT ''",
    "rank_score": "REAL",
    "uncertainty": "REAL",
    "model_scores_json": "TEXT DEFAULT '{}'",
    "metadata_json": "TEXT DEFAULT '{}'",
}

_SWIPE_COLUMNS = {
    "sampling_component": "TEXT DEFAULT ''",
}


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(PATHS.db_file, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db() -> None:
    with _lock, _connect() as con:
        con.executescript(SCHEMA)
        cols = {r[1] for r in con.execute("PRAGMA table_info(corpora)")}
        if "verdict" not in cols:
            con.execute("ALTER TABLE corpora ADD COLUMN verdict TEXT DEFAULT ''")
        if "kind" not in cols:
            con.execute("ALTER TABLE corpora ADD COLUMN kind TEXT DEFAULT 'ingested'")
        con.execute("UPDATE corpora SET kind='ingested' WHERE kind IS NULL OR kind=''")
        _ensure_columns(con, "reviews", _REVIEW_COLUMNS)
        _ensure_columns(con, "review_annotations", _ANNOTATION_COLUMNS)
        _ensure_columns(con, "review_swipes", _SWIPE_COLUMNS)
        con.execute("UPDATE reviews SET status='completed' WHERE status IS NULL OR status=''")


def _ensure_columns(con: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    for name, definition in columns.items():
        if name not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def q(sql: str, args: tuple = ()) -> list[dict]:
    with _connect() as con:
        return [dict(r) for r in con.execute(sql, args)]


def q1(sql: str, args: tuple = ()) -> dict | None:
    rows = q(sql, args)
    return rows[0] if rows else None


def execute(sql: str, args: tuple = ()) -> int:
    with _lock, _connect() as con:
        cur = con.execute(sql, args)
        return cur.rowcount


def executemany(sql: str, seq) -> int:
    with _lock, _connect() as con:
        cur = con.executemany(sql, seq)
        return cur.rowcount


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Run a group of writes atomically while serializing local writer threads."""
    with _lock:
        con = _connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            yield con
        except BaseException:
            con.rollback()
            raise
        else:
            con.commit()
        finally:
            con.close()


def replace_corpus_hashes(corpus_id: str, staged_path) -> int:
    """Atomically make ``corpus_id`` own exactly the hashes staged in ``staged_path``.

    ``staged_path`` is a SQLite file with a single ``hashes(hash)`` table
    produced by a re-import.  Hashes already owned by another corpus stay with
    that corpus (``INSERT OR IGNORE``), matching the ordinary ingest rule that
    the first corpus to register a hash keeps it.
    """

    with _lock:
        con = sqlite3.connect(PATHS.db_file, timeout=30)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("ATTACH DATABASE ? AS staged", (str(staged_path),))
            con.execute("BEGIN IMMEDIATE")
            con.execute("DELETE FROM doc_hashes WHERE corpus_id=?", (corpus_id,))
            cur = con.execute(
                "INSERT OR IGNORE INTO doc_hashes(hash, corpus_id) SELECT hash, ? FROM staged.hashes",
                (corpus_id,),
            )
            inserted = int(cur.rowcount)
            con.commit()
            con.execute("DETACH DATABASE staged")
            return inserted
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()


def new_id(prefix: str) -> str:
    return f"{prefix}_{dt.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


init_db()
