from __future__ import annotations

import datetime as dt
import sqlite3
import threading
import uuid

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
"""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(PATHS.db_file, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db() -> None:
    with _lock, _connect() as con:
        con.executescript(SCHEMA)


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


def new_id(prefix: str) -> str:
    return f"{prefix}_{dt.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


init_db()
