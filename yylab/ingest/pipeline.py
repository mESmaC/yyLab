from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import orjson
import zstandard as zstd

from .. import db
from ..artifacts import corpus_dir, write_json
from ..config import get_settings
from ..jobs import ProgressFn
from .adapters.loader import discover_adapters, dispatch, walk_files
from .dedup import MinHashDedup, norm_sha256_hex, sha256_hex
from .domains import classify_content
from .language import detect_language
from .quality import score_quality
from .records import doc_ok, normalize_text
from .tokenize_stats import SPEC_KEYS, TokenCounter

SHARD_DOCS = 20_000
GLOBAL_BATCH = 1_024
TOK_BATCH = 2_048

LANG_SKIP_DOMAINS = {"code", "math"}
QUALITY_SKIP_DOMAINS = {"code", "web", "math"}


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
        self._comp.write(orjson.dumps(line))
        self._comp.write(b"\n")
        self.count += 1
        if self.count >= SHARD_DOCS:
            self._close()
            self.index += 1
            self.count = 0
            self._open()

    def _close(self) -> None:
        self._comp.close()
        self._fh.close()

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        if self.count == 0 and self.index > 0:
            (self.data_dir / self.shards.pop()).unlink(missing_ok=True)
            self._fh = None
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
            reader = dctx.stream_reader(fh)
            buf = b""
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        yield orjson.loads(line)
            for line in buf.split(b"\n"):
                if line.strip():
                    yield orjson.loads(line)


def _global_existing(hashes: list[str]) -> set[str]:
    existing: set[str] = set()
    for i in range(0, len(hashes), 500):
        chunk = hashes[i : i + 500]
        ph = ",".join("?" * len(chunk))
        rows = db.q(f"SELECT hash FROM doc_hashes WHERE hash IN ({ph})", tuple(chunk))
        existing.update(r["hash"] for r in rows)
    return existing


def _rejected_existing(hashes: list[str]) -> set[str]:
    existing: set[str] = set()
    for i in range(0, len(hashes), 500):
        chunk = hashes[i : i + 500]
        ph = ",".join("?" * len(chunk))
        try:
            rows = db.q(f"SELECT hash FROM rejected_hashes WHERE hash IN ({ph})", tuple(chunk))
        except Exception:  # noqa: BLE001
            return existing
        existing.update(r["hash"] for r in rows)
    return existing


def run_ingest(progress: ProgressFn, cfg: IngestConfig) -> str:
    cid = db.new_id("cor")
    cdir = corpus_dir(cid)
    data_dir = cdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    now = db.now_iso()
    db.execute(
        "INSERT INTO corpora(id, name, created_at, updated_at, status, source_path) VALUES(?,?,?,?,?,?)",
        (cid, cfg.name, now, now, "processing", cfg.source_dir),
    )

    stats: dict = {
        "files_found": 0,
        "files_processed": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "docs_seen": 0,
        "docs_written": 0,
        "exact_dupes_local": 0,
        "exact_dupes_global": 0,
        "near_dupes": 0,
        "filtered": 0,
        "non_english_docs": 0,
        "quarantined_docs": 0,
        "rejected_known_bad": 0,
        "langs": {},
        "quarantine_reasons": {},
        "chars_written": 0,
        "bytes_on_disk": 0,
        "domains": {},
        "totals": {"tokens": {}},
        "tokenizer_failures": {},
        "shards": [],
        "nonen_shards": [],
        "quarantine_shards": [],
        "errors": [],
    }
    local_hashes: set[str] = set()
    pending: list[tuple[str, str, dict]] = []
    side_hashes: list[str] = []
    writer = _ShardWriter(data_dir)
    nonen_writer = _LazyShardWriter(cdir / "data_nonen")
    quarantine_writer = _LazyShardWriter(cdir / "data_quarantine")
    mh = MinHashDedup() if cfg.near_dedup else None

    def flush_pending(final: bool = False) -> None:
        nonlocal pending
        if not pending:
            return
        hashes = [h for h, _, _ in pending]
        blocked: set[str] = set()
        rejected: set[str] = set()
        if cfg.global_dedup:
            blocked = _global_existing(hashes)
        rejected = _rejected_existing(hashes)
        kept_rows: list[tuple[str, str]] = []
        for h, hn, line in pending:
            if h in rejected or hn in rejected:
                stats["rejected_known_bad"] += 1
                continue
            if h in blocked or hn in blocked:
                stats["exact_dupes_global"] += 1
                continue
            writer.write(line)
            kept_rows.append((h, cid))
            kept_rows.append((hn, cid))
            dom = line["domain"]
            d = stats["domains"].setdefault(dom, {"docs": 0, "chars": 0, "tokens": {}})
            d["docs"] += 1
            d["chars"] += len(line["text"])
            stats["chars_written"] += len(line["text"])
            stats["docs_written"] += 1
        if cfg.global_dedup and kept_rows:
            db.executemany("INSERT OR IGNORE INTO doc_hashes(hash, corpus_id) VALUES(?,?)", kept_rows)
        pending = [] if not final else pending

    def register_side_hashes() -> None:
        if side_hashes and cfg.global_dedup:
            db.executemany(
                "INSERT OR IGNORE INTO doc_hashes(hash, corpus_id) VALUES(?,?)",
                [(h, cid) for h in side_hashes],
            )
            side_hashes.clear()

    def finalize(error: str = "") -> None:
        writer.finish()
        stats["nonen_shards"] = nonen_writer.finish()
        stats["quarantine_shards"] = quarantine_writer.finish()
        stats["shards"] = writer.shards
        stats["bytes_on_disk"] = sum(
            (data_dir / s).stat().st_size for s in writer.shards if (data_dir / s).exists()
        )
        stats_path = cdir / "stats.json"
        write_json(stats_path, stats)
        status = "failed" if error else "ready"
        db.execute(
            "UPDATE corpora SET status=?, stats_json=?, error=?, updated_at=? WHERE id=?",
            (status, orjson.dumps(stats).decode(), error, db.now_iso(), cid),
        )

    try:
        adapters = discover_adapters()
        root = Path(cfg.source_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"source folder not found: {cfg.source_dir}")
        files = list(walk_files(root, cfg.max_file_mb))
        stats["files_found"] = len(files)
        progress("scan", 0.0, f"{len(files)} files found")

        for fi, path in enumerate(files):
            adapter = dispatch(path, adapters)
            if adapter is None:
                stats["files_skipped"] += 1
                continue
            try:
                for doc in adapter.extract(path):
                    stats["docs_seen"] += 1
                    text = normalize_text(doc.text)
                    if text is None or not doc_ok(text, cfg.min_chars):
                        stats["filtered"] += 1
                        continue
                    h = sha256_hex(text)
                    hn = norm_sha256_hex(text)
                    if h in local_hashes or hn in local_hashes:
                        stats["exact_dupes_local"] += 1
                        continue
                    if mh is not None and mh.is_duplicate(text):
                        stats["near_dupes"] += 1
                        continue
                    local_hashes.add(h)
                    local_hashes.add(hn)
                    domain = cfg.domain_override or classify_content(text, doc.domain)

                    lang = ""
                    if domain not in LANG_SKIP_DOMAINS:
                        lang = detect_language(text)
                        stats["langs"][lang] = stats["langs"].get(lang, 0) + 1
                        if cfg.language_filter and lang not in {"en", "unknown"}:
                            stats["non_english_docs"] += 1
                            nonen_writer.write(
                                {"text": text, "domain": domain, "source": doc.source, "lang": lang}
                            )
                            side_hashes.extend((h, hn))
                            continue

                    line = {"text": text, "domain": domain, "source": doc.source, "lang": lang}
                    if domain not in QUALITY_SKIP_DOMAINS:
                        rep = score_quality(text)
                        line["quality"] = round(rep.score, 3)
                        if cfg.quality_filter and rep.score < cfg.quality_threshold:
                            stats["quarantined_docs"] += 1
                            for reason in rep.reasons:
                                stats["quarantine_reasons"][reason] = (
                                    stats["quarantine_reasons"].get(reason, 0) + 1
                                )
                            line["quarantine_reasons"] = rep.reasons
                            quarantine_writer.write(line)
                            side_hashes.extend((h, hn))
                            continue
                    pending.append((h, hn, line))
                    if len(pending) >= GLOBAL_BATCH:
                        flush_pending()
            except Exception as e:  # noqa: BLE001
                stats["files_failed"] += 1
                if len(stats["errors"]) < 50:
                    stats["errors"].append(f"{path.name}: {str(e)[:200]}")
                continue
            stats["files_processed"] += 1
            progress("extract", (fi + 1) / max(len(files), 1), path.name)
        flush_pending(final=True)
        register_side_hashes()
        writer.finish()

        if cfg.tokenizer_keys:
            settings = get_settings()
            enabled = settings.get("tokenizers_enabled", {})
            keys = [k for k in cfg.tokenizer_keys if enabled.get(k, True)]
            counter = TokenCounter(keys, hf_token=settings.get("hf_token") or None)
            counter.available_specs()
            stats["tokenizer_failures"] = dict(counter.failures)
            batch_texts: list[str] = []
            batch_domains: list[str] = []

            def tok_flush() -> None:
                if not batch_texts:
                    return
                counts = counter.count(batch_texts)
                for key, per_doc in counts.items():
                    stats["totals"]["tokens"][key] = stats["totals"]["tokens"].get(key, 0) + sum(per_doc)
                    for dom, n in zip(batch_domains, per_doc):
                        stats["domains"][dom]["tokens"][key] = (
                            stats["domains"][dom]["tokens"].get(key, 0) + n
                        )
                batch_texts.clear()
                batch_domains.clear()

            total_docs = max(stats["docs_written"], 1)
            seen = 0
            for line in _iter_shards(data_dir, writer.shards):
                batch_texts.append(line["text"])
                batch_domains.append(line["domain"])
                seen += 1
                if len(batch_texts) >= TOK_BATCH:
                    tok_flush()
                    progress("tokenize", seen / total_docs, f"{seen}/{stats['docs_written']} docs")
            tok_flush()
            progress("tokenize", 1.0, "done")

        finalize()
        progress("done", 1.0, "complete")
    except Exception as e:
        finalize(error=str(e)[:500])
        raise
    return cid
