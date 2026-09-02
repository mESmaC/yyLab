from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from sqlite3 import Connection
from typing import Any

import orjson
import zstandard as zstd

from .. import db
from ..artifacts import corpus_dir, write_json
from ..config import get_settings
from ..jobs import ProgressFn
from .adapters.loader import discover_adapters, dispatch, walk_files
from .catalog import (
    SELECTION_PROFILES,
    AssetObservation,
    CorpusCatalog,
    analyze_document,
    asset_policy_reasons,
    document_logical_path,
    profile_quarantine_reasons,
)
from .dedup import MinHashDedup, dedup_fingerprints, sha256_hex, training_hash_key
from .domains import classify_content, domain_for_ext
from .language import detect_language
from .quality import score_quality
from .records import doc_ok, infer_source_split, normalize_chat_messages, normalize_text
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
    selection_profile: str = "dataset"
    include_path_regex: str = ""
    exclude_path_regex: str = ""


@dataclass(slots=True)
class _PendingDocument:
    raw_hash: str
    line: dict[str, Any]
    asset: AssetObservation
    locator: str
    source_text: str
    domain: str
    language: str
    reasons: list[str]
    roles: list[str]
    signals: dict[str, Any]
    metadata: dict[str, Any]
    logical_path: str


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


def run_ingest(progress: ProgressFn, cfg: IngestConfig) -> str:
    if cfg.selection_profile not in SELECTION_PROFILES:
        raise ValueError(
            f"selection_profile must be one of: {', '.join(SELECTION_PROFILES)}"
        )
    include_path = re.compile(cfg.include_path_regex, re.IGNORECASE) if cfg.include_path_regex else None
    exclude_path = re.compile(cfg.exclude_path_regex, re.IGNORECASE) if cfg.exclude_path_regex else None
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
        "ingest_config": {
            "near_dedup": bool(cfg.near_dedup),
            "global_dedup": bool(cfg.global_dedup),
            "language_filter": bool(cfg.language_filter),
            "quality_filter": bool(cfg.quality_filter),
            "quality_threshold": float(cfg.quality_threshold),
            "min_chars": int(cfg.min_chars),
            "tokenizer_keys": list(cfg.tokenizer_keys),
            "global_hash_storage": "training_sha256_v3_namespaced",
            "training_normalization": "nfc_whitespace_v2",
            "selection_profile": cfg.selection_profile,
            "include_path_regex": cfg.include_path_regex,
            "exclude_path_regex": cfg.exclude_path_regex,
            "max_file_mb": float(cfg.max_file_mb),
        },
    }
    local_hash_splits: dict[str, set[str]] = {}
    pending: list[_PendingDocument] = []
    pending_bytes = 0
    writer = _ShardWriter(data_dir)
    nonen_writer = _LazyShardWriter(cdir / "data_nonen")
    quarantine_writer = _LazyShardWriter(cdir / "data_quarantine")
    mh = MinHashDedup() if cfg.near_dedup else None
    token_counter: TokenCounter | None = None
    batch_texts: list[str] = []
    batch_domains: list[str] = []
    batch_splits: list[str] = []
    catalog: CorpusCatalog | None = None

    def tok_flush() -> None:
        if token_counter is None or not batch_texts:
            return
        counts = token_counter.count(batch_texts)
        for key, per_doc in counts.items():
            stats["totals"]["tokens"][key] = stats["totals"]["tokens"].get(key, 0) + sum(per_doc)
            for dom, n in zip(batch_domains, per_doc, strict=True):
                stats["domains"][dom]["tokens"][key] = (
                    stats["domains"][dom]["tokens"].get(key, 0) + n
                )
            for split, dom, n in zip(batch_splits, batch_domains, per_doc, strict=True):
                split_tokens = stats["splits"][split]["tokens"]
                split_tokens[key] = split_tokens.get(key, 0) + n
                split_domain_tokens = stats["splits"][split]["domains"][dom].setdefault(
                    "tokens", {}
                )
                split_domain_tokens[key] = split_domain_tokens.get(key, 0) + n
        batch_texts.clear()
        batch_domains.clear()
        batch_splits.clear()

    def queue_token_stats(line: dict) -> None:
        if token_counter is None:
            return
        batch_texts.append(line["text"])
        batch_domains.append(line["domain"])
        batch_splits.append(str(line.get("split") or "unspecified"))
        if len(batch_texts) >= TOK_BATCH:
            tok_flush()

    def flush_pending() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        exact_hashes = [training_hash_key(item.raw_hash) for item in pending]
        raw_hashes = [item.raw_hash for item in pending]
        kept_rows: list[tuple[str, str]] = []
        token_lines: list[dict] = []
        # One connection and transaction handles every membership chunk plus
        # registration. The old helpers opened a new SQLite connection for each
        # 500 hashes and then another for inserts.
        with db.transaction() as con:
            blocked = (
                _external_document_hashes(con, exact_hashes, cid)
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
                storage_hash = training_hash_key(item.raw_hash)
                if storage_hash in rejected:
                    stats["rejected_known_bad"] += 1
                    if catalog is not None:
                        catalog.record_document(
                            asset=item.asset,
                            locator=item.locator,
                            source_text=item.source_text,
                            training_text=line["text"],
                            domain=item.domain,
                            language=item.language,
                            selection="rejected_known_bad",
                            reasons=[*item.reasons, "known_bad_hash"],
                            roles=item.roles,
                            signals=item.signals,
                            metadata=item.metadata,
                        )
                    continue
                if storage_hash in blocked:
                    stats["exact_dupes_global"] += 1
                    if catalog is not None:
                        catalog.record_document(
                            asset=item.asset,
                            locator=item.locator,
                            source_text=item.source_text,
                            training_text=line["text"],
                            domain=item.domain,
                            language=item.language,
                            selection="duplicate_global",
                            reasons=[*item.reasons, "duplicate:global_exact"],
                            roles=item.roles,
                            signals=item.signals,
                            metadata=item.metadata,
                        )
                    continue
                writer.write(line)
                kept_rows.append((storage_hash, cid))
                dom = line["domain"]
                d = stats["domains"].setdefault(dom, {"docs": 0, "chars": 0, "tokens": {}})
                d["docs"] += 1
                d["chars"] += len(line["text"])
                split = str(line.get("split") or "unspecified")
                split_stats = stats["splits"].setdefault(
                    split,
                    {"docs": 0, "chars": 0, "tokens": {}, "domains": {}},
                )
                split_stats["docs"] += 1
                split_stats["chars"] += len(line["text"])
                split_domain = split_stats["domains"].setdefault(
                    dom,
                    {"docs": 0, "chars": 0},
                )
                split_domain["docs"] += 1
                split_domain["chars"] += len(line["text"])
                stats["chars_written"] += len(line["text"])
                stats["docs_written"] += 1
                source_stats = stats["sources"].setdefault(
                    item.logical_path,
                    {"docs": 0, "chars": 0, "domain": dom, "split": split},
                )
                source_stats["docs"] += 1
                source_stats["chars"] += len(line["text"])
                token_lines.append(line)
                if catalog is not None:
                    catalog.record_document(
                        asset=item.asset,
                        locator=item.locator,
                        source_text=item.source_text,
                        training_text=line["text"],
                        domain=item.domain,
                        language=item.language,
                        selection="kept",
                        reasons=item.reasons,
                        roles=item.roles,
                        signals=item.signals,
                        metadata=item.metadata,
                    )
            if cfg.global_dedup and kept_rows:
                con.executemany(
                    "INSERT OR IGNORE INTO doc_hashes(hash, corpus_id) VALUES(?,?)",
                    kept_rows,
                )
        # Tokenizers can use their own worker threads and may take seconds for
        # large batches. Never hold yyLab's global SQLite writer lock over them.
        for line in token_lines:
            queue_token_stats(line)
        pending = []
        pending_bytes = 0

    def finalize(error: str = "") -> None:
        writer.finish()
        stats["nonen_shards"] = nonen_writer.finish()
        stats["quarantine_shards"] = quarantine_writer.finish()
        stats["shards"] = writer.shards
        stats["bytes_on_disk"] = sum(
            (data_dir / s).stat().st_size for s in writer.shards if (data_dir / s).exists()
        )
        if catalog is not None and not catalog.closed:
            stats["cdas"] = catalog.summary()
            write_json(
                cdir / "ingest_manifest.json",
                {
                    "schema_version": 1,
                    "kind": "yylab_cdas_compatible_corpus",
                    "corpus_id": cid,
                    "name": cfg.name,
                    "source_root": str(Path(cfg.source_dir).resolve()),
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
            catalog.close()
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
        catalog = CorpusCatalog(
            cdir / "inventory.sqlite3",
            source_root=root,
            source_name=cfg.name,
        )
        # Discovery inventories every visible corpus asset. Size limits are an
        # explicit catalog decision below, never a silent pre-scan omission.
        files = list(walk_files(root))
        stats["files_found"] = len(files)
        progress("scan", 0.0, f"{len(files)} files found")

        if cfg.tokenizer_keys:
            settings = get_settings()
            enabled = settings.get("tokenizers_enabled", {})
            keys = [key for key in cfg.tokenizer_keys if enabled.get(key, True)]
            candidate = TokenCounter(keys, hf_token=settings.get("hf_token") or None)
            if candidate.available_specs():
                token_counter = candidate
            stats["tokenizer_failures"] = dict(candidate.failures)

        for fi, path in enumerate(files):
            adapter = dispatch(path, adapters)
            try:
                file_size = path.stat().st_size
            except OSError as error:
                stats["files_failed"] += 1
                if len(stats["errors"]) < 50:
                    stats["errors"].append(f"{path.name}: {str(error)[:200]}")
                continue
            size_limit = float(cfg.max_file_mb) * 1024 * 1024
            oversized = size_limit > 0 and file_size > size_limit
            try:
                asset = catalog.observe_asset(
                    path,
                    adapter_name=adapter.name if adapter is not None else "",
                    hash_contents=not oversized,
                )
            except (OSError, ValueError) as error:
                stats["files_failed"] += 1
                if len(stats["errors"]) < 50:
                    stats["errors"].append(f"{path.name}: {str(error)[:200]}")
                continue
            if oversized:
                stats["files_skipped"] += 1
                stats["assets_size_skipped"] += 1
                reason = f"asset_too_large:{cfg.max_file_mb:g}MiB"
                stats["quarantine_reasons"][reason] = (
                    stats["quarantine_reasons"].get(reason, 0) + 1
                )
                catalog.record_issue(
                    asset.asset_id,
                    "$",
                    "asset_too_large",
                    f"{file_size} bytes exceeds {cfg.max_file_mb:g} MiB",
                )
                catalog.finish_asset(
                    asset,
                    status="excluded_by_limit",
                    reasons=[reason],
                )
                continue
            path_reasons = asset_policy_reasons(cfg.selection_profile, asset.logical_path)
            if include_path is not None and include_path.search(asset.logical_path) is None:
                path_reasons.append("cdas:path_not_included")
            if exclude_path is not None and exclude_path.search(asset.logical_path) is not None:
                path_reasons.append("cdas:path_excluded")
            if adapter is None:
                stats["files_skipped"] += 1
                catalog.finish_asset(
                    asset,
                    status="inventory_only",
                    reasons=["adapter_unavailable"],
                )
                continue
            if path_reasons:
                stats["files_skipped"] += 1
                stats["assets_policy_skipped"] += 1
                for reason in sorted(set(path_reasons)):
                    stats["quarantine_reasons"][reason] = (
                        stats["quarantine_reasons"].get(reason, 0) + 1
                    )
                catalog.finish_asset(
                    asset,
                    status="excluded_by_policy",
                    reasons=path_reasons,
                )
                continue
            try:
                estimated_rows = adapter.estimated_records(path)
            except (OSError, ValueError):
                estimated_rows = None
            file_rows_seen = 0
            sample_ids: set[str] = set()

            def capture_adapter_issues() -> None:
                drain = getattr(adapter, "drain_issues", None)
                if not callable(drain):
                    return
                for issue_locator, reason_code, detail in drain():
                    stats["records_failed"] += 1
                    catalog.record_issue(
                        asset.asset_id,
                        issue_locator,
                        reason_code,
                        detail,
                    )

            try:
                for doc in adapter.extract(path):
                    stats["docs_seen"] += 1
                    file_rows_seen += 1
                    if file_rows_seen % PROGRESS_ROWS == 0:
                        within_file = (
                            min(file_rows_seen / estimated_rows, 0.999)
                            if estimated_rows
                            else 0.0
                        )
                        progress(
                            "extract",
                            (fi + within_file) / max(len(files), 1),
                            f"{path.name} · {file_rows_seen:,}"
                            + (f"/{estimated_rows:,}" if estimated_rows else "")
                            + f" rows · {stats['docs_written']:,} kept",
                        )
                    source_text = doc.text
                    text = normalize_text(source_text)
                    meta = doc.meta if isinstance(doc.meta, dict) else {}
                    locator = doc.locator or f"record={file_rows_seen}"
                    sample_id = meta.get("sample_id")
                    if isinstance(sample_id, str) and sample_id:
                        if sample_id in sample_ids:
                            stats["duplicate_sample_ids"] += 1
                            catalog.record_issue(
                                asset.asset_id,
                                locator,
                                "duplicate_sample_id",
                                sample_id,
                            )
                        else:
                            sample_ids.add(sample_id)
                    logical_path = document_logical_path(asset.logical_path, meta)
                    path_hint = domain_for_ext(PurePosixPath(logical_path).suffix)
                    domain = cfg.domain_override or classify_content(
                        text or source_text,
                        path_hint or doc.domain,
                    )
                    signals, roles = analyze_document(
                        source_text,
                        logical_path=logical_path,
                        format_name=doc.format_name or asset.format_name,
                        metadata=meta,
                        domain=domain,
                    )
                    signals["dedup"] = dedup_fingerprints(
                        source_text,
                        mode=str(signals["mode"]),
                        logical_path=logical_path,
                        language=str(meta.get("language") or meta.get("lang") or ""),
                    )
                    declared_hash = meta.get("content_sha256")
                    if declared_hash is None and isinstance(meta.get("provenance"), dict):
                        declared_hash = meta["provenance"].get("content_sha256")
                    content_hash_mismatch = False
                    if isinstance(declared_hash, str) and len(declared_hash) == 64:
                        actual_source_hash = signals["dedup"]["content_sha256"]
                        content_hash_mismatch = declared_hash.casefold() != actual_source_hash
                        signals["declared_content_sha256"] = declared_hash.casefold()
                        signals["declared_content_sha256_match"] = not content_hash_mismatch
                    if text is None or not doc_ok(text, cfg.min_chars):
                        stats["filtered"] += 1
                        catalog.record_document(
                            asset=asset,
                            locator=locator,
                            source_text=source_text,
                            training_text=text,
                            domain=domain,
                            language="",
                            selection="filtered",
                            reasons=["minimum_document_hygiene"],
                            roles=roles,
                            signals=signals,
                            metadata=meta,
                        )
                        continue
                    split = infer_source_split(logical_path, meta.get("split"))
                    line = {
                        "text": text,
                        "domain": domain,
                        "source": logical_path,
                        "lang": "",
                        "split": split,
                    }
                    messages = normalize_chat_messages(meta.get("messages"))
                    if messages:
                        line["messages"] = messages
                    for field_name in ("sample_id", "provenance", "labels"):
                        field_value = meta.get(field_name)
                        if field_value is not None:
                            line[field_name] = field_value

                    lang = ""
                    if domain not in LANG_SKIP_DOMAINS:
                        lang = detect_language(text)
                        line["lang"] = lang
                        stats["langs"][lang] = stats["langs"].get(lang, 0) + 1
                        if cfg.language_filter and lang not in {"en", "unknown"}:
                            stats["non_english_docs"] += 1
                            nonen_writer.write(line)
                            catalog.record_document(
                                asset=asset,
                                locator=locator,
                                source_text=source_text,
                                training_text=text,
                                domain=domain,
                                language=lang,
                                selection="non_english",
                                reasons=[f"language:{lang}"],
                                roles=roles,
                                signals=signals,
                                metadata=meta,
                            )
                            continue

                    quarantine_reasons = profile_quarantine_reasons(
                        cfg.selection_profile,
                        signals,
                        roles,
                    )
                    if content_hash_mismatch:
                        quarantine_reasons.append("cdas:content_hash_mismatch")
                    record_path_reasons = (
                        asset_policy_reasons(cfg.selection_profile, logical_path)
                        if logical_path != asset.logical_path
                        else []
                    )
                    if record_path_reasons:
                        stats["docs_policy_skipped"] += 1
                        quarantine_reasons.extend(record_path_reasons)
                    if domain not in QUALITY_SKIP_DOMAINS:
                        rep = score_quality(text)
                        line["quality"] = round(rep.score, 3)
                        if cfg.quality_filter and rep.score < cfg.quality_threshold:
                            quarantine_reasons.extend(rep.reasons)
                    quarantine_reasons = sorted(set(quarantine_reasons))
                    if quarantine_reasons:
                        stats["quarantined_docs"] += 1
                        for reason in quarantine_reasons:
                            stats["quarantine_reasons"][reason] = (
                                stats["quarantine_reasons"].get(reason, 0) + 1
                            )
                        line["quarantine_reasons"] = quarantine_reasons
                        quarantine_writer.write(line)
                        catalog.record_document(
                            asset=asset,
                            locator=locator,
                            source_text=source_text,
                            training_text=text,
                            domain=domain,
                            language=lang,
                            selection="quarantined",
                            reasons=quarantine_reasons,
                            roles=roles,
                            signals=signals,
                            metadata=meta,
                        )
                        continue

                    h = sha256_hex(text)
                    previous_splits = local_hash_splits.get(h, set())
                    candidate_reasons: list[str] = []
                    if split in previous_splits:
                        stats["exact_dupes_local"] += 1
                        reasons = ["duplicate:local_exact"]
                        catalog.record_document(
                            asset=asset,
                            locator=locator,
                            source_text=source_text,
                            training_text=text,
                            domain=domain,
                            language=lang,
                            selection="duplicate_local",
                            reasons=reasons,
                            roles=roles,
                            signals=signals,
                            metadata=meta,
                        )
                        continue
                    if previous_splits:
                        stats["split_leakage_dupes"] += 1
                        previous = ",".join(sorted(previous_splits))
                        candidate_reasons.append(f"split_leakage:{previous}:{split}")
                    if mh is not None and mh.is_duplicate(text):
                        stats["near_duplicate_candidates"] += 1
                        candidate_reasons.append("candidate:minhash_lsh")
                    local_hash_splits.setdefault(h, set()).add(split)
                    pending.append(
                        _PendingDocument(
                            raw_hash=h,
                            line=line,
                            asset=asset,
                            locator=locator,
                            source_text=source_text,
                            domain=domain,
                            language=lang,
                            reasons=candidate_reasons,
                            roles=roles,
                            signals=signals,
                            metadata=meta,
                            logical_path=logical_path,
                        )
                    )
                    pending_bytes += (
                        int(signals.get("utf8_bytes") or 0) + len(text) * 2 + 2048
                    )
                    if len(pending) >= GLOBAL_BATCH or pending_bytes >= GLOBAL_BATCH_BYTES:
                        flush_pending()
            except Exception as e:  # noqa: BLE001
                capture_adapter_issues()
                stats["files_failed"] += 1
                catalog.record_issue(
                    asset.asset_id,
                    "$",
                    "adapter_exception",
                    f"{type(e).__name__}: {e}",
                )
                catalog.finish_asset(
                    asset,
                    status="extraction_failed",
                    reasons=["adapter_exception"],
                )
                if len(stats["errors"]) < 50:
                    stats["errors"].append(f"{path.name}: {str(e)[:200]}")
                raise RuntimeError(f"asset extraction failed for {asset.logical_path}: {e}") from e
            capture_adapter_issues()
            catalog.finish_asset(asset, status="normalized")
            stats["files_processed"] += 1
            progress("extract", (fi + 1) / max(len(files), 1), path.name)
        flush_pending()
        writer.finish()

        if token_counter is not None:
            tok_flush()
            stats["tokenizer_failures"] = dict(token_counter.failures)
            progress("tokenize", 1.0, "done")

        finalize()
        progress("done", 1.0, "complete")
    except Exception as e:
        finalize(error=str(e)[:500])
        raise
    return cid
