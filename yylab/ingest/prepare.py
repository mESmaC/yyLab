"""Order-independent document preparation for corpus ingestion.

Everything yyLab computes for a document that depends only on the document
itself (normalization, CDAS evidence, dedup fingerprints, language, quality,
MinHash bands) lives here so it can run in parallel worker processes and be
cached.  Decisions that depend on ingest policy or on other documents (hygiene
thresholds, filters, exact/near duplicate suppression, global hashes) stay in
the single-threaded coordinator in :mod:`yylab.ingest.pipeline`.

This module must stay importable without :mod:`yylab.db` or the UI so that
spawned workers start quickly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import orjson
import zstandard as zstd

from .adapters.base import Adapter, ChunkSpec
from .catalog import (
    IDENTITY_VERSION,
    QUALITY_SIGNAL_VERSION,
    SOURCE_NORMALIZATION_VERSION,
    analyze_document,
    compact_metadata,
    digest_file,
    document_logical_path,
    source_digest,
)
from .dedup import (
    decode_band_fingerprints,
    dedup_fingerprints,
    encode_band_fingerprints,
    minhash_band_fingerprints,
    sha256_hex,
)
from .domains import classify_content, domain_for_ext
from .language import detect_language
from .quality import score_quality
from .records import Doc, infer_source_split, normalize_chat_messages, normalize_text, printable_ratio

PREPARED_SCHEMA_VERSION = 1
TRAINING_NORMALIZATION = "nfc_whitespace_v2"
PRINTABLE_MIN_RATIO = 0.85
LANG_SKIP_DOMAINS = frozenset({"code", "math"})
QUALITY_SKIP_DOMAINS = frozenset({"code", "web", "math"})
_LINE_EXTRA_FIELDS = ("sample_id", "provenance", "labels")
_LOCATOR_RE = re.compile(r"^(record|line|row)=(\d+)(.*)$", re.DOTALL)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class AnalysisParams:
    """The only ingest settings that influence preparation output.

    ``min_chars`` is a hint: language, quality, and MinHash work is skipped for
    documents that cannot pass hygiene at this threshold.  The coordinator
    computes those lazily if a later re-import lowers the threshold.
    """

    domain_override: str | None = None
    near_dedup: bool = False
    min_chars: int = 40

    def analysis_key(self) -> dict[str, Any]:
        """Versions and settings that change prepared output (cache validity)."""

        return {
            "prepared_schema_version": PREPARED_SCHEMA_VERSION,
            "quality_signal_version": QUALITY_SIGNAL_VERSION,
            "source_normalization_version": SOURCE_NORMALIZATION_VERSION,
            "identity_version": IDENTITY_VERSION,
            "training_normalization": TRAINING_NORMALIZATION,
            "domain_override": self.domain_override or "",
        }


@dataclass(slots=True)
class PreparedDoc:
    locator: str
    text: str | None
    source_sha256: str
    source_utf8_bytes: int
    training_sha256: str
    domain: str
    logical_path: str
    split: str
    printable_ratio: float
    signals: dict[str, Any]
    roles: list[str]
    metadata: dict[str, Any]
    content_hash_mismatch: bool
    sample_id: str | None
    extras: dict[str, Any]
    lang: str | None = None
    quality: float | None = None
    quality_reasons: list[str] = field(default_factory=list)
    quality_computed: bool = False
    minhash_bands: tuple[int, ...] | None = None
    minhash_computed: bool = False
    signals_json: str = ""
    roles_json: str = ""
    metadata_json: str = ""

    def hygiene_ok(self, min_chars: int) -> bool:
        return (
            self.text is not None
            and len(self.text) >= min_chars
            and self.printable_ratio >= PRINTABLE_MIN_RATIO
        )

    def ensure_json(self) -> None:
        if not self.signals_json:
            self.signals_json = _json(self.signals)
        if not self.roles_json:
            self.roles_json = _json(self.roles)
        if not self.metadata_json:
            self.metadata_json = _json(self.metadata)

    def shard_line(self) -> dict[str, Any]:
        """Rebuild the training-shard line in the historical key order."""

        line: dict[str, Any] = {
            "text": self.text,
            "domain": self.domain,
            "source": self.logical_path,
            "lang": "",
            "split": self.split,
        }
        messages = self.extras.get("messages")
        if messages:
            line["messages"] = messages
        for name in _LINE_EXTRA_FIELDS:
            if name in self.extras:
                line[name] = self.extras[name]
        return line

    def to_record(self) -> dict[str, Any]:
        """Cache representation.

        The catalog JSON columns are stored as the exact strings the
        coordinator writes (stdlib ``json``), so values that orjson would alter
        (``NaN``/``Infinity`` from permissive source decoding) round-trip
        byte-identically between a fresh ingest and a cached re-import.
        """

        self.ensure_json()
        return {
            "locator": self.locator,
            "text": self.text,
            "source_sha256": self.source_sha256,
            "source_utf8_bytes": self.source_utf8_bytes,
            "training_sha256": self.training_sha256,
            "domain": self.domain,
            "logical_path": self.logical_path,
            "split": self.split,
            "printable_ratio": self.printable_ratio,
            "signals": self.signals,
            "roles": self.roles,
            "signals_json": self.signals_json,
            "roles_json": self.roles_json,
            "metadata_json": self.metadata_json,
            "content_hash_mismatch": self.content_hash_mismatch,
            "sample_id": self.sample_id,
            "extras": self.extras,
            "lang": self.lang,
            "quality": self.quality,
            "quality_reasons": self.quality_reasons,
            "quality_computed": self.quality_computed,
            "minhash_bands": encode_band_fingerprints(self.minhash_bands),
            "minhash_computed": self.minhash_computed,
        }

    def approx_bytes(self) -> int:
        """Estimated resident size once decoded (measured at roughly 3x the raw text)."""

        return (len(self.text) if self.text else 0) * 4 + 2 * (len(self.signals_json) + len(self.metadata_json)) + 1024

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> PreparedDoc:
        quality = record.get("quality")
        doc = cls(
            locator=str(record.get("locator") or ""),
            text=record.get("text"),
            source_sha256=str(record["source_sha256"]),
            source_utf8_bytes=int(record["source_utf8_bytes"]),
            training_sha256=str(record.get("training_sha256") or ""),
            domain=str(record["domain"]),
            logical_path=str(record["logical_path"]),
            split=str(record["split"]),
            printable_ratio=float(record.get("printable_ratio") or 0.0),
            signals=dict(record["signals"]),
            roles=list(record.get("roles") or []),
            metadata=dict(record.get("metadata") or {}),
            content_hash_mismatch=bool(record.get("content_hash_mismatch")),
            sample_id=record.get("sample_id"),
            extras=dict(record.get("extras") or {}),
            lang=record.get("lang"),
            quality=None if quality is None else float(quality),
            quality_reasons=list(record.get("quality_reasons") or []),
            quality_computed=bool(record.get("quality_computed")),
            minhash_bands=decode_band_fingerprints(record.get("minhash_bands")),
            minhash_computed=bool(record.get("minhash_computed")),
            signals_json=str(record.get("signals_json") or ""),
            roles_json=str(record.get("roles_json") or ""),
            metadata_json=str(record.get("metadata_json") or ""),
        )
        doc.ensure_json()
        return doc


@dataclass(slots=True)
class ChunkResult:
    docs: list[PreparedDoc]
    issues: list[tuple[str, str, str]]
    records_seen: int
    lines_seen: int
    raw_sha256: str = ""
    error: str = ""
    error_type: str = ""
    hash_failed: bool = False
    payload: bytes | None = None
    #: Rough in-memory size of ``docs`` (text plus evidence), used by the
    #: coordinator to bound how many finished results it keeps queued.
    approx_bytes: int = 0
    #: Records the adapter yielded, including one whose preparation raised.
    docs_yielded: int = 0


def rebase_locator(locator: str, record_base: int, line_base: int) -> str:
    """Shift a chunk-local ``record=``/``row=``/``line=`` locator to file scope."""

    if not record_base and not line_base:
        return locator
    match = _LOCATOR_RE.match(locator)
    if match is None:
        return locator
    kind, number, rest = match.group(1), int(match.group(2)), match.group(3)
    base = line_base if kind == "line" else record_base
    return f"{kind}={number + base}{rest}"


def prepare_document(
    doc: Doc,
    *,
    asset_logical_path: str,
    asset_format_name: str,
    params: AnalysisParams,
) -> PreparedDoc:
    source_text = doc.text
    text = normalize_text(source_text)
    meta = doc.meta if isinstance(doc.meta, dict) else {}
    logical_path = document_logical_path(asset_logical_path, meta)
    path_hint = domain_for_ext(PurePosixPath(logical_path).suffix)
    domain = params.domain_override or classify_content(
        text or source_text,
        path_hint or doc.domain,
    )
    signals, roles = analyze_document(
        source_text,
        logical_path=logical_path,
        format_name=doc.format_name or asset_format_name,
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
    source_sha256, source_utf8_bytes = source_digest(source_text)
    sample_id = meta.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        sample_id = None
    extras: dict[str, Any] = {}
    if text is not None:
        messages = normalize_chat_messages(meta.get("messages"))
        if messages:
            extras["messages"] = messages
        for name in _LINE_EXTRA_FIELDS:
            value = meta.get(name)
            if value is not None:
                extras[name] = value
    prepared = PreparedDoc(
        locator=doc.locator or "",
        text=text,
        source_sha256=source_sha256,
        source_utf8_bytes=source_utf8_bytes,
        training_sha256=sha256_hex(text) if text is not None else "",
        domain=domain,
        logical_path=logical_path,
        split=infer_source_split(logical_path, meta.get("split")) if text is not None else "unspecified",
        printable_ratio=printable_ratio(text) if text else 0.0,
        signals=signals,
        roles=roles,
        metadata=compact_metadata(meta),
        content_hash_mismatch=content_hash_mismatch,
        sample_id=sample_id,
        extras=extras,
    )
    augment_document(prepared, params)
    prepared.ensure_json()
    return prepared


def augment_document(prepared: PreparedDoc, params: AnalysisParams) -> bool:
    """Fill language, quality, and MinHash bands that the policy stage may need.

    Returns ``True`` when anything was computed.  Work is only spent on
    documents that can pass hygiene at ``params.min_chars``.
    """

    if not prepared.hygiene_ok(params.min_chars):
        return False
    text = prepared.text or ""
    changed = False
    if prepared.lang is None and prepared.domain not in LANG_SKIP_DOMAINS:
        prepared.lang = detect_language(text)
        changed = True
    if not prepared.quality_computed and prepared.domain not in QUALITY_SKIP_DOMAINS:
        report = score_quality(text)
        prepared.quality = report.score
        prepared.quality_reasons = list(report.reasons)
        prepared.quality_computed = True
        changed = True
    if params.near_dedup and not prepared.minhash_computed:
        prepared.minhash_bands = minhash_band_fingerprints(text)
        prepared.minhash_computed = True
        changed = True
    return changed


def augment_documents(docs: list[PreparedDoc], params: AnalysisParams) -> list[PreparedDoc]:
    for prepared in docs:
        augment_document(prepared, params)
    return docs


def needs_augmentation(docs: Iterable[PreparedDoc], params: AnalysisParams) -> bool:
    for prepared in docs:
        if not prepared.hygiene_ok(params.min_chars):
            continue
        if prepared.lang is None and prepared.domain not in LANG_SKIP_DOMAINS:
            return True
        if not prepared.quality_computed and prepared.domain not in QUALITY_SKIP_DOMAINS:
            return True
        if params.near_dedup and not prepared.minhash_computed:
            return True
    return False


def encode_payload(docs: Iterable[PreparedDoc]) -> bytes:
    """Serialize prepared documents as zstd-compressed JSON lines."""

    buffer = bytearray()
    for prepared in docs:
        buffer += orjson.dumps(prepared.to_record())
        buffer += b"\n"
    return zstd.ZstdCompressor(level=3).compress(bytes(buffer))


def decode_payload(payload: bytes) -> list[PreparedDoc]:
    raw = zstd.ZstdDecompressor().decompress(payload, max_output_size=1 << 31)
    return [PreparedDoc.from_record(orjson.loads(line)) for line in raw.split(b"\n") if line]


def prepare_chunk(
    path: Path,
    adapter: Adapter | None,
    chunk: ChunkSpec | None,
    *,
    asset_logical_path: str,
    asset_format_name: str,
    params: AnalysisParams,
    hash_file: bool = False,
    build_payload: bool = False,
) -> ChunkResult:
    """Extract and prepare one chunk of an asset.

    ``adapter=None`` hashes the asset without extracting it (inventory-only and
    policy-excluded assets still record their content hash).  Adapter
    exceptions are captured together with the documents prepared before the
    failure so the coordinator can reproduce the sequential partial-flush and
    corpus-failure semantics in order.
    """

    raw_sha256 = ""
    if hash_file:
        try:
            raw_sha256 = digest_file(path)
        except OSError as exc:
            return ChunkResult(
                docs=[],
                issues=[],
                records_seen=0,
                lines_seen=0,
                error=str(exc),
                error_type=type(exc).__name__,
                hash_failed=True,
            )
    if adapter is None:
        return ChunkResult(docs=[], issues=[], records_seen=0, lines_seen=0, raw_sha256=raw_sha256)
    docs: list[PreparedDoc] = []
    error = ""
    error_type = ""
    docs_yielded = 0
    try:
        iterator = adapter.extract(path) if chunk is None else adapter.extract_chunk(path, chunk)
        for doc in iterator:
            docs_yielded += 1
            docs.append(
                prepare_document(
                    doc,
                    asset_logical_path=asset_logical_path,
                    asset_format_name=asset_format_name,
                    params=params,
                )
            )
    except Exception as exc:  # noqa: BLE001 - reported to the catalog by the coordinator
        error = str(exc)
        error_type = type(exc).__name__
    drain = getattr(adapter, "drain_issues", None)
    issues = list(drain()) if callable(drain) else []
    stats = getattr(adapter, "chunk_stats", None)
    records_seen, lines_seen = stats() if callable(stats) else (0, 0)
    payload = encode_payload(docs) if build_payload and not error else None
    return ChunkResult(
        docs=docs,
        issues=issues,
        records_seen=int(records_seen),
        lines_seen=int(lines_seen),
        raw_sha256=raw_sha256,
        error=error,
        error_type=error_type,
        payload=payload,
        approx_bytes=sum(doc.approx_bytes() for doc in docs) + (len(payload) if payload else 0),
        docs_yielded=docs_yielded,
    )
