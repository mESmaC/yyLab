from __future__ import annotations

import heapq
import math
import random
from collections.abc import Mapping
from typing import Any

import orjson
import zstandard as zstd

from . import db
from .artifacts import corpus_dir, write_json
from .config import PATHS
from .ingest.dedup import norm_sha256_hex, sha256_hex
from .ingest.pipeline import _iter_shards
from .ingest.pipeline import _ShardWriter as ShardWriter

REJECT_BAD_RATE = 0.40
MIN_REVIEWED = 5

STORE_NAMES = {"approved": "approved", "rejected": "rejected"}
SAMPLING_STRATEGIES = {"auto", "uniform", "active"}
DEFAULT_EXPLORATION_FRACTION = 0.20
DEFAULT_AUDIT_FRACTION = 0.10
_SCORE_BATCH_SIZE = 256


def _buckets(cid: str, include_quarantine: bool) -> list[tuple[str, list[str]]]:
    cdir = corpus_dir(cid)
    stats_path = cdir / "stats.json"
    stats = {}
    if stats_path.exists():
        stats = orjson.loads(stats_path.read_bytes())
    out: list[tuple[str, list[str]]] = [("data", stats.get("shards") or [])]
    if include_quarantine:
        out.append(("quarantine", stats.get("quarantine_shards") or []))
    return [(b, s) for b, s in out if s]


def _iter_review_docs(cid: str, include_quarantine: bool, reviewed: set[str]):
    for bucket, shards in _buckets(cid, include_quarantine):
        for line in _iter_shards(corpus_dir(cid) / bucket, shards):
            text = line.get("text", "")
            if not text:
                continue
            doc_hash = sha256_hex(text)
            if doc_hash in reviewed:
                continue
            item = {
                "bucket": bucket,
                "text": text,
                "domain": line.get("domain", "other"),
                "lang": line.get("lang", ""),
                "quality": line.get("quality"),
                "quality_report": line.get("quality_report"),
                "quarantine_reasons": line.get("quarantine_reasons"),
                "source": line.get("source", ""),
                "hash": doc_hash,
            }
            yield item


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _score_batch(ranker: object, docs: list[dict]) -> None:
    score_docs = getattr(ranker, "score_docs", None)
    if not callable(score_docs):
        raise TypeError("ranker must provide score_docs(docs)")
    scored = score_docs(docs)
    if len(scored) != len(docs):
        raise ValueError("ranker.score_docs returned a different number of scores")
    default_version = str(getattr(ranker, "model_version", "") or "")
    for doc, values in zip(docs, scored, strict=True):
        if not isinstance(values, Mapping):
            raise TypeError("ranker.score_docs entries must be mappings")
        rank_score = _finite_float(values.get("rank_score"))
        uncertainty = _finite_float(values.get("uncertainty"))
        if uncertainty is None and rank_score is not None:
            uncertainty = 1.0 - min(1.0, abs(rank_score - 0.5) * 2.0)
        doc["rank_score"] = rank_score
        doc["uncertainty"] = uncertainty
        doc["model_version"] = str(values.get("model_version") or default_version)
        doc["model_scores"] = {
            str(key): value
            for key, value in values.items()
            if key not in {"model_version"} and isinstance(value, int | float) and math.isfinite(float(value))
        }


def _keep_smallest_random(
    heap: list[tuple[float, int, dict]], item: dict, *, key: float, counter: int, limit: int
) -> None:
    entry = (-key, counter, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif key < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def _uniform_sample(
    docs,
    *,
    n: int,
    rng: random.Random,
    requested_strategy: str,
    fallback_reason: str = "",
) -> list[dict]:
    heap: list[tuple[float, int, dict]] = []
    seen = 0
    for seen, doc in enumerate(docs, start=1):
        _keep_smallest_random(heap, doc, key=rng.random(), counter=seen, limit=n)
    selected = [entry[2] for entry in heap]
    propensity = min(1.0, n / seen) if seen else None
    for doc in selected:
        doc.update(
            {
                "sampling_strategy": "uniform",
                "sampling_component": "uniform",
                "sampling_propensity": propensity,
                "sampling_candidate_count": seen,
                "sampling_requested_strategy": requested_strategy,
            }
        )
        if fallback_reason:
            doc["sampling_fallback_reason"] = fallback_reason
    rng.shuffle(selected)
    return selected


def _active_sample(
    docs,
    *,
    n: int,
    rng: random.Random,
    ranker: object,
    requested_strategy: str,
    exploration_fraction: float,
    audit_fraction: float,
) -> list[dict]:
    exploration_target = min(n, round(n * exploration_fraction))
    if exploration_fraction > 0 and n > 1:
        exploration_target = max(1, exploration_target)
    audit_target = min(n - exploration_target, round(n * audit_fraction))
    if audit_fraction > 0 and n - exploration_target > 1:
        audit_target = max(1, audit_target)
    uncertainty_target = n - exploration_target - audit_target
    deterministic_target = uncertainty_target + audit_target

    uncertainty_heap: list[tuple[float, int, dict]] = []
    audit_heap: list[tuple[float, int, dict]] = []
    random_heap: list[tuple[float, int, dict]] = []
    pending: list[tuple[int, dict]] = []
    seen = 0
    score_error = ""

    def consider_scored(batch: list[tuple[int, dict]]) -> None:
        nonlocal score_error
        if not batch or score_error:
            return
        batch_docs = [doc for _, doc in batch]
        try:
            _score_batch(ranker, batch_docs)
        except Exception as exc:  # a missing/corrupt optional model must not strand review
            score_error = f"{type(exc).__name__}: {exc}"
            return
        for counter, doc in batch:
            uncertainty = doc.get("uncertainty")
            if uncertainty is None:
                score_error = "ranker returned no usable uncertainty score"
                return
            high_entry = (float(uncertainty), counter, doc)
            if len(uncertainty_heap) < uncertainty_target:
                heapq.heappush(uncertainty_heap, high_entry)
            elif uncertainty_target and high_entry[0] > uncertainty_heap[0][0]:
                heapq.heapreplace(uncertainty_heap, high_entry)

            low_entry = (-float(uncertainty), counter, doc)
            low_limit = audit_target + uncertainty_target
            if len(audit_heap) < low_limit:
                heapq.heappush(audit_heap, low_entry)
            elif low_limit and low_entry[0] > audit_heap[0][0]:
                heapq.heapreplace(audit_heap, low_entry)

    random_limit = max(n, n + deterministic_target)
    for seen, doc in enumerate(docs, start=1):
        _keep_smallest_random(random_heap, doc, key=rng.random(), counter=seen, limit=random_limit)
        if not score_error:
            pending.append((seen, doc))
            if len(pending) >= _SCORE_BATCH_SIZE:
                consider_scored(pending)
                pending = []
    consider_scored(pending)

    if score_error:
        selected = [entry[2] for entry in sorted(random_heap, key=lambda entry: -entry[0])[:n]]
        propensity = min(1.0, n / seen) if seen else None
        for doc in selected:
            doc.update(
                {
                    "sampling_strategy": "uniform",
                    "sampling_component": "uniform",
                    "sampling_propensity": propensity,
                    "sampling_candidate_count": seen,
                    "sampling_requested_strategy": requested_strategy,
                    "sampling_fallback_reason": score_error,
                }
            )
        rng.shuffle(selected)
        return selected

    uncertainty_docs = [entry[2] for entry in sorted(uncertainty_heap, reverse=True)]
    selected_hashes = {doc["hash"] for doc in uncertainty_docs}
    audit_docs: list[dict] = []
    for _, _, doc in sorted(audit_heap, reverse=True):
        if doc["hash"] not in selected_hashes:
            audit_docs.append(doc)
            selected_hashes.add(doc["hash"])
            if len(audit_docs) >= audit_target:
                break

    random_docs: list[dict] = []
    random_needed = min(n - len(selected_hashes), exploration_target)
    for _, _, doc in sorted(random_heap, key=lambda entry: -entry[0]):
        if doc["hash"] not in selected_hashes:
            random_docs.append(doc)
            selected_hashes.add(doc["hash"])
            if len(random_docs) >= random_needed:
                break

    # Small candidate sets can collapse the planned components; fill any remaining slots uniformly.
    if len(selected_hashes) < min(n, seen):
        for _, _, doc in sorted(random_heap, key=lambda entry: -entry[0]):
            if doc["hash"] not in selected_hashes:
                random_docs.append(doc)
                selected_hashes.add(doc["hash"])
                if len(selected_hashes) >= min(n, seen):
                    break


    model_version = str(getattr(ranker, "model_version", "") or "")
    for component, component_docs in (
        ("uncertainty", uncertainty_docs),
        ("confidence_audit", audit_docs),
    ):
        for doc in component_docs:
            doc.update(
                {
                    "sampling_strategy": "active",
                    "sampling_component": component,
                    # Deterministic top/bottom-k selection has no meaningful random propensity.
                    "sampling_propensity": None,
                    "sampling_candidate_count": seen,
                    "sampling_requested_strategy": requested_strategy,
                    "model_version": doc.get("model_version") or model_version,
                }
            )

    random_denominator = max(1, seen - len(uncertainty_docs) - len(audit_docs))
    random_propensity = min(1.0, len(random_docs) / random_denominator)
    for doc in random_docs:
        doc.update(
            {
                "sampling_strategy": "active",
                "sampling_component": "random_exploration",
                "sampling_propensity": random_propensity,
                "sampling_candidate_count": seen,
                "sampling_requested_strategy": requested_strategy,
                "model_version": doc.get("model_version") or model_version,
            }
        )

    selected = uncertainty_docs + audit_docs + random_docs
    rng.shuffle(selected)
    return selected


def sample_review_docs(
    cid: str,
    n: int = 50,
    include_quarantine: bool = False,
    seed: int = 0,
    *,
    strategy: str = "auto",
    ranker: object | None = None,
    skip_reviewed: bool = True,
    exploration_fraction: float = DEFAULT_EXPLORATION_FRACTION,
    audit_fraction: float = DEFAULT_AUDIT_FRACTION,
) -> list[dict]:
    """Select a review queue without coupling review storage to a ranker implementation.

    A ranker is optional and only needs ``score_docs(list[dict]) -> list[Mapping]``.
    ``auto`` uses active selection when one is supplied and exact uniform sampling otherwise.
    """
    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"unknown review sampling strategy: {strategy}")
    if n <= 0:
        return []
    if not 0.0 <= exploration_fraction <= 1.0:
        raise ValueError("exploration_fraction must be between 0 and 1")
    if not 0.0 <= audit_fraction <= 1.0 or exploration_fraction + audit_fraction > 1.0:
        raise ValueError("audit_fraction must be non-negative and fit beside exploration_fraction")

    reviewed = set()
    if skip_reviewed:
        reviewed = {row["hash"] for row in db.q("SELECT hash FROM review_annotations")}
    docs = _iter_review_docs(cid, include_quarantine, reviewed)
    rng = random.Random(seed)
    use_active = strategy in {"active", "auto"} and ranker is not None
    if not use_active:
        fallback = "ranker unavailable" if strategy in {"active", "auto"} else ""
        return _uniform_sample(docs, n=n, rng=rng, requested_strategy=strategy, fallback_reason=fallback)
    return _active_sample(
        docs,
        n=n,
        rng=rng,
        ranker=ranker,
        requested_strategy=strategy,
        exploration_fraction=exploration_fraction,
        audit_fraction=audit_fraction,
    )


def _get_or_create_store(kind: str) -> str:
    row = db.q1("SELECT id FROM corpora WHERE kind=?", (kind,))
    if row:
        return row["id"]
    cid = db.new_id("cor")
    cdir = corpus_dir(cid)
    (cdir / "data").mkdir(parents=True, exist_ok=True)
    stats = {
        "docs_written": 0,
        "chars_written": 0,
        "domains": {},
        "shards": [],
        "bytes_on_disk": 0,
        "totals": {"tokens": {}},
        "review_store": kind,
    }
    write_json(cdir / "stats.json", stats)
    now = db.now_iso()
    db.execute(
        "INSERT INTO corpora(id, name, created_at, updated_at, status, source_path, stats_json, verdict, kind)"
        " VALUES(?,?,?,?,?,?,?,?,?)",
        (
            cid,
            STORE_NAMES[kind],
            now,
            now,
            "ready",
            "",
            orjson.dumps(stats).decode(),
            kind,
            kind,
        ),
    )
    return cid


def _append_docs(cid: str, docs: list[dict]) -> None:
    if not docs:
        return
    cdir = corpus_dir(cid)
    data_dir = cdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in data_dir.glob("shard-*.jsonl.zst"))
    writer = ShardWriter(data_dir, start_index=len(existing))
    domains: dict = {}
    chars = 0
    for d in docs:
        writer.write(
            {
                "text": d["text"],
                "domain": d.get("domain", "other"),
                "source": d.get("source", ""),
                "lang": d.get("lang", ""),
            }
        )
        dom = domains.setdefault(d.get("domain", "other"), {"docs": 0, "chars": 0, "tokens": {}})
        dom["docs"] += 1
        dom["chars"] += len(d["text"])
        chars += len(d["text"])
    writer.finish()

    stats_path = cdir / "stats.json"
    stats = orjson.loads(stats_path.read_bytes()) if stats_path.exists() else {}
    stats["docs_written"] = stats.get("docs_written", 0) + len(docs)
    stats["chars_written"] = stats.get("chars_written", 0) + chars
    for dom, d in domains.items():
        cur = stats.setdefault("domains", {}).setdefault(dom, {"docs": 0, "chars": 0, "tokens": {}})
        cur["docs"] += d["docs"]
        cur["chars"] += d["chars"]
    stats["shards"] = existing + writer.shards
    stats["bytes_on_disk"] = sum((data_dir / s).stat().st_size for s in stats["shards"])
    write_json(stats_path, stats)
    db.execute(
        "UPDATE corpora SET stats_json=?, updated_at=? WHERE id=?",
        (orjson.dumps(stats).decode(), db.now_iso(), cid),
    )


def _write_annotations(review_id: str, source_cid: str, annotated: list[dict]) -> str:
    PATHS.annotations.mkdir(parents=True, exist_ok=True)
    name = f"ann_{db.now_iso().replace(':', '')}_{review_id[-6:]}.jsonl.zst"
    path = PATHS.annotations / name
    comp = zstd.ZstdCompressor(level=3)
    now = db.now_iso()
    with path.open("wb") as fh, comp.stream_writer(fh) as out:
        for v in annotated:
            rec = {
                "hash": v["hash"],
                "verdict": v["verdict"],
                "corpus_id": source_cid,
                "review_id": review_id,
                "created_at": now,
                "domain": v.get("domain", ""),
                "lang": v.get("lang", ""),
                "quality": v.get("quality"),
                "source": v.get("source", ""),
                "bucket": v.get("bucket", ""),
                "sampling_strategy": v.get("sampling_strategy", "uniform"),
                "sampling_component": v.get("sampling_component", ""),
                "sampling_propensity": v.get("sampling_propensity"),
                "sampling_candidate_count": v.get("sampling_candidate_count"),
                "model_version": v.get("model_version", ""),
                "rank_score": v.get("rank_score"),
                "uncertainty": v.get("uncertainty"),
                "model_scores": v.get("model_scores", {}),
                "text": v["text"],
            }
            out.write(orjson.dumps(rec))
            out.write(b"\n")
    return str(path)


def _json_text(value: Any) -> str:
    return orjson.dumps(value, default=str).decode()


def _prepared_verdict(cid: str, value: Mapping[str, Any]) -> dict:
    verdict = value.get("verdict")
    if verdict not in {"good", "bad"}:
        raise ValueError("review verdict must be 'good' or 'bad'")
    text = value.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("review verdict requires non-empty text")
    calculated_hash = sha256_hex(text)
    supplied_hash = value.get("hash")
    if supplied_hash and supplied_hash != calculated_hash:
        raise ValueError("review document hash does not match its text")

    propensity = _finite_float(value.get("sampling_propensity"))
    if propensity is not None and not 0.0 <= propensity <= 1.0:
        raise ValueError("sampling propensity must be between 0 and 1")
    quality = _finite_float(value.get("quality"))
    rank_score = _finite_float(value.get("rank_score"))
    uncertainty = _finite_float(value.get("uncertainty"))
    model_scores = value.get("model_scores")
    if not isinstance(model_scores, Mapping):
        model_scores = {}

    metadata_keys = (
        "quality_report",
        "quarantine_reasons",
        "sampling_candidate_count",
        "sampling_requested_strategy",
        "sampling_fallback_reason",
    )
    metadata = {key: value.get(key) for key in metadata_keys if value.get(key) is not None}
    return {
        "hash": calculated_hash,
        "corpus_id": cid,
        "verdict": verdict,
        "text": text,
        "domain": str(value.get("domain") or "other"),
        "lang": str(value.get("lang") or ""),
        "quality": quality,
        "source": str(value.get("source") or ""),
        "bucket": str(value.get("bucket") or ""),
        "sampling_strategy": str(value.get("sampling_strategy") or "uniform"),
        "sampling_component": str(value.get("sampling_component") or ""),
        "sampling_propensity": propensity,
        "model_version": str(value.get("model_version") or ""),
        "rank_score": rank_score,
        "uncertainty": uncertainty,
        "model_scores_json": _json_text(dict(model_scores)),
        "metadata_json": _json_text(metadata),
    }


_LABEL_FIELDS = (
    "text",
    "domain",
    "lang",
    "quality",
    "source",
    "bucket",
    "sampling_strategy",
    "sampling_component",
    "sampling_propensity",
    "model_version",
    "rank_score",
    "uncertainty",
    "model_scores_json",
    "metadata_json",
)


def start_session(cid: str, *, sampling_strategy: str = "auto", model_version: str = "") -> str:
    if sampling_strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"unknown review sampling strategy: {sampling_strategy}")
    if not db.q1("SELECT id FROM corpora WHERE id=?", (cid,)):
        raise ValueError(f"unknown corpus: {cid}")
    review_id = db.new_id("rev")
    now = db.now_iso()
    with db.transaction() as con:
        con.execute(
            "INSERT INTO reviews(id, corpus_id, created_at, updated_at, status, sampling_strategy, model_version)"
            " VALUES(?,?,?,?,?,?,?)",
            (review_id, cid, now, now, "active", sampling_strategy, model_version),
        )
    return review_id


def _persist_verdicts(review_id: str, cid: str, verdicts: list[Mapping[str, Any]]) -> list[dict]:
    prepared = [_prepared_verdict(cid, value) for value in verdicts]
    if not prepared:
        return []
    now = db.now_iso()
    with db.transaction() as con:
        session = con.execute("SELECT corpus_id, status FROM reviews WHERE id=?", (review_id,)).fetchone()
        if session is None:
            raise ValueError(f"unknown review session: {review_id}")
        if session["corpus_id"] != cid:
            raise ValueError("review session belongs to a different corpus")
        if session["status"] != "active":
            raise ValueError("review session is already completed")

        for value in prepared:
            annotation_values = (
                value["hash"],
                value["verdict"],
                cid,
                review_id,
                now,
                *(value[field] for field in _LABEL_FIELDS),
            )
            inserted = con.execute(
                "INSERT OR IGNORE INTO review_annotations("
                "hash, verdict, corpus_id, review_id, created_at, "
                + ", ".join(_LABEL_FIELDS)
                + ") VALUES(?,?,?,?,?,"
                + ",".join("?" for _ in _LABEL_FIELDS)
                + ")",
                annotation_values,
            ).rowcount

            # Historical rows predate in-database text payloads. A later swipe can safely
            # hydrate their metadata without changing the original unique human verdict.
            if not inserted:
                assignments = ", ".join(f"{field}=?" for field in _LABEL_FIELDS)
                con.execute(
                    f"UPDATE review_annotations SET {assignments} WHERE hash=? AND text=''",
                    (*(value[field] for field in _LABEL_FIELDS), value["hash"]),
                )

            swipe_id = db.new_id("swp")
            swipe_values = (
                swipe_id,
                review_id,
                value["hash"],
                cid,
                value["verdict"],
                now,
                now,
                *(value[field] for field in _LABEL_FIELDS),
                int(bool(inserted)),
            )
            con.execute(
                "INSERT INTO review_swipes("
                "id, review_id, hash, corpus_id, verdict, created_at, updated_at, "
                + ", ".join(_LABEL_FIELDS)
                + ", is_new_annotation) VALUES(?,?,?,?,?,?,?,"
                + ",".join("?" for _ in _LABEL_FIELDS)
                + ",?) ON CONFLICT(review_id, hash) DO UPDATE SET "
                "verdict=excluded.verdict, updated_at=excluded.updated_at, "
                + ", ".join(f"{field}=excluded.{field}" for field in _LABEL_FIELDS)
                + ", is_new_annotation=MAX(review_swipes.is_new_annotation, excluded.is_new_annotation)",
                swipe_values,
            )

        con.execute(
            "UPDATE reviews SET updated_at=?, "
            "docs_reviewed=(SELECT COUNT(*) FROM review_swipes WHERE review_id=?), "
            "docs_good=(SELECT COUNT(*) FROM review_swipes WHERE review_id=? AND verdict='good'), "
            "docs_bad=(SELECT COUNT(*) FROM review_swipes WHERE review_id=? AND verdict='bad') WHERE id=?",
            (now, review_id, review_id, review_id, review_id),
        )
    return prepared


def record_verdict(review_id: str, cid: str, document: Mapping[str, Any], verdict: str) -> dict:
    """Durably record one human swipe before the UI advances to the next document."""
    prepared = _persist_verdicts(review_id, cid, [{**document, "verdict": verdict}])
    return prepared[0]


def _row_to_verdict(row: Mapping[str, Any]) -> dict:
    result = {field: row.get(field) for field in _LABEL_FIELDS if field not in {"model_scores_json", "metadata_json"}}
    result.update(
        {
            "hash": row["hash"],
            "verdict": row["verdict"],
            "corpus_id": row["corpus_id"],
            "review_id": row["review_id"],
        }
    )
    for source, target in (("model_scores_json", "model_scores"), ("metadata_json", "metadata")):
        try:
            result[target] = orjson.loads(row.get(source) or "{}")
        except orjson.JSONDecodeError:
            result[target] = {}
    result.update(result["metadata"])
    return result


def _backfill_historical_label_payloads() -> None:
    missing = db.q("SELECT hash, review_id FROM review_annotations WHERE text='' OR text IS NULL")
    wanted = {(row["hash"], row["review_id"]) for row in missing}
    if not wanted:
        return

    updates: list[dict] = []
    for path in sorted(PATHS.annotations.glob("ann_*.jsonl.zst")):
        with path.open("rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as reader:
            for line in reader.read().splitlines():
                if not line:
                    continue
                record = orjson.loads(line)
                key = (record.get("hash"), record.get("review_id"))
                if key not in wanted:
                    continue
                try:
                    updates.append(_prepared_verdict(str(record.get("corpus_id") or ""), record))
                except ValueError:
                    continue
                wanted.remove(key)
        if not wanted:
            break

    if not updates:
        return
    assignments = ", ".join(f"{field}=?" for field in _LABEL_FIELDS)
    with db.transaction() as con:
        for value in updates:
            con.execute(
                f"UPDATE review_annotations SET {assignments} WHERE hash=? AND text=''",
                (*(value[field] for field in _LABEL_FIELDS), value["hash"]),
            )


def load_review_labels() -> list[dict]:
    """Return one durable human label per document for classifier training.

    Older installations kept text in annotation archives. They are hydrated into SQLite
    on first load so later deletion/rebuilding of derived corpora cannot remove labels.
    """
    _backfill_historical_label_payloads()
    rows = db.q(
        "SELECT hash, verdict, corpus_id, review_id, created_at, "
        + ", ".join(_LABEL_FIELDS)
        + " FROM review_annotations WHERE text!='' ORDER BY created_at, hash"
    )
    labels = []
    for row in rows:
        label = _row_to_verdict(row)
        label["group"] = row["corpus_id"]
        label["created_at"] = row["created_at"]
        labels.append(label)
    return labels


def _block_future_ingest(review_id: str, bad_docs: list[dict]) -> None:
    rows = []
    for d in bad_docs:
        rows.append((d["hash"], db.now_iso(), f"review:{review_id}"))
        rows.append((norm_sha256_hex(d["text"]), db.now_iso(), f"review:{review_id}"))
    db.executemany("INSERT OR IGNORE INTO rejected_hashes(hash, added_at, source) VALUES(?,?,?)", rows)


def _completed_session_result(review_id: str) -> dict:
    row = db.q1("SELECT * FROM reviews WHERE id=?", (review_id,))
    if row is None:
        raise ValueError(f"unknown review session: {review_id}")
    counts = db.q1(
        "SELECT COUNT(*) AS total, SUM(is_new_annotation) AS claimed FROM review_swipes WHERE review_id=?",
        (review_id,),
    ) or {"total": 0, "claimed": 0}
    return {
        "review_id": review_id,
        "docs_reviewed": row["docs_reviewed"],
        "docs_good": row["docs_good"],
        "docs_bad": row["docs_bad"],
        "dupes": int(counts["total"] or 0) - int(counts["claimed"] or 0),
        "verdict": row["verdict"],
        "approved_corpus_id": row["approved_corpus_id"],
        "rejected_corpus_id": row.get("rejected_corpus_id", ""),
        "annotation_path": row.get("annotation_path", ""),
    }


def finish_session(cid: str, verdicts: list[dict] | None = None, *, review_id: str | None = None) -> dict:
    reviewed_input = [v for v in (verdicts or []) if v.get("verdict") in {"good", "bad"}]
    if review_id is None:
        first = reviewed_input[0] if reviewed_input else {}
        review_id = start_session(
            cid,
            sampling_strategy=str(first.get("sampling_requested_strategy") or first.get("sampling_strategy") or "auto"),
            model_version=str(first.get("model_version") or ""),
        )
    else:
        existing = db.q1("SELECT corpus_id, status FROM reviews WHERE id=?", (review_id,))
        if existing is None:
            raise ValueError(f"unknown review session: {review_id}")
        if existing["corpus_id"] != cid:
            raise ValueError("review session belongs to a different corpus")
        if existing["status"] == "completed":
            return _completed_session_result(review_id)

    if reviewed_input:
        _persist_verdicts(review_id, cid, reviewed_input)
    rows = db.q("SELECT * FROM review_swipes WHERE review_id=? ORDER BY created_at, id", (review_id,))
    reviewed = [_row_to_verdict(row) for row in rows]
    annotation_path = _write_annotations(review_id, cid, reviewed) if reviewed else ""
    claimed = [doc for doc, row in zip(reviewed, rows, strict=True) if row["is_new_annotation"]]
    good = [doc for doc in claimed if doc["verdict"] == "good"]
    bad = [doc for doc in claimed if doc["verdict"] == "bad"]
    dupes = len(reviewed) - len(claimed)

    approved_id = ""
    if good:
        approved_id = _get_or_create_store("approved")
        _append_docs(approved_id, good)
    rejected_id = ""
    if bad:
        rejected_id = _get_or_create_store("rejected")
        _append_docs(rejected_id, bad)
        _block_future_ingest(review_id, bad)

    verdict = ""
    if len(reviewed) >= MIN_REVIEWED:
        n_bad = sum(1 for v in reviewed if v["verdict"] == "bad")
        verdict = "rejected" if n_bad / len(reviewed) > REJECT_BAD_RATE else "approved"
    now = db.now_iso()
    with db.transaction() as con:
        if verdict:
            con.execute("UPDATE corpora SET verdict=?, updated_at=? WHERE id=?", (verdict, now, cid))
        con.execute(
            "UPDATE reviews SET updated_at=?, status='completed', docs_reviewed=?, docs_good=?, docs_bad=?, "
            "verdict=?, approved_corpus_id=?, rejected_corpus_id=?, annotation_path=?, notes=? WHERE id=?",
            (
                now,
                len(reviewed),
                len(good),
                len(bad),
                verdict,
                approved_id,
                rejected_id,
                annotation_path,
                f"annotations={annotation_path} rejected_store={rejected_id} dupes={dupes}",
                review_id,
            ),
        )
    return {
        "review_id": review_id,
        "docs_reviewed": len(reviewed),
        "docs_good": len(good),
        "docs_bad": len(bad),
        "dupes": dupes,
        "verdict": verdict,
        "approved_corpus_id": approved_id,
        "rejected_corpus_id": rejected_id,
        "annotation_path": annotation_path,
    }
