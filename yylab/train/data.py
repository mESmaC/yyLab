from __future__ import annotations

import math
import random
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from .. import db
from ..artifacts import corpus_dir
from ..ingest.pipeline import _iter_shards

CHARS_PER_TOKEN = 3.5
DEFAULT_SHUFFLE_BUFFER = 128
DEFAULT_SHUFFLE_BUFFER_BYTES = 64 * 2**20
DEFAULT_TOKENIZE_CHUNK = 8
# Keep each fast-tokenizer input comfortably below the very large raw chunks
# accepted by corpus ingestion.  Packing still joins every returned token into
# fixed model sequences, so this bounds transient CPU/RAM use without dropping
# corpus content.
DEFAULT_TEXT_CHUNK_CHARS = 65_536
CORPUS_MIX_CONCATENATE = "concatenate"
CORPUS_MIX_ROUND_ROBIN = "round_robin"
CORPUS_MIX_MODES = frozenset({CORPUS_MIX_CONCATENATE, CORPUS_MIX_ROUND_ROBIN})


def normalize_corpus_ids(corpus_ids: object) -> list[str]:
    """Return a deterministic, ordered, duplicate-free corpus selection.

    Older base-run payloads may contain one corpus ID directly instead of a
    list. Treating that string as an iterable would stream one-character IDs,
    so normalize it at the data boundary as well as in the worker. JSON lists
    keep their user-selected order; unordered Python sets are sorted so tests
    and programmatic callers remain reproducible.
    """

    if corpus_ids is None:
        values: Iterable[object] = ()
    elif isinstance(corpus_ids, str):
        values = (corpus_ids,)
    elif isinstance(corpus_ids, (set, frozenset)):
        values = sorted(corpus_ids, key=str)
    elif isinstance(corpus_ids, Iterable):
        values = corpus_ids
    else:
        raise TypeError("corpora must be a corpus ID or an iterable of corpus IDs")

    selected: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("every corpus ID must be a string")
        corpus_id = value.strip()
        if corpus_id and corpus_id not in seen:
            selected.append(corpus_id)
            seen.add(corpus_id)
    return selected


def normalize_corpus_mix(value: object) -> str:
    """Resolve an explicit mixing strategy, defaulting old runs to concatenation."""

    mode = str(value or CORPUS_MIX_CONCATENATE).strip().casefold()
    if mode not in CORPUS_MIX_MODES:
        choices = ", ".join(sorted(CORPUS_MIX_MODES))
        raise ValueError(f"unknown corpus mix mode {mode!r}; expected one of: {choices}")
    return mode


def corpus_data_provenance(
    corpus_ids: object,
    domains: list[str] | None = None,
    *,
    corpus_mix: object = CORPUS_MIX_CONCATENATE,
) -> dict[str, Any]:
    """Snapshot the selected corpus identities and aggregate intake statistics."""

    selected_ids = normalize_corpus_ids(corpus_ids)
    selected_domains = sorted({str(domain) for domain in domains or () if str(domain)})
    domain_set = set(selected_domains)
    corpora: list[dict[str, Any]] = []
    total_docs = 0
    total_chars = 0
    for corpus_id in selected_ids:
        row = db.q1(
            "SELECT id, name, source_path, status, stats_json FROM corpora WHERE id=?",
            (corpus_id,),
        )
        entry: dict[str, Any] = {"id": corpus_id}
        if row:
            import orjson

            stats = orjson.loads(row.get("stats_json") or b"{}")
            domain_stats = stats.get("domains", {})
            chosen = [
                values
                for domain, values in domain_stats.items()
                if not domain_set or domain in domain_set
            ]
            docs = sum(int(values.get("docs", 0)) for values in chosen)
            chars = sum(int(values.get("chars", 0)) for values in chosen)
            if not domain_stats and not domain_set:
                docs = int(stats.get("docs_written", 0))
                chars = int(stats.get("chars_written", 0))
            entry.update(
                name=str(row.get("name") or corpus_id),
                source_path=str(row.get("source_path") or ""),
                status=str(row.get("status") or ""),
                estimated_docs=docs,
                estimated_chars=chars,
            )
            total_docs += docs
            total_chars += chars
        else:
            # Programmatic and legacy runs may point at an artifact whose DB row
            # has since been removed. Keep its ID in the durable provenance; the
            # stream will still fail naturally if no shards exist.
            entry["missing_catalog_entry"] = True
        corpora.append(entry)
    return {
        "schema_version": 1,
        "corpus_ids": selected_ids,
        "corpus_count": len(selected_ids),
        "corpus_mix": normalize_corpus_mix(corpus_mix),
        "domains": selected_domains,
        "estimated_docs": total_docs,
        "estimated_chars": total_chars,
        "corpora": corpora,
    }


def tokenizer_stats_key(model_type: str | None) -> str | None:
    """Return the closest intake-tokenizer counter for a model family."""

    normalized = (model_type or "").casefold().replace("_", "").replace("-", "")
    if normalized.startswith("qwen"):
        return "qwen25"
    if normalized.startswith("llama"):
        return "llama3"
    if normalized.startswith("mistral"):
        return "mistral"
    if normalized.startswith("phi3"):
        return "phi3"
    if normalized.startswith("gemma"):
        return "gemma3"
    if normalized.startswith("gpt2"):
        return "gpt2"
    return None


@dataclass(frozen=True)
class ExpertDataPlan:
    """One isolated expert's deterministic slice of the selected corpora."""

    name: str
    domains: tuple[str, ...]
    estimated_chars: int
    shard_index: int | None = None
    num_shards: int | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domains": list(self.domains),
            "estimated_chars": self.estimated_chars,
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
        }

    @classmethod
    def from_dict(cls, value: dict) -> ExpertDataPlan:
        return cls(
            name=str(value["name"]),
            domains=tuple(value.get("domains") or ()),
            estimated_chars=max(0, int(value.get("estimated_chars") or 0)),
            shard_index=(None if value.get("shard_index") is None else int(value["shard_index"])),
            num_shards=(None if value.get("num_shards") is None else int(value["num_shards"])),
        )


def corpus_shards(corpus_id: str) -> list[str]:
    cdir = corpus_dir(corpus_id) / "data"
    return sorted(p.name for p in cdir.glob("shard-*.jsonl.zst"))


def iter_corpus_docs(
    corpus_ids: object,
    domains: list[str] | None = None,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
    seed: int = 0,
    shard_index: int | None = None,
    num_shards: int | None = None,
    shuffle_buffer_bytes: int = DEFAULT_SHUFFLE_BUFFER_BYTES,
    corpus_mix: str = CORPUS_MIX_CONCATENATE,
) -> Iterator[dict]:
    if (shard_index is None) != (num_shards is None):
        raise ValueError("shard_index and num_shards must be supplied together")
    if num_shards is not None and (num_shards < 1 or not 0 <= shard_index < num_shards):
        raise ValueError("document shard must be within [0, num_shards)")
    selected_corpora = normalize_corpus_ids(corpus_ids)
    mix_mode = normalize_corpus_mix(corpus_mix)
    domain_set = set(domains) if domains else None

    def corpus_stream(corpus_id: str) -> Iterator[dict]:
        shards = corpus_shards(corpus_id)
        if not shards:
            return
        for line in _iter_shards(corpus_dir(corpus_id) / "data", shards):
            if domain_set is None or line.get("domain") in domain_set:
                yield line

    def shuffled_stream(
        items: Iterator[dict],
        *,
        rng: random.Random,
        max_items: int,
        max_bytes: int,
    ) -> Iterator[dict]:
        if max_items <= 1:
            yield from items
            return
        buf: list[dict] = []
        buffered_bytes = 0
        for item in items:
            buf.append(item)
            buffered_bytes += sys.getsizeof(item) + sys.getsizeof(item.get("text", ""))
            if len(buf) >= max_items or buffered_bytes >= max_bytes:
                index = rng.randrange(len(buf))
                selected = buf[index]
                buffered_bytes -= sys.getsizeof(selected) + sys.getsizeof(
                    selected.get("text", "")
                )
                yield selected
                buf[index] = buf[-1]
                buf.pop()
        rng.shuffle(buf)
        yield from buf

    def raw_stream() -> Iterator[dict]:
        if mix_mode == CORPUS_MIX_CONCATENATE:
            for corpus_id in selected_corpora:
                yield from corpus_stream(corpus_id)
            return

        # Round-robin at the document boundary rather than opening every corpus
        # into memory. Exhausted inputs fall out while remaining iterators
        # continue, so an epoch still visits every selected document exactly
        # once and keeps the existing full-epoch step estimate valid.
        source_count = max(1, len(selected_corpora))
        per_source_items = max(1, shuffle_buffer // source_count)
        per_source_bytes = max(1, shuffle_buffer_bytes // source_count)
        active = [
            iter(
                shuffled_stream(
                    corpus_stream(corpus_id),
                    rng=random.Random(
                        seed if source_count == 1 else f"{seed}:{index}:{corpus_id}"
                    ),
                    max_items=per_source_items,
                    max_bytes=per_source_bytes,
                )
            )
            for index, corpus_id in enumerate(selected_corpora)
        ]
        while active:
            remaining: list[Iterator[dict]] = []
            for iterator in active:
                try:
                    yield next(iterator)
                except StopIteration:
                    continue
                remaining.append(iterator)
            active = remaining

    def stream() -> Iterator[dict]:
        eligible_index = 0
        for line in raw_stream():
            selected = num_shards is None or eligible_index % num_shards == shard_index
            eligible_index += 1
            if selected:
                yield line

    if mix_mode == CORPUS_MIX_ROUND_ROBIN:
        # Each source was shuffled independently above; preserving the strict
        # interleave here prevents a later global buffer from reclustering it.
        yield from stream()
        return
    if shuffle_buffer <= 1:
        yield from stream()
        return
    rng = random.Random(seed)
    buf: list[dict] = []
    buffered_bytes = 0
    for item in stream():
        buf.append(item)
        buffered_bytes += sys.getsizeof(item) + sys.getsizeof(item.get("text", ""))
        if len(buf) >= shuffle_buffer or buffered_bytes >= shuffle_buffer_bytes:
            i = rng.randrange(len(buf))
            selected = buf[i]
            buffered_bytes -= sys.getsizeof(selected) + sys.getsizeof(selected.get("text", ""))
            yield selected
            buf[i] = buf[-1]
            buf.pop()
    rng.shuffle(buf)
    yield from buf


class PackedBatches:
    def __init__(
        self,
        docs: Iterable[dict],
        tokenize: Callable[[list[str]], list[list[int]]],
        seq_len: int,
        eos_token_id: int,
        micro_batch: int,
        max_batches: int | None = None,
        tokenize_chunk: int = DEFAULT_TOKENIZE_CHUNK,
        max_text_chars: int = DEFAULT_TEXT_CHUNK_CHARS,
    ) -> None:
        if seq_len < 1 or micro_batch < 1 or tokenize_chunk < 1 or max_text_chars < 1:
            raise ValueError("sequence, batch, tokenization chunk, and text chunk sizes must be positive")
        self.docs = docs
        self.tokenize = tokenize
        self.seq_len = seq_len
        self.eos = [eos_token_id] if eos_token_id is not None else []
        self.micro_batch = micro_batch
        self.max_batches = max_batches
        self.tokenize_chunk = tokenize_chunk
        self.max_text_chars = max_text_chars
        self.sequences_made = 0
        self.tokens_seen = 0
        self.docs_seen = 0

    def __iter__(self) -> Iterator[dict]:
        carry: list[int] = []
        carry_start = 0
        batch: list[list[int]] = []
        # Keep the document boundary alongside each bounded raw-text chunk.
        # Tokenization may happen in several calls for one long document, but
        # packing must still add exactly one EOS for that original document.
        pending_texts: list[tuple[str, bool]] = []
        batches_made = 0

        def flush_texts() -> Iterator[dict]:
            nonlocal carry, carry_start, batches_made
            if not pending_texts:
                return
            texts = [text for text, _is_doc_end in pending_texts]
            boundaries = [is_doc_end for _text, is_doc_end in pending_texts]
            for ids, is_doc_end in zip(self.tokenize(texts), boundaries, strict=True):
                self.tokens_seen += len(ids)
                carry.extend(ids)
                if is_doc_end:
                    carry.extend(self.eos)
                while len(carry) - carry_start >= self.seq_len:
                    end = carry_start + self.seq_len
                    batch.append(carry[carry_start:end])
                    carry_start = end
                    self.sequences_made += 1
                    if len(batch) >= self.micro_batch:
                        yield {"input_ids": batch[: self.micro_batch]}
                        del batch[: self.micro_batch]
                        batches_made += 1
                        if self.max_batches is not None and batches_made >= self.max_batches:
                            pending_texts.clear()
                            return
                # Avoid repeatedly copying the whole remainder for every sequence,
                # while still releasing consumed token references on long documents.
                if carry_start and (carry_start >= 65_536 or carry_start * 2 >= len(carry)):
                    carry = carry[carry_start:]
                    carry_start = 0
            pending_texts.clear()

        for doc in self.docs:
            self.docs_seen += 1
            text = doc["text"]
            for start in range(0, max(len(text), 1), self.max_text_chars):
                end = min(start + self.max_text_chars, len(text))
                pending_texts.append((text[start:end], end >= len(text)))
                if len(pending_texts) >= self.tokenize_chunk:
                    yield from flush_texts()
                    if self.max_batches is not None and batches_made >= self.max_batches:
                        return

        yield from flush_texts()


def _corpus_domain_chars(corpus_ids: object, domains: list[str] | None = None) -> dict[str, int]:
    allowed = set(domains) if domains else None
    totals: dict[str, int] = {}
    for cid in normalize_corpus_ids(corpus_ids):
        row = db.q1("SELECT stats_json FROM corpora WHERE id=?", (cid,))
        if not row:
            continue
        import orjson

        stats = orjson.loads(row["stats_json"] or b"{}")
        for domain, values in stats.get("domains", {}).items():
            if allowed is None or domain in allowed:
                totals[domain] = totals.get(domain, 0) + int(values.get("chars", 0))
    return {domain: chars for domain, chars in totals.items() if chars > 0}


def plan_expert_data(
    corpus_ids: object, num_experts: int, domains: list[str] | None = None
) -> list[ExpertDataPlan]:
    """Build reproducible expert data slices without loading corpus documents.

    When there are enough labelled domains, whole domains are greedily balanced
    across experts. Otherwise the selected stream is split deterministically by
    document so every expert still receives an isolated slice.
    """

    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    totals = _corpus_domain_chars(corpus_ids, domains)
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) >= num_experts:
        groups = [{"domains": [], "chars": 0} for _ in range(num_experts)]
        for domain, chars in ordered:
            target = min(range(num_experts), key=lambda index: (groups[index]["chars"], index))
            groups[target]["domains"].append(domain)
            groups[target]["chars"] += chars
        plans = []
        for index, group in enumerate(groups):
            assigned = tuple(sorted(group["domains"]))
            label = "+".join(assigned) or f"expert-{index + 1:02d}"
            plans.append(ExpertDataPlan(label, assigned, int(group["chars"])))
        return plans

    selected_domains = tuple(sorted(totals))
    total_chars = sum(totals.values())
    return [
        ExpertDataPlan(
            name=f"shard-{index + 1:02d}",
            domains=selected_domains,
            estimated_chars=math.ceil(total_chars / num_experts),
            shard_index=index,
            num_shards=num_experts,
        )
        for index in range(num_experts)
    ]


def allocate_expert_steps(total_steps: int, plans: list[ExpertDataPlan]) -> list[int]:
    """Allocate one global budget proportionally while giving every expert work."""

    if total_steps < len(plans):
        raise ValueError("max_steps must be at least the number of experts")
    if not plans:
        return []
    remaining = total_steps - len(plans)
    weights = [max(0, plan.estimated_chars) for plan in plans]
    weight_sum = sum(weights)
    if weight_sum == 0:
        weights = [1] * len(plans)
        weight_sum = len(plans)
    raw = [remaining * weight / weight_sum for weight in weights]
    extra = [math.floor(value) for value in raw]
    leftovers = remaining - sum(extra)
    order = sorted(range(len(plans)), key=lambda index: (-(raw[index] - extra[index]), index))
    for index in order[:leftovers]:
        extra[index] += 1
    return [value + 1 for value in extra]


def estimate_steps(
    corpus_ids: object,
    domains: list[str] | None,
    epochs: int,
    seq_len: int,
    micro_batch: int,
    grad_accum: int,
    token_stats_key: str | None = None,
) -> int:
    """Estimate optimizer steps, preferring recorded tokenizer counts.

    Intake records per-domain counts for several tokenizer families.  A model
    page can select the closest compatible family (for example ``qwen25`` for
    Qwen2/Qwen3); older corpora transparently retain the character fallback.
    """

    estimated_tokens = 0.0
    domain_set = set(domains or ())
    for cid in normalize_corpus_ids(corpus_ids):
        row = db.q1("SELECT stats_json FROM corpora WHERE id=?", (cid,))
        if not row:
            continue
        import orjson

        stats = orjson.loads(row["stats_json"] or b"{}")
        selected = [
            values
            for domain, values in stats.get("domains", {}).items()
            if not domain_set or domain in domain_set
        ]
        if selected:
            for values in selected:
                recorded = (values.get("tokens") or {}).get(token_stats_key) if token_stats_key else None
                estimated_tokens += (
                    float(recorded)
                    if recorded is not None
                    else float(values.get("chars", 0)) / CHARS_PER_TOKEN
                )
        elif not domain_set:
            recorded = (stats.get("totals", {}).get("tokens", {}) or {}).get(token_stats_key)
            estimated_tokens += (
                float(recorded)
                if recorded is not None
                else float(stats.get("chars_written", 0)) / CHARS_PER_TOKEN
            )
    per_step = seq_len * micro_batch * grad_accum
    return max(1, math.ceil(epochs * estimated_tokens / max(per_step, 1)))
