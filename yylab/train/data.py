from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Iterator

from .. import db
from ..artifacts import corpus_dir
from ..ingest.pipeline import _iter_shards

CHARS_PER_TOKEN = 3.5


def corpus_shards(corpus_id: str) -> list[str]:
    cdir = corpus_dir(corpus_id) / "data"
    return sorted(p.name for p in cdir.glob("shard-*.jsonl.zst"))


def iter_corpus_docs(
    corpus_ids: list[str],
    domains: list[str] | None = None,
    shuffle_buffer: int = 2048,
    seed: int = 0,
) -> Iterator[dict]:
    domain_set = set(domains) if domains else None

    def stream() -> Iterator[dict]:
        for cid in corpus_ids:
            shards = corpus_shards(cid)
            if not shards:
                continue
            for line in _iter_shards(corpus_dir(cid) / "data", shards):
                if domain_set is None or line.get("domain") in domain_set:
                    yield line

    if shuffle_buffer <= 1:
        yield from stream()
        return
    rng = random.Random(seed)
    buf: list[dict] = []
    for item in stream():
        buf.append(item)
        if len(buf) >= shuffle_buffer:
            i = rng.randrange(len(buf))
            yield buf[i]
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
        tokenize_chunk: int = 64,
    ) -> None:
        self.docs = docs
        self.tokenize = tokenize
        self.seq_len = seq_len
        self.eos = [eos_token_id] if eos_token_id is not None else []
        self.micro_batch = micro_batch
        self.max_batches = max_batches
        self.tokenize_chunk = tokenize_chunk
        self.sequences_made = 0
        self.tokens_seen = 0
        self.docs_seen = 0

    def __iter__(self) -> Iterator[dict]:
        carry: list[int] = []
        batch: list[list[int]] = []
        pending_texts: list[str] = []

        def flush_texts() -> None:
            nonlocal carry
            if not pending_texts:
                return
            for ids in self.tokenize(pending_texts):
                self.tokens_seen += len(ids)
                carry.extend(ids)
                carry.extend(self.eos)
            pending_texts.clear()

        def maybe_yield() -> dict | None:
            if len(batch) >= self.micro_batch:
                out = {"input_ids": batch[: self.micro_batch]}
                del batch[: self.micro_batch]
                return out
            return None

        for doc in self.docs:
            pending_texts.append(doc["text"])
            self.docs_seen += 1
            if len(pending_texts) >= self.tokenize_chunk:
                flush_texts()
            while len(carry) >= self.seq_len:
                batch.append(carry[: self.seq_len])
                carry = carry[self.seq_len :]
                self.sequences_made += 1
                out = maybe_yield()
                if out is not None:
                    yield out
                    if self.max_batches is not None and self.sequences_made // self.micro_batch >= self.max_batches:
                        return

        flush_texts()
        while len(carry) >= self.seq_len:
            batch.append(carry[: self.seq_len])
            carry = carry[self.seq_len :]
            self.sequences_made += 1
            out = maybe_yield()
            if out is not None:
                yield out
                if self.max_batches is not None and self.sequences_made // self.micro_batch >= self.max_batches:
                    return


def estimate_steps(
    corpus_ids: list[str],
    domains: list[str] | None,
    epochs: int,
    seq_len: int,
    micro_batch: int,
    grad_accum: int,
) -> int:
    total_chars = 0
    for cid in corpus_ids:
        row = db.q1("SELECT stats_json FROM corpora WHERE id=?", (cid,))
        if not row:
            continue
        import orjson

        stats = orjson.loads(row["stats_json"] or b"{}")
        if domains:
            total_chars += sum(
                d.get("chars", 0) for dom, d in stats.get("domains", {}).items() if dom in set(domains)
            )
        else:
            total_chars += stats.get("chars_written", 0)
    est_tokens = total_chars / CHARS_PER_TOKEN
    per_step = seq_len * micro_batch * grad_accum
    return max(1, math.ceil(epochs * est_tokens / max(per_step, 1)))
