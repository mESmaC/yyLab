"""Repeating, exactly weighted causal windows from existing yyLab token caches.

Only explicit training records are read. Each source keeps its own shuffled
document stream and one-token overlap, and repeats independently when exhausted.
Source choice uses shuffled exact-ratio cycles, never exhaustion-dependent mixing.
TokenStore owns checksum verification and bounded shard handles; no text is copied.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import operator
import random
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from .data import PREPARED_FIELD
from .prepared import TokenStore, tokenizer_fingerprint


def _positive(value: int, name: str) -> int:
    value = operator.index(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class _SourceStream:
    def __init__(self, corpus_id, store, *, seed, shuffle_buffer, read_chunk_tokens, eos, vocab_size):
        self.corpus_id, self.store = corpus_id, store
        self.rng = random.Random(f"cached-spalmer:{seed}:source:{corpus_id}")
        self.shuffle_buffer = shuffle_buffer
        self.read_chunk_tokens = read_chunk_tokens
        self.eos, self.vocab_size = eos, vocab_size
        self.train_documents = 0
        self.train_content_tokens = 0
        self.excluded_documents = 0
        records = store.records()
        try:
            for record in records:
                if record.get("split") != "train":
                    self.excluded_documents += 1
                    continue
                prepared = record[PREPARED_FIELD]
                if "chat" in prepared or "labels" in prepared or "mask" in prepared:
                    raise ValueError(f"cached source {corpus_id} contains special supervision")
                self.train_documents += 1
                for unit in prepared["units"]:
                    for shard, offset, count in unit.spans:
                        if any(type(value) is not int or value < 0 for value in (shard, offset, count)):
                            raise ValueError(f"invalid token span in {corpus_id}")
                        self.train_content_tokens += count
        finally:
            records.close()
        if not self.train_documents or not self.train_content_tokens:
            raise ValueError(f"cached source {corpus_id} has no nonempty explicit train split")
        self.docs_seen = 0
        self.epochs_started = 0
        self.epochs_completed = 0
        self.content_tokens_read = 0
        self.stream_tokens_consumed = 0
        self.served_sequences = 0
        self.served_target_tokens = 0
        self.last_token = None
        self.tokens = self._tokens()

    def _shuffled_records(self):
        records = self.store.records()
        selected = (record for record in records if record.get("split") == "train")
        try:
            buffer = list(itertools.islice(selected, self.shuffle_buffer))
            for record in selected:
                index = self.rng.randrange(len(buffer))
                yield buffer[index]
                buffer[index] = record
            while buffer:
                yield buffer.pop(self.rng.randrange(len(buffer)))
        finally:
            selected.close()
            records.close()

    def _tokens(self):
        while True:
            self.epochs_started += 1
            records = self._shuffled_records()
            try:
                for record in records:
                    self.docs_seen += 1
                    for unit in record[PREPARED_FIELD]["units"]:
                        for shard, offset, count in unit.spans:
                            for start in range(0, count, self.read_chunk_tokens):
                                take = min(count - start, self.read_chunk_tokens)
                                ids = self.store.read([[shard, offset + start, take]], "tokens", 4)
                                if any(token < 0 or token >= self.vocab_size for token in ids):
                                    raise ValueError(f"out-of-vocabulary cached token in {self.corpus_id}")
                                self.content_tokens_read += len(ids)
                                yield from ids
                    # Units are tokenizer chunks of one document, not boundaries.
                    yield self.eos
            finally:
                records.close()
            self.epochs_completed += 1

    def window(self, length):
        first = self.last_token
        consumed = length
        if first is None:
            first = next(self.tokens)
            consumed += 1
        row = [first, *itertools.islice(self.tokens, length)]
        if len(row) != length + 1:
            raise RuntimeError(f"repeating cached source ended unexpectedly: {self.corpus_id}")
        self.last_token = row[-1]
        self.stream_tokens_consumed += consumed
        self.served_sequences += 1
        self.served_target_tokens += length
        return row

    def stats(self):
        return {
            "directory": str(self.store.root),
            "train_documents": self.train_documents,
            "train_content_tokens": self.train_content_tokens,
            "excluded_nontrain_documents": self.excluded_documents,
            "docs_seen": self.docs_seen,
            "epochs_started": self.epochs_started,
            "epochs_completed": self.epochs_completed,
            "content_tokens_read": self.content_tokens_read,
            "stream_tokens_consumed": self.stream_tokens_consumed,
            "served_sequences": self.served_sequences,
            "served_target_tokens": self.served_target_tokens,
        }

    def close(self):
        self.tokens.close()
        self.store.close()


class CachedSPALMERBatchSource:
    """Native CausalBatch source with exact-ratio repetition and replayable state.

    ``cache_directories`` and ``weights`` use registered corpus IDs as keys.
    Weights are exact decimal/rational values: 70/15/15 produces a shuffled
    twenty-row cycle with fourteen/three/three rows. Every row has N targets,
    so completed cycles preserve the requested target-token ratios exactly.
    A partial final cycle is reported through actual served target counts.

    At most ``shuffle_buffer`` record metadata entries plus one replacement
    are retained per source; token payload reads are bounded by
    ``read_chunk_tokens``. No full document or corpus token list is retained.
    The caller owns this single-threaded source and must close it.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        cache_directories: Mapping[str, Path],
        *,
        weights: Mapping[str, float],
        tokenizer: Any,
        seed: int = 42,
        shuffle_buffer: int = 128,
        read_chunk_tokens: int = 16384,
        causal_batch_type: type | None = None,
    ):
        if not cache_directories or set(cache_directories) != set(weights):
            raise ValueError("cache directories and positive weights must have the same corpus IDs")
        self._directories = {str(cid): Path(path).resolve(strict=True) for cid, path in cache_directories.items()}
        self._weights = dict(weights)
        self._tokenizer = tokenizer
        self.seed = operator.index(seed)
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        self.shuffle_buffer = _positive(shuffle_buffer, "shuffle_buffer")
        self.read_chunk_tokens = _positive(read_chunk_tokens, "read_chunk_tokens")
        self._causal_batch_type = causal_batch_type
        self.tokenizer_fingerprint = tokenizer_fingerprint(tokenizer)
        raw = getattr(tokenizer, "tokenizer", tokenizer)
        vocabulary = getattr(tokenizer, "vocab_size", None)
        self.vocab_size = _positive(vocabulary if vocabulary is not None else len(raw), "vocab_size")
        special = getattr(tokenizer, "special_tokens", None)
        eos = getattr(special, "eos_id", None)
        if eos is None:
            eos = getattr(raw, "eos_token_id", None)
        if eos is None:
            eos = getattr(special, "eod_id", None)
        if eos is None or not 0 <= operator.index(eos) < self.vocab_size:
            raise ValueError("cached causal packing requires a valid tokenizer EOS/EOD identity")
        self.eos_token_id = operator.index(eos)
        fractions = {cid: Fraction(str(weights[cid])) for cid in self._directories}
        if any(weight <= 0 for weight in fractions.values()):
            raise ValueError("every source weight must be positive")
        total = sum(fractions.values())
        fractions = {cid: value / total for cid, value in fractions.items()}
        cycle_size = math.lcm(*(value.denominator for value in fractions.values()))
        if cycle_size > 10000:
            raise ValueError("weight ratios require more than 10000 rows per exact cycle")
        self.cycle_counts = {cid: int(value * cycle_size) for cid, value in fractions.items()}
        self._schedule_rng = random.Random(f"cached-spalmer:{self.seed}:mixture")
        self._cycle = []
        self._cycle_position = 0
        self.cycles_completed = 0
        self.consumed_batches = 0
        self.discarded_incomplete_batches = 0
        self.delivered_sequences = 0
        self.delivered_input_tokens = 0
        self.delivered_target_tokens = 0
        self._shape = None
        self._closed = False
        self._failed = False
        self._sources = {}
        manifests = {}
        try:
            for cid, path in self._directories.items():
                store = TokenStore(path)
                try:
                    identity = store.manifest.get("identity", {})
                    if identity.get("source", {}).get("corpus_id") != cid:
                        raise ValueError(f"prepared cache corpus identity does not match {cid}")
                    if identity.get("tokenizer") != self.tokenizer_fingerprint:
                        raise ValueError(f"prepared cache tokenizer does not match actual tokenizer: {cid}")
                    if identity.get("objective") != "causal_lm" or any(
                        name.startswith("mask-") for name in store.files
                    ):
                        raise ValueError(f"prepared cache must have ordinary causal supervision: {cid}")
                    manifests[cid] = _file_hash(path / "manifest.json")
                    self._sources[cid] = _SourceStream(
                        cid, store, seed=self.seed, shuffle_buffer=self.shuffle_buffer,
                        read_chunk_tokens=self.read_chunk_tokens, eos=self.eos_token_id,
                        vocab_size=self.vocab_size,
                    )
                except BaseException:
                    store.close()
                    raise
        except BaseException:
            self.close()
            raise
        self._recipe = {
            "source_order": list(self._directories),
            "directories": {cid: str(path) for cid, path in self._directories.items()},
            "manifest_sha256": manifests,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "seed": self.seed,
            "shuffle_buffer": self.shuffle_buffer,
            "read_chunk_tokens": self.read_chunk_tokens,
            "cycle_counts": self.cycle_counts,
        }
        self._recipe_hash = hashlib.sha256(json.dumps(self._recipe, sort_keys=True).encode()).hexdigest()

    @property
    def docs_seen(self):
        return sum(source.docs_seen for source in self._sources.values())

    @property
    def tokens_seen(self):
        """Delivered input positions, including EOS and the repeated context token."""
        return self.delivered_input_tokens

    def _next_source(self):
        if self._cycle_position == len(self._cycle):
            self._cycle = [cid for cid, count in self.cycle_counts.items() for _ in range(count)]
            self._schedule_rng.shuffle(self._cycle)
            self._cycle_position = 0
        cid = self._cycle[self._cycle_position]
        self._cycle_position += 1
        if self._cycle_position == len(self._cycle):
            self.cycles_completed += 1
        return cid

    def next_batch(self, *, batch_size: int, sequence_length: int):
        if self._closed or self._failed:
            raise RuntimeError("cached batch source is closed or failed; restore state before reuse")
        shape = (_positive(batch_size, "batch_size"), _positive(sequence_length, "sequence_length"))
        if self._shape is not None and shape != self._shape:
            raise ValueError("cached batch shape cannot change after the first batch")
        self._shape = shape
        try:
            sources = [self._next_source() for _ in range(batch_size)]
            rows = [self._sources[cid].window(sequence_length) for cid in sources]
            import torch

            if self._causal_batch_type is None:
                from spalmer.training.engine import CausalBatch

                self._causal_batch_type = CausalBatch
            batch = self._causal_batch_type(input_ids=torch.tensor(rows, dtype=torch.long), strata=tuple(sources))
        except BaseException:
            self._failed = True
            raise
        self.consumed_batches += 1
        self.delivered_sequences += batch_size
        self.delivered_input_tokens += batch_size * (sequence_length + 1)
        self.delivered_target_tokens += batch_size * sequence_length
        return batch

    def data_stats(self):
        sources = {cid: source.stats() for cid, source in self._sources.items()}
        rendered = sum(source.stream_tokens_consumed for source in self._sources.values())
        return {
            "schema": "yylab-cached-spalmer-v1",
            "split": "train",
            "objective": "causal_lm",
            "repeats_sources_independently": True,
            "docs_seen": self.docs_seen,
            "rendered_tokens": rendered,
            "sequences": self.delivered_sequences,
            "consumed_batches": self.consumed_batches,
            "cycle_counts": dict(self.cycle_counts),
            "cycles_completed": self.cycles_completed,
            "rows_into_cycle": self._cycle_position % len(self._cycle) if self._cycle else 0,
            "recipe": self._recipe,
            "sources": sources,
            "per_corpus": sources,
            "prepared_tokens": {
                cid: {
                    "directory": str(source.store.root),
                    "documents": source.store.manifest["documents"],
                    "tokens": source.store.manifest["tokens"],
                    "train_documents": source.train_documents,
                    "train_content_tokens": source.train_content_tokens,
                }
                for cid, source in self._sources.items()
            },
            "tokens": {
                "rendered_token_positions": rendered,
                "packing_replay_inputs": self.delivered_input_tokens - rendered,
                "forwarded_model_slots": self.delivered_input_tokens,
                "forwarded_real_inputs": self.delivered_input_tokens,
                "model_target_slots": self.delivered_target_tokens,
                "optimizer_supervised_targets": self.delivered_target_tokens,
                "padding_inputs": 0,
                "ignored_target_slots": 0,
            },
        }

    def state_dict(self):
        if self._failed:
            raise RuntimeError("cannot checkpoint a partially failed cached batch")
        return {
            "version": self.STATE_VERSION,
            "recipe_sha256": self._recipe_hash,
            "consumed_batches": self.consumed_batches,
            "shape": list(self._shape) if self._shape is not None else None,
        }

    def load_state_dict(self, state: Mapping[str, Any]):
        if state.get("version") != self.STATE_VERSION or state.get("recipe_sha256") != self._recipe_hash:
            raise ValueError("incompatible cached batch-source state or source identity")
        consumed = operator.index(state.get("consumed_batches", -1))
        shape = state.get("shape")
        if consumed < 0 or (consumed and (not isinstance(shape, (list, tuple)) or len(shape) != 2)):
            raise ValueError("invalid cached batch replay position")
        if shape is not None:
            shape = tuple(_positive(value, "saved batch shape") for value in shape)
            if len(shape) != 2:
                raise ValueError("invalid saved batch shape")
        old_recipe_hash = self._recipe_hash
        self.close()
        self.__init__(
            self._directories, weights=self._weights, tokenizer=self._tokenizer, seed=self.seed,
            shuffle_buffer=self.shuffle_buffer, read_chunk_tokens=self.read_chunk_tokens,
            causal_batch_type=self._causal_batch_type,
        )
        if self._recipe_hash != old_recipe_hash:
            self.close()
            raise ValueError("cached source manifest changed before replay")
        for _ in range(consumed):
            self.next_batch(batch_size=shape[0], sequence_length=shape[1])
        if shape is not None:
            self._shape = shape

    def close(self):
        for source in self._sources.values():
            source.close()
        self._closed = True


__all__ = ["CachedSPALMERBatchSource"]
