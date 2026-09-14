"""Bounded, array-based causal packing for Sprint's on-disk row cache.

The emitted stream and final accounting match ``PackedBatches`` with one row
per batch. Only the transport changes: many complete rows share one contiguous
array, so callers can validate, convert and write them in a single operation.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

from .data import (
    CORPUS_ID_FIELD,
    DEFAULT_TEXT_CHUNK_CHARS,
    DEFAULT_TOKENIZE_CHUNK,
    PREPARED_FIELD,
    PackedBatches,
)

MAX_TOKENIZATION_TEXT_CHARS = 8 * 2**20


def make_sprint_tokenize(tokenizer: Any) -> Callable[[list[str]], list[list[int]]]:
    """Return an IDs-only encoder without mutating tokenizer cache identity.

    The generic HF fast tokenizer delegates directly to the Rust backend. Its
    IDs-only variant can skip offset bookkeeping and the HF output wrappers.
    Custom tokenizers keep their original call path, including any overrides.
    """

    def standard(texts: list[str]) -> list[list[int]]:
        return tokenizer(
            texts, add_special_tokens=False, return_attention_mask=False, verbose=False
        )["input_ids"]

    from transformers import PreTrainedTokenizerFast

    # Restrict the shortcut to the generic implementation. A subclass can
    # preprocess input or adjust tokens in __call__/_encode_plus, and bypassing
    # that behavior would change the training data.
    if type(tokenizer) is not PreTrainedTokenizerFast:
        return standard
    raw_backend = getattr(tokenizer, "backend_tokenizer", None)
    if not callable(getattr(raw_backend, "encode_batch_fast", None)):
        return standard

    from tokenizers import Tokenizer

    backend = Tokenizer.from_str(raw_backend.to_str())
    backend.no_padding()
    backend.no_truncation()
    backend.encode_special_tokens = bool(getattr(tokenizer, "split_special_tokens", False))

    def ids_only(texts: list[str]) -> list[list[int]]:
        return [encoding.ids for encoding in backend.encode_batch_fast(texts, add_special_tokens=False)]

    ids_only.supports_batched_tokenization = True
    return ids_only


class SprintCausalBatches(PackedBatches):
    """Pack causal rows without allocating a Python list for every sequence.

    ``rows_per_chunk`` controls output buffering, not data order or the row
    budget. Returned arrays own their memory and remain valid after advancing
    the iterator. Memory use is bounded by one output chunk and one bounded
    tokenizer batch; long documents retain the standard text-chunk boundaries.
    Physical tokenization batches may contain several logical ``tokenize_chunk``
    groups. Document counters advance at the original logical boundaries, so
    reading ahead for throughput does not inflate recorded prefix exposure.
    """

    def __init__(
        self,
        docs: Iterable[dict],
        tokenize: Callable[[list[str]], list[list[int]]],
        *,
        seq_len: int,
        eos_token_id: int | None,
        max_rows: int | None = None,
        rows_per_chunk: int = 256,
        tokenize_chunk: int = DEFAULT_TOKENIZE_CHUNK,
        tokenization_batch_size: int = 128,
        max_text_chars: int = DEFAULT_TEXT_CHUNK_CHARS,
    ) -> None:
        if rows_per_chunk < 1 or (max_rows is not None and max_rows < 1):
            raise ValueError("output chunk size and row budget must be positive")
        if tokenization_batch_size < 1:
            raise ValueError("tokenization batch size must be positive")
        super().__init__(
            docs,
            tokenize,
            seq_len=seq_len,
            eos_token_id=eos_token_id,
            micro_batch=1,
            max_batches=max_rows,
            tokenize_chunk=tokenize_chunk,
            max_text_chars=max_text_chars,
        )
        self.rows_per_chunk = int(rows_per_chunk)
        self.max_rows = max_rows
        # Use complete logical groups to retain the reference read-ahead
        # accounting. The standard 65,536-character chunks cap this at 128 inputs;
        # unusually large explicitly configured groups retain their old bound.
        bounded = min(int(tokenization_batch_size), MAX_TOKENIZATION_TEXT_CHARS // max_text_chars)
        self.tokenization_batch_size = max(tokenize_chunk, bounded // tokenize_chunk * tokenize_chunk)

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        pending_texts: list[tuple[Any, bool, str]] = []
        pending_documents: list[tuple[int, Counter[str]]] = []
        source_runs: deque[list[Any]] = deque()
        rows_made = 0
        filled = 0

        def allocate() -> np.ndarray:
            remaining = self.rows_per_chunk
            if self.max_rows is not None:
                remaining = min(remaining, self.max_rows - rows_made)
            return np.empty((remaining, self.seq_len), dtype=np.int64)

        output = allocate()

        def append_source(source: str, count: int) -> None:
            if not count:
                return
            if source_runs and source_runs[-1][0] == source:
                source_runs[-1][1] += count
            else:
                source_runs.append([source, count])

        def mark_forwarded(rows: int) -> None:
            nonlocal rows_made
            rows_made += rows
            self.sequences_made += rows
            self.sequences_forwarded += rows
            remaining = rows * self.seq_len
            self.forwarded_tokens += remaining
            while remaining:
                if not source_runs:
                    raise RuntimeError("packing source attribution fell behind token buffer")
                source, available = source_runs[0]
                take = min(remaining, available)
                self.source_forwarded_tokens[source] += take
                remaining -= take
                available -= take
                if available:
                    source_runs[0][1] = available
                else:
                    source_runs.popleft()

        def consume_texts(*, complete_groups_only: bool = False) -> Iterator[dict[str, np.ndarray]]:
            nonlocal output, filled
            document_cursor = 0

            def record_documents(end: int, *, include_boundary: bool = False) -> None:
                nonlocal document_cursor
                while document_cursor < len(pending_documents):
                    position, counts = pending_documents[document_cursor]
                    if position > end or (position == end and not include_boundary):
                        break
                    self.docs_seen += sum(counts.values())
                    self.source_docs_seen.update(counts)
                    document_cursor += 1

            count = len(pending_texts)
            if complete_groups_only:
                count -= count % self.tokenize_chunk
            consumable = pending_texts[:count]
            if not consumable:
                record_documents(len(pending_texts), include_boundary=True)
                pending_texts.clear()
                pending_documents.clear()
                return
            def encode(group: list[tuple[Any, bool, str]]) -> list[list[int]]:
                texts = [text for text, _end, _source in group]
                prepared = not isinstance(texts[0], str)
                return [value.read_ids() for value in texts] if prepared else self.tokenize(texts)

            def encode_groups() -> Iterator[list[int]]:
                try:
                    encodings = encode(consumable)
                except Exception:
                    # Physical read-ahead may include a bad record outside the
                    # requested row budget. Retry only as each original logical
                    # group is needed, so unused lookahead cannot fail a run.
                    for start in range(0, len(consumable), self.tokenize_chunk):
                        end = min(start + self.tokenize_chunk, len(consumable))
                        record_documents(end, include_boundary=end - start < self.tokenize_chunk)
                        group = consumable[start:end]
                        for ids, _ in zip(encode(group), group, strict=True):
                            yield ids
                else:
                    yield from encodings

            encodings = encode_groups()
            for index, (ids, (_text, is_end, source)) in enumerate(zip(encodings, consumable, strict=True)):
                if index % self.tokenize_chunk == 0:
                    end = min(index + self.tokenize_chunk, len(consumable))
                    record_documents(end, include_boundary=end - index < self.tokenize_chunk)
                # Count the entire tokenization unit before the budget check,
                # exactly as PackedBatches does, including an EOS outside the
                # final forwarded row. Read-ahead therefore remains auditable.
                self.tokens_seen += len(ids)
                self.source_content_tokens[source] += len(ids)
                boundary = len(self.eos) if is_end else 0
                self.boundary_tokens += boundary
                self.source_boundary_tokens[source] += boundary
                append_source(source, len(ids) + boundary)
                parts = (ids, self.eos) if boundary else (ids,)
                for part in parts:
                    values = np.asarray(part, dtype=np.int64)
                    if values.ndim != 1:
                        raise ValueError("causal tokenizer must return one-dimensional token sequences")
                    position = 0
                    while position < values.size:
                        take = min(values.size - position, output.size - filled)
                        output.reshape(-1)[filled : filled + take] = values[position : position + take]
                        filled += take
                        position += take
                        if filled == output.size:
                            mark_forwarded(output.shape[0])
                            yield {"input_ids": output}
                            if self.max_rows is not None and rows_made >= self.max_rows:
                                pending_texts.clear()
                                pending_documents.clear()
                                return
                            output = allocate()
                            filled = 0
            record_documents(len(pending_texts), include_boundary=True)
            pending_texts.clear()
            pending_documents.clear()

        def final_rows() -> Iterator[dict[str, np.ndarray]]:
            complete_rows = filled // self.seq_len
            if complete_rows:
                mark_forwarded(complete_rows)
                # Drop the incomplete terminal sequence, keeping the same
                # no-pad contract. Also release unused output capacity.
                yield {"input_ids": output[:complete_rows].copy()}

        def flush_texts(*, complete_groups_only: bool = False) -> Iterator[dict[str, np.ndarray]]:
            try:
                yield from consume_texts(complete_groups_only=complete_groups_only)
            except Exception:
                # Rows from successful logical groups precede a needed failure,
                # even when they do not fill a physical output chunk.
                yield from final_rows()
                raise

        def document_events() -> Iterator[tuple[str, Any]]:
            for doc in self.docs:
                source = str(doc.get(CORPUS_ID_FIELD) or "unknown")
                yield "document", source
                prepared = doc.get(PREPARED_FIELD)
                if prepared:
                    chunks = prepared["units"]
                    parts = ((chunk, index == len(chunks) - 1) for index, chunk in enumerate(chunks))
                else:
                    text = doc["text"]
                    parts = (
                        (text[start : start + self.max_text_chars], start + self.max_text_chars >= len(text))
                        for start in range(0, max(len(text), 1), self.max_text_chars)
                    )
                for part, is_end in parts:
                    yield "part", (part, is_end, source)

        events = document_events()
        try:
            while True:
                try:
                    kind, value = next(events)
                except StopIteration:
                    break
                except Exception:
                    # A source or record may fail beyond a satisfied logical
                    # prefix. Consume pending groups first and surface the
                    # failure only if more source data was actually required.
                    yield from flush_texts(complete_groups_only=True)
                    if self.max_rows is not None and rows_made >= self.max_rows:
                        return
                    yield from final_rows()
                    raise
                if kind == "document":
                    # Empty prepared records have no units. Coalesce their
                    # events so any number of them still uses bounded space.
                    position = len(pending_texts)
                    if pending_documents and pending_documents[-1][0] == position:
                        pending_documents[-1][1][value] += 1
                    else:
                        pending_documents.append((position, Counter({value: 1})))
                else:
                    pending_texts.append(value)
                    if len(pending_texts) >= self.tokenization_batch_size:
                        yield from flush_texts()
                        if self.max_rows is not None and rows_made >= self.max_rows:
                            return
        finally:
            events.close()
        yield from flush_texts()
        if self.max_rows is not None and rows_made >= self.max_rows:
            return
        yield from final_rows()
