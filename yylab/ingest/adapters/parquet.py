from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..records import Doc
from .base import Adapter, ChunkSpec
from .jsonl import (
    MESSAGE_FIELDS,
    PROMPT_FIELDS,
    RESPONSE_FIELDS,
    TEXT_FIELDS,
    _record_domain,
    _record_format_name,
    _record_locator,
    _record_messages,
    _record_metadata,
    _record_text,
    _unsupported_detail,
)

BATCH_ROWS = 8_192
MSG_BATCH_ROWS = 256
#: Row groups are the smallest independently readable Parquet unit; chunks
#: aggregate consecutive groups up to roughly this many rows or bytes.
CHUNK_ROWS = 16_384
CHUNK_BYTES = 32 * 2**20
MESSAGE_COLUMNS = MESSAGE_FIELDS
SPLIT_COLUMNS = ("split", "source_split", "partition")
PROVENANCE_COLUMNS = (
    "domain",
    "language",
    "lang",
    "source_id",
    "sample_id",
    "schema_version",
    "provenance",
    "labels",
    "metrics",
    "sequence_id",
    "document_id",
    "asset_id",
    "logical_path",
    "locator",
    "format",
    "format_name",
    "path",
    "source_path",
    "content_sha256",
    "mixture_group",
    "selection_rank",
    "sample_weight",
    "estimated_tokens",
    "target_tokens",
    "sequence_boundary",
)


def _string_columns(schema: pa.Schema) -> list[str]:
    cols = []
    for field in schema:
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            cols.append(field.name)
    return cols


def _message_columns(schema: pa.Schema) -> list[str]:
    names = set(schema.names)
    return [column for column in MESSAGE_COLUMNS if column in names]


def _split_columns(schema: pa.Schema) -> list[str]:
    names = set(schema.names)
    return [column for column in SPLIT_COLUMNS if column in names]


def _scalar_text_columns(cols: list[str]) -> list[str]:
    def norm(c: str) -> str:
        return c[:-5] if c.endswith("_text") else c

    supported = {*TEXT_FIELDS, *PROMPT_FIELDS, *RESPONSE_FIELDS, "system"}
    return [column for column in cols if norm(column) in supported]


class ParquetAdapter(Adapter):
    name = "parquet"
    version = 2
    extensions = {".parquet", ".pq"}
    default_domain = "other"

    def __init__(self) -> None:
        self._records_seen = 0

    def estimated_records(self, path: Path) -> int | None:
        return int(pq.ParquetFile(str(path)).metadata.num_rows)

    def chunk_stats(self) -> tuple[int, int]:
        return (int(self._records_seen), 0)

    def plan_chunks(self, path: Path) -> list[ChunkSpec] | None:
        """Group small row groups and sub-split oversized ones by row range.

        Files written with pyarrow defaults can hold a million rows in one
        row group, so a plan must not depend on row-group boundaries alone.
        """

        pf = pq.ParquetFile(str(path))
        metadata = pf.metadata
        if not self._columns(pf.schema_arrow)[0]:
            # An unsupported schema records exactly one issue; keep it whole.
            return None
        plan: list[tuple[tuple[int, ...], int, int, int]] = []  # groups, start, end, rows
        current: list[int] = []
        rows = 0
        nbytes = 0
        for index in range(metadata.num_row_groups):
            row_group = metadata.row_group(index)
            group_rows = int(row_group.num_rows)
            group_bytes = int(row_group.total_byte_size)
            if group_bytes > CHUNK_BYTES and group_rows > 1:
                if current:
                    plan.append((tuple(current), 0, 0, rows))
                    current, rows, nbytes = [], 0, 0
                pieces = min(group_rows, math.ceil(group_bytes / CHUNK_BYTES))
                step = math.ceil(group_rows / pieces)
                for start in range(0, group_rows, step):
                    end = min(group_rows, start + step)
                    plan.append(((index,), start, end, end - start))
                continue
            current.append(index)
            rows += group_rows
            nbytes += group_bytes
            if rows >= CHUNK_ROWS or nbytes >= CHUNK_BYTES:
                plan.append((tuple(current), 0, 0, rows))
                current, rows, nbytes = [], 0, 0
        if current:
            plan.append((tuple(current), 0, 0, rows))
        if len(plan) < 2:
            return None
        return [
            ChunkSpec(
                index=index,
                kind="row_groups",
                row_groups=groups,
                start=start,
                end=end,
                estimated_records=estimate,
            )
            for index, (groups, start, end, estimate) in enumerate(plan)
        ]

    @staticmethod
    def _columns(schema: pa.Schema) -> tuple[list[str], list[str], list[str]]:
        message_columns = _message_columns(schema)
        split_columns = _split_columns(schema)
        scalar_columns = _scalar_text_columns(_string_columns(schema))
        metadata_columns = [name for name in PROVENANCE_COLUMNS if name in schema.names]
        content_columns = [*message_columns, *scalar_columns]
        columns = list(dict.fromkeys([*content_columns, *split_columns, *metadata_columns]))
        return content_columns, message_columns, columns

    def extract(self, path: Path) -> Iterator[Doc]:
        yield from self._extract(path, None)

    def extract_chunk(self, path: Path, chunk: ChunkSpec) -> Iterator[Doc]:
        if chunk.kind == "whole":
            yield from self._extract(path, None)
            return
        if chunk.kind != "row_groups":
            raise ValueError(f"parquet adapter cannot extract {chunk.kind} chunks")
        yield from self._extract(path, chunk)

    def _extract(self, path: Path, chunk: ChunkSpec | None) -> Iterator[Doc]:
        self._reset_issues()
        self._records_seen = 0
        pf = pq.ParquetFile(str(path))
        content_columns, message_columns, columns = self._columns(pf.schema_arrow)
        if not content_columns:
            fields = ", ".join(
                f"{field.name}:{field.type}" for field in pf.schema_arrow
            )
            self._record_issue(
                "$schema",
                "unsupported_parquet_schema",
                f"no supported text or message columns; fields=[{fields}]",
            )
            return

        record_index = 0
        batch_rows = MSG_BATCH_ROWS if message_columns else BATCH_ROWS
        row_groups = list(chunk.row_groups) if chunk is not None else None
        # A sub-split row group carries a [start, end) row range; batches before
        # the range are skipped without materializing Python rows.
        row_start = chunk.start if chunk is not None and chunk.end > 0 else 0
        row_end = chunk.end if chunk is not None and chunk.end > 0 else None
        position = 0
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns, row_groups=row_groups):
            batch_start = position
            position += batch.num_rows
            if row_end is not None and batch_start >= row_end:
                break
            if position <= row_start:
                continue
            if batch_start < row_start or (row_end is not None and position > row_end):
                offset = max(0, row_start - batch_start)
                length = (min(position, row_end) if row_end is not None else position) - batch_start - offset
                batch = batch.slice(offset, length)
            for row in batch.to_pylist():
                record_index += 1
                self._records_seen = record_index
                messages = _record_messages(row)
                text = _record_text(row, messages)
                locator = _record_locator(row, record_index, "row")
                if not text:
                    self._record_issue(
                        locator,
                        "unsupported_record_schema",
                        _unsupported_detail(row),
                    )
                    continue
                yield Doc(
                    text=text,
                    source=str(path),
                    domain=_record_domain(row, messages),
                    meta=_record_metadata(row, path, messages),
                    locator=locator,
                    format_name=_record_format_name(row, "parquet"),
                )
