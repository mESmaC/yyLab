from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..records import Doc
from .base import Adapter
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
    extensions = {".parquet", ".pq"}
    default_domain = "other"

    def estimated_records(self, path: Path) -> int | None:
        return int(pq.ParquetFile(str(path)).metadata.num_rows)

    def extract(self, path: Path) -> Iterator[Doc]:
        self._reset_issues()
        pf = pq.ParquetFile(str(path))
        message_columns = _message_columns(pf.schema_arrow)
        split_columns = _split_columns(pf.schema_arrow)
        scalar_columns = _scalar_text_columns(_string_columns(pf.schema_arrow))
        metadata_columns = [
            name for name in PROVENANCE_COLUMNS if name in pf.schema_arrow.names
        ]
        content_columns = [*message_columns, *scalar_columns]
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

        columns = list(
            dict.fromkeys(
                [
                    *content_columns,
                    *split_columns,
                    *metadata_columns,
                ]
            )
        )
        record_index = 0
        batch_rows = MSG_BATCH_ROWS if message_columns else BATCH_ROWS
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            for row in batch.to_pylist():
                record_index += 1
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
