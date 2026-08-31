from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..records import Doc
from .base import Adapter
from .jsonl import TEXT_FIELDS, _record_text, format_messages

BATCH_ROWS = 2_048
MSG_BATCH_ROWS = 256
MESSAGE_COLUMNS = ("messages", "conversations")


def _string_columns(schema: pa.Schema) -> list[str]:
    cols = []
    for field in schema:
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            cols.append(field.name)
    return cols


def _message_column(schema: pa.Schema) -> str | None:
    names = set(schema.names)
    for c in MESSAGE_COLUMNS:
        if c in names:
            return c
    return None


def _scalar_text_columns(cols: list[str]) -> list[str]:
    def norm(c: str) -> str:
        return c[:-5] if c.endswith("_text") else c

    preferred = [c for c in cols if norm(c) in TEXT_FIELDS]
    return preferred if preferred else cols[:4]


class ParquetAdapter(Adapter):
    name = "parquet"
    extensions = {".parquet", ".pq"}
    default_domain = "other"

    def extract(self, path: Path) -> Iterator[Doc]:
        pf = pq.ParquetFile(str(path))
        msg_col = _message_column(pf.schema_arrow)
        if msg_col is not None:
            for batch in pf.iter_batches(batch_size=MSG_BATCH_ROWS, columns=[msg_col]):
                for row in batch.to_pylist():
                    msgs = row.get(msg_col)
                    if isinstance(msgs, list) and msgs:
                        text = format_messages(msgs)
                        if text:
                            yield Doc(text=text, source=str(path), domain=self.default_domain)
            return
        cols = _string_columns(pf.schema_arrow)
        if not cols:
            return
        selected = _scalar_text_columns(cols)
        for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=selected):
            rows = batch.to_pylist()
            for rec in rows:
                text = _record_text({k: v for k, v in rec.items() if isinstance(v, str)})
                if text:
                    yield Doc(text=text, source=str(path), domain=self.default_domain)
