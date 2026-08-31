from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..records import Doc
from .base import Adapter

SPLIT_THRESHOLD = 200_000


class DocxAdapter(Adapter):
    name = "docx"
    extensions = {".docx"}
    default_domain = "language"

    def extract(self, path: Path) -> Iterator[Doc]:
        import docx

        document = docx.Document(str(path))
        parts: list[str] = []
        for para in document.paragraphs:
            if para.text.strip():
                parts.append(para.text.rstrip())
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        if not parts:
            return
        buf: list[str] = []
        buf_len = 0
        for part in parts:
            buf.append(part)
            buf_len += len(part)
            if buf_len >= SPLIT_THRESHOLD:
                yield Doc(text="\n\n".join(buf), source=str(path), domain=self.default_domain)
                buf, buf_len = [], 0
        if buf:
            yield Doc(text="\n\n".join(buf), source=str(path), domain=self.default_domain)
