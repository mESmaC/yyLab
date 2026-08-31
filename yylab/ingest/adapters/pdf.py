from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from ..records import Doc
from .base import Adapter

logging.getLogger("pypdf").setLevel(logging.ERROR)

CHUNK_TARGET = 4_000


class PdfAdapter(Adapter):
    name = "pdf"
    extensions = {".pdf"}
    default_domain = "language"

    def extract(self, path: Path) -> Iterator[Doc]:
        from pypdf import PdfReader

        reader = PdfReader(str(path), strict=False)
        buf: list[str] = []
        buf_len = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            buf.append(text)
            buf_len += len(text)
            if buf_len >= CHUNK_TARGET:
                yield Doc(text="\n\n".join(buf), source=str(path), domain=self.default_domain)
                buf, buf_len = [], 0
        if buf:
            yield Doc(text="\n\n".join(buf), source=str(path), domain=self.default_domain)
