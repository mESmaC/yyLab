from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..records import Doc
from .base import Adapter

SPLIT_THRESHOLD = 1_000_000


class TextAdapter(Adapter):
    name = "text"
    extensions = {".txt", ".text"}
    default_domain = "language"

    def extract(self, path: Path) -> Iterator[Doc]:
        raw = self._read_text(path)
        if len(raw) > SPLIT_THRESHOLD and "\n\n" in raw:
            for chunk in raw.split("\n\n"):
                if chunk.strip():
                    yield Doc(text=chunk, source=str(path), domain=self.default_domain)
        elif raw.strip():
            yield Doc(text=raw, source=str(path), domain=self.default_domain)
