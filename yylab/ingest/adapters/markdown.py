from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from ..records import Doc
from .base import Adapter

FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


class MarkdownAdapter(Adapter):
    name = "markdown"
    extensions = {".md", ".markdown", ".rst"}
    default_domain = "language"

    def extract(self, path: Path) -> Iterator[Doc]:
        raw = FRONTMATTER_RE.sub("", self._read_text(path))
        if raw.strip():
            yield Doc(text=raw, source=str(path), domain=self.default_domain)
