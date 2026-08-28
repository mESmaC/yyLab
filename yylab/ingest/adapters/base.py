from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..records import Doc


class Adapter:
    name = "base"
    extensions: set[str] = set()
    default_domain = "other"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def extract(self, path: Path) -> Iterator[Doc]:
        raise NotImplementedError

    def _read_text(self, path: Path) -> str:
        return path.read_text("utf-8", errors="ignore")
