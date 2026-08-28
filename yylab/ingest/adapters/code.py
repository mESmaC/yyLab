from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..domains import CODE_EXTS, WEB_EXTS
from ..records import Doc
from .base import Adapter


class CodeAdapter(Adapter):
    name = "code"
    extensions = CODE_EXTS | WEB_EXTS
    default_domain = "code"

    def extract(self, path: Path) -> Iterator[Doc]:
        raw = self._read_text(path)
        if raw.strip():
            domain = "web" if path.suffix.lower() in WEB_EXTS else "code"
            yield Doc(
                text=raw,
                source=str(path),
                domain=domain,
                meta={"language": path.suffix.lower().lstrip(".")},
            )
