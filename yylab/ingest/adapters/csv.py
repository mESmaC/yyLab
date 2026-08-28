from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from ..records import Doc
from .base import Adapter


class CsvAdapter(Adapter):
    name = "csv"
    extensions = {".csv", ".tsv"}
    default_domain = "other"

    def extract(self, path: Path) -> Iterator[Doc]:
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f, delimiter=delim)
            try:
                header = next(reader)
            except StopIteration:
                return
            for row in reader:
                if not row:
                    continue
                text = "\n".join(f"{h}: {v}" for h, v in zip(header, row) if v.strip())
                if text.strip():
                    yield Doc(text=text, source=str(path), domain=self.default_domain)
