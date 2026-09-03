from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..records import Doc


@dataclass(frozen=True, slots=True)
class ChunkSpec:
    """One independently extractable slice of an asset.

    ``kind`` is ``whole`` (the entire asset), ``lines`` (a byte range of a
    line-oriented file whose boundaries sit exactly on line starts) or
    ``row_groups`` (a set of Parquet row groups).  Record and line numbers
    produced inside a chunk are local to that chunk; the ingest coordinator
    rebases them using the ``records_seen``/``lines_seen`` counts reported by
    the preceding chunks, so the resulting locators equal a sequential read.
    """

    index: int = 0
    kind: str = "whole"
    start: int = 0
    end: int = 0
    row_groups: tuple[int, ...] = ()
    estimated_records: int = 0
    estimated_bytes: int = 0


class Adapter:
    name = "base"
    #: Bump when an adapter's extraction output changes so cached prepared
    #: documents produced by the old behaviour are not reused.
    version = 1
    extensions: set[str] = set()
    default_domain = "other"

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def extract(self, path: Path) -> Iterator[Doc]:
        raise NotImplementedError

    def estimated_records(self, path: Path) -> int | None:
        """Return a cheap record-count estimate when the container provides one."""

        return None

    def plan_chunks(self, path: Path) -> list[ChunkSpec] | None:
        """Split an asset into independently extractable chunks, or ``None``.

        ``None`` means the asset is extracted with :meth:`extract` as a single
        unit.  Adapters that return a plan must implement :meth:`extract_chunk`
        so that concatenating the chunk outputs in order reproduces
        :meth:`extract` exactly.
        """

        return None

    def extract_chunk(self, path: Path, chunk: ChunkSpec) -> Iterator[Doc]:
        if chunk.kind != "whole":
            raise ValueError(f"{self.name} adapter cannot extract {chunk.kind} chunks")
        yield from self.extract(path)

    def chunk_stats(self) -> tuple[int, int]:
        """Return ``(records_seen, lines_seen)`` for the most recent extraction.

        Only record-oriented adapters whose locators embed a record or line
        number need to report real values; the coordinator rebases locators of
        later chunks by these counts.
        """

        return (0, 0)

    def _reset_issues(self) -> None:
        self._extraction_issues: list[tuple[str, str, str]] = []

    def _record_issue(self, locator: str, reason_code: str, detail: str) -> None:
        if not hasattr(self, "_extraction_issues"):
            self._reset_issues()
        self._extraction_issues.append((locator, reason_code, detail[:2000]))

    def drain_issues(self) -> list[tuple[str, str, str]]:
        issues = list(getattr(self, "_extraction_issues", ()))
        self._extraction_issues = []
        return issues

    def _read_text(self, path: Path) -> str:
        text = path.read_text("utf-8", errors="ignore").removeprefix("﻿")
        return text.replace("\r\n", "\n").replace("\r", "\n")
