from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .base import Adapter
from .code import CodeAdapter
from .csv import CsvAdapter
from .jsonl import JsonlAdapter
from .markdown import MarkdownAdapter
from .text import TextAdapter

_ADAPTER_CLASSES: tuple[type[Adapter], ...] = (
    JsonlAdapter,
    CsvAdapter,
    CodeAdapter,
    MarkdownAdapter,
    TextAdapter,
)

SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".hg", ".svn"}


def discover_adapters() -> list[Adapter]:
    return [cls() for cls in _ADAPTER_CLASSES]


def dispatch(path: Path, adapters: list[Adapter]) -> Adapter | None:
    for a in adapters:
        if a.can_handle(path):
            return a
    return None


def walk_files(root: Path, max_file_mb: float = 300.0) -> Iterator[Path]:
    limit = max_file_mb * 1024 * 1024

    def _walk(d: Path) -> Iterator[Path]:
        try:
            entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.stat().st_size if p.is_file() else 0))
        except (PermissionError, OSError):
            return
        for p in entries:
            if p.name.startswith("."):
                continue
            if p.is_dir():
                if p.name not in SKIP_DIRS:
                    yield from _walk(p)
            elif p.is_file():
                try:
                    if p.stat().st_size <= limit:
                        yield p
                except OSError:
                    continue

    yield from _walk(root)
