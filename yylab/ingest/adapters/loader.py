from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .base import Adapter
from .code import CodeAdapter
from .csv import CsvAdapter
from .docx import DocxAdapter
from .html import HtmlAdapter
from .jsonl import JsonlAdapter
from .markdown import MarkdownAdapter
from .parquet import ParquetAdapter
from .pdf import PdfAdapter
from .text import TextAdapter

_ADAPTER_CLASSES: tuple[type[Adapter], ...] = (
    JsonlAdapter,
    ParquetAdapter,
    CsvAdapter,
    PdfAdapter,
    DocxAdapter,
    HtmlAdapter,
    CodeAdapter,
    MarkdownAdapter,
    TextAdapter,
)

SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".hg", ".svn"}
FALLBACK_TEXT_EXTENSIONS = {
    "",
    ".bat",
    ".bzl",
    ".cfg",
    ".cmd",
    ".conf",
    ".core",
    ".diff",
    ".hjson",
    ".ini",
    ".log",
    ".mdx",
    ".mem",
    ".mif",
    ".mk",
    ".mlir",
    ".org",
    ".patch",
    ".sdc",
    ".sv",
    ".svh",
    ".tcl",
    ".tex",
    ".toml",
    ".vhd",
    ".vhdl",
    ".vh",
    ".wat",
    ".yaml",
    ".yml",
}


def discover_adapters() -> list[Adapter]:
    return [cls() for cls in _ADAPTER_CLASSES]


def dispatch(path: Path, adapters: list[Adapter]) -> Adapter | None:
    name = path.name.casefold()
    if name == ".env" or name.startswith(".env.") or path.suffix.casefold() == ".env":
        return None
    try:
        with path.open("rb") as stream:
            sample = stream.read(8192)
    except OSError:
        return None
    signature_adapter = None
    if sample.startswith(b"%PDF-"):
        signature_adapter = "pdf"
    elif sample.startswith(b"PAR1"):
        signature_adapter = "parquet"
    elif sample.startswith((b"SQLite format 3\x00", b"\x1f\x8b")):
        return None
    elif sample.startswith(b"PK\x03\x04") and path.suffix.casefold() not in {
        ".docx",
        ".epub",
        ".pptx",
        ".xlsx",
    }:
        return None
    if signature_adapter is not None:
        return next((adapter for adapter in adapters if adapter.name == signature_adapter), None)
    for a in adapters:
        if a.can_handle(path):
            return a
    # CDAS inventories extensionless and uncommon UTF-8 source files as text.
    # Keep binary detection conservative and route through yyLab's existing
    # streaming/extraction adapters instead of silently discarding such files.
    if path.suffix.casefold() in FALLBACK_TEXT_EXTENSIONS and b"\x00" not in sample:
        try:
            decoded = sample.decode("utf-8")
        except UnicodeDecodeError:
            decoded = ""
        if decoded:
            controls = sum(
                1 for character in decoded if ord(character) < 32 and character not in "\n\r\t\f\b"
            )
            if controls / max(len(decoded), 1) < 0.01:
                return next((adapter for adapter in adapters if adapter.name == "text"), None)
    return None


def walk_files(root: Path, max_file_mb: float = 300.0) -> Iterator[Path]:
    """Enumerate every local asset without applying Git or size policy.

    ``max_file_mb`` remains in the signature for callers using the historical
    API. Discovery intentionally does not enforce it: ingestion must inventory
    an asset before deciding whether its size permits extraction. Likewise,
    corpus data ignored by Git is still corpus data and must remain visible.
    """

    _ = max_file_mb

    def _walk(d: Path) -> Iterator[Path]:
        try:
            entries = sorted(d.iterdir(), key=lambda path: path.name.casefold())
        except (PermissionError, OSError):
            return
        for p in entries:
            try:
                if p.is_symlink():
                    continue
                if p.is_dir():
                    if p.name not in SKIP_DIRS:
                        yield from _walk(p)
                elif p.is_file():
                    yield p
            except OSError:
                continue

    yield from _walk(root)
