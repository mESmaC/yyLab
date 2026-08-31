from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .. import db

SKIP_DIRS = {"__pycache__", ".git", ".hg", ".svn", "node_modules", ".venv"}


@dataclass(slots=True)
class DatasetFolder:
    name: str
    path: str
    files: int
    size_bytes: int


def scan_dataset_folders(root: str) -> list[DatasetFolder]:
    root_p = Path(root)
    if not root_p.is_dir():
        return []
    out: list[DatasetFolder] = []
    for entry in sorted(root_p.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        files = 0
        size = 0
        for dirpath, dirnames, filenames in os.walk(entry):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS]
            for f in filenames:
                if f.startswith("."):
                    continue
                files += 1
                try:
                    size += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    continue
        out.append(DatasetFolder(name=entry.name, path=str(entry), files=files, size_bytes=size))
    return out


def ingested_by_source() -> dict[str, dict]:
    rows = db.q("SELECT id, name, status, source_path FROM corpora")
    return {r["source_path"]: r for r in rows if r["source_path"]}
