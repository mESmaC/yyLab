from __future__ import annotations

import re
from pathlib import Path

import orjson

from .config import PATHS


def corpus_dir(cid: str) -> Path:
    return PATHS.corpora / cid


def checkpoint_dir(cid: str) -> Path:
    return PATHS.checkpoints / cid


def adapter_dir(aid: str) -> Path:
    return PATHS.adapters / aid


def router_dir(rid: str) -> Path:
    return PATHS.routers / rid


def run_dir(rid: str) -> Path:
    return PATHS.runs / rid


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_")
    return slug[:64] or "unnamed"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def read_json(path: Path):
    return orjson.loads(path.read_bytes())


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"
