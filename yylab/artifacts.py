from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path

import orjson

from .config import PATHS


def corpus_dir(cid: str) -> Path:
    return PATHS.corpora / cid


# -- corpus generations -------------------------------------------------------
#
# A freshly ingested corpus keeps its outputs at the corpus root (``data/``,
# ``data_nonen/``, ``data_quarantine/``, ``inventory.sqlite3``,
# ``ingest_manifest.json``).  A re-import writes a complete new generation into
# ``gen-<n>/`` and publishes it by atomically replacing ``current.json``.
# Readers resolve the generation once when they start streaming, so a training
# run or review job that is mid-way through the old shards keeps reading them;
# retired generations are reclaimed later, only when nothing holds them open.

CURRENT_FILE = "current.json"
CORPUS_BUCKETS = ("data", "data_nonen", "data_quarantine")
GENERATION_ENTRIES = (*CORPUS_BUCKETS, "inventory.sqlite3", "ingest_manifest.json")


def read_corpus_layout(root: Path) -> dict:
    path = root / CURRENT_FILE
    if not path.exists():
        return {"generation": "", "retired": []}
    try:
        layout = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return {"generation": "", "retired": []}
    if not isinstance(layout, dict):
        return {"generation": "", "retired": []}
    generation = str(layout.get("generation") or "")
    retired = [str(item) for item in layout.get("retired") or [] if isinstance(item, str)]
    return {"generation": generation, "retired": retired}


def _write_layout(root: Path, layout: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f"{CURRENT_FILE}.tmp"
    temporary.write_bytes(orjson.dumps(layout, option=orjson.OPT_INDENT_2))
    os.replace(temporary, root / CURRENT_FILE)


def corpus_generation_dir(cid: str, root: Path | None = None) -> Path:
    root = root if root is not None else corpus_dir(cid)
    generation = read_corpus_layout(root)["generation"]
    return root / generation if generation else root


def corpus_bucket_dir(cid: str, bucket: str = "data", root: Path | None = None) -> Path:
    """Directory holding the live shards of ``bucket`` for corpus ``cid``."""

    if bucket not in CORPUS_BUCKETS:
        raise ValueError(f"unknown corpus bucket {bucket!r}")
    return corpus_generation_dir(cid, root) / bucket


def corpus_inventory_path(cid: str) -> Path:
    return corpus_generation_dir(cid) / "inventory.sqlite3"


def next_corpus_generation(cid: str) -> str:
    current = read_corpus_layout(corpus_dir(cid))["generation"]
    number = 0
    if current.startswith("gen-"):
        try:
            number = int(current[4:])
        except ValueError:
            number = 0
    return f"gen-{number + 1}"


def publish_corpus_generation(cid: str, generation: str, stats: dict) -> None:
    """Make ``generation`` live and mark the previous outputs as retired.

    ``stats`` is also written to the root ``stats.json`` so readers that look
    there keep seeing the live corpus statistics.
    """

    root = corpus_dir(cid)
    layout = read_corpus_layout(root)
    previous = layout["generation"]
    retired = [item for item in layout["retired"] if item != generation]
    if previous:
        if previous != generation:
            retired.append(previous)
    else:
        retired.extend(entry for entry in GENERATION_ENTRIES if (root / entry).exists())
    # The pointer flips first: readers resolve the generation, then list
    # shards inside it, so they see either the old or the new state whole.
    _write_layout(
        root,
        {"generation": generation, "retired": sorted(set(retired)), "updated_at": _now()},
    )
    write_json(root / "stats.json.tmp", stats)
    os.replace(root / "stats.json.tmp", root / "stats.json")


def reclaim_retired_generations(cid: str) -> list[str]:
    """Delete retired outputs that nothing holds open; return what was reclaimed.

    On Windows a directory rename fails while any file inside it is open, so
    renaming first is a reliable cross-process probe and entries still in use
    stay retired for a later pass.  On POSIX the rename succeeds regardless of
    open descriptors and only the shard currently open survives deletion, so
    there this must only run when no reader can be pinned to a retired
    generation (app start, before a new re-import begins), never right after
    a publish.
    """

    root = corpus_dir(cid)
    layout = read_corpus_layout(root)
    if not layout["retired"]:
        return []
    reclaimed: list[str] = []
    remaining: list[str] = []
    for entry in layout["retired"]:
        path = root / entry
        if not path.exists():
            reclaimed.append(entry)
            continue
        probe = root / f".reclaim-{entry}-{uuid.uuid4().hex[:6]}"
        try:
            path.rename(probe)
        except OSError:
            remaining.append(entry)
            continue
        if probe.is_dir():
            shutil.rmtree(probe, ignore_errors=True)
        else:
            probe.unlink(missing_ok=True)
        reclaimed.append(entry)
    if reclaimed:
        _write_layout(
            root,
            {"generation": layout["generation"], "retired": remaining, "updated_at": _now()},
        )
    return reclaimed


def reclaim_all_retired_generations() -> dict[str, list[str]]:
    reclaimed: dict[str, list[str]] = {}
    if not PATHS.corpora.exists():
        return reclaimed
    for root in PATHS.corpora.iterdir():
        if root.is_dir() and (root / CURRENT_FILE).exists():
            try:
                entries = reclaim_retired_generations(root.name)
            except OSError:
                continue
            if entries:
                reclaimed[root.name] = entries
    return reclaimed


def _now() -> str:
    import datetime as dt

    return dt.datetime.now().isoformat(timespec="seconds")


def checkpoint_dir(cid: str) -> Path:
    return PATHS.checkpoints / cid


def adapter_dir(aid: str) -> Path:
    return PATHS.adapters / aid


def router_dir(rid: str) -> Path:
    return PATHS.routers / rid


def tokenizer_dir(tid: str) -> Path:
    return PATHS.tokenizers / tid


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
