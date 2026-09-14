"""Register GCCF receipts as source indexes; corpus text stays in place."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import db
from ..artifacts import corpus_dir
from ..config import PATHS
from ..jobs import ProgressFn
from .pipeline import IngestConfig, _exclusive_ingest


def _receipt_tools():
    try:
        from spalmer.data.gccf import audit_receipt
    except ImportError as exc:
        raise RuntimeError(
            "GCCF import requires the SPALMER checkout with receipt support. "
            "Install it into this yyLab environment: pip install --no-deps -e ../spalmer"
        ) from exc
    return audit_receipt


def gccf_ingest_config(*, name: str, source_dir: str, workers: int | None = None) -> IngestConfig:
    # Existing raw corpora must not make newly approved records disappear under
    # global dedup. Cross-corpus dedup still applies when training mixes are read.
    return IngestConfig(
        name=name,
        source_dir=source_dir,
        near_dedup=False,
        global_dedup=False,
        language_filter=False,
        quality_filter=False,
        min_chars=1,
        max_file_mb=0,
        tokenizer_keys=[],
        selection_profile="preserve",
        workers=workers,
        prepared_cache=False,
    )


def import_export(progress: ProgressFn, export_dir: str | Path, *, workers: int | None = None) -> dict:
    """Reject the retired copy-based importer without reading or writing anything."""
    raise ValueError(
        "GCCF export copying is unsupported. Register the original GCCF receipt "
        "with run_gccf_ingest; its index reads data from the original source files. "
        "Existing registered corpora are unchanged."
    )


def run_gccf_ingest(
    progress: ProgressFn,
    receipt_path: str,
    *,
    output: str | None = None,
    limit_per_file: int | None = None,
    workers: int | None = None,
    materialize: bool = False,
) -> dict:
    """Index a receipt's selected source records without copying their text.

    ``output`` is an optional metadata-index directory. ``workers`` and an
    explicit ``materialize=False`` remain accepted for existing callers.
    Copy requests fail before receipt inspection, dependency loading or writes.
    """
    if materialize:
        raise ValueError(
            "GCCF materialization is unsupported: receipts locate and process "
            "original source data without copying it. Omit materialize=True "
            "to register a metadata-only source index."
        )
    audit_receipt = _receipt_tools()
    audit = audit_receipt(receipt_path)
    suffix = f"-sample-{limit_per_file}" if limit_per_file else ""
    from spalmer.data.gccf_index import build_receipt_index

    destination = (
        Path(output) if output else (PATHS.cache / "gccf-index" / (audit["receipt_sha256"][:20] + suffix))
    )
    if not destination.exists():
        build_receipt_index(
            receipt_path,
            destination,
            limit_per_file=limit_per_file,
            progress=lambda message: progress("gccf-index", 0, message),
        )
    manifest = json.loads((destination / "gccf-index.json").read_text(encoding="utf-8"))
    if (
        manifest["audit"]["receipt_sha256"] != audit["receipt_sha256"]
        or manifest["limit_per_file"] != limit_per_file
    ):
        raise ValueError("existing index does not match receipt and sampling limit")
    return register_source_index(progress, destination)


def register_source_index(progress, index_directory):
    """Register lightweight corpus pointers; no extracted text is written."""
    from spalmer.data.gccf_index import INDEX_SCHEMA

    root = Path(index_directory).resolve()
    manifest = json.loads((root / "gccf-index.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != INDEX_SCHEMA:
        raise ValueError("unsupported GCCF source index")
    registered = {}
    with _exclusive_ingest():
        for name, source_stats in manifest["corpora"].items():
            source = (root / "sources" / name).resolve()
            if not source.is_relative_to(root / "sources"):
                raise ValueError("GCCF indexed source is outside its index")
            existing = db.q1("SELECT id,status FROM corpora WHERE source_path=?", (str(source),))
            if existing:
                if existing["status"] != "ready":
                    raise ValueError("existing indexed corpus is not ready")
                registered[name] = existing["id"]
                continue
            cid = db.new_id("cor")
            cdir = corpus_dir(cid)
            cdir.mkdir(parents=True, exist_ok=True)
            pointer = {"index_directory": str(root), "source": name}
            (cdir / "gccf-source.json").write_text(json.dumps(pointer), encoding="utf-8")
            provenance = {
                "receipt_sha256": manifest["audit"]["receipt_sha256"],
                "index_manifest": str(root / "gccf-index.json"),
                "source": name,
                "storage": "original_sources",
                "classified": True,
                "complete_source_scan": manifest["complete_source_scan"],
                "approval_scope": "file_sampled",
            }
            cfg = gccf_ingest_config(name="GCCF · " + name, source_dir=str(source))
            stats = {
                **source_stats,
                "gccf": provenance,
                "bytes_on_disk": 0,
                "ingest_config": cfg.ingest_config_stats(workers=1),
            }
            (cdir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
            (cdir / "gccf-provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
            now = db.now_iso()
            db.execute(
                "INSERT INTO corpora(id,name,created_at,updated_at,status,source_path,stats_json) "
                "VALUES(?,?,?,?,?,?,?)",
                (cid, cfg.name, now, now, "ready", str(source), json.dumps(stats)),
            )
            registered[name] = cid
            progress("gccf-index", 1, f"{cfg.name}: reads original files")
    result = {
        "index_directory": str(root),
        "storage": "original_sources",
        "corpora": registered,
        "receipt_sha256": manifest["audit"]["receipt_sha256"],
        "complete_source_scan": manifest["complete_source_scan"],
        "text_bytes_copied": 0,
    }
    (root / "yylab-import.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def iter_source_corpus(corpus_id, *, splits=None):
    from spalmer.data.gccf_index import iter_indexed_records

    pointer = json.loads((corpus_dir(corpus_id) / "gccf-source.json").read_text(encoding="utf-8"))
    stream = iter_indexed_records(pointer["index_directory"], source=pointer["source"], splits=splits)
    try:
        for row in stream:
            yield {**row, "lang": row["language"], "sample_id": row["document_id"]}
    finally:
        stream.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Register a GCCF receipt as a metadata-only index of original source files",
        allow_abbrev=False,
    )
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=str, help="metadata-index directory; corpus text is not copied")
    parser.add_argument("--limit-per-file", type=int)
    parser.add_argument("--workers", type=int)
    arguments = list(sys.argv[1:] if argv is None else argv)
    retired = {"--materialize", "--export-dir"}
    for argument in arguments:
        if argument == "--":
            break
        if argument.split("=", 1)[0] in retired:
            parser.error(
                f"{argument.split('=', 1)[0]} is unsupported: GCCF receipts reference original "
                "source data without copying it; provide the original receipt path"
            )
    args = parser.parse_args(arguments)
    db.init_db()

    def progress(stage, fraction, message):
        print(f"{stage}: {message}", flush=True)

    result = run_gccf_ingest(
        progress,
        str(args.receipt),
        output=args.output,
        limit_per_file=args.limit_per_file,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
