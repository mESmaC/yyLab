"""Register an unclassified text supplement backed by original files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .. import db
from ..artifacts import corpus_dir
from ..config import PATHS
from .pipeline import _exclusive_ingest


def run_text_source(
    progress,
    source_directory,
    *,
    validation_files=(),
    test_files=(),
    end_before_lines=None,
    output=None,
    preprocessing="prose_v1",
):
    from spalmer.data.text_index import build_text_index, preprocessing_identity

    profile = preprocessing_identity(preprocessing)
    source = Path(source_directory).resolve()
    inventory = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(source.glob("*.txt"))]
    request = {
        "source": str(source),
        "files": inventory,
        "validation": sorted(validation_files),
        "test": sorted(test_files),
        "boundaries": end_before_lines or {},
    }
    if profile is not None:
        request["preprocessing"] = profile
    identity = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:20]
    destination = Path(output) if output else PATHS.cache / "text-sources" / identity
    if not destination.exists():
        progress("text-index", 0, "Indexing original text files without copying them")
        build_text_index(
            source,
            destination,
            validation_files=validation_files,
            test_files=test_files,
            end_before_lines=end_before_lines,
            preprocessing=preprocessing,
        )
    else:
        manifest = json.loads((destination / "text-index.json").read_text(encoding="utf-8"))
        indexed_files = manifest["files"]
        indexed_request = {
            "source": manifest["source_directory"],
            "files": sorted((f["filename"], f["size_bytes"], f["mtime_ns"]) for f in indexed_files),
            "validation": sorted(f["filename"] for f in indexed_files if f["split"] == "validation"),
            "test": sorted(f["filename"] for f in indexed_files if f["split"] == "test"),
            "boundaries": {
                f["filename"]: f["end_before_line"] for f in indexed_files if f["end_before_line"] is not None
            },
        }
        if manifest.get("preprocessing") is not None:
            indexed_request["preprocessing"] = manifest["preprocessing"]
        request["files"] = sorted(inventory)
        if indexed_request != request:
            raise ValueError(
                "existing text index does not match the requested source selection; use a new output"
            )
    return register_text_index(progress, destination)


def register_text_index(progress, index_directory):
    from spalmer.data.text_index import TEXT_INDEX_SCHEMAS, iter_indexed_text

    root = Path(index_directory).resolve()
    path = root / "text-index.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in TEXT_INDEX_SCHEMAS or manifest.get("classified") is not False:
        raise ValueError("expected an unclassified text source index")
    verified = sum(1 for _ in iter_indexed_text(root))
    if verified != manifest["stats"]["docs_written"]:
        raise ValueError("text source count does not match its index")
    with _exclusive_ingest():
        existing = db.q1("SELECT id,status FROM corpora WHERE source_path=?", (str(path),))
        if existing:
            if existing["status"] != "ready":
                raise ValueError("existing text source is not ready")
            cid = existing["id"]
        else:
            cid = db.new_id("cor")
            cdir = corpus_dir(cid)
            cdir.mkdir(parents=True, exist_ok=True)
            provenance = {
                "storage": "original_sources",
                "selection_origin": "user_supplied_raw",
                "classified": False,
                "index_manifest": str(path),
                "index_sha256": manifest["index_sha256"],
                "source_directory": manifest["source_directory"],
                "split_policy": manifest["split_policy"],
                "preprocessing": manifest.get("preprocessing"),
            }
            stats = {**manifest["stats"], "source_index": provenance, "bytes_on_disk": 0}
            (cdir / "text-source.json").write_text(
                json.dumps({"index_directory": str(root)}), encoding="utf-8"
            )
            (cdir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
            now = db.now_iso()
            db.execute(
                "INSERT INTO corpora(id,name,created_at,updated_at,status,source_path,stats_json) VALUES(?,?,?,?,?,?,?)",
                (
                    cid,
                    ("Prepared prose · " if manifest.get("preprocessing") else "Raw prose · ")
                    + manifest["name"],
                    now,
                    now,
                    "ready",
                    str(path),
                    json.dumps(stats),
                ),
            )
    result = {
        "corpus_id": cid,
        "index_directory": str(root),
        "classified": False,
        "text_bytes_copied": 0,
        "documents": verified,
        "stats": manifest["stats"],
        "preprocessing": manifest.get("preprocessing"),
    }
    (root / "yylab-import.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    progress("text-index", 1, f"{verified:,} documents available from original files")
    return result


def iter_text_source_corpus(corpus_id, *, splits=None):
    from spalmer.data.text_index import iter_indexed_text

    pointer = json.loads((corpus_dir(corpus_id) / "text-source.json").read_text(encoding="utf-8"))
    records = iter_indexed_text(pointer["index_directory"], splits=splits)
    try:
        for row in records:
            yield {**row, "lang": row["language"], "sample_id": row["document_id"]}
    finally:
        records.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Use original prose files as an unclassified supplement")
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--selection", type=Path, help="JSON with validation_files, test_files, end_before_lines"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preprocessing", choices=["none", "prose_v1"], default="prose_v1")
    args = parser.parse_args(argv)
    selection = json.loads(args.selection.read_text(encoding="utf-8")) if args.selection else {}
    selection["preprocessing"] = None if args.preprocessing == "none" else args.preprocessing
    db.init_db()
    result = run_text_source(
        lambda stage, fraction, message: print(message, flush=True),
        args.source,
        output=args.output,
        **selection,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
