"""Import GCCF-approved exports into yyLab's ordinary corpus/training machinery."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import db
from ..artifacts import corpus_dir
from ..config import PATHS
from ..jobs import ProgressFn
from .pipeline import IngestConfig, _exclusive_ingest, _ShardWriter


def _receipt_tools():
    try:
        from spalmer.data.gccf import EXPORT_SCHEMA, audit_receipt, export_receipt
    except ImportError as exc:
        raise RuntimeError(
            "GCCF import requires the SPALMER checkout with receipt support. "
            "Install it into this yyLab environment: pip install --no-deps -e ../spalmer"
        ) from exc
    return EXPORT_SCHEMA, audit_receipt, export_receipt


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
    """Register each source independently so token-balanced mixes can cap large roots."""
    schema, _, _ = _receipt_tools()
    root = Path(export_dir).resolve()
    manifest = json.loads((root / "gccf-export.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != schema:
        raise ValueError("unsupported GCCF export manifest")
    registered = {}
    for index, (name, stats) in enumerate(manifest["corpora"].items()):
        source = (root / "data" / name).resolve()
        if not source.is_relative_to(root / "data") or not source.is_dir():
            raise ValueError("GCCF export source is missing or outside export directory")
        existing = db.q1("SELECT id,status FROM corpora WHERE source_path=?", (str(source),))
        if existing:
            if existing["status"] != "ready":
                raise RuntimeError(f"existing GCCF corpus {existing['id']} is {existing['status']}")
            registered[name] = existing["id"]
            continue
        progress("gccf-ingest", index / len(manifest["corpora"]), name)
        cfg = gccf_ingest_config(name="GCCF · " + name, source_dir=str(source), workers=workers)
        provenance = {
            "receipt_sha256": manifest["audit"]["receipt_sha256"],
            "export_manifest": str(root / "gccf-export.json"),
            "complete_source_scan": manifest["complete_source_scan"],
            "source": name,
            "export_stats": stats,
            "approval_scope": "file_sampled",
            "classified": True,
        }
        cid = _import_source(progress, root, cfg, stats, provenance)
        registered[name] = cid
    result = {
        "export_directory": str(root),
        "corpora": registered,
        "receipt_sha256": manifest["audit"]["receipt_sha256"],
        "complete_source_scan": manifest["complete_source_scan"],
    }
    (root / "yylab-import.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _import_source(progress, export_root, cfg, exported_stats, provenance):
    """Compress already selected records without repeating raw-data classification.

    GCCF and the shared exporter own selection/dedup/splits. Native yyLab shards
    and statistics retain those decisions; an explicit later Re-import can run
    yyLab's ordinary document analysis if desired.
    """
    with _exclusive_ingest():
        started = time.perf_counter()
        cid = db.new_id("cor")
        cdir = corpus_dir(cid)
        data_dir = cdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        now = db.now_iso()
        db.execute(
            "INSERT INTO corpora(id,name,created_at,updated_at,status,source_path) VALUES(?,?,?,?,?,?)",
            (cid, cfg.name, now, now, "processing", cfg.source_dir),
        )
        writer = _ShardWriter(data_dir)
        stats = {
            "docs_written": 0,
            "chars_written": 0,
            "domains": {},
            "splits": {},
            "langs": {},
            "errors": [],
            "gccf": provenance,
            "ingest_config": cfg.ingest_config_stats(workers=1),
        }

        def add_count(bucket, domain, chars):
            count = bucket.setdefault(domain, {"docs": 0, "chars": 0, "tokens": {}})
            count["docs"] += 1
            count["chars"] += chars

        try:
            for shard in exported_stats["shards"]:
                path = (export_root / shard).resolve()
                if not path.is_relative_to(Path(cfg.source_dir)):
                    raise ValueError("GCCF shard is outside its declared source")
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        record_provenance = row.get("provenance", {})
                        if (
                            record_provenance.get("receipt_sha256") != provenance["receipt_sha256"]
                            or record_provenance.get("verdict") != "include"
                        ):
                            raise ValueError("GCCF row does not belong to this approved receipt")
                        split = row["split"]
                        if split not in {"train", "validation", "test"}:
                            raise ValueError("GCCF row has no explicit training/validation/test split")
                        text = row["text"]
                        if not isinstance(text, str):
                            raise ValueError("GCCF row text must be a string")
                        domain = row["domain"]
                        language = row["language"]
                        native = {
                            "text": text,
                            "domain": domain,
                            "lang": language,
                            "source": row["source"],
                            "split": split,
                            "sample_id": row["document_id"],
                            "provenance": record_provenance,
                        }
                        if row.get("messages"):
                            native["messages"] = row["messages"]
                        writer.write(native)
                        chars = len(text)
                        stats["docs_written"] += 1
                        stats["chars_written"] += chars
                        stats["langs"][language] = stats["langs"].get(language, 0) + 1
                        add_count(stats["domains"], domain, chars)
                        split_counts = stats["splits"].setdefault(
                            split, {"docs": 0, "chars": 0, "domains": {}}
                        )
                        split_counts["docs"] += 1
                        split_counts["chars"] += chars
                        add_count(split_counts["domains"], domain, chars)
                        if stats["docs_written"] % 20000 == 0:
                            progress(
                                "gccf-ingest",
                                stats["docs_written"] / exported_stats["documents"],
                                f"{cfg.name}: {stats['docs_written']:,} approved documents",
                            )
            if stats["docs_written"] != exported_stats["documents"]:
                raise ValueError("GCCF export document count changed")
            writer.finish()
            stats["shards"] = writer.shards
            stats["bytes_on_disk"] = sum((data_dir / name).stat().st_size for name in writer.shards)
            seconds = time.perf_counter() - started
            stats["timing"] = {
                "seconds": seconds,
                "workers": 1,
                "docs_per_second": stats["docs_written"] / max(seconds, 1e-9),
            }
            (cdir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
            (cdir / "gccf-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            db.execute(
                "UPDATE corpora SET status='ready',stats_json=?,updated_at=? WHERE id=?",
                (json.dumps(stats), db.now_iso(), cid),
            )
        except BaseException as exc:
            writer.finish()
            db.execute(
                "UPDATE corpora SET status='failed',error=?,updated_at=? WHERE id=?",
                (str(exc)[:500], db.now_iso(), cid),
            )
            raise
        return cid


def run_gccf_ingest(
    progress: ProgressFn,
    receipt_path: str,
    *,
    output: str | None = None,
    limit_per_file: int | None = None,
    workers: int | None = None,
) -> dict:
    _, audit_receipt, export_receipt = _receipt_tools()
    audit = audit_receipt(receipt_path)
    suffix = f"-sample-{limit_per_file}" if limit_per_file else ""
    destination = Path(output) if output else (PATHS.cache / "gccf" / (audit["receipt_sha256"][:20] + suffix))
    if destination.exists():
        manifest = json.loads((destination / "gccf-export.json").read_text(encoding="utf-8"))
        if (
            manifest["audit"]["receipt_sha256"] != audit["receipt_sha256"]
            or manifest["limit_per_file"] != limit_per_file
        ):
            raise ValueError("existing export does not match the receipt and sampling limit")
    else:
        export_receipt(
            receipt_path,
            destination,
            limit_per_file=limit_per_file,
            progress=lambda message: progress("gccf-export", 0, message),
        )
    return import_export(progress, destination, workers=workers)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Import a GCCF receipt into yyLab")
    parser.add_argument("receipt", type=Path, nargs="?")
    parser.add_argument("--export-dir", type=Path, help="import an existing SPALMER GCCF export")
    parser.add_argument("--output", type=str)
    parser.add_argument("--limit-per-file", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    if bool(args.receipt) == bool(args.export_dir):
        parser.error("provide either a receipt or --export-dir")
    db.init_db()

    def progress(stage, fraction, message):
        print(f"{stage}: {message}", flush=True)

    result = (
        import_export(progress, args.export_dir, workers=args.workers)
        if args.export_dir
        else run_gccf_ingest(
            progress,
            str(args.receipt),
            output=args.output,
            limit_per_file=args.limit_per_file,
            workers=args.workers,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
