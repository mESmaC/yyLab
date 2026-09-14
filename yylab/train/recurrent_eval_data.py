"""Freeze a validation view with one window per distinct document for depth evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import corpus_data_dir, iter_corpus_docs


def prepare_view(tokenizer, sources, *, documents=128, targets=512, pool=2048, seed=20260905):
    rows, references, reports = [], [], []
    seen = set()
    for source in sources:
        candidates = []
        stream = iter_corpus_docs(
            [source["corpus_id"]],
            domains=source.get("domains"),
            splits=["validation"],
            shuffle_buffer=1,
            cross_corpus_dedup=False,
        )
        scanned = 0
        try:
            for record in stream:
                scanned += 1
                text = record.get("text", "")
                if text and len(text) <= 262144:
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    priority = hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()
                    candidates.append((priority, digest, text, record.get("source")))
                if scanned >= pool:
                    break
        finally:
            stream.close()
        selected = 0
        for priority, digest, text, locator in sorted(candidates, key=lambda item: item[0]):
            if digest in seen:
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) < targets + 1:
                continue
            start = int(priority, 16) % (len(token_ids) - targets)
            window = token_ids[start : start + targets + 1]
            rows.append(window)
            references.append(
                {
                    "source": source["name"],
                    "corpus_id": source["corpus_id"],
                    "split": "validation",
                    "locator": locator,
                    "text_sha256": digest,
                    "document_tokens": len(token_ids),
                    "window_start": start,
                    "window_sha256": hashlib.sha256(np.asarray(window, dtype="<u4").tobytes()).hexdigest(),
                }
            )
            seen.add(digest)
            selected += 1
            if selected == documents:
                break
        if selected < documents:
            raise ValueError(f"{source['name']}: only {selected}/{documents} eligible documents")
        reports.append(
            {
                **source,
                "data_directory": str(corpus_data_dir(source["corpus_id"])),
                "validation_records_scanned": scanned,
                "selected_documents": selected,
            }
        )
        print(json.dumps({"prepared": source["name"], "documents": selected}), flush=True)
    return np.asarray(rows, dtype="<u4"), {
        "schema": "spalmer.depth-evaluation-data/1",
        "tokenizer": tokenizer.identity.to_dict(),
        "seed": seed,
        "targets_per_document": targets,
        "sources": reports,
        "sampling": (
            "Hash-ranked documents from a bounded prefix of each validation stream; one random "
            "in-document window per distinct text. Short (<513 tokens at default length) and "
            ">262144-character documents excluded. This is not a corpus-wide random sample."
        ),
        "references": references,
    }


def main():
    from spalmer.tokenizer.backends import HFTokenizerAdapter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=128)
    parser.add_argument("--targets", type=int, default=512)
    parser.add_argument("--pool", type=int, default=2048)
    args = parser.parse_args()
    if min(args.documents, args.targets, args.pool) < 1 or args.pool < args.documents:
        parser.error("positive sizes are required, with pool >= documents")
    if args.output.exists():
        parser.error("output already exists; reuse its frozen tokens or choose another directory")
    tokenizer = HFTokenizerAdapter.from_pretrained(args.tokenizer, local_files_only=True)
    tokens, manifest = prepare_view(
        tokenizer,
        json.loads(args.sources.read_text(encoding="utf-8")),
        documents=args.documents,
        targets=args.targets,
        pool=args.pool,
    )
    args.output.mkdir(parents=True)
    token_path = args.output / "tokens.npy"
    np.save(token_path, tokens, allow_pickle=False)
    manifest["token_file_sha256"] = hashlib.sha256(token_path.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "shape": tokens.shape, "bytes": tokens.nbytes}))


if __name__ == "__main__":
    main()
