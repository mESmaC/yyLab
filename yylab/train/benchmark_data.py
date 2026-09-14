"""CPU-only delivery benchmark; it does not construct or train a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
from transformers import AutoTokenizer

from ..tokenizer_registry import get_tokenizer, resolve_tokenizer_artifact
from ..workers.train import build_batches
from .prefetch import batch_stream
from .prepared import tokenizer_fingerprint


def measure(cfg, tokenizer, batches, consumer_ms):
    start = time.perf_counter()
    packed = build_batches(cfg, tokenizer, batches)
    setup_seconds = time.perf_counter() - start
    digest = hashlib.sha256()
    waits = []
    tokens = 0
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    with batch_stream(packed) as iterator:
        while True:
            before = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            waits.append(time.perf_counter() - before)
            ids = np.asarray(batch["input_ids"], dtype="<u4")
            digest.update(ids.tobytes())
            tokens += ids.size
            if consumer_ms:
                time.sleep(consumer_ms / 1000)
            if len(waits) % 100 == 0:
                peak_rss = max(peak_rss, process.memory_info().rss)
    elapsed = time.perf_counter() - started
    return {
        "setup_seconds": setup_seconds, "drain_seconds": elapsed,
        "next_batch_wait_seconds": sum(waits), "batches": len(waits), "input_tokens": tokens,
        "delivery_tokens_per_second": tokens / max(sum(waits), 1e-9),
        "next_batch_p50_ms": statistics.median(waits) * 1000 if waits else 0,
        "next_batch_p95_ms": float(np.percentile(waits, 95)) * 1000 if waits else 0,
        "sha256": digest.hexdigest(), "sampled_peak_process_rss_bytes": peak_rss,
        "prepared_tokens": packed.data_stats().get("prepared_tokens", {}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--shard-mib", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("--prefetch", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--batches", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--consumer-ms", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batches, args.repeat, args.sequence_length, *args.shard_mib) < 1 or min(args.prefetch) < 0 or args.consumer_ms < 0:
        parser.error("sizes and repeats must be positive; prefetch and consumer time must be non-negative")
    ref = str(resolve_tokenizer_artifact(args.tokenizer)) if get_tokenizer(args.tokenizer) else args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ref, local_files_only=True)
    cfg = {
        "corpora": args.corpora, "seq_len": args.sequence_length,
        "micro_batch": 1, "grad_accum": 1, "splits": ["train"], "seed": 42,
    }
    result = {
        "kind": "cpu_data_delivery_benchmark", "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "config": cfg, "consumer_ms": args.consumer_ms,
        "note": "No model/GPU training. OS page cache is not flushed. First setup includes preparation; later setups reuse tokens. Delivery throughput excludes consumer/hash work; drain time includes it. RSS is sampled, not a guaranteed process peak.",
        "cases": [],
    }
    expected = None
    for shard_mib in [None, *args.shard_mib]:
        for depth in args.prefetch:
            case_cfg = {**cfg, "token_cache": shard_mib is not None, "data_prefetch_batches": depth}
            if shard_mib is not None:
                case_cfg["token_shard_bytes"] = shard_mib * 2**20
            runs = []
            for repeat in range(args.repeat):
                print(f"Benchmark shard_mib={shard_mib}, prefetch={depth}, repeat={repeat + 1}", flush=True)
                run = measure(case_cfg, tokenizer, args.batches, args.consumer_ms)
                identity = (run["sha256"], run["input_tokens"], run["batches"])
                if expected is None:
                    expected = identity
                if identity != expected:
                    raise RuntimeError("data delivery benchmark changed the training token stream")
                runs.append(run)
            result["cases"].append({"shard_mib": shard_mib, "prefetch": depth, "runs": runs})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"All delivery streams identical; results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
