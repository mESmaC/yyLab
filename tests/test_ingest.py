from __future__ import annotations

import json
from pathlib import Path

from yylab import db
from yylab.artifacts import corpus_dir, read_json
from yylab.ingest.pipeline import IngestConfig, run_ingest

PARA = (
    "Machine learning systems improve with data quality. Deduplication removes repeated "
    "examples that would otherwise bias gradient updates toward memorization of duplicates."
)

CODE = (
    "def fibonacci(n):\n"
    "    a, b = 0, 1\n"
    "    for _ in range(n):\n"
    "        a, b = b, a + b\n"
    "    return a\n"
)

MATH = "\n".join(
    [
        r"We define the loss as \frac{1}{N} \sum_i e_i where each term is computed below.",
        r"The integral \int_0^1 x^2 dx evaluates to one third.",
        r"Using \sqrt{n} scaling we get the standard error.",
        r"Let \alpha = 0.1 and \beta = 0.9 for the momentum update.",
        r"Then \lim_{n \to \infty} a_n exists by monotone convergence.",
        r"Finally \nabla_\theta J gives the policy gradient.",
    ]
)


def _make_source(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text(PARA, "utf-8")
    (src / "dup.txt").write_text(PARA, "utf-8")
    (src / "notes.md").write_text("# Notes\n\nSome markdown notes that are long enough to pass filters.", "utf-8")
    (src / "code.py").write_text(CODE, "utf-8")
    (src / "math.txt").write_text(MATH, "utf-8")
    records = [{"text": PARA}, {"text": "A completely unique record with enough characters to survive."}, {"text": PARA}]
    (src / "data.jsonl").write_text("\n".join(json.dumps(r) for r in records), "utf-8")
    (src / "mystery.xyz").write_text("unsupported", "utf-8")
    return src


def test_ingest_pipeline(tmp_path):
    src = _make_source(tmp_path)
    cfg = IngestConfig(
        name="unit",
        source_dir=str(src),
        tokenizer_keys=[],
        near_dedup=False,
        global_dedup=True,
        language_filter=False,
        quality_filter=False,
    )
    cid = run_ingest(lambda *a: None, cfg)

    row = db.q1("SELECT * FROM corpora WHERE id=?", (cid,))
    assert row["status"] == "ready"
    stats = read_json(corpus_dir(cid) / "stats.json")
    assert stats["files_found"] == 7
    assert stats["files_skipped"] == 1
    assert stats["docs_written"] == 5
    assert stats["exact_dupes_local"] == 3
    assert stats["files_processed"] == 6
    assert stats["shards"]
    assert (corpus_dir(cid) / "data" / stats["shards"][0]).exists()
    domains = set(stats["domains"].keys())
    assert "code" in domains
    assert "math" in domains

    cfg2 = IngestConfig(
        name="unit2",
        source_dir=str(src),
        tokenizer_keys=[],
        near_dedup=False,
        global_dedup=True,
        language_filter=False,
        quality_filter=False,
    )
    cid2 = run_ingest(lambda *a: None, cfg2)
    stats2 = read_json(corpus_dir(cid2) / "stats.json")
    assert stats2["docs_written"] == 0
    assert stats2["exact_dupes_global"] >= 5


def test_near_dedup():
    words = [f"w{i}" for i in range(300)]
    text_a = " ".join(words)
    altered = list(words)
    altered[50] = "zebra1"
    altered[150] = "zebra2"
    altered[250] = "zebra3"
    text_b = " ".join(altered)
    from yylab.ingest.dedup import MinHashDedup

    mh = MinHashDedup()
    assert not mh.is_duplicate(text_a)
    assert mh.is_duplicate(text_b)
    assert not mh.is_duplicate(" ".join(f"q{i}" for i in range(300)))


def test_chunked_minhash_signature_matches_reference_formula():
    import numpy as np

    from yylab.ingest.dedup import _MERSENNE_PRIME, _PERM_A, _PERM_B, _signature

    hashes = np.arange(1, 4_097, dtype=np.uint64) * np.uint64(7_919)
    reference = (
        (_PERM_A[:, None] * hashes[None, :] + _PERM_B[:, None]) % _MERSENNE_PRIME
    ).min(axis=1)
    assert np.array_equal(_signature(hashes), reference)


def test_compact_lsh_matches_datasketch_banding_without_false_duplicates():
    import numpy as np
    from datasketch import MinHash, MinHashLSH

    from yylab.ingest.dedup import NUM_PERM, MinHashDedup

    rng = np.random.default_rng(917)
    unique = [
        rng.integers(0, np.iinfo(np.uint64).max, size=NUM_PERM, dtype=np.uint64)
        for _ in range(256)
    ]
    signatures: list[np.ndarray] = []
    expected_duplicates: list[bool] = []
    for index, signature in enumerate(unique):
        signatures.append(signature)
        expected_duplicates.append(False)
        if index % 31 == 0:
            band_match = rng.integers(
                0,
                np.iinfo(np.uint64).max,
                size=NUM_PERM,
                dtype=np.uint64,
            )
            band_match[:13] = signature[:13]
            signatures.append(band_match)
            expected_duplicates.append(True)

    compact = MinHashDedup()
    compact_decisions = [compact._is_duplicate_signature(signature) for signature in signatures]

    legacy = MinHashLSH(threshold=0.8, num_perm=NUM_PERM)
    legacy_decisions: list[bool] = []
    inserted = 0
    for signature in signatures:
        minhash = MinHash(num_perm=NUM_PERM)
        minhash.hashvalues = signature
        duplicate = bool(legacy.query(minhash))
        legacy_decisions.append(duplicate)
        if not duplicate:
            legacy.insert(f"d{inserted}", minhash)
            inserted += 1

    assert legacy.hashranges == [(start, start + 13) for start in range(0, 117, 13)]
    assert compact_decisions == legacy_decisions == expected_duplicates


def test_compact_lsh_quantifies_index_memory_and_throughput():
    import sys
    import time

    import numpy as np
    from datasketch import MinHash, MinHashLSH

    from yylab.ingest.dedup import NUM_PERM, MinHashDedup

    def deep_size(value, seen: set[int] | None = None) -> int:
        seen = seen if seen is not None else set()
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(value)
        if isinstance(value, dict):
            return size + sum(
                deep_size(key, seen) + deep_size(item, seen)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple, set, frozenset)):
            return size + sum(deep_size(item, seen) for item in value)
        return size

    rng = np.random.default_rng(20260831)
    signatures = [
        rng.integers(0, np.iinfo(np.uint64).max, size=NUM_PERM, dtype=np.uint64)
        for _ in range(1_000)
    ]

    compact = MinHashDedup()
    started = time.perf_counter()
    assert not any(compact._is_duplicate_signature(signature) for signature in signatures)
    compact_seconds = time.perf_counter() - started

    legacy = MinHashLSH(threshold=0.8, num_perm=NUM_PERM)
    started = time.perf_counter()
    for index, signature in enumerate(signatures):
        minhash = MinHash(num_perm=NUM_PERM)
        minhash.hashvalues = signature
        assert not legacy.query(minhash)
        legacy.insert(f"d{index}", minhash)
    legacy_seconds = time.perf_counter() - started
    legacy_bytes = deep_size(
        [legacy.keys._dict, *(hashtable._dict for hashtable in legacy.hashtables)]
    )

    compact_bytes_per_document = compact.estimated_index_bytes / len(signatures)
    legacy_bytes_per_document = legacy_bytes / len(signatures)
    compact_docs_per_second = len(signatures) / compact_seconds
    legacy_docs_per_second = len(signatures) / legacy_seconds
    print(
        "compact MinHash LSH benchmark: "
        f"{compact_bytes_per_document:.1f} vs {legacy_bytes_per_document:.1f} bytes/doc; "
        f"{compact_docs_per_second:,.0f} vs {legacy_docs_per_second:,.0f} signatures/s"
    )

    assert compact.indexed_documents == len(signatures)
    assert compact.band_fingerprint_count == 9 * len(signatures)
    assert compact_bytes_per_document < legacy_bytes_per_document * 0.6
    assert compact_docs_per_second > legacy_docs_per_second * 2
