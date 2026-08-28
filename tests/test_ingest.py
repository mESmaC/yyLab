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
    cfg = IngestConfig(name="unit", source_dir=str(src), tokenizer_keys=[], near_dedup=False, global_dedup=True)
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

    cfg2 = IngestConfig(name="unit2", source_dir=str(src), tokenizer_keys=[], near_dedup=False, global_dedup=True)
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
