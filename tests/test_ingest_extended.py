from __future__ import annotations

from pathlib import Path

import pytest

from yylab import db
from yylab.artifacts import corpus_dir, read_json
from yylab.ingest.pipeline import IngestConfig, _iter_shards, run_ingest


@pytest.fixture(autouse=True)
def _clean_dedup_state():
    db.execute("DELETE FROM doc_hashes")
    db.execute("DELETE FROM rejected_hashes")

GOOD_EN = (
    "The maintenance team reviewed the overnight sensor readings and confirmed that every "
    "channel reported values within the expected range. They noted that the calibration cycle "
    "was complete and scheduled the next audit for the following week without incident."
)
GERMAN = (
    "Die Wartungsmannschaft überprüfte die nächtlichen Sensorwerte und bestätigte, dass alle "
    "Kanäle Werte im erwarteten Bereich meldeten. Die Kalibrierung war abgeschlossen und die "
    "nächste Prüfung wurde für die folgende Woche ohne Zwischenfälle angesetzt."
)
JUNK = (
    "buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now "
    "buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now "
    "buy now buy now buy now buy now buy now buy now buy now buy now buy now buy now"
)


def _ingest(tmp_path: Path, files: dict[str, str], **kw) -> tuple[str, dict]:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    for name, content in files.items():
        (src / name).write_text(content, "utf-8")
    cfg = IngestConfig(name="t", source_dir=str(src), tokenizer_keys=[], near_dedup=False, **kw)
    cid = run_ingest(lambda *a: None, cfg)
    return cid, read_json(corpus_dir(cid) / "stats.json")


def test_language_routing(tmp_path):
    cid, stats = _ingest(
        tmp_path,
        {"en.txt": GOOD_EN, "de.txt": GERMAN},
        language_filter=True,
        quality_filter=False,
    )
    assert stats["docs_written"] == 1
    assert stats["non_english_docs"] == 1
    assert stats["langs"].get("de") == 1
    assert stats["langs"].get("en") == 1
    nonen = corpus_dir(cid) / "data_nonen"
    assert stats["nonen_shards"], "expected non-english side shards"
    lines = list(_iter_shards(nonen, stats["nonen_shards"]))
    assert len(lines) == 1 and lines[0]["lang"] == "de"


def test_quality_quarantine(tmp_path):
    cid, stats = _ingest(
        tmp_path,
        {"good.txt": GOOD_EN, "junk.txt": JUNK},
        language_filter=False,
        quality_filter=True,
    )
    assert stats["docs_written"] == 1
    assert stats["quarantined_docs"] == 1
    assert stats["quarantine_reasons"], "expected quarantine reasons"
    qdir = corpus_dir(cid) / "data_quarantine"
    lines = list(_iter_shards(qdir, stats["quarantine_shards"]))
    assert len(lines) == 1
    assert lines[0]["quarantine_reasons"]
    main = list(_iter_shards(corpus_dir(cid) / "data", stats["shards"]))
    assert main[0]["quality"] >= 0.35


def test_normalized_dedup(tmp_path):
    base = "The sensor array reported stable values across all sixteen channels overnight."
    variant = "the   sensor ARRAY reported stable values across all sixteen channels overnight."
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text(base, "utf-8")
    (src / "b.txt").write_text(variant, "utf-8")
    cfg = IngestConfig(
        name="normdedup",
        source_dir=str(src),
        tokenizer_keys=[],
        near_dedup=False,
        language_filter=False,
        quality_filter=False,
    )
    cid = run_ingest(lambda *a: None, cfg)
    stats = read_json(corpus_dir(cid) / "stats.json")
    assert stats["docs_written"] == 1
    assert stats["exact_dupes_local"] == 1


def test_pdf_docx_html_parquet_adapters(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import (
        DictionaryObject,
        NameObject,
        StreamObject,
    )

    src = tmp_path / "src"
    src.mkdir()

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject()
    font.update({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                 NameObject("/BaseFont"): NameObject("/Helvetica")})
    font_ref = writer._add_object(font)
    content = StreamObject()
    content.set_data(
        b"BT /F1 12 Tf 50 700 Td (The quarterly report summarized throughput gains across "
        b"every processing line and confirmed that all targets were met this period.) Tj ET"
    )
    content_ref = writer._add_object(content)
    resources = DictionaryObject()
    font_dict = DictionaryObject()
    font_dict[NameObject("/F1")] = font_ref
    resources[NameObject("/Font")] = font_dict
    page[NameObject("/Resources")] = resources
    page[NameObject("/Contents")] = content_ref
    pdf_path = src / "report.pdf"
    with pdf_path.open("wb") as f:
        writer.write(f)

    import docx

    d = docx.Document()
    d.add_paragraph(
        "The field team completed the installation ahead of schedule and documented every "
        "step of the procedure for the operations handbook."
    )
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "channel"
    table.rows[0].cells[1].text = "7"
    d.save(src / "notes.docx")

    (src / "page.html").write_text(
        "<html><head><title>Grid status</title><script>var x=1;</script></head>"
        "<body><h1>Grid status</h1><p>All substations reported normal operation throughout "
        "the night and no faults were recorded by the monitoring systems.</p></body></html>",
        "utf-8",
    )

    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table({"text": ["Archived discussion notes about the rollout plan and the revised "
                           "timeline agreed by the steering committee."], "id": [1]}),
        src / "rows.parquet",
    )

    cfg = IngestConfig(
        name="adapters",
        source_dir=str(src),
        tokenizer_keys=[],
        near_dedup=False,
        language_filter=False,
        quality_filter=False,
    )
    cid = run_ingest(lambda *a: None, cfg)
    stats = read_json(corpus_dir(cid) / "stats.json")
    assert stats["files_processed"] == 4, stats["errors"]
    lines = list(_iter_shards(corpus_dir(cid) / "data", stats["shards"]))
    joined = "\n".join(line["text"] for line in lines)
    assert "quarterly report" in joined
    assert "field team" in joined
    assert "substations" in joined
    assert "steering committee" in joined
    html_doc = next(line for line in lines if "substations" in line["text"])
    assert "var x=1" not in html_doc["text"]
    assert html_doc["text"].startswith("Grid status")


def test_jsonl_agent_trace_serialization(tmp_path):
    import json as _json

    rec = {
        "prompt": "Fix the bug",
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix the bug"},
            {
                "role": "assistant",
                "content": "Checking the code first.",
                "reasoning_content": "The bug is likely in the parser.",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": {"path": "src/parser.py"}},
                    }
                ],
            },
            {"role": "tool", "name": "read_file", "tool_call_id": "t1", "content": "def parse(x):\n    pass"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "apply_patch", "arguments": {"patch": "+ fixed"}}}
            ]},
            {"role": "assistant", "content": "Fixed the parser bug."},
        ],
    }
    src = tmp_path / "src"
    src.mkdir()
    (src / "traces.jsonl").write_text(_json.dumps(rec), "utf-8")
    cfg = IngestConfig(
        name="traces",
        source_dir=str(src),
        tokenizer_keys=[],
        near_dedup=False,
        language_filter=False,
        quality_filter=False,
    )
    cid = run_ingest(lambda *a: None, cfg)
    stats = read_json(corpus_dir(cid) / "stats.json")
    lines = list(_iter_shards(corpus_dir(cid) / "data", stats["shards"]))
    assert len(lines) == 1
    text = lines[0]["text"]
    assert text.startswith("system: You are a coding agent.")
    assert "reasoning: The bug is likely in the parser." in text
    assert 'tool_call read_file: {"path": "src/parser.py"}' in text
    assert "tool_result read_file: def parse(x):\n    pass" in text
    assert "tool_call apply_patch" in text
    assert "assistant: Fixed the parser bug." in text
    assert len(text) > 200, "trace must be serialized, not just the prompt"


def test_conversation_classification(tmp_path):
    chat = "\n".join(f"user{i % 3}: this is message number {i} in the ongoing discussion" for i in range(30))
    cid, stats = _ingest(tmp_path, {"chat.txt": chat}, language_filter=False, quality_filter=False)
    assert stats["docs_written"] == 1
    assert "conversation" in stats["domains"]


def test_parquet_messages_struct_column(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    msg_type = pa.struct([("role", pa.string()), ("content", pa.string())])
    trace = [
        {"role": "user", "content": "Summarize the repository layout and the main entry points."},
        {"role": "assistant", "content": "The repository is organized around the lab modules."},
    ]
    table = pa.table(
        {
            "id": ["a", "b"],
            "messages": pa.array([trace, trace], type=pa.list_(msg_type)),
        }
    )
    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(table, src / "traces.parquet")
    cfg = IngestConfig(
        name="pq-traces",
        source_dir=str(src),
        tokenizer_keys=[],
        near_dedup=False,
        language_filter=False,
        quality_filter=False,
    )
    cid = run_ingest(lambda *a: None, cfg)
    stats = read_json(corpus_dir(cid) / "stats.json")
    lines = list(_iter_shards(corpus_dir(cid) / "data", stats["shards"]))
    assert len(lines) == 1, "identical traces must dedup to one doc"
    text = lines[0]["text"]
    assert text.startswith("user: Summarize the repository layout")
    assert "assistant: The repository is organized" in text


def test_scan_dataset_folders(tmp_path):
    from yylab.ingest.library import ingested_by_source, scan_dataset_folders

    root = tmp_path / "datasets"
    (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "train.jsonl").write_text('{"text": "hello world from alpha"}', "utf-8")
    (root / "beta" / "sub").mkdir(parents=True)
    (root / "beta" / "sub" / "part1.txt").write_text("beta data", "utf-8")
    (root / ".hidden").mkdir()
    (root / "stray.txt").write_text("not a folder", "utf-8")

    folders = scan_dataset_folders(str(root))
    assert [f.name for f in folders] == ["alpha", "beta"]
    assert folders[0].files == 1
    assert folders[1].files == 1
    assert folders[1].size_bytes == len("beta data")

    cfg = IngestConfig(
        name="alpha",
        source_dir=str(root / "alpha"),
        tokenizer_keys=[],
        near_dedup=False,
        language_filter=False,
        quality_filter=False,
    )
    run_ingest(lambda *a: None, cfg)
    known = ingested_by_source()
    assert str(root / "alpha") in known
    assert str(root / "beta") not in known


def test_rejected_hashes_block_ingest(tmp_path):
    bad_doc = (
        "This particular paragraph is known to be bad data and it has been rejected by a "
        "human reviewer during a swiping session in the review lab."
    )
    from yylab.ingest.dedup import norm_sha256_hex, sha256_hex

    db.executemany(
        "INSERT OR IGNORE INTO rejected_hashes(hash, added_at, source) VALUES(?,?,?)",
        [(sha256_hex(bad_doc), db.now_iso(), "test"), (norm_sha256_hex(bad_doc), db.now_iso(), "test")],
    )
    cid, stats = _ingest(
        tmp_path, {"bad.txt": bad_doc, "good.txt": GOOD_EN}, language_filter=False, quality_filter=False
    )
    assert stats["rejected_known_bad"] == 1
    assert stats["docs_written"] == 1
