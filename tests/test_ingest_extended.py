from __future__ import annotations

import sqlite3
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


def test_semantic_normalization_is_evidence_not_a_hard_dedup_rule(tmp_path):
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
    assert stats["docs_written"] == 2
    assert stats["exact_dupes_local"] == 0


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
    assert html_doc["domain"] == "language"


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
    line = lines[0]
    text = line["text"]
    assert text.startswith("system: You are a coding agent.")
    assert "reasoning: The bug is likely in the parser." in text
    assert 'tool_call read_file: {"path": "src/parser.py"}' in text
    assert "tool_result read_file: def parse(x):\n    pass" in text
    assert "tool_call apply_patch" in text
    assert "assistant: Fixed the parser bug." in text
    assert len(text) > 200, "trace must be serialized, not just the prompt"
    assert line["split"] == "unspecified"
    assert [message["role"] for message in line["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "assistant",
    ]
    assert line["messages"][2]["reasoning_content"] == "The bug is likely in the parser."
    assert line["messages"][3]["tool_call_id"] == "t1"
    assert line["domain"] == "conversation"
    assert stats["splits"]["unspecified"]["docs"] == 1


def test_jsonl_normalizes_instruction_and_question_pairs_without_losing_answers(tmp_path):
    import json as _json

    records = [
        {
            "instruction": "Explain why the sky appears blue in clear daylight.",
            "input": "Keep the explanation concise but scientifically accurate.",
            "output": "Air molecules scatter shorter blue wavelengths more strongly.",
        },
        {
            "question": "What process converts light energy into chemical energy in plants?",
            "answer": "Photosynthesis converts light into stored chemical energy.",
        },
    ]
    cid, stats = _ingest(
        tmp_path,
        {"train.jsonl": "\n".join(_json.dumps(record) for record in records)},
        language_filter=False,
        quality_filter=False,
    )
    lines = list(_iter_shards(corpus_dir(cid) / "data", stats["shards"]))

    assert len(lines) == 2
    assert lines[0]["messages"][-1]["content"].startswith("Air molecules")
    assert "assistant: Air molecules" in lines[0]["text"]
    assert lines[1]["messages"] == [
        {
            "role": "user",
            "content": "What process converts light energy into chemical energy in plants?",
        },
        {
            "role": "assistant",
            "content": "Photosynthesis converts light into stored chemical energy.",
        },
    ]


def test_bom_prefixed_jsonl_keeps_its_first_record(tmp_path):
    import json as _json

    record = {
        "text": "A byte order mark must not erase the first valid training record in a dataset. "
        * 2
    }
    cid, stats = _ingest(
        tmp_path,
        {"train.jsonl": "\ufeff" + _json.dumps(record)},
        language_filter=False,
        quality_filter=False,
    )
    lines = list(_iter_shards(corpus_dir(cid) / "data", stats["shards"]))

    assert len(lines) == 1
    assert not lines[0]["text"].startswith("\ufeff")


def test_parquet_mixed_messages_and_normalized_trace_columns_keep_every_row(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = tmp_path / "src"
    src.mkdir()
    table = pa.table(
        {
            "messages_json": [
                '[{"role":"user","content":"First prompt"},'
                '{"role":"assistant","content":"First answer with enough detail."}]',
                None,
                None,
            ],
            "prompt_text": [None, "Second prompt asks for a normalized trace.", None],
            "completion_text": [None, "Second answer remains attached to its prompt.", None],
            "text": [
                None,
                None,
                "A plain fallback text row remains available when the messages column is null. "
                * 2,
            ],
        }
    )
    pq.write_table(table, src / "mixed.parquet")
    cfg = IngestConfig(
        name="mixed-parquet",
        source_dir=str(src),
        tokenizer_keys=[],
        near_dedup=False,
        language_filter=False,
        quality_filter=False,
    )
    cid = run_ingest(lambda *args: None, cfg)
    stats = read_json(corpus_dir(cid) / "stats.json")
    lines = list(_iter_shards(corpus_dir(cid) / "data", stats["shards"]))

    assert len(lines) == 3
    assert lines[0]["messages"][-1]["content"].startswith("First answer")
    assert lines[1]["messages"][-1]["content"].startswith("Second answer")
    assert lines[2]["text"].startswith("A plain fallback text row")


def test_pretty_json_chat_normalizes_messages_and_infers_train_split(tmp_path):
    import json as _json

    rec = {
        "conversations": [
            {
                "from": "human",
                "value": "Explain why retaining exact turn boundaries matters during chat fine-tuning.",
            },
            {
                "from": "gpt",
                "value": "Exact boundaries keep supervision aligned with assistant responses and context.",
            },
        ]
    }
    src = tmp_path / "src"
    src.mkdir()
    (src / "train.json").write_text(_json.dumps(rec, indent=2), "utf-8")
    cfg = IngestConfig(
        name="json-chat",
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
    assert lines[0]["split"] == "train"
    assert lines[0]["messages"] == [
        {
            "role": "user",
            "content": "Explain why retaining exact turn boundaries matters during chat fine-tuning.",
        },
        {
            "role": "assistant",
            "content": "Exact boundaries keep supervision aligned with assistant responses and context.",
        },
    ]
    assert stats["splits"]["train"]["docs"] == 1


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
            "split": ["testing", "testing"],
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
    line = lines[0]
    text = line["text"]
    assert text.startswith("user: Summarize the repository layout")
    assert "assistant: The repository is organized" in text
    assert line["split"] == "test"
    assert line["messages"] == trace
    assert stats["splits"]["test"]["docs"] == 1


@pytest.mark.parametrize("column", ["messages", "messages_json"])
def test_parquet_json_encoded_messages_columns(tmp_path, column):
    import json as _json

    import pyarrow as pa
    import pyarrow.parquet as pq

    trace = [
        {"role": "user", "content": "Inspect the failing parser and explain the correction."},
        {"role": "assistant", "content": "I inspected the parser and corrected its boundary check."},
    ]
    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(
        pa.table({column: [_json.dumps(trace)], "split": ["train"]}),
        src / "encoded-trace.parquet",
    )
    cid = run_ingest(
        lambda *args: None,
        IngestConfig(
            name="encoded-trace",
            source_dir=str(src),
            tokenizer_keys=[],
            near_dedup=False,
            language_filter=False,
            quality_filter=False,
        ),
    )
    stats = read_json(corpus_dir(cid) / "stats.json")
    lines = list(_iter_shards(corpus_dir(cid) / "data", stats["shards"]))
    assert len(lines) == 1
    assert lines[0]["messages"] == trace
    assert lines[0]["split"] == "train"


def test_parquet_adapter_reports_metadata_row_count(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from yylab.ingest.adapters.parquet import ParquetAdapter

    path = tmp_path / "records.parquet"
    pq.write_table(pa.table({"content": ["one", "two", "three"]}), path)

    assert ParquetAdapter().estimated_records(path) == 3


def test_jsonl_messages_json_string_is_normalized(tmp_path):
    import json as _json

    trace = [
        {"from": "human", "value": "Inspect the trace serialization path."},
        {"from": "gpt", "value": "The trace serialization path preserves both turns."},
    ]
    src = tmp_path / "src"
    src.mkdir()
    (src / "train.jsonl").write_text(
        _json.dumps({"messages_json": _json.dumps(trace)}),
        "utf-8",
    )
    cid = run_ingest(
        lambda *args: None,
        IngestConfig(
            name="jsonl-encoded-trace",
            source_dir=str(src),
            tokenizer_keys=[],
            near_dedup=False,
            language_filter=False,
            quality_filter=False,
        ),
    )
    stats = read_json(corpus_dir(cid) / "stats.json")
    lines = list(_iter_shards(corpus_dir(cid) / "data", stats["shards"]))
    assert [message["role"] for message in lines[0]["messages"]] == ["user", "assistant"]


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
    from yylab.ingest.dedup import norm_sha256_hex, sha256_hex, training_hash_key

    db.executemany(
        "INSERT OR IGNORE INTO rejected_hashes(hash, added_at, source) VALUES(?,?,?)",
        [
            (training_hash_key(sha256_hex(bad_doc)), db.now_iso(), "test"),
            (norm_sha256_hex(bad_doc), db.now_iso(), "legacy-test"),
        ],
    )
    cid, stats = _ingest(
        tmp_path, {"bad.txt": bad_doc, "good.txt": GOOD_EN}, language_filter=False, quality_filter=False
    )
    assert stats["rejected_known_bad"] == 1
    assert stats["docs_written"] == 1


def test_global_dedup_only_suppresses_exact_model_visible_text(tmp_path):
    original = "The Sensor Array reported stable values across all sixteen channels overnight."
    variant = "the   sensor array reported stable values across all sixteen channels overnight."
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "train.txt").write_text(original, "utf-8")
    (second / "train.txt").write_text(variant, "utf-8")
    common = dict(
        tokenizer_keys=[],
        near_dedup=False,
        global_dedup=True,
        language_filter=False,
        quality_filter=False,
    )
    first_id = run_ingest(
        lambda *args: None,
        IngestConfig(name="first", source_dir=str(first), **common),
    )
    assert db.q1(
        "SELECT COUNT(*) AS n FROM doc_hashes WHERE corpus_id=?",
        (first_id,),
    )["n"] == 1
    second_id = run_ingest(
        lambda *args: None,
        IngestConfig(name="second", source_dir=str(second), **common),
    )
    stats = read_json(corpus_dir(second_id) / "stats.json")
    assert stats["docs_written"] == 1
    assert stats["exact_dupes_global"] == 0

    third = tmp_path / "third"
    third.mkdir()
    (third / "train.txt").write_text(original, "utf-8")
    third_id = run_ingest(
        lambda *args: None,
        IngestConfig(name="third", source_dir=str(third), **common),
    )
    third_stats = read_json(corpus_dir(third_id) / "stats.json")
    assert third_stats["docs_written"] == 0
    assert third_stats["exact_dupes_global"] == 1


def test_jsonl_adapter_streams_line_records_without_whole_file_read(tmp_path, monkeypatch):
    import json as _json

    from yylab.ingest.adapters.jsonl import JsonlAdapter

    path = tmp_path / "large.jsonl"
    path.write_text(
        "\n".join(
            _json.dumps({"text": f"streamed training document number {index}"})
            for index in range(1_000)
        ),
        "utf-8",
    )

    def fail_read_bytes(self):  # noqa: ANN001
        raise AssertionError(f"streaming JSONL must not call read_bytes: {self}")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    docs = list(JsonlAdapter().extract(path))
    assert len(docs) == 1_000
    assert docs[0].text.endswith("0")
    assert docs[-1].text.endswith("999")


def test_jsonl_adapter_retains_pretty_json_and_malformed_line_recovery(tmp_path):
    import json as _json

    from yylab.ingest.adapters.jsonl import JsonlAdapter

    pretty = tmp_path / "pretty.jsonl"
    pretty.write_text(_json.dumps({"text": GOOD_EN}, indent=2), "utf-8")
    assert [doc.text for doc in JsonlAdapter().extract(pretty)] == [GOOD_EN]

    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        "not json\n" + _json.dumps({"text": GOOD_EN}) + "\n{broken\n",
        "utf-8",
    )
    adapter = JsonlAdapter()
    assert [doc.text for doc in adapter.extract(mixed)] == [GOOD_EN]
    assert [reason for _locator, reason, _detail in adapter.drain_issues()] == [
        "json_decode_error",
        "json_decode_error",
    ]


def test_jsonl_decode_failures_are_catalogued_without_losing_valid_rows(tmp_path):
    import json as _json

    cid, stats = _ingest(
        tmp_path,
        {"mixed.jsonl": "broken first row\n" + _json.dumps({"text": GOOD_EN})},
        language_filter=False,
        quality_filter=False,
    )

    assert stats["docs_written"] == 1
    assert stats["records_failed"] == 1
    with sqlite3.connect(corpus_dir(cid) / "inventory.sqlite3") as connection:
        issue = connection.execute(
            "SELECT locator,reason_code FROM issues"
        ).fetchone()
    assert issue == ("line=1", "json_decode_error")


def test_exact_shard_boundary_closes_empty_frame_before_unlink(tmp_path, monkeypatch):
    from yylab.ingest import pipeline

    monkeypatch.setattr(pipeline, "SHARD_DOCS", 2)
    cid, stats = _ingest(
        tmp_path,
        {"two.jsonl": '{"text":"' + GOOD_EN + '"}\n{"text":"' + GOOD_EN + " Unique." + '"}'},
        language_filter=False,
        quality_filter=False,
    )
    assert stats["docs_written"] == 2
    assert len(stats["shards"]) == 1
    assert len(list(_iter_shards(corpus_dir(cid) / "data", stats["shards"]))) == 2


def test_token_stats_are_counted_inline_without_rereading_shards(tmp_path, monkeypatch):
    from yylab.ingest import pipeline

    class FakeCounter:
        def __init__(self, keys, hf_token=None):  # noqa: ANN001, ARG002
            self.failures = {}

        def available_specs(self):
            return [object()]

        def count(self, texts):
            return {"gpt2": [len(text.split()) for text in texts]}

    monkeypatch.setattr(pipeline, "TokenCounter", FakeCounter)

    def fail_iter(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("newly written shards should not be decompressed for token stats")

    monkeypatch.setattr(pipeline, "_iter_shards", fail_iter)
    src = tmp_path / "inline"
    src.mkdir()
    (src / "train.txt").write_text(GOOD_EN, "utf-8")
    cid = run_ingest(
        lambda *args: None,
        IngestConfig(
            name="inline-token-count",
            source_dir=str(src),
            tokenizer_keys=["gpt2"],
            near_dedup=False,
            language_filter=False,
            quality_filter=False,
        ),
    )
    stats = read_json(corpus_dir(cid) / "stats.json")
    assert stats["totals"]["tokens"]["gpt2"] == len(GOOD_EN.split())


def test_curated_library_ingest_avoids_recleaning_published_datasets():
    from yylab.ui.pages.corpora import curated_library_ingest_config

    config = curated_library_ingest_config(path=r"D:\datasets\published", name="published")

    assert config.global_dedup is True
    assert config.near_dedup is False
    assert config.language_filter is False
    assert config.quality_filter is False
    assert config.tokenizer_keys == []
