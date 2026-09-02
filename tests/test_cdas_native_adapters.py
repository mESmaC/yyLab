from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from yylab.ingest.adapters.jsonl import JsonlAdapter
from yylab.ingest.adapters.parquet import ParquetAdapter


def _turns(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def test_native_cdas_jsonl_preserves_lineage_split_locator_and_format(tmp_path):
    sequence_boundary = {
        "kind": "document",
        "cross_sequence_attention": False,
        "loss_normalization": "shifted_nonignored_target_tokens",
    }
    record = {
        "schema_version": 1,
        "sequence_id": "doc-native-7",
        "document_id": "doc-native-7",
        "source_id": "source-native",
        "asset_id": "asset-native",
        "logical_path": "sources/train/example.md",
        "locator": "page=3;item=7",
        "format": "Markdown",
        "content_sha256": "a" * 64,
        "mixture_group": "technical",
        "selection_rank": 7,
        "sample_weight": 0.75,
        "estimated_tokens": 41,
        "target_tokens": 40,
        "sequence_boundary": sequence_boundary,
        "provenance": {"rights_status": "licensed"},
        "text": "A native CDAS record whose row-level lineage must survive ingestion.",
    }
    path = tmp_path / "shard-00000.jsonl"
    path.write_text(json.dumps(record), encoding="utf-8")

    adapter = JsonlAdapter()
    docs = list(adapter.extract(path))

    assert adapter.drain_issues() == []
    assert len(docs) == 1
    doc = docs[0]
    assert doc.format_name == "markdown"
    assert doc.meta["split"] == "train"
    assert doc.meta["logical_path"] == record["logical_path"]
    assert doc.meta["sequence_boundary"] == sequence_boundary
    assert "document_id=doc-native-7" in doc.locator
    assert "sequence_id=doc-native-7" in doc.locator
    assert "native_locator=page=3%3Bitem=7" in doc.locator
    for field in (
        "schema_version",
        "sequence_id",
        "document_id",
        "source_id",
        "asset_id",
        "logical_path",
        "locator",
        "format",
        "content_sha256",
        "mixture_group",
        "selection_rank",
        "sample_weight",
        "estimated_tokens",
        "target_tokens",
        "sequence_boundary",
    ):
        assert doc.meta[field] == record[field]
        assert doc.meta["provenance"][field] == record[field]
    assert doc.meta["provenance"]["rights_status"] == "licensed"


def test_jsonl_alias_pairs_full_documents_and_unsupported_rows(tmp_path):
    records = [
        {"query": "Why?", "response": "Because."},
        {"request": "Summarize this.", "reply": "A concise summary."},
        {"document": "A complete document value."},
        {"article": "A complete article value."},
        {"unrecognized": "must not be guessed as training text"},
    ]
    path = tmp_path / "aliases.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    adapter = JsonlAdapter()
    docs = list(adapter.extract(path))

    assert len(docs) == 4
    assert docs[0].meta["messages"] == _turns("Why?", "Because.")
    assert docs[1].meta["messages"] == _turns(
        "Summarize this.",
        "A concise summary.",
    )
    assert docs[2].text == "A complete document value."
    assert docs[3].text == "A complete article value."
    issues = adapter.drain_issues()
    assert [(locator, reason) for locator, reason, _detail in issues] == [
        ("record=5", "unsupported_record_schema")
    ]


def test_parquet_loads_every_message_alias_and_preserves_native_metadata(tmp_path):
    sparse_records = [
        {
            "messages": _turns("messages user", "messages assistant"),
            "format": "Trace",
            "document_id": "native-parquet-doc",
            "sequence_id": "native-parquet-sequence",
            "logical_path": "exports/validation/trace.jsonl",
            "source_id": "native-parquet-source",
        },
        {"conversations": _turns("conversations user", "conversations assistant")},
        {"messages_json": json.dumps(_turns("json user", "json assistant"))},
        {
            "conversations_json": json.dumps(
                _turns("conversation json user", "conversation json assistant")
            )
        },
        {"document": "A scalar document fallback in a message-bearing table."},
        {},
    ]
    columns = {
        "messages",
        "conversations",
        "messages_json",
        "conversations_json",
        "document",
        "format",
        "document_id",
        "sequence_id",
        "logical_path",
        "source_id",
    }
    records = [
        {**dict.fromkeys(columns), **record}
        for record in sparse_records
    ]
    path = tmp_path / "mixed.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)

    adapter = ParquetAdapter()
    docs = list(adapter.extract(path))

    assert len(docs) == 5
    expected_prompts = (
        "messages user",
        "conversations user",
        "json user",
        "conversation json user",
    )
    for doc, prompt in zip(docs[:4], expected_prompts, strict=True):
        assert doc.meta["messages"][0]["content"] == prompt
    assert docs[4].text == "A scalar document fallback in a message-bearing table."
    first = docs[0]
    assert first.format_name == "trace"
    assert first.meta["split"] == "validation"
    assert first.meta["provenance"]["document_id"] == "native-parquet-doc"
    assert "document_id=native-parquet-doc" in first.locator
    assert "sequence_id=native-parquet-sequence" in first.locator
    issues = adapter.drain_issues()
    assert [(locator, reason) for locator, reason, _detail in issues] == [
        ("row=6", "unsupported_record_schema")
    ]


def test_parquet_reports_an_unsupported_schema(tmp_path):
    path = tmp_path / "opaque.parquet"
    pq.write_table(pa.table({"opaque": ["not a recognized content column"]}), path)

    adapter = ParquetAdapter()
    assert list(adapter.extract(path)) == []
    issues = adapter.drain_issues()

    assert len(issues) == 1
    assert issues[0][0] == "$schema"
    assert issues[0][1] == "unsupported_parquet_schema"
    assert "opaque:string" in issues[0][2]
