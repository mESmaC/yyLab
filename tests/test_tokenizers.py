from __future__ import annotations

import json

from transformers import AutoTokenizer

from yylab import db
from yylab.artifacts import tokenizer_dir
from yylab.tokenizer_registry import (
    artifact_checksum,
    create_tokenizer_record,
    get_tokenizer,
    resolve_tokenizer_artifact,
)
from yylab.train.compact_tokenizer import CompactTokenizerSpec, train_compact_tokenizer
from yylab.ui.pages.tokenizers import (
    build_tokenizer_import_config,
    build_tokenizer_train_config,
)
from yylab.workers import tokenizer as tokenizer_worker
from yylab.workers.tokenizer import evaluate_tokenizer, import_tokenizer_artifact


def test_tokenizer_registry_roundtrip_and_resolver():
    tokenizer_id = db.new_id("tok")
    output = tokenizer_dir(tokenizer_id)
    train_compact_tokenizer(
        ["registry probe"],
        output,
        CompactTokenizerSpec(
            vocab_size=257,
            min_frequency=1,
            max_documents=1,
            special_token_profile="causal",
        ),
    )
    create_tokenizer_record(
        tokenizer_id=tokenizer_id,
        name="registry probe",
        kind="byte_bpe",
        corpora=["cor_a"],
        splits=["train"],
        config={"probe": True},
        status="ready",
    )
    db.execute(
        "UPDATE tokenizers SET vocab_size=?,path=?,stats_json=? WHERE id=?",
        (257, str(output), json.dumps({"bytes_per_token": 2.0}), tokenizer_id),
    )

    row = get_tokenizer(tokenizer_id, parsed=True)
    assert row["corpora"] == ["cor_a"]
    assert row["splits"] == ["train"]
    assert resolve_tokenizer_artifact(tokenizer_id) == output.resolve()


def test_tokenizer_payload_checksum_is_reproducible_after_manifest(tmp_path):
    train_compact_tokenizer(
        ["checksum probe"],
        tmp_path,
        CompactTokenizerSpec(
            vocab_size=257,
            min_frequency=1,
            max_documents=1,
            special_token_profile="causal",
        ),
    )
    before = artifact_checksum(tmp_path)
    (tmp_path / "yylab_tokenizer.json").write_text(
        json.dumps({"stats": {"tokenizer_payload_sha256": before}}),
        encoding="utf-8",
    )
    assert artifact_checksum(tmp_path) == before


def test_raw_tokenizer_json_import_builds_hf_wrapper_without_mutating_source(tmp_path):
    source = tmp_path / "source"
    train_compact_tokenizer(
        ["ASCII ＦＵＬＬＷＩＤＴＨ ① λ 数学 🧪"],
        source,
        CompactTokenizerSpec(
            vocab_size=320,
            min_frequency=1,
            max_documents=1,
            special_token_profile="causal",
            normalization="none",
        ),
    )
    before = (source / "tokenizer.json").read_bytes()
    destination = tmp_path / "imported"
    tokenizer, special = import_tokenizer_artifact(source / "tokenizer.json", destination)

    assert (source / "tokenizer.json").read_bytes() == before
    assert (destination / "tokenizer_config.json").is_file()
    assert (destination / "special_tokens_map.json").is_file()
    assert (destination / "tokenizer.json").is_file()
    assert special["profile"] == "causal"
    assert tokenizer.eos_token == "<|endoftext|>"
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    reloaded = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    probe = "ASCII ＦＵＬＬＷＩＤＴＨ ① λ 数学 🧪"
    assert reloaded.decode(reloaded.encode(probe, add_special_tokens=False)) == probe


def test_evaluate_tokenizer_reports_compression_utilization_and_roundtrip(tmp_path):
    tokenizer = train_compact_tokenizer(
        ["alpha beta", "gamma delta"],
        tmp_path,
        CompactTokenizerSpec(
            vocab_size=280,
            min_frequency=1,
            max_documents=2,
            special_token_profile="causal",
        ),
    )
    metrics = evaluate_tokenizer(
        tokenizer,
        [{"text": "alpha βeta 🧪", "source": "probe"}],
        max_documents=1,
    )
    assert metrics["documents"] == 1
    assert metrics["tokens"] > 0
    assert metrics["bytes_per_token"] > 0
    assert 0 < metrics["vocab_utilization"] <= 1
    assert metrics["round_trip_failures"] == 0


def test_ui_config_accepts_arbitrary_small_vocab_and_split_defaults(tmp_path):
    train = build_tokenizer_train_config(
        tokenizer_id="tok_test",
        corpora=["cor_a", "cor_b"],
        splits=[],
        evaluation_splits=["validation", "test"],
        domains=["code"],
        vocab_size=257,
        min_frequency=1,
        max_documents=10,
        max_training_bytes=1024,
        normalization="none",
        special_token_profile="causal",
    )
    assert train["splits"] == ["train", "unspecified"]
    assert train["corpus_mix"] == "round_robin"
    assert train["spec"]["vocab_size"] == 257
    assert train["spec"]["normalization"] == "none"

    source = tmp_path / "tokenizer.json"
    source.write_text("{}", encoding="utf-8")
    imported = build_tokenizer_import_config(
        tokenizer_id="tok_import",
        source=str(source),
    )
    assert imported["source"] == str(source.resolve())


def test_streaming_worker_publishes_ready_registry_artifact(monkeypatch, tmp_path):
    tokenizer_id = db.new_id("tok")
    config = build_tokenizer_train_config(
        tokenizer_id=tokenizer_id,
        corpora=["cor_stream"],
        splits=["train"],
        evaluation_splits=["validation"],
        domains=None,
        vocab_size=280,
        min_frequency=1,
        max_documents=10,
        max_training_bytes=0,
        normalization="none",
        special_token_profile="causal",
        evaluation_max_documents=2,
    )
    create_tokenizer_record(
        tokenizer_id=tokenizer_id,
        name="stream worker",
        kind="byte_bpe",
        corpora=config["corpora"],
        splits=config["splits"],
        config=config,
    )

    def fake_stream(_config, *, splits, seed_offset=0):
        del seed_offset
        label = splits[0]
        return iter(
            [
                {"text": f"{label} α fullwidth Ａ sample one", "source": f"{label}.jsonl"},
                {"text": f"{label} β sample two 🧪", "source": f"{label}.jsonl"},
            ]
        )

    monkeypatch.setattr(tokenizer_worker, "_document_stream", fake_stream)
    monkeypatch.setattr(
        tokenizer_worker,
        "corpus_data_provenance",
        lambda *_args, **_kwargs: {"corpus_ids": ["cor_stream"], "estimated_docs": 2},
    )
    stats = tokenizer_worker.run_train_job(config, tmp_path / "run")

    row = get_tokenizer(tokenizer_id, parsed=True)
    assert row["status"] == "ready"
    assert row["vocab_size"] == stats["actual_vocab_size"]
    assert stats["evaluation_mode"] == "held_out"
    assert stats["evaluation"]["documents"] == 2
    assert artifact_checksum(row["path"]) == stats["tokenizer_payload_sha256"]
