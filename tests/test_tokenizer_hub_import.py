from __future__ import annotations

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from yylab import db
from yylab.artifacts import run_dir, tokenizer_dir
from yylab.tokenizer_registry import (
    HF_TOKENIZER_PRESETS,
    HUB_REPO_RE,
    create_tokenizer_record,
    get_tokenizer,
    hub_preset,
    hub_preset_options,
    is_hub_reference,
    parse_tokenizer_source,
)
from yylab.train.compact_tokenizer import CompactTokenizerSpec, train_compact_tokenizer
from yylab.ui.pages.tokenizers import build_tokenizer_import_config
from yylab.workers import tokenizer as worker
from yylab.workers.tokenizer import effective_vocab_size, run_import_job


def test_parse_tokenizer_source_distinguishes_local_paths_and_hub_ids(tmp_path):
    local = tmp_path / "tok"
    local.mkdir()
    assert parse_tokenizer_source(str(local)) == ("path", str(local.resolve()))
    assert parse_tokenizer_source("openai/gpt-oss-20b") == ("hub", "openai/gpt-oss-20b")
    assert parse_tokenizer_source("hf:Qwen/Qwen3-0.6B") == ("hub", "Qwen/Qwen3-0.6B")
    assert is_hub_reference("NousResearch/Llama-3.2-1B")
    assert not is_hub_reference(r"C:\tokenizers\missing")
    assert not is_hub_reference("./relative/dir")
    assert not is_hub_reference("no-slash")
    with pytest.raises(FileNotFoundError):
        parse_tokenizer_source(str(tmp_path / "does-not-exist"))
    with pytest.raises(ValueError):
        parse_tokenizer_source("")


def test_import_config_records_hub_source_and_revision():
    config = build_tokenizer_import_config(
        tokenizer_id="tok_hub",
        source="openai/gpt-oss-20b",
        hub_revision=" main ",
    )
    assert config["source"] == "openai/gpt-oss-20b"
    assert config["source_kind"] == "hub"
    assert config["hub_revision"] == "main"
    assert config["special_token_profile"] == "auto"


def test_presets_are_well_formed_and_larger_than_small_byte_bpes():
    repo_ids = [preset.repo_id for preset in HF_TOKENIZER_PRESETS]
    assert len(repo_ids) == len(set(repo_ids))
    for preset in HF_TOKENIZER_PRESETS:
        assert HUB_REPO_RE.match(preset.repo_id), preset.repo_id
        assert preset.vocab_size is None or preset.vocab_size > 100_000
        assert preset.repo_id in hub_preset_options()
        assert hub_preset(preset.repo_id) is preset
    o200k = hub_preset("openai/gpt-oss-20b")
    assert o200k is not None and o200k.vocab_size == 200_019
    assert hub_preset("google/gemma-3-1b-pt").gated
    assert hub_preset("nobody/nothing") is None


def test_effective_vocab_size_follows_the_spalmer_rule():
    class Stub:
        vocab_size = 100

        def __len__(self) -> int:
            return 120

        def get_vocab(self) -> dict[str, int]:
            return {"a": 0, "b": 150}

    assert effective_vocab_size(Stub()) == 151

    class Contiguous:
        vocab_size = 4

        def __len__(self) -> int:
            return 4

        def get_vocab(self) -> dict[str, int]:
            return {"a": 0, "b": 1, "c": 2, "d": 3}

    assert effective_vocab_size(Contiguous()) == 4


def test_hub_import_job_publishes_registered_artifact_without_network(monkeypatch, tmp_path):
    local = tmp_path / "hub-stand-in"
    train_compact_tokenizer(
        ["alpha beta gamma", "δ ε ζ 数学 🧪"],
        local,
        CompactTokenizerSpec(
            vocab_size=300,
            min_frequency=1,
            max_documents=2,
            special_token_profile="causal",
            normalization="none",
        ),
    )
    real_from_pretrained = AutoTokenizer.from_pretrained
    hub_calls: list[tuple[str, dict]] = []

    def fake_from_pretrained(source, *args, **kwargs):
        if Path(str(source)).exists():
            return real_from_pretrained(source, *args, **kwargs)
        hub_calls.append((str(source), kwargs))
        return real_from_pretrained(local, use_fast=True, local_files_only=True)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(worker, "_hub_snapshot_dir", lambda *args, **kwargs: local)
    monkeypatch.setattr(worker, "get_settings", lambda: {"hf_token": "hf_test_token"})

    tokenizer_id = db.new_id("tok")
    config = build_tokenizer_import_config(
        tokenizer_id=tokenizer_id,
        source="example/large-tokenizer",
        hub_revision="v1",
    )
    create_tokenizer_record(
        tokenizer_id=tokenizer_id, name="hub import", kind="imported", config=config
    )
    rd = run_dir(db.new_id("run"))
    rd.mkdir(parents=True, exist_ok=True)

    stats = run_import_job(config, rd)

    assert hub_calls and hub_calls[0][0] == "example/large-tokenizer"
    assert hub_calls[0][1]["revision"] == "v1"
    assert hub_calls[0][1]["token"] == "hf_test_token"
    assert hub_calls[0][1]["trust_remote_code"] is False
    row = get_tokenizer(tokenizer_id, parsed=True)
    assert row["status"] == "ready"
    assert row["kind"] == "imported"
    assert row["vocab_size"] == stats["effective_vocab_size"] == stats["actual_vocab_size"] > 256
    assert row["config"]["source_kind"] == "hub"
    assert row["config"]["source"] == "example/large-tokenizer"
    assert row["config"]["hub_commit"] == local.name
    assert stats["source_sha256"]
    artifact = tokenizer_dir(tokenizer_id)
    assert (artifact / "tokenizer.json").is_file()
    manifest = json.loads((artifact / "yylab_tokenizer.json").read_text("utf-8"))
    assert manifest["config"]["source_kind"] == "hub"
    reloaded = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
    probe = "alpha δ 数学 🧪"
    assert reloaded.decode(reloaded.encode(probe, add_special_tokens=False)) == probe
