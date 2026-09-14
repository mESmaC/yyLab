"""Finite token quotas must survive source exhaustion, reuse and stopped runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zstandard as zstd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from yylab import db
from yylab.train import data, prepared, sprint, sprint_mix


@pytest.fixture
def quota_recipe(tmp_path, monkeypatch):
    db.init_db()
    corpora = [f"quota-short-{tmp_path.name}", f"quota-long-{tmp_path.name}"]
    paths = {cid: tmp_path / str(index) for index, cid in enumerate(corpora)}
    monkeypatch.setattr(data, "corpus_dir", paths.__getitem__)
    monkeypatch.setattr(prepared, "corpus_dir", paths.__getitem__)

    def write_source(cid, texts):
        root = paths[cid] / "data"
        root.mkdir(parents=True, exist_ok=True)
        records = [{"id": str(i), "text": text, "split": "train"} for i, text in enumerate(texts)]
        (root / "shard-00000.jsonl.zst").write_bytes(zstd.ZstdCompressor().compress(
            "\n".join(json.dumps(record) for record in records).encode()))

    for cid, text in zip(corpora, ["short " * 32, "long " * 800], strict=True):
        db.execute("INSERT INTO corpora(id,name,created_at,updated_at,status) VALUES(?,?,?,?,?)",
                   (cid, cid, db.now_iso(), db.now_iso(), "ready"))
        write_source(cid, [text])
    raw = Tokenizer(models.WordLevel(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    raw.train_from_iterator(["short long unique first second"], trainers.WordLevelTrainer(
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>",
                                       bos_token="<bos>", eos_token="<eos>")
    return {"corpora": corpora, "splits": ["train"], "corpus_mix": sprint_mix.MODE,
            "objective": "causal_lm", "seq_len": 8, "micro_batch": 2, "grad_accum": 1,
            "token_budget": 64 * 8, "max_steps": 0, "device": "cpu", "token_cache": False,
            "cross_corpus_dedup": True, "packed_cache_dir": str(tmp_path / "packed"),
            "packed_cache_verify": "sha256", "seed": 7}, tokenizer, write_source


def normalized(raw, **changes):
    return sprint.normalize_sprint_config({**raw, **changes})


def test_optional_row_filter_identity_reuses_raw_pools_and_maps_original_rows(quota_recipe, monkeypatch):
    raw, tokenizer, _ = quota_recipe
    cfg = normalized(raw, token_budget=16)
    before = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    raw_key = before.manifest["row_provenance"]["pool_key"]
    raw_cache_key = before.key
    before.close()
    root = Path(cfg["packed_cache_dir"])
    raw_pool_root = root / "quota-source-pools" / raw_key
    raw_bytes = [(raw_pool_root / f"source-{i}.bin").read_bytes() for i in range(2)]
    disabled = normalized(raw, token_budget=16, quota_row_filter={"enabled": False})
    assert "quota_row_filter" not in disabled
    assert sprint._packed_identity(cfg, tokenizer) == sprint._packed_identity(disabled, tokenizer)
    enabled = normalized(raw, token_budget=16, quota_row_filter={"enabled": True})
    enabled_identity = sprint._packed_identity(enabled, tokenizer)
    assert enabled_identity != sprint._packed_identity(cfg, tokenizer)
    assert sprint_mix._pool_identity(enabled_identity) != sprint_mix._pool_identity(sprint._packed_identity(cfg, tokenizer))

    def must_reuse(*args, **kwargs):
        raise AssertionError("enabling or reusing row filtering must not repack raw data")

    monkeypatch.setattr(sprint_mix, "iter_corpus_docs", must_reuse)
    cleaned = sprint.prepare_sprint_cache(enabled, tokenizer, progress=lambda _: None)
    try:
        provenance = cleaned.manifest["row_provenance"]
        assert cleaned.key != raw_cache_key
        assert provenance["unfiltered_pool_key"] == raw_key
        assert provenance["pool_key"] != raw_key
        assert "retained rows" in provenance["row_position_scope"]
        cleaned_root = root / "quota-source-pools" / provenance["pool_key"]
        for i, cid in enumerate(cfg["corpora"]):
            pool = provenance["pools"][cid]
            assert pool["rows"] == 1
            assert pool["row_filter"]["input_rows"] == (4 if i == 0 else 100)
            assert pool["data_stats"]["tokens"]["forwarded_real_inputs"] == 8
            assert pool["raw_data_stats"]
            assert np.fromfile(cleaned_root / f"source-{i}.original_rows.bin", dtype="<u8").tolist() == [0]
            assert (raw_pool_root / f"source-{i}.bin").read_bytes() == raw_bytes[i]
        assert cleaned.prefix_data_stats(1)["unique_position_scope"] == "retained_source_pool_positions_after_configured_row_filter"
        prefix = cleaned.prefix_data_stats(1)
        assert prefix["row_filter"]["scope"] == "complete_source_pool_preparation_not_consumed_prefix"
        assert prefix["tokens"]["forwarded_real_inputs"] == 8
        assert prefix["row_filter"]["per_corpus"][cfg["corpora"][1]]["rejected_rows"] == 99
    finally:
        cleaned.close()
    reused = sprint.prepare_sprint_cache(enabled, tokenizer, progress=lambda _: None)
    assert reused.preparation["mode"] == "exact_cache"
    reused.close()
    # A new quota schedule reuses the cleaned pools, not just the final cache.
    changed = normalized(raw, token_budget=16, seed=8, quota_row_filter={"enabled": True})
    rebuilt = sprint.prepare_sprint_cache(changed, tokenizer, progress=lambda _: None)
    assert rebuilt.manifest["row_provenance"]["pool_key"] == provenance["pool_key"]
    rebuilt.close()


def test_row_filter_admission_uses_retained_availability(quota_recipe):
    raw, tokenizer, _ = quota_recipe
    cfg = normalized(raw, token_budget=32, quota_row_filter={"enabled": True})
    with pytest.raises(ValueError, match="available pool has 1 rows"):
        sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    explicit_replay = normalized(raw, token_budget=32, quota_row_filter={"enabled": True}, quota_max_source_epochs=2)
    cache = sprint.prepare_sprint_cache(explicit_replay, tokenizer, progress=lambda _: None)
    assert cache.data_stats["tokens"]["packing_replay_inputs"] == 16
    cache.close()


def test_row_filter_exception_leaves_raw_pool_and_removes_partial_derived_state(quota_recipe, monkeypatch):
    raw, tokenizer, _ = quota_recipe
    cfg = normalized(raw, token_budget=16)
    cache = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    pool_key = cache.manifest["row_provenance"]["pool_key"]
    cache.close()
    parent = Path(cfg["packed_cache_dir"]) / "quota-source-pools"

    def fail(*args, **kwargs):
        raise RuntimeError("intentional filter failure")

    monkeypatch.setattr(sprint_mix.QuotaRowFilter, "filter_batch", fail)
    cfg = normalized(raw, token_budget=16, quota_row_filter={"enabled": True})
    with pytest.raises(RuntimeError, match="intentional filter failure"):
        sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    assert [p.name for p in parent.iterdir()] == [pool_key]


def test_filtered_original_row_mapping_has_integrity_checks(quota_recipe):
    raw, tokenizer, _ = quota_recipe
    cfg = normalized(raw, token_budget=16, quota_row_filter={"enabled": True})
    cache = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    pool_key = cache.manifest["row_provenance"]["pool_key"]
    cache.close()
    parent = Path(cfg["packed_cache_dir"]) / "quota-source-pools" / pool_key
    (parent / "source-0.original_rows.bin").write_bytes(b"bad")
    # A new materialization must load/check the existing derived pool group.
    changed = normalized(raw, token_budget=16, seed=9, quota_row_filter={"enabled": True})
    with pytest.raises(ValueError, match="size mismatch"):
        sprint.prepare_sprint_cache(changed, tokenizer, progress=lambda _: None)


def test_row_filter_removes_cross_source_token_duplicates_after_different_text_rendering(quota_recipe):
    raw, tokenizer, write_source = quota_recipe
    first, second = raw["corpora"]
    write_source(first, ["short long unique first " * 4])
    write_source(second, ["\n".join(["short", "long", "unique", "first"] * 4) + " second" * 16])
    cfg = normalized(raw, token_budget=16, quota_row_filter={"enabled": True})
    cache = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    try:
        pools = cache.manifest["row_provenance"]["pools"]
        samples = pools[second]["row_filter"]["rejected_samples"]
        assert any(s.get("duplicate_of", {}).get("source") == first for s in samples)
        assert pools[first]["rows"] == pools[second]["rows"] == 1
        assert cache.rows == 2
    finally:
        cache.close()


def test_largest_remainder_allocation_and_seeded_cycle_coverage():
    assert sprint_mix.allocate_rows(11, {"a": 1, "b": 1, "c": 1}) == {"a": 4, "b": 4, "c": 3}
    assert sprint_mix.allocate_rows(1000, {"a": 1, "b": 99}) == {"a": 10, "b": 990}
    first = sprint_mix.sample_pool_rows(5, 12, np.random.default_rng(9))
    second = sprint_mix.sample_pool_rows(5, 12, np.random.default_rng(9))
    np.testing.assert_array_equal(first, second)
    assert set(first[:5]) == set(first[5:10]) == set(range(5))
    assert len(set(first[10:])) == 2


def test_small_source_exhaustion_fails_instead_of_becoming_large_source(quota_recipe):
    raw, tokenizer, _ = quota_recipe
    cfg = normalized(raw)
    with pytest.raises(ValueError, match="other sources will not silently replace it"):
        sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    assert cfg["quota_max_source_epochs"] == dict.fromkeys(cfg["corpora"], 1.0)
    # Complete pools survive admission failure so changing the replay budget
    # does not tokenize and pack all input data a second time.
    pool_manifests = list((Path(cfg["packed_cache_dir"]) / "quota-source-pools").glob("*/manifest.json"))
    assert len(pool_manifests) == 1


def test_exact_quota_replay_provenance_prefix_and_resume_seek(quota_recipe):
    raw, tokenizer, _ = quota_recipe
    cfg = normalized(raw, quota_max_source_epochs={raw["corpora"][0]: 8})
    cache = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    try:
        assert cache.rows == 64
        stats = cache.data_stats
        short, long = cfg["corpora"]
        assert [stats["per_corpus"][cid]["forwarded_real_inputs"] for cid in cfg["corpora"]] == [256, 256]
        assert stats["per_corpus"][short]["packing_replay_inputs"] == 28 * 8
        assert stats["per_corpus"][long]["packing_replay_inputs"] == 0
        assert stats["tokens"]["optimizer_supervised_targets"] == 64 * 7
        # The first epoch already contains a seeded global mixture.
        assert set(cache.provenance["row_source"][:16]) == {0, 1}
        for count in (0, 1, 9, 31, 64):
            prefix = cache.prefix_data_stats(count)
            assert prefix["tokens"]["forwarded_real_inputs"] == count * 8
            assert prefix["tokens"]["optimizer_supervised_targets"] == count * 7
            pairs = list(zip(cache.provenance["row_source"][:count].tolist(),
                             cache.provenance["row_pool_index"][:count].tolist(), strict=True))
            assert prefix["tokens"]["covered_rendered_positions"] == len(set(pairs)) * 8
            assert prefix["tokens"]["packing_replay_inputs"] == (count - len(set(pairs))) * 8
        full = cache.read_batch(0, 64)["input_ids"]
        resumed = cache.read_batch(17, 47)["input_ids"]
        assert full[17:].equal(resumed)
        for source in (0, 1):
            token = tokenizer.convert_tokens_to_ids("short" if source == 0 else "long")
            assert np.all(cache.arrays["input_ids"][cache.provenance["row_source"] == source] == token)
        first_bytes = (cache.root / "input_ids.bin").read_bytes()
        first_key = cache.key
    finally:
        cache.close()
    exact = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    try:
        assert exact.preparation["mode"] == "exact_cache"
        assert exact.key == first_key
        assert (exact.root / "input_ids.bin").read_bytes() == first_bytes
    finally:
        exact.close()


def test_changed_budget_weights_seed_reuse_full_pools_without_repacking(quota_recipe, monkeypatch):
    raw, tokenizer, _ = quota_recipe
    cfg = normalized(raw, quota_max_source_epochs=8)
    cache = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    pool_key = cache.manifest["row_provenance"]["pool_key"]
    key = cache.key
    cache.close()
    monkeypatch.setattr(sprint_mix, "iter_corpus_docs", lambda *a, **k: pytest.fail("full source pool was repacked"))
    monkeypatch.setattr(sprint_mix, "prepare_for_training", lambda *a, **k: pytest.fail("tokenizer was reopened"))
    changed = normalized(raw, token_budget=17 * 8, seed=99,
                         corpus_weights=dict(zip(raw["corpora"], [1, 3], strict=True)), quota_max_source_epochs=2)
    repicked = sprint.prepare_sprint_cache(changed, tokenizer, progress=lambda _: None)
    try:
        assert repicked.rows == 17 and repicked.key != key
        assert repicked.manifest["row_provenance"]["pool_key"] == pool_key
        assert [repicked.data_stats["per_corpus"][cid]["sequences"] for cid in raw["corpora"]] == [4, 13]
    finally:
        repicked.close()


def test_global_exact_dedup_precedes_quotas_and_reports_empty_source(quota_recipe):
    raw, tokenizer, write_source = quota_recipe
    for cid in raw["corpora"]:
        write_source(cid, ["unique " * 32])
    cfg = normalized(raw, token_budget=32)
    with pytest.raises(ValueError, match="available pool has 0 rows"):
        sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    without_dedup = normalized(raw, token_budget=32, cross_corpus_dedup=False)
    cache = sprint.prepare_sprint_cache(without_dedup, tokenizer, progress=lambda _: None)
    try:
        assert cache.rows == 4
        assert cache.data_stats["tokens"]["packing_replay_inputs"] == 0
    finally:
        cache.close()


@pytest.mark.parametrize("changes", [
    {"objective": "chat_sft"}, {"epochs": 2}, {"quota_max_source_epochs": 0},
    {"quota_max_source_epochs": float("inf")}, {"quota_max_source_epochs": True},
    {"quota_version": "future"}, {"source_max_ratio": 0.75},
    {"quota_max_source_epochs": {"not-selected": 2}},
])
def test_invalid_quota_contract_rejected(quota_recipe, changes):
    raw, _, _ = quota_recipe
    with pytest.raises(ValueError):
        normalized(raw, **changes)


def test_legacy_identity_unchanged_and_quota_normalization_idempotent(quota_recipe):
    raw, tokenizer, _ = quota_recipe
    legacy = normalized(raw, corpus_mix="token_balanced")
    identity = sprint._packed_identity(legacy, tokenizer)
    assert "fixed_token_quota" not in identity["selection"]
    assert "quota_version" not in legacy
    assert identity == sprint._packed_identity(sprint.normalize_sprint_config(legacy), tokenizer)
    quota = normalized(raw, corpus_weights=dict(zip(raw["corpora"], [1, 7], strict=True)))
    assert quota == sprint.normalize_sprint_config(quota)
    assert sprint._packed_identity(quota, tokenizer) != identity


def test_quota_cache_requires_valid_provenance(quota_recipe):
    raw, tokenizer, _ = quota_recipe
    cfg = normalized(raw, quota_max_source_epochs=8)
    cache = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    root = cache.root
    cache.close()
    with (root / "row_pool_index.bin").open("r+b") as stream:
        stream.write(np.asarray([2**63], dtype="<u8").tobytes())
    with pytest.raises(ValueError, match="checksum mismatch"):
        sprint.SprintPackedCache(root, verify="sha256")
    with pytest.raises(ValueError, match="allocated source rows"):
        sprint.SprintPackedCache(root, verify="size")
