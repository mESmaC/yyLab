"""Fixed token allocations cannot silently fall back to ordinary source mixing."""
import hashlib
import json

import pytest
from test_sprint_ple_training import one_cpu_thread as one_cpu_thread
from test_sprint_ple_training import ple_recipe as ple_recipe

from yylab.train import sprint


def fixed_descriptor(cfg):
    corpus = cfg["corpora"][0]
    return {"version": "fixed_token_rows_v1", "seed": cfg["seed"],
            "rows_by_corpus": {corpus: cfg["max_rows"]},
            "pool_sha256": {corpus: "a" * 64}, "dedup": "global_exact_text_v1"}


def test_fixed_mix_is_distinct_and_never_rebuilt_as_ordinary_schedule(ple_recipe):
    raw, tokenizer = ple_recipe
    cfg = sprint.normalize_sprint_config({**raw, "cross_corpus_dedup": True})
    normal = sprint.prepare_sprint_cache(cfg, tokenizer)
    fixed_cfg = {**cfg, "fixed_token_mix": fixed_descriptor(cfg)}
    assert sprint._packed_identity(cfg, tokenizer) != sprint._packed_identity(fixed_cfg, tokenizer)
    with pytest.raises(ValueError, match="materialized fixed-token dataset is missing"):
        sprint.prepare_sprint_cache(fixed_cfg, tokenizer)
    # Materialize the fixture's same rows under its explicit fixed-allocation identity.
    identity = sprint._packed_identity(fixed_cfg, tokenizer)
    key = hashlib.sha256(sprint._json(identity).encode()).hexdigest()
    root = normal.root.parent / key
    root.mkdir()
    (root / "input_ids.bin").write_bytes((normal.root / "input_ids.bin").read_bytes())
    manifest = {**normal.manifest, "identity": identity, "key": key}
    (root / "manifest.json").write_text(json.dumps(manifest))
    normal.close()
    with pytest.raises(ValueError, match="token budget"):
        sprint._packed_identity({**fixed_cfg, "max_rows": cfg["max_rows"] + 1}, tokenizer)
    verified = sprint.prepare_sprint_cache({**fixed_cfg, "packed_cache_verify": "sha256"}, tokenizer)
    assert verified.key == key and verified.preparation["mode"] == "exact_cache"
    verified.close()
    prefix = sprint.prepare_sprint_cache({**fixed_cfg, "max_rows": 2}, tokenizer)
    assert prefix.rows == 2 and prefix.preparation["mode"] == "compatible_cache"
    assert "packing_replay_inputs" not in prefix.data_stats["tokens"]
    assert "covered_rendered_positions" not in prefix.data_stats["tokens"]
    prefix.close()


@pytest.mark.parametrize("override", [
    {"splits": ["validation"]}, {"cross_corpus_dedup": False}, {"corpus_mix": "token_balanced"},
    {"objective": "chat_sft"}, {"epochs": 2},
])
def test_fixed_mix_rejects_changed_selection(ple_recipe, override):
    raw, tokenizer = ple_recipe
    cfg = sprint.normalize_sprint_config({**raw, "cross_corpus_dedup": True})
    cfg["fixed_token_mix"] = fixed_descriptor(cfg)
    with pytest.raises(ValueError, match="fixed_token_mix"):
        sprint._packed_identity({**cfg, **override}, tokenizer)
