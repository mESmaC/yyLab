"""Real, tiny CPU checkpoint and exact-resume coverage for production PLE."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import zstandard as zstd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from yylab import db
from yylab.models.sprint import SprintConfig, SprintForCausalLM, estimate_parameter_breakdown
from yylab.train import data, prepared, sprint


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def ple_recipe(tmp_path, monkeypatch):
    db.init_db()
    corpus = f"sprint-ple-{tmp_path.name}"
    db.execute("INSERT INTO corpora(id,name,created_at,updated_at,status) VALUES(?,?,?,?,?)",
               (corpus, "PLE CPU fixture", db.now_iso(), db.now_iso(), "ready"))
    root = tmp_path / "corpus"
    monkeypatch.setattr(data, "corpus_dir", lambda _: root)
    monkeypatch.setattr(prepared, "corpus_dir", lambda _: root)
    (root / "data").mkdir(parents=True)
    text = "alpha beta gamma delta epsilon zeta eta theta " * 16
    (root / "data/shard-00000.jsonl.zst").write_bytes(zstd.ZstdCompressor().compress(
        json.dumps({"id": "ple-cpu", "text": text, "split": "train", "domain": "code"}).encode()))
    raw = Tokenizer(models.WordLevel(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    raw.train_from_iterator([text], trainers.WordLevelTrainer(special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>",
                                       bos_token="<bos>", eos_token="<eos>")
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    return {"corpora": [corpus], "splits": ["train"], "domains": ["code"], "corpus_mix": "concatenate",
            "objective": "causal_lm", "cross_corpus_dedup": False, "tokenizer_ref": str(tmp_path / "tokenizer"),
            "seq_len": 8, "micro_batch": 2, "grad_accum": 2, "max_steps": 3, "warmup_steps": 1,
            "lr": .002, "seed": 41, "log_interval": 1, "save_interval": 1, "eval_interval": 0,
            "device": "cpu", "torch_compile": False, "token_cache": False, "data_prefetch_batches": 0,
            "packed_cache_dir": str(tmp_path / "packed"), "shuffle_block_rows": 1,
            "parameter_dtype": "float32", "optimizer_state_offload": "none",
            "model_config": SprintConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                                         num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8,
                                         loss_backend="torch").to_dict()}, tokenizer


@pytest.mark.parametrize("kind", ["full", "qr"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_ple_training_reload_sample_exact_resume_and_continuation(ple_recipe, tmp_path, kind, dtype):
    raw, tokenizer = ple_recipe
    cfg = {**raw, "parameter_dtype": dtype, "optimizer_state_offload": "cpu" if dtype == "bfloat16" else "none",
           "model_config": {**raw["model_config"], "ple_type": kind, "ple_lanes": 4}}
    identifier = f"ple-{tmp_path.name}"
    result = sprint.run_sprint_training({**cfg, "out_checkpoint_id": identifier}, tmp_path / "full")
    assert result["tokens"] == 3 * 2 * 2 * 8
    assert result["supervised_tokens"] == 3 * 2 * 2 * 7
    assert result["runtime"]["ple_type"] == kind and result["runtime"]["ple_lanes"] == 4
    assert result["runtime"]["parameter_dtype"] == dtype
    loaded = SprintForCausalLM.from_pretrained(result["checkpoint_path"], local_files_only=True, dtype="auto")
    assert loaded.config.ple_type == kind and loaded.config.ple_lanes == 4
    assert result["runtime"]["parameter_breakdown"] == estimate_parameter_breakdown(loaded.config)
    assert sum(parameter.numel() for parameter in loaded.parameters()) == result["runtime"]["parameter_breakdown"]["total"]
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
    metadata = json.loads(db.q1("SELECT meta_json FROM checkpoints WHERE id=?", (identifier,))["meta_json"])
    assert metadata["model_config"]["ple_type"] == kind
    assert metadata["parameter_breakdown"]["ple"] > 0
    saved_config = json.loads((Path(result["checkpoint_path"]) / "sprint_training.json").read_text())["config"]
    assert saved_config["model_config"]["ple_type"] == kind
    with torch.no_grad():
        prompt = tokenizer("alpha beta", return_tensors="pt")
        sampled = loaded.generate(**prompt, max_new_tokens=3, do_sample=False, use_cache=True)
    assert sampled.shape[0] == 1 and prompt["input_ids"].shape[1] < sampled.shape[1] <= 5
    assert int(sampled.max()) < len(tokenizer)
    periodic = tmp_path / "full/checkpoints/step-00000001"
    resumed = sprint.run_sprint_training({"resume_from": str(periodic), "out_checkpoint_id": identifier + "-resume"}, tmp_path / "resumed")
    assert resumed["tokens"] == result["tokens"] and resumed["new_steps"] == 2
    restored = SprintForCausalLM.from_pretrained(resumed["checkpoint_path"], local_files_only=True, dtype="auto")
    for name, expected in loaded.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    continued = sprint.run_sprint_training(
        {**cfg, "model_config": {}, "initialization": {"mode": "checkpoint", "checkpoint_ref": result["checkpoint_path"]},
         "max_steps": 1, "save_interval": 0}, tmp_path / "continued")
    assert continued["steps"] == 1 and continued["runtime"]["ple_type"] == kind
    assert json.loads((tmp_path / "continued/resolved_config.json").read_text())["model_config"]["ple_type"] == kind
    for override in ({"ple_type": "none"}, {"ple_type": "qr" if kind == "full" else "full"}, {"ple_lanes": 2}):
        with pytest.raises(ValueError, match="architecture"):
            sprint.run_sprint_training({"resume_from": str(periodic), "model_config": override,
                                       "out_checkpoint_id": identifier + "-bad"}, tmp_path / "bad-resume")


def test_missing_legacy_ple_fields_resume_as_none(ple_recipe, tmp_path):
    cfg, _ = ple_recipe
    full = sprint.run_sprint_training(cfg, tmp_path / "legacy-full")
    periodic = tmp_path / "legacy-full/checkpoints/step-00000001"
    config_path = periodic / "config.json"
    config = json.loads(config_path.read_text())
    for key in ("ple_type", "ple_lanes"):
        config.pop(key)
    config_path.write_text(json.dumps(config))
    state = torch.load(periodic / "trainer.pt", map_location="cpu", weights_only=True)
    for section in (state["contract"]["model_config"], state["config"]["model_config"]):
        for key in ("ple_type", "ple_lanes"):
            section.pop(key)
    torch.save(state, periodic / "trainer.pt")
    resumed = sprint.run_sprint_training({"resume_from": str(periodic)}, tmp_path / "legacy-resumed")
    expected = SprintForCausalLM.from_pretrained(full["checkpoint_path"], local_files_only=True)
    actual = SprintForCausalLM.from_pretrained(resumed["checkpoint_path"], local_files_only=True)
    assert actual.config.ple_type == "none" and actual.config.ple_lanes == 4
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], tensor, rtol=0, atol=0)


def test_ple_normalization_inheritance_and_data_identity(ple_recipe):
    raw, tokenizer = ple_recipe
    legacy = {**raw, "model_config": {key: value for key, value in raw["model_config"].items() if key not in {"ple_type", "ple_lanes"}}}
    none = sprint.normalize_sprint_config(legacy)
    assert none["model_config"]["ple_type"] == "none" and none["model_config"]["ple_lanes"] == 4
    inherited = sprint.normalize_sprint_config({**raw, "model_config": {}, "initialization": {"mode": "checkpoint", "path": "saved"}})
    assert inherited["model_config"] == {}
    for kind in ("full", "qr"):
        selected = sprint.normalize_sprint_config({**raw, "model_config": {**raw["model_config"], "ple_type": kind}})
        assert sprint._packed_identity(selected, tokenizer) == sprint._packed_identity(none, tokenizer)
        for key in ("token_budget", "max_rows", "max_steps", "parameter_dtype", "optimizer_state_offload"):
            assert selected[key] == none[key]
    for invalid in ({"ple_type": "unknown"}, {"ple_lanes": 0}, {"ple_lanes": True}, {"ple_lanes": 1.5}):
        with pytest.raises(ValueError):
            sprint.normalize_sprint_config({**raw, "model_config": {**raw["model_config"], **invalid}})


@pytest.mark.parametrize("kind", ["full", "qr"])
def test_training_target_planner_includes_ple_cost(ple_recipe, kind):
    raw, tokenizer = ple_recipe
    cfg = sprint.normalize_sprint_config({**raw, "target_parameters": 10000,
                                         "model_config": {"ple_type": kind, "ple_lanes": 2, "loss_backend": "torch"}})
    model, loaded_tokenizer, parent = sprint._load_training_inputs(cfg, None)
    assert model.config.ple_type == kind and model.config.ple_lanes == 2
    assert len(loaded_tokenizer) == len(tokenizer) and parent is None
    breakdown = estimate_parameter_breakdown(model.config)
    assert 0 < breakdown["ple"] < breakdown["total"] <= 10000
    assert sum(parameter.numel() for parameter in model.parameters()) == breakdown["total"]
