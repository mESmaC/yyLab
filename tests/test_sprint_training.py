from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import zstandard as zstd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from yylab import db
from yylab.models.sprint import SprintConfig, SprintForCausalLM
from yylab.train import data, prepared, sprint


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def training_fixture(tmp_path, monkeypatch):
    db.init_db()
    corpus_id = f"sprint-{tmp_path.name}"
    db.execute("INSERT INTO corpora(id,name,created_at,updated_at,status) VALUES(?,?,?,?,?)",
               (corpus_id, "Sprint CPU fixture", db.now_iso(), db.now_iso(), "ready"))
    corpus_root = tmp_path / "corpus"
    monkeypatch.setattr(data, "corpus_dir", lambda _: corpus_root)
    monkeypatch.setattr(prepared, "corpus_dir", lambda _: corpus_root)
    (corpus_root / "data").mkdir(parents=True)
    source = corpus_root / "data" / "shard-00000.jsonl.zst"
    rows = [{"id": "one", "text": "alpha beta gamma delta " * 5, "split": "train", "domain": "code"},
            {"id": "two", "text": "epsilon zeta eta theta " * 3, "split": "validation", "domain": "code"}]

    def write_source(records):
        source.write_bytes(zstd.ZstdCompressor().compress("\n".join(json.dumps(row) for row in records).encode()))

    write_source(rows)
    raw = Tokenizer(models.WordLevel(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    raw.train_from_iterator([row["text"] for row in rows], trainers.WordLevelTrainer(
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>",
                                       bos_token="<bos>", eos_token="<eos>")
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    cfg = {
        "corpora": [corpus_id], "splits": ["train"], "domains": ["code"],
        "corpus_mix": "concatenate", "objective": "causal_lm", "cross_corpus_dedup": False,
        "tokenizer_ref": str(tokenizer_path), "seq_len": 8, "micro_batch": 2, "grad_accum": 2,
        "max_steps": 4, "seed": 0, "warmup_steps": 1, "lr": 0.002, "log_interval": 2,
        "save_interval": 2, "eval_interval": 2, "eval_steps": 1, "eval_splits": ["validation"],
        "device": "cpu", "torch_compile": False, "token_cache": False, "data_prefetch_batches": 2,
        "packed_cache_dir": str(tmp_path / "packed"), "shuffle_block_rows": 1,
        "model_config": SprintConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
                                      loss_backend="torch").to_dict(),
    }
    yield cfg, tokenizer, write_source, rows


def tiny_model():
    return SprintForCausalLM(SprintConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                                         num_hidden_layers=1, num_attention_heads=2,
                                         max_position_embeddings=8, loss_backend="torch"))


def test_accumulation_weights_supervised_targets_and_partial_batches():
    torch.manual_seed(8)
    initial = tiny_model()
    a, b = copy.deepcopy(initial), copy.deepcopy(initial)
    cfg = {"lr": 0.001, "max_steps": 3, "grad_clip": 0, "weight_decay": 0.0}
    split, combined = sprint.SprintStepEngine(a, cfg, "cpu"), sprint.SprintStepEngine(b, cfg, "cpu")
    ids = torch.randint(0, 16, (3, 8))
    labels = ids.clone()
    labels[0, 1:6] = -100
    labels[2, 1:4] = -100
    separate = split.train_step([{"input_ids": ids[:2], "labels": labels[:2]},
                                 {"input_ids": ids[2:], "labels": labels[2:]}])
    together = combined.train_step([{"input_ids": ids, "labels": labels}])
    assert separate.tokens == 24
    assert separate.supervised_tokens == 13
    torch.testing.assert_close(separate.loss, together.loss)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb, atol=1e-6, rtol=1e-5)
    for parameter, state in split.optimizer.state.items():
        assert parameter.dtype == torch.float32
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32


def test_evaluation_weighting_and_mode_restore():
    engine = sprint.SprintStepEngine(tiny_model(), {"grad_clip": 0}, "cpu")
    ids = torch.randint(0, 16, (3, 8))
    labels = ids.clone()
    labels[0, 1:4] = -100
    engine.model.train()
    result = engine.evaluate([{"input_ids": ids[:2], "labels": labels[:2]},
                              {"input_ids": ids[2:], "labels": labels[2:]}])
    expected = engine.model(input_ids=ids, labels=labels).loss.item()
    assert result["eval_loss"] == pytest.approx(expected, abs=1e-6)
    assert result["eval_supervised_tokens"] == 18
    assert engine.model.training
    with pytest.raises(ValueError, match="no supervised"):
        engine.train_step([{"input_ids": ids, "labels": torch.full_like(ids, -100)}])


def test_learning_rate_warmup_and_final_decay():
    cfg = {"lr": 1.0, "warmup_steps": 2, "max_steps": 6, "min_lr_ratio": 0.1}
    rates = [sprint.sprint_learning_rate(cfg, index) for index in range(6)]
    assert rates[:3] == [0.5, 1.0, 1.0]
    assert rates[-1] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(rates[2:], rates[3:]))


def test_config_aliases_budget_and_evaluation_admission(training_fixture):
    cfg, _, _, _ = training_fixture
    original = {**cfg, "token_budget": 83, "max_steps": 0, "activation_checkpointing": True}
    resolved = sprint.normalize_sprint_config(original)
    assert resolved["max_steps"] == 3
    assert resolved["max_rows"] == 10
    assert resolved["gradient_checkpointing"]
    assert original["max_steps"] == 0
    for bad in ({"seq_len": 1}, {"token_budget": 7}, {"lr": float("nan")},
                {"optimizer_state_offload": "disk"}, {"splits": ["validation"]},
                {"eval_splits": []}, {"grad_accum": 1.5}):
        with pytest.raises(ValueError):
            sprint.normalize_sprint_config({**cfg, **bad})


def test_packed_cache_exact_seek_epoch_permutation_and_reuse(training_fixture, monkeypatch):
    raw, tokenizer, _, _ = training_fixture
    cfg = sprint.normalize_sprint_config(raw)
    cache = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    assert cache.rows == 2
    assert cache.arrays["input_ids"].dtype == np.dtype("<u2")
    batches = list(sprint._CacheBatches(cache, cfg, 0, 11, pin_memory=False))
    full = torch.cat([batch["input_ids"] for batch in batches])
    assert full.shape == (11, 8)
    assert [len(batch["input_ids"]) for batch in batches] == [2, 2, 2, 2, 2, 1]
    resumed = torch.cat([batch["input_ids"] for batch in sprint._CacheBatches(cache, cfg, 3, 11,
                                                                            pin_memory=False)])
    torch.testing.assert_close(full[3:], resumed)
    root = cache.root
    cache.close()
    monkeypatch.setattr(sprint, "iter_corpus_docs", lambda *a, **k: pytest.fail("cache reuse repacked corpus"))
    reused = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    assert reused.root == root
    reused.close()


def test_cache_crosses_epoch_without_short_optimizer_batches(training_fixture):
    raw, tokenizer, _, _ = training_fixture
    cfg = sprint.normalize_sprint_config({**raw, "micro_batch": 3, "grad_accum": 1})
    cache = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    try:
        batches = list(sprint._CacheBatches(cache, cfg, 0, 12, pin_memory=False))
        assert len(batches) == 4
        assert all(len(batch["input_ids"]) == 3 for batch in batches)
        assert sum(batch["num_targets"] for batch in batches) == 12 * 7
    finally:
        cache.close()


def test_cache_change_identity_and_truncation_refusal(training_fixture):
    raw, tokenizer, write_source, rows = training_fixture
    cfg = sprint.normalize_sprint_config(raw)
    cache = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    old_root = cache.root
    cache.close()
    write_source([{**rows[0], "text": rows[0]["text"] + " more"}, rows[1]])
    changed = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    assert changed.root != old_root
    changed.close()
    (old_root / "input_ids.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="size mismatch"):
        sprint.SprintPackedCache(old_root)


def test_larger_packed_cache_reuses_exact_prefix_with_logical_identity(training_fixture, monkeypatch, tmp_path):
    raw, tokenizer, write_source, records = training_fixture
    write_source([{**records[0], "text": records[0]["text"] * 50}])
    cfg = sprint.normalize_sprint_config({**raw, "eval_interval": 0})
    large = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    backing_root, backing_key = large.root, large.key
    expected = large.read_batch(0, 3)["input_ids"]
    original_manifest = (large.root / "manifest.json").read_bytes()
    large.close()
    small_cfg = {**cfg, "max_rows": 3}
    # A fresh small build must have the same logical identity and row stream.
    fresh = sprint.prepare_sprint_cache({**small_cfg, "packed_cache_dir": str(tmp_path / "fresh")},
                                       tokenizer, progress=lambda _: None)
    expected_key = fresh.key
    torch.testing.assert_close(fresh.read_batch(0, 3)["input_ids"], expected)
    fresh.close()
    monkeypatch.setattr(sprint, "iter_corpus_docs", lambda *a, **k: pytest.fail("prefix reuse repacked"))
    reused = sprint.prepare_sprint_cache(small_cfg, tokenizer, progress=lambda _: None)
    try:
        assert reused.root == backing_root
        assert reused.rows == 3 and reused.stored_rows == 16
        assert reused.key == expected_key and reused.key != backing_key
        assert reused.preparation["mode"] == "compatible_cache"
        assert reused.data_stats["sequences"] == 3
        assert reused.data_stats["tokens"]["forwarded_model_slots"] == 24
        assert "docs_seen" not in reused.data_stats and "per_corpus" not in reused.data_stats
        stream = sprint._CacheBatches(reused, cfg, 0, 7, pin_memory=False)
        actual = torch.cat([batch["input_ids"] for batch in stream])
        torch.testing.assert_close(actual, expected.repeat(3, 1)[:7])
        assert (backing_root / "manifest.json").read_bytes() == original_manifest
    finally:
        reused.close()


def test_complete_packed_pass_reused_for_larger_budget(training_fixture, monkeypatch):
    raw, tokenizer, _, _ = training_fixture
    cfg = sprint.normalize_sprint_config(raw)
    original = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    root, key, rows = original.root, original.key, original.rows
    assert rows < cfg["max_rows"]
    original.close()
    monkeypatch.setattr(sprint, "iter_corpus_docs", lambda *a, **k: pytest.fail("complete pass repacked"))
    reused = sprint.prepare_sprint_cache({**cfg, "max_rows": 50}, tokenizer, progress=lambda _: None)
    try:
        assert reused.root == root and reused.rows == rows
        assert reused.key != key
        assert reused.data_stats == reused.manifest["data_stats"]
    finally:
        reused.close()


def test_truncated_packed_pass_cannot_satisfy_larger_budget(training_fixture):
    raw, tokenizer, _, _ = training_fixture
    cfg = sprint.normalize_sprint_config(raw)
    truncated = sprint.prepare_sprint_cache({**cfg, "max_rows": 1}, tokenizer, progress=lambda _: None)
    root = truncated.root
    with pytest.raises(ValueError, match="does not cover"):
        truncated.select_request(sprint._packed_identity(cfg, tokenizer))
    truncated.close()
    complete = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    try:
        assert complete.root != root
        assert complete.rows == 2
        assert complete.preparation["mode"] == "built"
    finally:
        complete.close()


def test_custom_tokenizer_keeps_original_call_boundaries(training_fixture, monkeypatch):
    raw, tokenizer, write_source, records = training_fixture
    write_source([records[0]] * 12)
    cfg = sprint.normalize_sprint_config(raw)
    calls = []

    def batch_sensitive(texts):
        calls.append(len(texts))
        return [[4 + index % 3] * 7 for index in range(len(texts))]

    monkeypatch.setattr(sprint, "make_sprint_tokenize", lambda _: batch_sensitive)
    cache = sprint.prepare_sprint_cache({**cfg, "max_rows": 12}, tokenizer, progress=lambda _: None)
    try:
        assert calls == [8, 4]
        assert cache.read_batch(8, 1)["input_ids"].tolist() == [[4] * 7 + [tokenizer.eos_token_id]]
    finally:
        cache.close()


@pytest.mark.parametrize("change", [{"seed": 5}, {"seq_len": 4}, {"splits": ["validation"]},
                                   {"cross_corpus_dedup": True}, {"max_text_chars": 9}])
def test_compatible_cache_preserves_data_selection(training_fixture, change):
    raw, tokenizer, _, _ = training_fixture
    cfg = sprint.normalize_sprint_config(raw)
    cache = sprint.prepare_sprint_cache(cfg, tokenizer, progress=lambda _: None)
    root = cache.root
    cache.close()
    changed = sprint.prepare_sprint_cache({**cfg, **change, "max_rows": 1}, tokenizer, progress=lambda _: None)
    try:
        assert changed.root != root
        assert changed.preparation["mode"] == "built"
    finally:
        changed.close()


def test_resume_with_compatible_cache_matches_fresh_training(training_fixture, tmp_path):
    raw, tokenizer, write_source, records = training_fixture
    write_source([{**records[0], "text": records[0]["text"] * 50}])
    cfg = {**raw, "eval_interval": 0, "max_steps": 2, "save_interval": 1}
    large_cfg = sprint.normalize_sprint_config({**cfg, "max_steps": 8})
    cache = sprint.prepare_sprint_cache(large_cfg, tokenizer, progress=lambda _: None)
    cache.close()
    full = sprint.run_sprint_training(cfg, tmp_path / "prefix-full")
    assert full["runtime"]["data_preparation"]["mode"] == "compatible_cache"
    fresh = sprint.run_sprint_training({**cfg, "packed_cache_dir": str(tmp_path / "new-physical-cache")},
                                      tmp_path / "fresh-full")
    assert full["runtime"]["packed_key"] == fresh["runtime"]["packed_key"]
    # Resume can use a different physical cache when its logical rows match.
    resumed = sprint.run_sprint_training({"resume_from": str(tmp_path / "prefix-full" / "checkpoints" /
                                                              "step-00000001"),
                                         "packed_cache_dir": str(tmp_path / "new-physical-cache")},
                                        tmp_path / "prefix-resumed")
    expected = SprintForCausalLM.from_pretrained(full["checkpoint_path"]).state_dict()
    for result in (fresh, resumed):
        actual = SprintForCausalLM.from_pretrained(result["checkpoint_path"]).state_dict()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("parameter_dtype", ["float32", "bfloat16"])
def test_real_training_registry_eval_and_exact_resume(training_fixture, tmp_path, parameter_dtype):
    cfg, _, _, _ = training_fixture
    cfg = {**cfg, "parameter_dtype": parameter_dtype, "optimizer_state_offload": "cpu"}
    out_id = f"checkpoint-{tmp_path.name}"
    full = sprint.run_sprint_training({**cfg, "out_checkpoint_id": out_id}, tmp_path / "full")
    assert full["steps"] == 4
    assert full["tokens"] == 4 * 2 * 2 * 8
    assert full["eval_supervised_tokens"] == 7
    assert full["runtime"]["parameter_dtype"] == parameter_dtype
    assert full["runtime"]["optimizer_state_offload"] == "cpu"
    assert full["runtime"]["master_weight_dtype"] == "float32"
    row = db.q1("SELECT * FROM checkpoints WHERE id=?", (out_id,))
    assert row["kind"] == "sprint"
    assert json.loads(row["meta_json"])["model_family"] == "hf"
    full_model = SprintForCausalLM.from_pretrained(full["checkpoint_path"])
    resume = sprint.run_sprint_training(
        {"resume_from": str(tmp_path / "full" / "checkpoints" / "step-00000002"),
         "out_checkpoint_id": out_id + "-resume"}, tmp_path / "resumed")
    assert resume["new_steps"] == 2
    assert resume["tokens"] == full["tokens"]
    resumed_model = SprintForCausalLM.from_pretrained(resume["checkpoint_path"])
    for key, expected in full_model.state_dict().items():
        torch.testing.assert_close(resumed_model.state_dict()[key], expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="resume configuration or data differs"):
        sprint.run_sprint_training(
            {"resume_from": str(tmp_path / "full" / "checkpoints" / "step-00000002"),
             "out_checkpoint_id": out_id + "-bad", "lr": 0.1}, tmp_path / "bad-resume")
    with pytest.raises(ValueError, match="resume configuration or data differs"):
        sprint.run_sprint_training(
            {"resume_from": str(tmp_path / "full" / "checkpoints" / "step-00000002"),
             "out_checkpoint_id": out_id + "-bad-checkpointing", "activation_checkpointing": True},
            tmp_path / "bad-resume-checkpointing")


def test_real_ui_recipe_runs_partial_budget_and_checkpoint_continuation(training_fixture, tmp_path):
    from yylab.ui.pages.train_sprint import build_sprint_run_config

    cfg, _, _, _ = training_fixture
    recipe = build_sprint_run_config({**cfg, "checkpoint_name": "Sprint UI CPU test",
                                     "token_budget": 83, "max_steps": 0, "save_interval": 0,
                                     "eval_interval": 0, "warmup_steps": 0})
    recipe.update(packed_cache_dir=str(tmp_path / "packed"), token_cache=False)
    result = sprint.run_sprint_training(recipe, tmp_path / "ui-run")
    assert result["tokens"] == 80
    assert result["supervised_tokens"] == 70
    assert result["steps"] == 3
    continued = sprint.run_sprint_training(
        {**cfg, "initialization": {"mode": "checkpoint", "checkpoint_ref": result["checkpoint_path"]},
         "max_steps": 1, "save_interval": 0, "eval_interval": 0, "model_config": {}}, tmp_path / "continued")
    assert continued["steps"] == 1
    assert continued["tokens"] == 32


def test_chat_cache_preserves_assistant_masks_and_padding(training_fixture, tmp_path):
    cfg, tokenizer, write_source, _ = training_fixture
    tokenizer.chat_template = (
        "{% for message in messages %}{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}"
        "{% else %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"
    )
    tokenizer.save_pretrained(cfg["tokenizer_ref"])
    write_source([{"text": "chat fixture", "split": "train", "domain": "code", "messages": [
        {"role": "user", "content": "alpha beta"}, {"role": "assistant", "content": "gamma delta"}]}])
    resolved = sprint.normalize_sprint_config({**cfg, "objective": "chat_sft", "eval_interval": 0,
                                               "max_steps": 1, "save_interval": 0})
    cache = sprint.prepare_sprint_cache(resolved, tokenizer, progress=lambda _: None)
    try:
        batch = cache.read_batch(0, 1)
        assert batch["num_targets"] == 3
        assert batch["real_tokens"] == 6
        assert batch["labels"][0, :3].tolist() == [-100, -100, -100]
        assert batch["labels"][0, 6:].tolist() == [-100, -100]
        assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
    finally:
        cache.close()
    result = sprint.run_sprint_training(resolved, tmp_path / "chat")
    assert result["supervised_tokens"] == 12
    assert result["padding_tokens"] == 8


def test_chat_prefix_reuse_preserves_labels_and_does_not_invent_statistics(training_fixture, monkeypatch):
    cfg, tokenizer, write_source, _ = training_fixture
    tokenizer.chat_template = (
        "{% for message in messages %}{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}"
        "{% else %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"
    )
    write_source([{"text": "chat fixture", "split": "train", "domain": "code", "messages": [
        {"role": "user", "content": "alpha beta"}, {"role": "assistant", "content": "gamma delta"}]}] * 3)
    resolved = sprint.normalize_sprint_config({**cfg, "objective": "chat_sft", "eval_interval": 0})
    full = sprint.prepare_sprint_cache(resolved, tokenizer, progress=lambda _: None)
    expected = full.read_batch(0, 1)
    full.close()
    monkeypatch.setattr(sprint, "iter_corpus_docs", lambda *a, **k: pytest.fail("SFT prefix repacked"))
    prefix = sprint.prepare_sprint_cache({**resolved, "max_rows": 1}, tokenizer, progress=lambda _: None)
    try:
        actual = prefix.read_batch(0, 1)
        for field in ("input_ids", "labels", "attention_mask"):
            torch.testing.assert_close(actual[field], expected[field])
        assert actual["num_targets"] == 3 and actual["real_tokens"] == 6
        assert "optimizer_supervised_targets" not in prefix.data_stats["tokens"]
        assert prefix.data_stats["tokens"]["forwarded_model_slots"] == 8
    finally:
        prefix.close()


@pytest.mark.parametrize("damage", ["seq_len", "fields", "dtype"])
def test_packed_manifest_rejects_inconsistent_metadata(training_fixture, damage):
    raw, tokenizer, _, _ = training_fixture
    cache = sprint.prepare_sprint_cache(sprint.normalize_sprint_config(raw), tokenizer, progress=lambda _: None)
    root, manifest = cache.root, cache.manifest
    cache.close()
    if damage == "seq_len":
        manifest["seq_len"] *= 2
    elif damage == "fields":
        manifest["files"]["labels"] = dict(manifest["files"]["input_ids"])
    else:
        manifest["files"]["input_ids"]["dtype"] = "<i2"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity|objective|dtype"):
        sprint.SprintPackedCache(root)


def test_activation_checkpointing_alias_and_cpu_dtype_contract():
    engine = sprint.SprintStepEngine(tiny_model(), {"activation_checkpointing": True, "grad_clip": 0}, "cpu")
    assert engine.model.gradient_checkpointing
    result = engine.train_step([{"input_ids": torch.randint(0, 16, (2, 8))}])
    assert torch.isfinite(result.loss)
    with pytest.raises(RuntimeError, match="BF16 training requires CUDA"):
        sprint.SprintStepEngine(tiny_model(), {"compute_dtype": "bf16"}, "cpu")


def test_budget_planner_accounts_for_swiglu_and_untied_embeddings(training_fixture):
    raw, _, _, _ = training_fixture
    cfg = sprint.normalize_sprint_config({**raw, "target_parameters": 100_000, "model_config": {
        "mlp_type": "swiglu", "tie_word_embeddings": False, "max_position_embeddings": 32,
    }})
    model, _, _ = sprint._load_training_inputs(cfg, None)
    assert model.config.mlp_type == "swiglu"
    assert not model.config.tie_word_embeddings
    assert model.config.max_position_embeddings == 32
    assert sum(parameter.numel() for parameter in model.parameters()) <= 100_000


@pytest.mark.parametrize("corrupt", ["parameter", "moment", "master"])
def test_checkpoint_refuses_nonfinite_updated_state(training_fixture, tmp_path, corrupt):
    _, tokenizer, _, _ = training_fixture
    engine = sprint.SprintStepEngine(tiny_model(), {"grad_clip": 0}, "cpu")
    result = engine.train_step([{"input_ids": torch.randint(0, 16, (2, 8))}])
    assert torch.isfinite(result.loss)
    parameter = next(engine.model.parameters())
    with torch.no_grad():
        if corrupt == "parameter":
            parameter.view(-1)[0] = float("nan")
        elif corrupt == "moment":
            engine.optimizer.state[parameter]["exp_avg"].view(-1)[0] = float("inf")
        else:
            engine.optimizer.state[parameter]["master_param"].view(-1)[0] = float("nan")
    output = tmp_path / "corrupt-checkpoint"
    with pytest.raises(FloatingPointError, match="nonfinite"):
        sprint.save_sprint_checkpoint(output, engine, tokenizer, {"step": 1})
    assert not output.exists()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bucketed_cpu_offload_matches_fp32_master_adam_and_preserves_missing_grad_steps(dtype):
    torch.manual_seed(19)
    originals = [torch.randn(3, 17), torch.randn(29)]
    parameters = [torch.nn.Parameter(value.to(dtype)) for value in originals]
    reference = [torch.nn.Parameter(value.clone()) for value in originals]
    opts = {"lr": 0.001, "weight_decay": 0.1, "betas": (0.9, 0.95)}
    offload = sprint.SprintCPUOffloadAdamW(parameters, **opts)
    offload.initialize_master_weights([value.clone() for value in originals])
    offload.bucket_elements = 17
    expected = torch.optim.AdamW(reference, fused=True, **opts)
    for step in range(4):
        for index, (parameter, master) in enumerate(zip(parameters, reference, strict=True)):
            grad = torch.randn_like(parameter) if index == 0 or step % 2 == 0 else None
            parameter.grad = grad
            master.grad = grad.float() if grad is not None else None
        offload.step()
        expected.step()
        for parameter, master in zip(parameters, reference, strict=True):
            state = offload.state[parameter]
            torch.testing.assert_close(state["master_param"], master, rtol=0, atol=0)
            torch.testing.assert_close(parameter, master.to(dtype), rtol=0, atol=0)
            for key in ("master_param", "exp_avg", "exp_avg_sq"):
                assert state[key].dtype == torch.float32
                assert state[key].device.type == "cpu"
    assert offload.state[parameters[0]]["step"] == 4
    assert offload.state[parameters[1]]["step"] == 2
    assert offload._staging.numel() <= 17
    assert offload._gradient.numel() <= 17


def test_bf16_offload_retains_updates_smaller_than_weight_ulp_and_restores_without_cloning():
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = sprint.SprintCPUOffloadAdamW([parameter], lr=1e-4, betas=(0.0, 0.0), weight_decay=0)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert parameter.tolist() == [1.0, 1.0]
    assert optimizer.state[parameter]["master_param"][0].item() < 1.0
    for _ in range(30):
        optimizer.step()
    assert parameter[0] < 1
    payload = optimizer.state_dict()
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = sprint.SprintCPUOffloadAdamW([restored_parameter])
    restored.load_state_dict(payload)
    source_state = next(iter(payload["state"].values()))
    for key in ("master_param", "exp_avg", "exp_avg_sq"):
        assert restored.state[restored_parameter][key].data_ptr() == source_state[key].data_ptr()
        assert restored.state[restored_parameter][key].dtype == torch.float32


def test_engine_explicit_gpu_resident_recipe_and_low_precision_guard():
    resident = sprint.SprintStepEngine(tiny_model(), {"parameter_dtype": "float32", "optimizer_state_offload": "none"}, "cpu")
    assert isinstance(resident.optimizer, torch.optim.AdamW)
    assert not isinstance(resident.optimizer, sprint.SprintCPUOffloadAdamW)
    with pytest.raises(ValueError, match="BF16 weights require CPU offload"):
        sprint.SprintStepEngine(tiny_model(), {"parameter_dtype": "bfloat16", "optimizer_state_offload": "none"}, "cpu")


def test_rng_checkpoint_roundtrip_restores_all_cpu_generators():
    import random

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = sprint._rng_state()
    expected = random.random(), np.random.rand(), torch.rand(3)
    sprint._restore_rng(state)
    actual = random.random(), np.random.rand(), torch.rand(3)
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
