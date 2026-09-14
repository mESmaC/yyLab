from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from test_sprint_token_quota import quota_recipe as quota_recipe
from test_sprint_training import tiny_model
from test_sprint_training import training_fixture as training_fixture

from yylab.infer.prompt_format import PROMPT_FORMAT_FILENAME, build_prompt_format_contract, load_prompt_format
from yylab.models.sprint import SprintConfig, SprintForCausalLM
from yylab.train import sprint, sprint_quality


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _health_config(enabled=True, interval=2):
    return {"enabled": enabled, "interval": interval, "options": {
        "max_new_tokens": 3, "modes": ["greedy", "sample"],
        "prompts": [{"name": "small", "kind": "prose", "prompt": "alpha"}],
    }}


def _train_only(training_fixture):
    cfg, tokenizer, write_source, records = training_fixture
    write_source([{**records[0], "text": records[0]["text"] * 50}])
    return {**cfg, "eval_interval": 0, "save_interval": 0}, tokenizer


@pytest.mark.parametrize("steps,expected", [(3, [2, 3]), (4, [2, 4])])
def test_generation_own_cadence_and_final_receipt_without_validation_data(
    training_fixture, tmp_path, monkeypatch, steps, expected,
):
    cfg, _ = _train_only(training_fixture)
    calls = []
    actual_evaluate = sprint_quality.evaluate_sprint_generation

    def evaluate(*args, **kwargs):
        assert args[0].get_input_embeddings().weight.device.type == "cpu"
        result = actual_evaluate(*args, **kwargs)
        assert result["summary"]["skipped_outputs"] == 0
        calls.append(result)
        return result

    monkeypatch.setattr(sprint_quality, "evaluate_sprint_generation", evaluate)
    run_dir = tmp_path / "cadence"
    summary = sprint.run_sprint_training({**cfg, "max_steps": steps, "save_interval": 2,
                                         "log_interval": 3, "generation_eval": _health_config()}, run_dir)
    receipts = sorted((run_dir / "generation-eval").glob("*.json"))
    assert len(calls) == len(expected)
    assert [_read(path)["step"] for path in receipts] == expected
    assert summary["generation_eval_step"] == steps
    assert "eval_loss" not in summary
    assert _read(summary["generation_eval_path"])["input_slots"] == summary["tokens"]
    assert _read(summary["generation_eval_path"])["supervised_targets"] == summary["supervised_tokens"]
    for path in [Path(summary["checkpoint_path"]), *sorted((run_dir / "checkpoints").iterdir())]:
        prompt_contract = _read(path / PROMPT_FORMAT_FILENAME)
        metadata = _read(path / "sprint_training.json")
        assert prompt_contract == metadata["config"]["prompt_format_contract"]
        assert load_prompt_format(path).default_mode == "raw"
        assert metadata["generation_evaluation"]["generation_eval_step"] == metadata["step"]


def test_generation_diagnostics_preserve_exact_training_updates_optimizer_and_rng(training_fixture, tmp_path):
    cfg, _ = _train_only(training_fixture)
    cfg.update(max_steps=3, save_interval=0, log_interval=2)
    enabled = sprint.run_sprint_training({**cfg, "generation_eval": _health_config(True, 1)}, tmp_path / "enabled")
    disabled = sprint.run_sprint_training({**cfg, "generation_eval": _health_config(False, 1)}, tmp_path / "disabled")
    assert enabled["steps"] == disabled["steps"] == 3
    assert enabled["tokens"] == disabled["tokens"]
    assert enabled["supervised_tokens"] == disabled["supervised_tokens"]
    assert enabled["loss"] == disabled["loss"]
    assert not (tmp_path / "disabled" / "generation-eval").exists()
    model_a = SprintForCausalLM.from_pretrained(enabled["checkpoint_path"]).state_dict()
    model_b = SprintForCausalLM.from_pretrained(disabled["checkpoint_path"]).state_dict()
    for name in model_a:
        torch.testing.assert_close(model_a[name], model_b[name], rtol=0, atol=0)
    state_a = torch.load(Path(enabled["checkpoint_path"]) / "trainer.pt", weights_only=True)
    state_b = torch.load(Path(disabled["checkpoint_path"]) / "trainer.pt", weights_only=True)
    for key in ("python", "numpy"):
        assert state_a["rng"][key] == state_b["rng"][key]
    assert len(state_a["rng"]["cuda"]) == len(state_b["rng"]["cuda"])
    for actual, expected in zip(state_a["rng"]["cuda"], state_b["rng"]["cuda"], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state_a["rng"]["torch"], state_b["rng"]["torch"], rtol=0, atol=0)
    assert state_a["optimizer"]["param_groups"] == state_b["optimizer"]["param_groups"]
    for parameter, tensors in state_a["optimizer"]["state"].items():
        for name, actual in tensors.items():
            torch.testing.assert_close(actual, state_b["optimizer"]["state"][parameter][name], rtol=0, atol=0)


def test_repetition_warning_persists_between_checks_and_into_checkpoint_summary(
    training_fixture, tmp_path, monkeypatch, capsys,
):
    cfg, _ = _train_only(training_fixture)

    def repeating(*args, **kwargs):
        return {"summary": {"status": "repetition_detected", "outputs": 2,
                            "nonfinite_outputs": 0, "repetition_flagged_outputs": 1},
                "outputs": [{"generated_text": "loop loop loop"}]}

    monkeypatch.setattr(sprint_quality, "evaluate_sprint_generation", repeating)
    summary = sprint.run_sprint_training({**cfg, "max_steps": 4, "log_interval": 1,
                                         "generation_eval": _health_config(True, 2)}, tmp_path / "warnings")
    metrics = [json.loads(line) for line in (tmp_path / "warnings" / "metrics.jsonl").read_text().splitlines()]
    between = [row for row in metrics if row["step"] == 3]
    assert len(between) == 1
    assert between[0]["generation_eval_step"] == 2
    for result in [between[0], summary, _read(tmp_path / "warnings" / "progress.json")]:
        assert result["generation_eval_status"] == "repetition_detected"
        assert "generated text is repeating" in result["msg"]
    assert "repetition in 1/2 samples" in capsys.readouterr().out
    metadata = _read(Path(summary["checkpoint_path"]) / "sprint_training.json")
    assert metadata["generation_evaluation"]["generation_eval_status"] == "repetition_detected"
    assert _read(summary["generation_eval_path"])["outputs"][0]["generated_text"] == "loop loop loop"


def test_reported_gradient_norm_is_before_clipping_with_target_weighted_accumulation():
    torch.manual_seed(91)
    model = tiny_model()
    reference = copy.deepcopy(model)
    clip = 0.01
    engine = sprint.SprintStepEngine(model, {"grad_clip": clip, "lr": 0.001, "cpu_threads": 1}, "cpu")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    labels = ids.clone()
    labels[0, 1:5] = -100
    reference(input_ids=ids, labels=labels).loss.backward()
    expected = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(parameter.grad) for parameter in reference.parameters() if parameter.grad is not None
    ]))
    result = engine.train_step([{"input_ids": ids[:1], "labels": labels[:1]},
                                {"input_ids": ids[1:], "labels": labels[1:]}])
    assert expected > clip * 10
    torch.testing.assert_close(result.preclip_grad_norm, expected)
    actual_clipped = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(parameter.grad) for parameter in model.parameters() if parameter.grad is not None
    ]))
    assert actual_clipped <= clip + 1e-6
    assert result.supervised_tokens == 10


def test_gradient_progress_reports_window_average_and_clip_coefficient(training_fixture, tmp_path, monkeypatch):
    cfg, _ = _train_only(training_fixture)
    norms = []
    actual_clip = torch.nn.utils.clip_grad_norm_

    def clip(*args, **kwargs):
        result = actual_clip(*args, **kwargs)
        norms.append(float(result))
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    summary = sprint.run_sprint_training({**cfg, "max_steps": 3, "log_interval": 2, "grad_clip": 0.0001,
                                         "generation_eval": _health_config(False)}, tmp_path / "gradients")
    metrics = [json.loads(line) for line in (tmp_path / "gradients" / "metrics.jsonl").read_text().splitlines()]
    assert len(norms) == 3
    first = metrics[0]
    assert first["step"] == 2
    assert first["gradient_norm_steps"] == 2
    assert first["gradient_norm"] == pytest.approx(norms[1])
    assert first["gradient_norm_mean"] == pytest.approx(sum(norms[:2]) / 2)
    assert first["gradient_clip_fraction"] == 1.0
    assert summary["gradient_norm_steps"] == 1
    assert summary["gradient_norm"] == pytest.approx(norms[-1])
    assert summary["gradient_norm_mean"] == pytest.approx(norms[-1])
    assert summary["gradient_clip_coefficient"] == pytest.approx(0.0001 / (norms[-1] + 1e-6))


def test_stopped_quota_run_saves_consumed_prefix_not_whole_planned_mix(quota_recipe, tmp_path):
    cfg, tokenizer, _ = quota_recipe
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    run_dir = tmp_path / "stopped"
    run_dir.mkdir()
    (run_dir / "STOP").touch()
    model_config = SprintConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                                num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
                                loss_backend="torch").to_dict()
    summary = sprint.run_sprint_training({**cfg, "quota_max_source_epochs": 8,
        "tokenizer_ref": str(tokenizer_path), "model_config": model_config,
        "torch_compile": False, "save_interval": 0, "eval_interval": 0, "cpu_threads": 1,
        "generation_eval": _health_config(False)}, run_dir)
    assert summary["stop_reason"] == "stop_file"
    assert summary["steps"] == 1 and summary["next_row"] == 2
    assert summary["runtime"]["packed_rows"] == 64
    consumed = summary["consumed_data_stats"]
    assert consumed["sequences"] == 2
    assert consumed["tokens"]["forwarded_real_inputs"] == summary["tokens"] == 16
    assert consumed["tokens"]["optimizer_supervised_targets"] == summary["supervised_tokens"] == 14
    cache = sprint.SprintPackedCache(summary["runtime"]["packed_cache"])
    try:
        assert consumed == cache.prefix_data_stats(2)
        source_indices = list(cache.provenance["row_source"][:2])
        for index, cid in enumerate(cfg["corpora"]):
            assert consumed["per_corpus"][cid]["sequences"] == source_indices.count(index)
    finally:
        cache.close()
    metadata = _read(Path(summary["checkpoint_path"]) / "sprint_training.json")
    state = torch.load(Path(summary["checkpoint_path"]) / "trainer.pt", weights_only=True)
    assert metadata["consumed_data_stats"] == state["consumed_data_stats"] == consumed


def _parent_format_fixture(cfg, tokenizer, path, *, objective, legacy=False):
    tokenizer.chat_template = (
        "{% for message in messages %}{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}"
        "{% else %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"
    )
    model = SprintForCausalLM(SprintConfig.from_dict(cfg["model_config"]))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    contract = build_prompt_format_contract({"objective": objective}, tokenizer)
    if not legacy:
        (path / PROMPT_FORMAT_FILENAME).write_text(json.dumps(contract), encoding="utf-8")
    (path / "sprint_training.json").write_text(json.dumps({"config": {"objective": objective}}), encoding="utf-8")
    return contract


def test_filtered_quota_training_keeps_evaluation_separate_and_saves_cleanup_receipt(training_fixture, tmp_path):
    cfg, _, write_source, records = training_fixture
    write_source([{**records[0], "text": records[0]["text"] * 50}, records[1]])
    cfg.update(corpus_mix="fixed_token_quota", quota_row_filter={"enabled": True},
               quota_max_source_epochs=8, max_steps=1, save_interval=0,
               eval_interval=1, generation_eval={"enabled": False})
    summary = sprint.run_sprint_training(cfg, tmp_path / "filtered")
    assert "eval_loss" in summary
    stats = summary["consumed_data_stats"]
    assert stats["tokens"]["forwarded_model_slots"] == summary["tokens"] == 32
    cleanup = stats["row_filter"]
    assert cleanup["scope"] == "complete_source_pool_preparation_not_consumed_prefix"
    receipt = cleanup["per_corpus"][cfg["corpora"][0]]
    assert receipt["rejected_rows"] > 0
    assert receipt["input_rows"] == receipt["retained_rows"] + receipt["rejected_rows"]
    metadata = _read(Path(summary["checkpoint_path"]) / "sprint_training.json")
    assert metadata["consumed_data_stats"]["row_filter"] == cleanup
    assert metadata["config"]["quota_row_filter"]["enabled"] is True


def _continuation_config(cfg, parent, *, objective):
    return {**cfg, "objective": objective, "tokenizer_ref": str(parent), "model_config": {},
            "initialization": {"mode": "checkpoint", "checkpoint_ref": str(parent)},
            "max_steps": 1, "save_interval": 0, "eval_interval": 0, "generation_eval": _health_config(False)}


@pytest.mark.parametrize("legacy", [False, True])
def test_causal_continuation_inherits_and_saves_verified_parent_chat_format(training_fixture, tmp_path, legacy):
    cfg, tokenizer = _train_only(training_fixture)
    parent = tmp_path / "chat-parent"
    expected = _parent_format_fixture(cfg, tokenizer, parent, objective="chat_sft", legacy=legacy)
    summary = sprint.run_sprint_training(_continuation_config(cfg, parent, objective="causal_lm"),
                                         tmp_path / "continued")
    assert _read(Path(summary["checkpoint_path"]) / PROMPT_FORMAT_FILENAME) == expected
    assert load_prompt_format(summary["checkpoint_path"]).default_mode == "chat"
    assert (parent / PROMPT_FORMAT_FILENAME).exists() is (not legacy)


def test_new_chat_sft_continuation_upgrades_raw_parent_and_saves_actual_trained_format(training_fixture, tmp_path):
    cfg, tokenizer, write_source, _ = training_fixture
    parent = tmp_path / "raw-parent"
    inherited = _parent_format_fixture(cfg, tokenizer, parent, objective="causal_lm")
    write_source([{"text": "chat fixture", "split": "train", "domain": "code", "messages": [
        {"role": "user", "content": "alpha beta"}, {"role": "assistant", "content": "gamma delta"}]}])
    recipe = _continuation_config(cfg, parent, objective="chat_sft")
    # Copying a parent's resolved recipe may already carry its raw contract.
    recipe["prompt_format_contract"] = inherited
    summary = sprint.run_sprint_training(recipe, tmp_path / "chat-sft")
    expected = build_prompt_format_contract({"objective": "chat_sft"}, tokenizer)
    assert _read(Path(summary["checkpoint_path"]) / PROMPT_FORMAT_FILENAME) == expected
    assert summary["supervised_tokens"] == 12


def test_chat_sft_template_conflict_fails_before_loading_model(training_fixture, tmp_path, monkeypatch):
    cfg, tokenizer, _, _ = training_fixture
    parent = tmp_path / "chat-parent"
    _parent_format_fixture(cfg, tokenizer, parent, objective="chat_sft")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(parent)
    recipe = sprint.normalize_sprint_config(_continuation_config(cfg, parent, objective="chat_sft"))
    monkeypatch.setattr(SprintForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("loaded model before format validation"))
    with pytest.raises(ValueError, match="tokenizer template differs"):
        sprint._load_training_inputs(recipe, None)


def test_new_causal_continuation_allows_explicit_format_override_but_exact_resume_rejects_it(
    training_fixture, tmp_path,
):
    cfg, tokenizer, _, _ = training_fixture
    parent = tmp_path / "chat-parent"
    _parent_format_fixture(cfg, tokenizer, parent, objective="chat_sft")
    override = build_prompt_format_contract({"objective": "causal_lm"}, tokenizer)
    recipe = sprint.normalize_sprint_config({**_continuation_config(cfg, parent, objective="causal_lm"),
                                             "prompt_format_contract": override})
    sprint._load_training_inputs(recipe, None)
    assert recipe["prompt_format_contract"] == override
    with pytest.raises(ValueError, match="exact resume cannot change"):
        sprint._load_training_inputs(recipe, parent)


def test_generation_chat_on_raw_contract_is_rejected_before_step_engine(training_fixture, tmp_path, monkeypatch):
    cfg, tokenizer = _train_only(training_fixture)
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(cfg["tokenizer_ref"])
    health = _health_config()
    health["options"]["prompt_format"] = "tokenizer_chat"
    health["options"]["prompts"][0]["kind"] = "instruction"
    monkeypatch.setattr(sprint, "SprintStepEngine", lambda *a, **k: pytest.fail("allocated step engine before format validation"))
    with pytest.raises(ValueError, match="tokenizer_chat requires a recorded trained chat format"):
        sprint.run_sprint_training({**cfg, "generation_eval": health}, tmp_path / "bad-health-format")


def test_generation_uses_inherited_recorded_chat_template_without_mutating_tokenizer(
    training_fixture, tmp_path, monkeypatch,
):
    from transformers import AutoTokenizer

    cfg, tokenizer = _train_only(training_fixture)
    parent = tmp_path / "chat-parent"
    inherited = _parent_format_fixture(cfg, tokenizer, parent, objective="chat_sft")
    recorded = inherited["chat_template"]
    current = "gamma {{ messages[0]['content'] }} delta"
    tokenizer.chat_template = current
    tokenizer.save_pretrained(parent)
    original_files = {path.name: path.read_bytes() for path in parent.iterdir()
                      if path.suffix in {".json", ".jinja"}}
    health = _health_config(True, 1)
    health["options"]["prompt_format"] = "tokenizer_chat"
    health["options"]["prompts"] = [{"name": "chat", "kind": "instruction", "prompt": "alpha"}]
    real_evaluate = sprint_quality.evaluate_sprint_generation
    calls = []

    def evaluate(model, active_tokenizer, **kwargs):
        assert active_tokenizer.chat_template == current
        assert kwargs["chat_template"] == recorded
        receipt = real_evaluate(model, active_tokenizer, **kwargs)
        assert active_tokenizer.chat_template == current
        calls.append(receipt)
        return receipt

    monkeypatch.setattr(sprint_quality, "evaluate_sprint_generation", evaluate)
    recipe = {**_continuation_config(cfg, parent, objective="causal_lm"), "generation_eval": health}
    summary = sprint.run_sprint_training(recipe, tmp_path / "bound-generation")
    assert len(calls) == 1
    expected = tokenizer.apply_chat_template([{"role": "user", "content": "alpha"}],
                                              tokenize=False, add_generation_prompt=True, chat_template=recorded)
    wrong = tokenizer.apply_chat_template([{"role": "user", "content": "alpha"}],
                                           tokenize=False, add_generation_prompt=True)
    assert expected != wrong
    receipt = _read(summary["generation_eval_path"])
    assert receipt["summary"]["skipped_outputs"] == 0
    assert all(output["prompt"] == expected for output in receipt["outputs"])
    assert all(output["prompt_token_ids"] == tokenizer.encode(expected, add_special_tokens=False)
               for output in receipt["outputs"])
    assert tokenizer.chat_template == current
    assert AutoTokenizer.from_pretrained(summary["checkpoint_path"], local_files_only=True).chat_template == current
    assert _read(Path(summary["checkpoint_path"]) / PROMPT_FORMAT_FILENAME) == inherited
    assert {name: (parent / name).read_bytes() for name in original_files} == original_files
