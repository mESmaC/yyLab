"""CPU coverage of native KDA training, exact checkpoint identity and UI recipes."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_sprint_ple_training import ple_recipe as ple_recipe

from yylab import db
from yylab.models.sprint import SprintConfig, SprintForCausalLM, estimate_parameter_breakdown
from yylab.train import sprint
from yylab.ui.pages import train_sprint

ATTENTION_DEFAULTS = {"attention_type": "sdpa", "kda_backend": "auto", "kda_disable_recompute": True}
KDA_OPTIONS = {"attention_type": "kda", "kda_backend": "reference", "kda_disable_recompute": False}


@pytest.fixture(autouse=True)
def one_cpu_thread(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def kda_recipe(raw):
    return {**raw, "cpu_threads": 1,
            "model_config": {**raw["model_config"], **KDA_OPTIONS, "ple_type": "full",
                             "num_hidden_layers": 4, "max_position_embeddings": 16}}


@pytest.mark.parametrize("checkpointing", [False, True])
def test_reference_kda_full_ple_train_save_exact_resume_and_continue(ple_recipe, tmp_path, checkpointing):
    raw, tokenizer = ple_recipe
    cfg = {**kda_recipe(raw), "gradient_checkpointing": checkpointing}
    identifier = f"kda-{tmp_path.name}"
    result = sprint.run_sprint_training({**cfg, "out_checkpoint_id": identifier}, tmp_path / "full")
    assert result["tokens"] == 3 * 2 * 2 * 8
    assert result["supervised_tokens"] == 3 * 2 * 2 * 7
    assert result["runtime"]["compute_dtype"] == "float32"
    assert result["runtime"]["parameter_dtype"] == "float32"
    assert result["runtime"]["ple_type"] == "full"
    assert result["runtime"]["gradient_checkpointing"] is checkpointing
    for key, value in KDA_OPTIONS.items():
        assert result["runtime"][key] == value
    loaded = SprintForCausalLM.from_pretrained(result["checkpoint_path"], local_files_only=True).eval()
    assert sum(hasattr(layer.self_attn, "A_log") for layer in loaded.layers) == 3
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
    assert all(torch.isfinite(parameter).all() for parameter in loaded.parameters())
    assert result["runtime"]["parameter_breakdown"] == estimate_parameter_breakdown(loaded.config)
    assert sum(parameter.numel() for parameter in loaded.parameters()) == result["runtime"]["parameter_breakdown"]["total"]
    metadata = json.loads(db.q1("SELECT meta_json FROM checkpoints WHERE id=?", (identifier,))["meta_json"])
    saved = json.loads((Path(result["checkpoint_path"]) / "sprint_training.json").read_text())
    for key, value in KDA_OPTIONS.items():
        assert metadata["model_config"][key] == saved["contract"]["model_config"][key] == value
        assert saved["config"]["model_config"][key] == value
    with torch.no_grad():
        prompt = tokenizer("alpha beta", return_tensors="pt")
        sampled = loaded.generate(**prompt, max_new_tokens=2, do_sample=False, use_cache=True)
    assert prompt["input_ids"].shape[1] < sampled.shape[1] <= 4
    periodic = tmp_path / "full/checkpoints/step-00000001"
    resumed = sprint.run_sprint_training({"resume_from": str(periodic), "out_checkpoint_id": identifier + "-resumed"},
                                         tmp_path / "resumed")
    assert resumed["tokens"] == result["tokens"] and resumed["new_steps"] == 2
    restored = SprintForCausalLM.from_pretrained(resumed["checkpoint_path"], local_files_only=True)
    for name, expected in loaded.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    continued = sprint.run_sprint_training(
        {**cfg, "model_config": {}, "initialization": {"mode": "checkpoint", "checkpoint_ref": result["checkpoint_path"]},
         "max_steps": 1, "save_interval": 0}, tmp_path / "continued")
    assert continued["steps"] == continued["new_steps"] == 1
    assert continued["tokens"] == 2 * 2 * 8
    resolved = json.loads((tmp_path / "continued/resolved_config.json").read_text())
    for key, value in KDA_OPTIONS.items():
        assert resolved["model_config"][key] == continued["runtime"][key] == value
    assert resolved["model_config"]["ple_type"] == "full"
    # Reject architecture and execution changes even when all tensor shapes would fit.
    for index, override in enumerate(({"attention_type": "sdpa"}, {"kda_backend": "auto"},
                                       {"kda_disable_recompute": True})):
        with pytest.raises(ValueError, match="architecture"):
            sprint.run_sprint_training({"resume_from": str(periodic), "model_config": override,
                                       "out_checkpoint_id": identifier + f"-bad-{index}"}, tmp_path / f"bad-{index}")


def test_legacy_sdpa_resume_and_continue_inherit_absent_attention_fields(ple_recipe, tmp_path):
    cfg, _ = ple_recipe
    trained = sprint.run_sprint_training(cfg, tmp_path / "legacy")
    periodic = tmp_path / "legacy/checkpoints/step-00000001"
    config_path = periodic / "config.json"
    config = json.loads(config_path.read_text())
    for key in ATTENTION_DEFAULTS:
        config.pop(key)
    config_path.write_text(json.dumps(config))
    state = torch.load(periodic / "trainer.pt", map_location="cpu", weights_only=True)
    for section in (state["contract"]["model_config"], state["config"]["model_config"]):
        for key in ATTENTION_DEFAULTS:
            section.pop(key)
    torch.save(state, periodic / "trainer.pt")
    resumed = sprint.run_sprint_training({"resume_from": str(periodic)}, tmp_path / "legacy-resumed")
    expected = SprintForCausalLM.from_pretrained(trained["checkpoint_path"], local_files_only=True)
    restored = SprintForCausalLM.from_pretrained(resumed["checkpoint_path"], local_files_only=True)
    for key, value in ATTENTION_DEFAULTS.items():
        assert getattr(restored.config, key) == value
    assert not any(hasattr(layer.self_attn, "A_log") for layer in restored.layers)
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    continued = sprint.normalize_sprint_config({**cfg, "model_config": {},
                                               "initialization": {"mode": "checkpoint", "path": str(periodic)}})
    inherited, _, _ = sprint._load_training_inputs(continued, None)
    assert continued["model_config"] == {}
    assert sprint._model_contract(inherited.config)["attention_type"] == "sdpa"


def test_attention_defaults_contract_and_packed_data_identity(ple_recipe):
    raw, tokenizer = ple_recipe
    legacy_model = {key: value for key, value in raw["model_config"].items() if key not in ATTENTION_DEFAULTS}
    ordinary = sprint.normalize_sprint_config({**raw, "model_config": legacy_model})
    for key, value in ATTENTION_DEFAULTS.items():
        assert ordinary["model_config"][key] == value
    normalized = sprint.normalize_sprint_config(kda_recipe(raw))
    assert sprint._packed_identity(normalized, tokenizer) == sprint._packed_identity(ordinary, tokenizer)
    contract = sprint._model_contract(SprintConfig(**normalized["model_config"]))
    for key, changed in (("attention_type", "sdpa"), ("kda_backend", "auto"), ("kda_disable_recompute", True)):
        altered = SimpleNamespace(**normalized["model_config"])
        setattr(altered, key, changed)
        assert sprint._model_contract(altered) != contract
    inherited = sprint.normalize_sprint_config({**raw, "model_config": {},
                                               "initialization": {"mode": "checkpoint", "path": "saved"}})
    assert inherited["model_config"] == {}


def test_training_target_planner_keeps_kda_and_ple_count(ple_recipe):
    raw, tokenizer = ple_recipe
    cfg = sprint.normalize_sprint_config({**raw, "target_parameters": 100_000,
                                         "model_config": {**KDA_OPTIONS, "ple_type": "full", "loss_backend": "torch"}})
    model, loaded_tokenizer, parent = sprint._load_training_inputs(cfg, None)
    assert parent is None and len(loaded_tokenizer) == len(tokenizer)
    for key, value in KDA_OPTIONS.items():
        assert getattr(model.config, key) == value
    assert model.config.num_hidden_layers % 4 == 0
    assert sum(hasattr(layer.self_attn, "A_log") for layer in model.layers) == model.config.num_hidden_layers * 3 // 4
    count = estimate_parameter_breakdown(model.config)
    assert 0 < count["ple"] < count["total"] <= 100_000
    assert count["total"] == sum(parameter.numel() for parameter in model.parameters())


def test_inherited_kda_rejects_bf16_parameter_storage_before_optimizer_mutation(ple_recipe, tmp_path):
    raw, tokenizer = ple_recipe
    recipe = kda_recipe(raw)
    path = tmp_path / "kda"
    SprintForCausalLM(SprintConfig(**recipe["model_config"])).save_pretrained(path)
    tokenizer.save_pretrained(path)
    cfg = sprint.normalize_sprint_config({**raw, "model_config": {}, "parameter_dtype": "bfloat16",
                                         "optimizer_state_offload": "cpu",
                                         "initialization": {"mode": "checkpoint", "path": str(path)}})
    model, _, _ = sprint._load_training_inputs(cfg, None)
    assert cfg["model_config"] == {} and model.config.attention_type == "kda"
    parameters = [(id(parameter), parameter.dtype, parameter.detach().clone()) for parameter in model.parameters()]
    with pytest.raises(ValueError, match="KDA training currently requires FP32 weights"):
        sprint.SprintStepEngine(model, cfg, "cpu")
    for parameter, (identity, dtype, expected) in zip(model.parameters(), parameters, strict=True):
        assert id(parameter) == identity and parameter.dtype == dtype
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="KDA training currently requires FP32 weights"):
        sprint.normalize_sprint_config({**recipe, "parameter_dtype": "bfloat16", "optimizer_state_offload": "cpu"})


def test_ui_planner_cache_and_recipe_retain_selected_kda_options():
    train_sprint._planned_config.cache_clear()
    try:
        args = {"vocab_size": 128, "context_length": 64, "target_parameters": 100_000, "ple_type": "full"}
        ordinary = train_sprint.resolve_sprint_shape(**args)
        selected = train_sprint.resolve_sprint_shape(**args, **KDA_OPTIONS)
        assert train_sprint.resolve_sprint_shape(**args) == ordinary
        assert ordinary["model_config"]["attention_type"] == "sdpa"
        for key, value in KDA_OPTIONS.items():
            assert selected["model_config"][key] == value
        allocation = estimate_parameter_breakdown(SprintConfig(**selected["model_config"]))
        assert selected["parameter_breakdown"] == allocation
        training = train_sprint.build_sprint_run_config(
            {**selected, "checkpoint_name": "kda-ui", "tokenizer_ref": "tokenizer", "corpora": ["corpus"],
             "seq_len": 8, "micro_batch": 1, "grad_accum": 1, "max_steps": 2,
             "parameter_dtype": "float32", "optimizer_state_offload": "none"})
        benchmark = train_sprint.build_sprint_benchmark_config(
            model_config=selected["model_config"], seq_len=8, batch_sizes="1", compile_modes=["eager"],
            effective_batch=1, warmup_steps=1, measure_steps=1, device="cpu",
            parameter_dtype="float32", optimizer_state_offload="none")
        assert training["model_config"] == benchmark["model_config"] == selected["model_config"]
        assert training["parameter_breakdown"] == benchmark["parameter_breakdown"] == allocation
        with pytest.raises(ValueError, match="FP32 weights"):
            train_sprint.build_sprint_run_config({**training, "parameter_dtype": "bfloat16", "optimizer_state_offload": "cpu"})
        with pytest.raises(ValueError, match="FP32 weights"):
            train_sprint.build_sprint_benchmark_config(
                model_config=selected["model_config"], seq_len=8, batch_sizes="1", compile_modes=["eager"],
                effective_batch=1, warmup_steps=1, measure_steps=1, device="cpu",
                parameter_dtype="bfloat16", optimizer_state_offload="cpu")
    finally:
        train_sprint._planned_config.cache_clear()


def test_ui_recommendations_reject_other_attention_and_accept_legacy_sdpa(monkeypatch):
    config = SprintConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                          num_attention_heads=2, max_position_embeddings=64).to_dict()
    legacy = {key: value for key, value in config.items() if key not in ATTENTION_DEFAULTS}
    execution = train_sprint.sprint_execution_config(parameter_dtype="float32", optimizer_state_offload="none")
    report = {"model_config": legacy, "sequence_length": 8, "device": "cpu", "execution": execution,
              "recommended": {"micro_batch": 1, "grad_accum": 1, "torch_compile": False, "compile_mode": "default"}}
    monkeypatch.setattr(db, "q", lambda *_: [{"id": "measured", "metrics_json": json.dumps(report)}])
    kwargs = {"device": "cpu", "parameter_dtype": "float32", "optimizer_state_offload": "none"}
    assert train_sprint.latest_sprint_recommendation(config, 8, **kwargs)["run_id"] == "measured"
    for change in ({"attention_type": "kda"}, {"kda_backend": "reference"}, {"kda_disable_recompute": False}):
        with pytest.raises(ValueError, match="No completed benchmark"):
            train_sprint.latest_sprint_recommendation({**config, **change}, 8, **kwargs)
    report["model_config"] = {**config, **KDA_OPTIONS}
    assert train_sprint.latest_sprint_recommendation(report["model_config"], 8, **kwargs)["run_id"] == "measured"
    with pytest.raises(ValueError, match="No completed benchmark"):
        train_sprint.latest_sprint_recommendation(config, 8, **kwargs)


def test_ui_attention_defaults_and_kda_selection_switches_gpu_optimizer(monkeypatch):
    from nicegui import Client
    from nicegui.page import page

    monkeypatch.setattr(train_sprint, "list_tokenizers", lambda **_: [])
    monkeypatch.setattr(train_sprint, "sprint_checkpoint_rows", lambda: [])
    monkeypatch.setattr(train_sprint, "corpus_options", lambda: {})
    with Client(page("/__sprint_kda_test")) as client:
        train_sprint.train_sprint_page()
        controls = {element._props.get("label"): element for element in client.elements.values()}
        attention = controls["Model attention"]
        optimizer = controls["Weights and optimizer"]
        assert attention.value == "sdpa" and optimizer.value == "cpu_offload"
        assert controls["Per-layer embeddings (PLE)"].value == "none"
        attention.set_value("kda")
        assert optimizer.value == "gpu"


def test_checkpoint_attention_reads_saved_config_metadata_and_legacy_defaults(tmp_path):
    checkpoint = tmp_path / "saved"
    checkpoint.mkdir()
    row = {"meta_json": json.dumps({"model_config": {"attention_type": "kda"}})}
    assert train_sprint.checkpoint_attention_type(row) == "kda"
    assert train_sprint.checkpoint_attention_type({"meta_json": json.dumps({"model_config": {"hidden_size": 16}})}) == "sdpa"
    assert train_sprint.checkpoint_attention_type({"meta_json": "broken"}) is None
    assert train_sprint.checkpoint_attention_type(resume_from=str(tmp_path / "missing")) is None
    config = checkpoint / "config.json"
    config.write_text(json.dumps({"model_type": "sprint", "attention_type": "kda"}))
    assert train_sprint.checkpoint_attention_type(resume_from=str(checkpoint)) == "kda"
    assert train_sprint.checkpoint_attention_type(resume_from=str(checkpoint / "trainer.pt")) == "kda"
    # Saved model config is authoritative over potentially stale registry metadata.
    config.write_text(json.dumps({"model_type": "sprint"}))
    assert train_sprint.checkpoint_attention_type({**row, "path": str(checkpoint)}) == "sdpa"


@pytest.mark.parametrize("parent_first", [False, True])
def test_ui_continue_checkpoint_selects_gpu_from_registered_kda_metadata(monkeypatch, parent_first):
    from nicegui import Client
    from nicegui.page import page

    row = {"id": "saved-kda", "name": "Saved KDA", "kind": "sprint",
           "meta_json": json.dumps({"model_config": {"attention_type": "kda", "ple_type": "full"}})}
    monkeypatch.setattr(train_sprint, "list_tokenizers", lambda **_: [])
    monkeypatch.setattr(train_sprint, "sprint_checkpoint_rows", lambda: [row])
    monkeypatch.setattr(train_sprint, "corpus_options", lambda: {})
    with Client(page("/__sprint_kda_continue_test")) as client:
        train_sprint.train_sprint_page()
        controls = {element._props.get("label"): element for element in client.elements.values()}
        if parent_first:
            controls["Sprint checkpoint"].set_value("saved-kda")
            assert controls["Weights and optimizer"].value == "cpu_offload"
            controls["Initialization"].set_value("checkpoint")
        else:
            controls["Initialization"].set_value("checkpoint")
            assert controls["Weights and optimizer"].value == "cpu_offload"
            controls["Sprint checkpoint"].set_value("saved-kda")
        assert controls["Weights and optimizer"].value == "gpu"
        texts = [getattr(element, "text", "") for element in client.elements.values()]
        assert any("This checkpoint uses Kimi linear attention" in text for text in texts)


def test_ui_resume_directory_detects_kda_and_unknown_path_explains_guard(monkeypatch, tmp_path):
    from nicegui import Client
    from nicegui.page import page

    saved = tmp_path / "step"
    saved.mkdir()
    (saved / "config.json").write_text(json.dumps({"model_type": "sprint", "attention_type": "kda"}))
    monkeypatch.setattr(train_sprint, "list_tokenizers", lambda **_: [])
    monkeypatch.setattr(train_sprint, "sprint_checkpoint_rows", lambda: [])
    monkeypatch.setattr(train_sprint, "corpus_options", lambda: {})
    with Client(page("/__sprint_kda_resume_test")) as client:
        train_sprint.train_sprint_page()
        controls = {element._props.get("label"): element for element in client.elements.values()}
        controls["Initialization"].set_value("resume")
        controls["Resume checkpoint directory"].set_value(str(tmp_path / "unknown"))
        assert controls["Weights and optimizer"].value == "cpu_offload"
        texts = [getattr(element, "text", "") for element in client.elements.values()]
        assert any("Select a readable checkpoint" in text and "GPU optimizer preset" in text for text in texts)
        controls["Resume checkpoint directory"].set_value(str(saved))
        assert controls["Weights and optimizer"].value == "gpu"
