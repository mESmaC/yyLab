from __future__ import annotations

import copy

import pytest

from yylab.ui.pages import train_sprint


def recipe(**changes):
    values = {
        "checkpoint_name": "new-sprint", "tokenizer_ref": "tiny",
        "model_config": {"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                         "num_hidden_layers": 2, "num_attention_heads": 2,
                         "max_position_embeddings": 128},
        "corpora": ["prose", "math"], "token_budget": 1024, "seq_len": 64,
    }
    return train_sprint.build_sprint_run_config(values, **changes)


def test_new_causal_recipe_defaults_to_exact_token_shares_and_no_replay():
    cfg = recipe(corpus_weights={"prose": 3})
    assert cfg["corpus_mix"] == "fixed_token_quota"
    assert cfg["corpus_weights"] == {"prose": 3.0, "math": 1.0}
    assert cfg["quota_token_shares"] == {"prose": .75, "math": .25}
    assert cfg["quota_max_source_epochs"] == {"prose": 1.0, "math": 1.0}
    assert cfg["source_max_ratio"] is None
    assert cfg["generation_eval"] == {"enabled": True, "interval": 500, "options": {"max_new_tokens": 64}}


@pytest.mark.parametrize("mix", ["token_balanced", "concatenate", "round_robin", None])
def test_explicit_legacy_recipe_selection_is_not_reinterpreted(mix):
    cfg = recipe(corpus_mix=mix)
    assert cfg["corpus_mix"] == (mix or "concatenate")
    assert "quota_token_shares" not in cfg
    assert "quota_max_source_epochs" not in cfg


def test_sft_defaults_to_legacy_and_rejects_explicit_quota():
    assert recipe(objective="chat_sft")["corpus_mix"] == "token_balanced"
    with pytest.raises(ValueError, match="causal_lm only"):
        recipe(objective="chat_sft", corpus_mix="fixed_token_quota")


def test_fixed_mix_rejects_legacy_caps_and_accepts_explicit_replay_limits():
    with pytest.raises(ValueError, match="remove source_max_ratio"):
        recipe(source_max_ratio=.75)
    assert recipe(quota_max_source_epochs=2.5)["quota_max_source_epochs"] == {"prose": 2.5, "math": 2.5}
    assert recipe(quota_max_source_epochs={"math": 2})["quota_max_source_epochs"] == {"prose": 1.0, "math": 2.0}
    for invalid in (.5, True, float("inf")):
        with pytest.raises(ValueError, match="quota_max_source_epochs"):
            recipe(quota_max_source_epochs=invalid)
    with pytest.raises(ValueError, match="Corpus weights"):
        recipe(corpus_weights={"math": True})


def test_generic_preflight_receives_only_supported_surrogate_and_recipe_is_unchanged(monkeypatch):
    cfg = recipe(corpus_weights={"prose": 3})
    snapshot = copy.deepcopy(cfg)
    calls = []

    def preflight(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs["corpus_mix"] == "token_balanced"
        return {"errors": [], "warnings": []}

    monkeypatch.setattr(train_sprint, "corpus_training_preflight", preflight)
    assert train_sprint.sprint_corpus_preflight(cfg)["errors"] == []
    assert calls[0][1]["corpus_weights"] == {"prose": 3.0, "math": 1.0}
    assert cfg == snapshot
    legacy = recipe(corpus_mix="round_robin")
    monkeypatch.setattr(train_sprint, "corpus_training_preflight", lambda *args, **kwargs: kwargs)
    assert train_sprint.sprint_corpus_preflight(legacy)["corpus_mix"] == "round_robin"


def test_generation_checks_do_not_require_loss_eval_or_held_out_splits():
    cfg = recipe(eval_interval=0, eval_splits=[], generation_eval={"interval": 100,
                 "options": {"max_new_tokens": 96, "prompt_format": "inst"}})
    assert cfg["generation_eval"] == {"enabled": True, "interval": 100,
                                      "options": {"max_new_tokens": 96, "prompt_format": "inst"}}
    assert recipe(generation_eval={"enabled": False})["generation_eval"]["enabled"] is False


@pytest.mark.parametrize("value", [
    False, {"enabled": "false"}, {"interval": 0}, {"interval": 1.5},
    {"options": []}, {"options": {"max_new_tokens": 97}}, {"options": {"repetition_penalty": 1.2}},
])
def test_invalid_generation_controls_are_rejected_before_queue(value):
    with pytest.raises(ValueError):
        recipe(generation_eval=value)


def test_fixed_recipe_roundtrip_keeps_allocation_and_explicit_generation_options():
    cfg = recipe(corpus_weights={"prose": 7, "math": 2}, quota_max_source_epochs={"math": 1.5},
                 generation_eval={"enabled": True, "interval": 40, "options": {"modes": ["greedy"]}})
    # Recipes contain both the resolved step ceiling and the original token
    # budget; the form chooses one budget mode when re-creating a recipe.
    replay = train_sprint.build_sprint_run_config({**cfg, "max_steps": 0})
    for key in ("corpus_mix", "corpus_weights", "quota_token_shares", "quota_max_source_epochs", "quota_version", "generation_eval"):
        assert replay[key] == cfg[key]


def test_pool_cleanup_is_explicit_in_recipes_and_preserved_on_roundtrip():
    assert "quota_row_filter" not in recipe()
    assert "quota_row_filter" not in recipe(quota_row_filter={"enabled": False})
    cfg = recipe(quota_row_filter={"enabled": True})
    assert cfg["quota_row_filter"] == {"enabled": True, "version": "sprint-quota-row-filter-v1"}
    replay = train_sprint.build_sprint_run_config({**cfg, "max_steps": 0})
    assert replay["quota_row_filter"] == cfg["quota_row_filter"]
    legacy = recipe(corpus_mix="token_balanced", quota_row_filter={"enabled": True})
    assert "quota_row_filter" not in legacy


@pytest.mark.parametrize("policy", [True, {}, {"enabled": "false"}, {"enabled": True, "version": "unknown"}])
def test_invalid_pool_cleanup_settings_fail_before_queue(policy):
    with pytest.raises(ValueError, match="row filter|row_filter"):
        recipe(quota_row_filter=policy)


def test_legacy_balancing_label_describes_character_budget():
    assert train_sprint.CORPUS_MIX_OPTIONS["fixed_token_quota"] == "Fixed token shares"
    assert "characters" in train_sprint.CORPUS_MIX_OPTIONS["token_balanced"]


def test_page_defaults_and_objective_changes_keep_unsupported_quota_out_of_sft(monkeypatch):
    from nicegui import Client
    from nicegui.page import page

    monkeypatch.setattr(train_sprint, "list_tokenizers", lambda **_: [])
    monkeypatch.setattr(train_sprint, "sprint_checkpoint_rows", lambda: [])
    monkeypatch.setattr(train_sprint, "corpus_options", lambda: {})
    with Client(page("/__sprint_repair_test")) as client:
        train_sprint.train_sprint_page()
        controls = {element._props.get("label"): element for element in client.elements.values()}
        mix = controls["Corpus mixing"]
        cap = controls["Maximum source ratio · 0 disables"]
        assert mix.value == "fixed_token_quota"
        assert controls["Maximum passes through each source"].value == 1
        cleanup = next(element for element in client.elements.values()
                       if getattr(element, "text", None) == "Remove duplicate and extremely repetitive sequences")
        assert cleanup.value is True
        repetition = next(element for element in client.elements.values()
                          if getattr(element, "text", None) == "Check generated text for repetition")
        assert repetition.value is True
        assert controls["Check generated text every N steps"].value == 500
        mix.set_value("token_balanced")
        cap.set_value(.75)
        mix.set_value("fixed_token_quota")
        assert cap.value == 0
        controls["Objective"].set_value("chat_sft")
        assert mix.value == "token_balanced"
        assert "fixed_token_quota" not in mix.options
        controls["Objective"].set_value("causal_lm")
        assert "fixed_token_quota" in mix.options


def test_worker_normalization_accepts_ui_fixed_mode_and_resolves_sft_probe_format(monkeypatch):
    from yylab import db
    from yylab.train.sprint import normalize_sprint_config

    monkeypatch.setattr(db, "q1", lambda *args, **kwargs: {"status": "ready", "kind": "text"})
    cfg = recipe(device="cpu", parameter_dtype="float32", optimizer_state_offload="none")
    normalized = normalize_sprint_config(cfg)
    assert normalized["corpus_mix"] == "fixed_token_quota"
    assert normalized["quota_token_shares"] == cfg["quota_token_shares"]
    assert normalized["generation_eval"]["options"]["prompt_format"] == "plain"
    sft = normalize_sprint_config(recipe(objective="chat_sft", device="cpu",
                                         parameter_dtype="float32", optimizer_state_offload="none"))
    assert sft["generation_eval"]["options"]["prompt_format"] == "tokenizer_chat"


def test_worker_rejects_explicit_pool_cleanup_for_unsupported_mixing(monkeypatch):
    from yylab import db
    from yylab.train.sprint import normalize_sprint_config

    monkeypatch.setattr(db, "q1", lambda *args, **kwargs: {"status": "ready", "kind": "text"})
    cfg = recipe(corpus_mix="token_balanced", device="cpu", parameter_dtype="float32", optimizer_state_offload="none")
    with pytest.raises(ValueError, match="requires fixed_token_quota"):
        normalize_sprint_config({**cfg, "quota_row_filter": {"enabled": True}})
    assert "quota_row_filter" not in normalize_sprint_config({**cfg, "quota_row_filter": {"enabled": False}})
