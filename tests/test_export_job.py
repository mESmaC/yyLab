from __future__ import annotations

import json

import pytest

from yylab.artifacts import read_json, write_json
from yylab.ui.pages.checkpoints import (
    STATIC_TOKEN_EXPORT_MODE,
    _adapter_router_type,
    _export_job_config,
)


def _joint_adapter(tmp_path) -> dict:
    artifact = tmp_path / "adapter"
    write_json(
        artifact / "mole_config.json",
        {
            "arch": "mole",
            "format_version": 3,
            "base_ref": "base-ref",
            "layers": ["model.layers.0.self_attn.q_proj"],
            "config": {"router_type": "per_token"},
        },
    )
    return {
        "id": "adapter-id",
        "path": str(artifact),
        "arch": "mole",
        "meta_json": "{}",
        "corpora_json": json.dumps(["corpus-a"]),
    }


def test_joint_export_config_ignores_legacy_router_and_marks_static_approximation(tmp_path):
    adapter = _joint_adapter(tmp_path)
    assert _adapter_router_type(adapter) == "per_token"

    config = _export_job_config(
        {"ref": "base-ref"},
        adapter,
        {"path": "legacy-shared-prompt-router"},
        "output-id",
    )

    assert config["router_path"] is None
    assert config["source_router_type"] == "per_token"
    assert config["export_mode"] == STATIC_TOKEN_EXPORT_MODE
    assert config["dynamic_token_routing_preserved"] is False
    assert config["corpora"] == ["corpus-a"]


def test_export_worker_persists_calibrated_static_metadata(tmp_path, monkeypatch):
    import yylab.artifacts as artifacts
    import yylab.export.merge as merge_module
    import yylab.models.loading as loading_module
    from yylab.workers import train as worker

    output = tmp_path / "output"
    merge_calls: list[dict] = []
    database_updates: list[tuple[str, tuple]] = []

    def fake_merge_mole(_base, _adapter, out, **kwargs):
        merge_calls.append(kwargs)
        write_json(out / "config.json", {"model_type": "tiny"})

    monkeypatch.setattr(artifacts, "checkpoint_dir", lambda _checkpoint_id: output)
    monkeypatch.setattr(merge_module, "merge_mole", fake_merge_mole)
    monkeypatch.setattr(
        worker,
        "load_mole_config",
        lambda *_args, **_kwargs: {"config": {"router_type": "per_token"}},
    )
    monkeypatch.setattr(worker, "_calib_ids", lambda *_args, **_kwargs: [[1, 2, 3]])
    monkeypatch.setattr(
        loading_module,
        "scan_checkpoint",
        lambda _path: {
            "arch": "TinyForCausalLM",
            "params_b": 0.01,
            "hidden_size": 16,
            "num_layers": 1,
        },
    )
    monkeypatch.setattr(
        worker.db,
        "execute",
        lambda sql, args=(): database_updates.append((sql, args)) or 1,
    )

    summary = worker.run_export(
        {
            "checkpoint_ref": "base-ref",
            "adapter_path": str(tmp_path / "adapter"),
            "adapter_arch": "mole",
            "router_path": None,
            "corpora": ["corpus-a"],
            "out_checkpoint_id": "output-id",
        },
        tmp_path / "run",
    )

    assert merge_calls == [{"router_path": None, "calib_ids": [[1, 2, 3]]}]
    metadata = summary["export"]
    assert metadata["export_mode"] == STATIC_TOKEN_EXPORT_MODE
    assert metadata["source_router_type"] == "per_token"
    assert metadata["dynamic_token_routing_preserved"] is False
    assert metadata["calibration"]["sequence_count"] == 1
    assert read_json(output / "config.json")["yylab_export"] == metadata
    assert read_json(output / "yylab_export.json") == metadata
    saved_metadata = json.loads(database_updates[-1][1][-2])
    assert saved_metadata == metadata


def test_export_worker_rejects_prompt_router_for_joint_adapter(tmp_path, monkeypatch):
    import yylab.artifacts as artifacts
    from yylab.workers import train as worker

    monkeypatch.setattr(artifacts, "checkpoint_dir", lambda _checkpoint_id: tmp_path / "out")
    monkeypatch.setattr(
        worker,
        "load_mole_config",
        lambda *_args, **_kwargs: {"config": {"router_type": "per_token"}},
    )

    with pytest.raises(ValueError, match="legacy shared-prompt router"):
        worker.run_export(
            {
                "checkpoint_ref": "base-ref",
                "adapter_path": str(tmp_path / "adapter"),
                "adapter_arch": "mole",
                "router_path": str(tmp_path / "legacy-router"),
                "corpora": ["corpus-a"],
                "out_checkpoint_id": "output-id",
            },
            tmp_path / "run",
        )
