from __future__ import annotations

import json
import sys

import pytest

from yylab.workers import train as worker


def test_benchmark_failure_never_mutates_the_adapter_under_test(monkeypatch):
    updates: list[tuple[str, tuple]] = []
    monkeypatch.setattr(
        worker.db,
        "execute",
        lambda sql, args=(): updates.append((sql, args)) or 1,
    )

    worker.mark_training_artifacts_failed(
        "benchmark_quality",
        {"adapter_id": "legacy-key", "adapter_benchmark_id": "trained-subject"},
    )
    worker.mark_training_artifacts_failed(
        "benchmark_train_perf",
        {"adapter_id": "legacy-key"},
    )

    assert updates == []


def test_training_failure_still_marks_owned_adapter_and_router(monkeypatch):
    updates: list[tuple[str, tuple]] = []
    monkeypatch.setattr(
        worker.db,
        "execute",
        lambda sql, args=(): updates.append((sql, args)) or 1,
    )

    worker.mark_training_artifacts_failed(
        "router_stage2",
        {"adapter_id": "adapter", "router_id": "router"},
    )

    assert updates == [
        ("UPDATE adapters SET status='failed' WHERE id=?", ("adapter",)),
        ("UPDATE routers SET status='failed' WHERE id=?", ("router",)),
    ]


def test_quality_worker_prefers_semantic_baseline_projection(tmp_path, monkeypatch):
    from yylab.benchmark import quality

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "summary.json").write_text("{}", encoding="utf-8")
    semantic_path = baseline_dir / "semantic-summary.json"
    semantic_path.write_text("{}", encoding="utf-8")
    captured: list[dict] = []
    monkeypatch.setattr(worker, "run_dir", lambda _run_id: baseline_dir)
    monkeypatch.setattr(
        quality,
        "run_quality_benchmark",
        lambda cfg, _rd, progress=None: captured.append(cfg) or {},
    )

    worker.run_benchmark_quality({"baseline_run_id": "run-base"}, tmp_path / "candidate")

    assert captured[0]["baseline_summary_path"] == str(semantic_path)


@pytest.mark.parametrize(
    ("kind", "runner_name"),
    [
        ("benchmark_quality", "run_benchmark_quality"),
        ("benchmark_train_perf", "run_benchmark_training_performance"),
    ],
)
def test_worker_main_dispatches_benchmark_run_kinds(tmp_path, monkeypatch, kind, runner_name):
    updates: list[tuple[str, tuple]] = []
    calls: list[tuple[dict, object]] = []
    config = {"label": "fixture benchmark"}
    monkeypatch.setattr(sys, "argv", ["train", "--run", "run-fixture"])
    monkeypatch.setattr(
        worker.db,
        "q1",
        lambda *_args, **_kwargs: {
            "id": "run-fixture",
            "kind": kind,
            "config_json": json.dumps(config),
        },
    )
    monkeypatch.setattr(
        worker.db,
        "execute",
        lambda sql, args=(): updates.append((sql, args)) or 1,
    )
    monkeypatch.setattr(worker, "run_dir", lambda _run_id: tmp_path)
    monkeypatch.setattr(worker, "log", lambda _message: None)
    monkeypatch.setattr(
        worker,
        runner_name,
        lambda cfg, rd: calls.append((cfg, rd)) or {"status": "ok", "metric": 7},
    )

    worker.main()

    assert calls == [(config, tmp_path)]
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == config
    assert any("status='running'" in sql for sql, _args in updates)
    done = [(sql, args) for sql, args in updates if "status='done'" in sql]
    assert len(done) == 1
    assert json.loads(done[0][1][1]) == {"status": "ok", "metric": 7}
