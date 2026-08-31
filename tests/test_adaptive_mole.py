from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import SafetensorError
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from yylab.mole.adaptive import (
    AdaptiveOccupancyConfig,
    AdaptiveOccupancyController,
    solve_expert_budget,
)
from yylab.mole.inject import MoLEConfig, inject_mole, mole_layers, resolve_adaptive_budget
from yylab.mole.layer import MoLELinear, SharedTokenRouter
from yylab.mole.store import (
    CONFIG_NAME,
    EXPERTS_DIR,
    WEIGHTS_NAME,
    LazyExpertStore,
    load_mole_config,
    save_mole,
)


class _TinyBlock(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.q_proj(hidden_states)


class _TinyBlockModel(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width)
        self.layers = nn.ModuleList([_TinyBlock(width)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layers[0](hidden_states)


def _adaptive_config(*, num_experts: int = 4) -> MoLEConfig:
    return MoLEConfig(
        num_experts=num_experts,
        rank=2,
        alpha=4.0,
        top_k=2,
        target_modules=("q_proj",),
        router_type="per_token",
        adaptive=AdaptiveOccupancyConfig(
            enabled=True,
            baseline_nll=2.25,
            preferred_experts=(1, 0),
        ),
    )


def _saved_adaptive_artifact(directory) -> tuple[_TinyBlockModel, MoLEConfig]:
    model = _TinyBlockModel()
    cfg = _adaptive_config()
    replaced = inject_mole(model, cfg, expert_dtype=torch.float32)
    assert len(replaced) == 1
    with torch.no_grad():
        for _, layer in mole_layers(model):
            for index, expert in enumerate(layer.experts):
                expert.lora_A.fill_(index + 1.0)
                expert.lora_B.fill_(index + 11.0)
    save_mole(directory, model, cfg, base_ref="tiny-adaptive")
    return model, cfg


def test_adaptive_config_round_trip_and_legacy_default() -> None:
    adaptive = AdaptiveOccupancyConfig(
        enabled=True,
        baseline_nll=1.75,
        preferred_experts=(3, 1),
        base_parameter_count=1_000,
        router_parameter_count=40,
        expert_parameter_count=10,
        resolved_total_parameter_count=1_240,
        resolved_expert_footprint_fraction=0.005,
    )
    cfg = MoLEConfig(
        num_experts=20,
        rank=5,
        alpha=10.0,
        top_k=2,
        target_modules=("q_proj", "v_proj"),
        router_type="per_token",
        adaptive=adaptive,
    )

    payload = cfg.to_dict()
    assert payload["adaptive"]["preferred_experts"] == [3, 1]
    restored = MoLEConfig.from_dict(payload)
    assert restored == cfg
    restored.adaptive.validate(num_experts=20)

    legacy = MoLEConfig.from_dict(
        {
            "num_experts": 4,
            "rank": 8,
            "top_k": 2,
            "router_type": "per_token",
        }
    )
    assert legacy.adaptive.enabled is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"score_semantics": "utility"}, "predicted_nll"),
        ({"expert_footprint_fraction": 0.0}, "footprint fraction"),
        ({"min_active_fraction": 0.011}, "integer expert quantum"),
        ({"max_active_fraction": 0.11}, "max <= 10%"),
        ({"preferred_experts": (1, 1)}, "must be unique"),
    ],
)
def test_adaptive_config_rejects_invalid_contract(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptiveOccupancyConfig(enabled=True, **overrides)


def test_adaptive_config_validates_bank_bounds() -> None:
    cfg = AdaptiveOccupancyConfig(enabled=True, preferred_experts=(3,))
    with pytest.raises(ValueError, match="needs at least 2 experts"):
        cfg.validate(num_experts=1)
    with pytest.raises(ValueError, match="preferred expert index exceeds"):
        cfg.validate(num_experts=3)


def test_exact_expert_budget_and_actual_fraction_drive_active_bounds() -> None:
    plan = solve_expert_budget(
        base_parameters=1_000,
        expert_rank_width=10,
        router_parameters_per_expert=0,
        target_total_multiplier=2.0,
        expert_footprint_fraction=0.005,
        minimum_experts=2,
        maximum_experts=196,
        maximum_rank=10,
    )

    assert plan.requested_total_parameters == 2_000
    assert plan.resolved_total_parameters == 2_000
    assert plan.num_experts == 100
    assert plan.rank == 1
    assert plan.parameters_per_expert == 10
    assert plan.expert_bank_parameters == 1_000
    assert plan.resolved_total_parameters == (
        plan.base_parameters + plan.router_parameters + plan.expert_bank_parameters
    )
    assert plan.expert_footprint_fraction == pytest.approx(0.005)
    assert plan.budget_error_fraction == 0.0

    # Integer rank/expert resolution can move the real footprint away from the
    # requested 0.5%. Runtime bounds must use that resolved fraction.
    cfg = AdaptiveOccupancyConfig(
        enabled=True,
        resolved_expert_footprint_fraction=0.006,
    )
    assert cfg.minimum_experts == 2  # ceil(1% / 0.6%)
    assert cfg.maximum_experts(100) == 16  # floor(10% / 0.6%)
    assert cfg.maximum_experts(12) == 12


def test_budget_resolution_uses_logical_count_for_packed_base_weights() -> None:
    model = _TinyBlockModel()
    physical_count = sum(parameter.numel() for parameter in model.parameters())
    model.num_parameters = lambda: 100_000
    cfg = MoLEConfig(
        num_experts=2,
        rank=2,
        alpha=4.0,
        top_k=2,
        target_modules=("q_proj",),
        router_type="per_token",
        adaptive=AdaptiveOccupancyConfig(enabled=True),
    )

    plan = resolve_adaptive_budget(model, cfg)

    assert physical_count != 100_000
    assert plan.base_parameters == 100_000
    assert cfg.adaptive.base_parameter_count == 100_000


def test_controller_expands_on_equality_and_contracts_with_hysteresis() -> None:
    cfg = AdaptiveOccupancyConfig(
        enabled=True,
        baseline_nll=2.0,
        surprise_ema_decay=0.999,
        surprise_ema_warmup=1,
        expand_margin_nats=0.0,
        contract_margin_nats=0.1,
        expand_patience=1,
        contract_patience=2,
    )
    controller = AdaptiveOccupancyController(cfg, num_experts=20)

    equality = controller.observe(2.0, "nll")
    assert equality.action == "expand"
    assert (equality.previous_experts, equality.active_experts) == (2, 3)

    first_low = controller.observe(1.0, "nll")
    assert first_low.action == "hold"
    assert first_low.reason == "contraction hysteresis"
    second_low = controller.observe(1.0, "nll")
    assert second_low.action == "contract"
    assert (second_low.previous_experts, second_low.active_experts) == (3, 2)


def test_controller_memory_cap_clamps_and_blocks_expansion() -> None:
    cfg = AdaptiveOccupancyConfig(
        enabled=True,
        baseline_nll=2.0,
        surprise_ema_warmup=1,
        expand_patience=1,
    )
    controller = AdaptiveOccupancyController(cfg, num_experts=20)
    assert controller.observe(2.0, "nll").active_experts == 3

    controller.set_memory_maximum(2)
    assert controller.current_experts == 2
    capped = controller.observe(3.0, "nll")
    assert capped.action == "hold"
    assert capped.active_experts == 2
    assert capped.reason == "memory occupancy cap"
    assert controller.telemetry()["memory_caps"] == 1

    with pytest.raises(MemoryError, match="below the required 2-expert minimum"):
        controller.set_memory_maximum(1)


def test_adaptive_execution_packs_only_active_expert_parameters(monkeypatch) -> None:
    torch.manual_seed(7)
    router = SharedTokenRouter(
        hidden_size=6,
        num_experts=4,
        top_k=2,
        adaptive=True,
    )
    layer = MoLELinear(
        nn.Linear(6, 5, bias=False),
        num_experts=4,
        rank=2,
        alpha=4.0,
        top_k=2,
        shared_token_router=router,
        adaptive=True,
    )
    resident = (1, 3)
    router.set_adaptive_expert_pool(resident)
    layer.set_resident_experts(resident)

    calls: list[int] = []
    for index, expert in enumerate(layer.experts):
        original = expert.forward

        def traced(x: torch.Tensor, *, _index=index, _original=original):
            if _index not in resident:
                raise AssertionError(f"inactive expert {_index} executed")
            calls.append(_index)
            return _original(x)

        monkeypatch.setattr(expert, "forward", traced)

    x = torch.randn(2, 3, 6)
    router.prepare_route(x, force=True)
    output = layer(x)
    output.sum().backward()

    assert output.shape == (2, 3, 5)
    # The optimized path bypasses Python expert calls and uses two K*r GEMMs.
    assert calls == []
    for index, expert in enumerate(layer.experts):
        if index in resident:
            assert expert.lora_B.grad is not None
        else:
            assert expert.lora_A.grad is None
            assert expert.lora_B.grad is None
    assert set(router.last_router_indices.flatten().tolist()) == set(resident)
    assert torch.count_nonzero(router.last_router_dense_weights[:, (0, 2)]) == 0


def test_v4_save_is_sharded_and_lazy_loads_only_requested_lane(tmp_path, monkeypatch) -> None:
    model, _cfg = _saved_adaptive_artifact(tmp_path)
    payload = load_mole_config(tmp_path, expected_base_ref="tiny-adaptive")

    assert payload["format_version"] == 4
    assert [entry["index"] for entry in payload["expert_manifest"]] == [0, 1, 2, 3]
    routing_state = load_file(str(tmp_path / WEIGHTS_NAME), device="cpu")
    assert routing_state
    assert not any(".expert_" in key for key in routing_state)
    assert all((tmp_path / entry["file"]).is_file() for entry in payload["expert_manifest"])

    store = LazyExpertStore(tmp_path, expected_base_ref="tiny-adaptive")
    store.validate_model(model)
    assert store.num_experts == 4
    assert store.bytes_per_expert == 16 * torch.tensor([], dtype=torch.float32).element_size()

    import yylab.mole.store as store_module

    real_load = store_module.load_file
    loaded_paths: list[str] = []

    def recording_load(path: str, *args, **kwargs):
        loaded_paths.append(str(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(store_module, "load_file", recording_load)
    lane = store.expert_weights(2)

    assert len(loaded_paths) == 1
    loaded_shard = Path(loaded_paths[0])
    assert loaded_shard.parent.name == EXPERTS_DIR
    assert loaded_shard.name == "expert_002.safetensors"
    assert lane
    assert all(".expert_2." in key for key in lane)
    assert not any(".expert_0." in key or ".expert_1." in key or ".expert_3." in key for key in lane)


def test_v4_missing_shard_is_rejected_at_manifest_load(tmp_path) -> None:
    _saved_adaptive_artifact(tmp_path)
    payload = load_mole_config(tmp_path)
    missing = tmp_path / payload["expert_manifest"][1]["file"]
    missing.unlink()

    with pytest.raises(ValueError, match="expert shard is missing"):
        load_mole_config(tmp_path)


def test_v4_corrupt_shard_fails_only_when_lane_is_materialized(tmp_path) -> None:
    _saved_adaptive_artifact(tmp_path)
    store = LazyExpertStore(tmp_path)
    corrupt = tmp_path / store.manifest[3]["file"]
    corrupt.write_bytes(b"not-a-safetensors-file")

    # Manifest loading is intentionally lazy: existence is checked up front,
    # while the corrupt payload is rejected when that lane is requested.
    assert LazyExpertStore(tmp_path).num_experts == 4
    with pytest.raises(SafetensorError):
        store.expert_weights(3)


def test_v4_export_rejects_before_loading_the_base(tmp_path, monkeypatch) -> None:
    from yylab.export import merge as merge_module

    _saved_adaptive_artifact(tmp_path)
    monkeypatch.setattr(
        merge_module,
        "load_base_full",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("adaptive export should reject before loading the base")
        ),
    )

    with pytest.raises(ValueError, match="adaptive MoLE v4 cannot be losslessly merged"):
        merge_module.merge_mole(
            "tiny-adaptive",
            str(tmp_path),
            tmp_path / "merged",
        )


def test_legacy_v3_save_remains_monolithic(tmp_path) -> None:
    model = _TinyBlockModel()
    cfg = MoLEConfig(
        num_experts=4,
        rank=2,
        alpha=4.0,
        top_k=2,
        target_modules=("q_proj",),
        router_type="per_token",
    )
    inject_mole(model, cfg)
    save_mole(tmp_path, model, cfg, base_ref="tiny-legacy")

    payload = load_mole_config(tmp_path, expected_base_ref="tiny-legacy")
    assert payload["format_version"] == 3
    assert "expert_manifest" not in payload
    assert not (tmp_path / EXPERTS_DIR).exists()
    state = load_file(str(tmp_path / WEIGHTS_NAME), device="cpu")
    assert any(".expert_0." in key for key in state)
    assert (tmp_path / CONFIG_NAME).is_file()


class _WorkerTinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.q_proj(hidden_states)


class _WorkerTinyLM(nn.Module):
    """A 100,000-parameter base whose only adaptive target is 8x8."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny", hidden_size=8, use_cache=False)
        self.embedding = nn.Embedding(32, 8)
        self.layers = nn.ModuleList([_WorkerTinyBlock()])
        self.lm_head = nn.Linear(8, 32, bias=False)
        # 576 useful parameters + this frozen capacity tensor gives the real
        # budget solver an exact, deterministic 100,000-parameter base shape.
        self.frozen_capacity = nn.Parameter(torch.zeros(99_424), requires_grad=False)

    def forward(self, input_ids=None, labels=None, **_kwargs):
        hidden = self.layers[0](self.embedding(input_ids))
        logits = self.lm_head(hidden)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return SimpleNamespace(loss=loss, logits=logits)


class _WorkerFixedBatches:
    docs_seen = 4
    tokens_seen = 16

    def __iter__(self):
        yield {"input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]]}
        yield {"input_ids": [[9, 10, 11, 12], [13, 14, 15, 16]]}


def test_adaptive_worker_resolves_trains_and_persists_v4_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    from yylab.workers import train as worker

    torch.manual_seed(19)
    loaded: dict[str, _WorkerTinyLM] = {}
    adapter_path = tmp_path / "adaptive-adapter"
    database_updates: list[tuple[str, tuple]] = []
    lock_events: list[tuple[str, str, str]] = []
    batch_configs: list[dict] = []

    def fake_load_base(_ref: str, quant: str):
        model = _WorkerTinyLM()
        assert sum(parameter.numel() for parameter in model.parameters()) == 100_000
        loaded["model"] = model
        return model, object(), False, quant, []

    def fake_build_batches(cfg: dict, _tokenizer, max_steps: int, **kwargs):
        batch_configs.append({"cfg": dict(cfg), "max_steps": max_steps, "kwargs": kwargs})
        return _WorkerFixedBatches()

    monkeypatch.setattr(worker, "load_base", fake_load_base)
    monkeypatch.setattr(worker, "build_batches", fake_build_batches)
    monkeypatch.setattr(worker, "adapter_dir", lambda _adapter_id: adapter_path)
    monkeypatch.setattr(
        worker,
        "enforce_quant_lock",
        lambda ckpt, quant: lock_events.append(("check", ckpt, quant)),
    )
    monkeypatch.setattr(
        worker,
        "commit_quant_lock",
        lambda ckpt, quant: lock_events.append(("commit", ckpt, quant)),
    )
    monkeypatch.setattr(
        worker.db,
        "execute",
        lambda sql, args=(): database_updates.append((sql, args)) or 1,
    )
    monkeypatch.setattr(worker, "log", lambda _message: None)
    monkeypatch.setattr(worker, "log_memory", lambda _label: None)
    monkeypatch.setattr(
        worker,
        "plan_expert_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("adaptive joint routing attempted a domain split")
        ),
    )

    cfg = {
        "training_strategy": "joint_adaptive_surprise_v1",
        "checkpoint_id": "checkpoint",
        "checkpoint_ref": "tiny-adaptive-base",
        "adapter_id": "adaptive-adapter",
        "quant": "bf16",
        # These are only valid construction seeds; the real integer solver
        # replaces E/r/K from the actual 100,000-parameter base model.
        "num_experts": 2,
        "rank": 2,
        "alpha": None,
        "top_k": 2,
        "target_modules": ["q_proj"],
        "corpora": ["corpus-a", "corpus-b"],
        "domains": ["code"],
        "expert_plan": [{"not": "a valid domain assignment"}],
        "seq_len": 4,
        "micro_batch": 2,
        "grad_accum": 1,
        "max_steps": 2,
        "lr": 1e-2,
        "aux_weight": 0.05,
        "grad_ckpt": False,
        "liger_fused_ce": False,
        "log_every": 1,
        "save_every": None,
        "adaptive": {
            "enabled": True,
            "target_total_multiplier": 1.25,
            "expert_footprint_fraction": 0.005,
            "min_active_fraction": 0.01,
            "max_active_fraction": 0.10,
            "surprise_ema_warmup": 1,
            "calibration_weight": 1.0,
            "nested_k_probe_every": 100,
            "exploration_experts": 0,
            "preferred_experts": [0, 1],
        },
    }

    run_dir = tmp_path / "run"
    summary = worker.run_mole(cfg, run_dir)

    assert summary["strategy"] == "joint_adaptive_surprise_v1"
    assert summary["steps"] == 2
    assert summary["docs_seen"] == _WorkerFixedBatches.docs_seen
    assert summary["tokens"] == _WorkerFixedBatches.tokens_seen
    assert len(batch_configs) == 1
    derived = batch_configs[0]
    assert derived["max_steps"] == 2
    assert derived["kwargs"] == {}
    assert derived["cfg"]["domains"] == ["code"]
    assert derived["cfg"]["expert_plan"] == cfg["expert_plan"]
    assert lock_events == [
        ("check", "checkpoint", "bf16"),
        ("commit", "checkpoint", "bf16"),
    ]

    saved = load_mole_config(adapter_path, expected_base_ref="tiny-adaptive-base")
    assert saved["format_version"] == 4
    saved_cfg = MoLEConfig.from_dict(saved["config"])
    capacity = saved["meta"]["adaptive_capacity"]
    # This shape resolves deterministically through solve_expert_budget; no
    # worker or solver monkeypatch is involved.
    assert (saved_cfg.num_experts, saved_cfg.rank, saved_cfg.top_k) == (39, 39, 2)
    assert (
        derived["cfg"]["num_experts"],
        derived["cfg"]["rank"],
        derived["cfg"]["top_k"],
    ) == (saved_cfg.num_experts, saved_cfg.rank, saved_cfg.top_k)
    assert capacity["num_experts"] == saved_cfg.num_experts
    assert capacity["rank"] == saved_cfg.rank
    assert capacity["base_parameters"] == 100_000
    assert capacity["parameters_per_expert"] == 624
    assert capacity["router_parameters"] == 351
    assert capacity["resolved_total_parameters"] == (
        capacity["base_parameters"]
        + capacity["router_parameters"]
        + saved_cfg.num_experts * capacity["parameters_per_expert"]
    )
    assert saved_cfg.adaptive.resolved_total_parameter_count == capacity[
        "resolved_total_parameters"
    ]
    assert saved_cfg.adaptive.resolved_expert_footprint_fraction == pytest.approx(
        capacity["expert_footprint_fraction"]
    )
    assert saved_cfg.adaptive.minimum_experts == saved_cfg.top_k == 2

    meta = saved["meta"]
    assert meta["training_strategy"] == "joint_adaptive_surprise_v1"
    assert meta["router_arch"] == "global_surprise_lanes_v1"
    assert meta["global_domain_filter"] == ["code"]
    assert meta["occupancy_semantics"] == (
        "incremental_expert_parameters_over_resolved_total"
    )
    assert meta["training_residency"] == (
        "full_bank_weights_resident_selected_lanes_execute"
    )
    assert meta["adaptive_controller"]["routing_mode"] == "adaptive_surprise"
    assert meta["adaptive_controller"]["signal_kind"] == "nll"
    assert meta["adaptive_controller"]["surprise"] is not None
    assert len(meta["preferred_experts"]) == saved_cfg.num_experts
    assert saved_cfg.adaptive.score_semantics == "predicted_nll"

    manifest = saved["expert_manifest"]
    assert len(manifest) == saved_cfg.num_experts
    assert [entry["index"] for entry in manifest] == list(range(saved_cfg.num_experts))
    for entry in manifest:
        shard = adapter_path / entry["file"]
        assert shard.is_file()
        assert entry["parameters"] == capacity["parameters_per_expert"]
        assert entry["bytes"] == capacity["parameters_per_expert"] * 4
        assert len(entry["sha256"]) == 64
        assert hashlib.sha256(shard.read_bytes()).hexdigest() == entry["sha256"]

    routing_tensors = load_file(str(adapter_path / WEIGHTS_NAME), device="cpu")
    assert not any(".expert_" in key for key in routing_tensors)
    gate_key = next(key for key in routing_tensors if key.endswith("gate.weight"))
    surprise_key = next(key for key in routing_tensors if key.endswith("surprise_bias"))
    assert routing_tensors[gate_key].shape == (saved_cfg.num_experts, 8)
    assert routing_tensors[surprise_key].shape == (saved_cfg.num_experts,)
    assert torch.isfinite(routing_tensors[gate_key]).all()
    assert torch.isfinite(routing_tensors[surprise_key]).all()
    assert not torch.allclose(
        routing_tensors[surprise_key],
        torch.full_like(routing_tensors[surprise_key], 1.8545866),
    )

    # LoRA-B starts at exactly zero, so any nonzero saved lane is direct
    # evidence that a selected expert executed and learned.
    learned_lanes = []
    for entry in manifest:
        lane = load_file(str(adapter_path / entry["file"]), device="cpu")
        if any(
            tensor.abs().sum().item() > 0
            for key, tensor in lane.items()
            if key.endswith("lora_B")
        ):
            learned_lanes.append(entry["index"])
    assert learned_lanes

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(metrics) == 2
    assert all("predicted_nll" in record for record in metrics)
    assert all(
        record["adaptive_controller"]["routing_mode"] == "adaptive_surprise"
        for record in metrics
    )
    assert all(
        saved_cfg.adaptive.minimum_experts
        <= record["training_active_experts"]
        <= saved_cfg.adaptive.maximum_experts(saved_cfg.num_experts)
        for record in metrics
    )

    assert len(database_updates) == 1
    sql, args = database_updates[0]
    assert "num_experts=?, top_k=?, rank=?" in sql
    assert json.loads(args[1]) == ["code"]
    db_meta = json.loads(args[2])
    assert db_meta["adaptive_capacity"] == capacity
    assert args[3:6] == (saved_cfg.num_experts, saved_cfg.top_k, saved_cfg.rank)
    assert args[6] == "adaptive-adapter"

    trainable = {
        name for name, parameter in loaded["model"].named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all("experts" in name or "router" in name for name in trainable)
