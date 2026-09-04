from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden_states):
        return self.q_proj(hidden_states)


class TinyTokenLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny", hidden_size=8, use_cache=False)
        self.embedding = nn.Embedding(32, 8)
        self.layers = nn.ModuleList([TinyBlock()])
        self.lm_head = nn.Linear(8, 32, bias=False)

    def forward(self, input_ids=None, labels=None, **_kwargs):
        hidden = self.layers[0](self.embedding(input_ids))
        logits = self.lm_head(hidden)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return SimpleNamespace(loss=loss, logits=logits)


class FixedBatches:
    docs_seen = 2
    tokens_seen = 16

    def __iter__(self):
        yield {"input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]]}
        yield {"input_ids": [[9, 10, 11, 12], [13, 14, 15, 16]]}


def test_joint_per_token_worker_trains_and_saves_complete_bank(tmp_path, monkeypatch):
    from yylab.mole.inject import MoLEConfig, inject_mole
    from yylab.mole.store import apply_mole_weights, load_mole_config
    from yylab.workers import train as worker

    torch.manual_seed(7)
    loaded: dict[str, TinyTokenLM] = {}
    adapter_path = tmp_path / "adapter"
    database_updates: list[tuple[str, tuple]] = []
    lock_events: list[tuple[str, str, str]] = []
    batch_configs: list[dict] = []

    def fake_load_base(_ref: str, quant: str):
        model = TinyTokenLM()
        loaded["model"] = model
        return model, object(), False, quant, []

    def fake_build_batches(cfg: dict, _tokenizer, max_steps: int, **kwargs):
        batch_configs.append({"cfg": dict(cfg), "max_steps": max_steps, "kwargs": kwargs})
        return FixedBatches()

    monkeypatch.setattr(worker, "load_base", fake_load_base)
    monkeypatch.setattr(worker, "build_batches", fake_build_batches)
    monkeypatch.setattr(worker, "adapter_dir", lambda _adapter_id: adapter_path)
    monkeypatch.setattr(worker, "enforce_quant_lock", lambda ckpt, quant: lock_events.append(("check", ckpt, quant)))
    monkeypatch.setattr(worker, "commit_quant_lock", lambda ckpt, quant: lock_events.append(("commit", ckpt, quant)))
    monkeypatch.setattr(worker.db, "execute", lambda sql, args=(): database_updates.append((sql, args)) or 1)
    monkeypatch.setattr(worker, "log", lambda _message: None)
    monkeypatch.setattr(worker, "log_memory", lambda _label: None)
    # If the joint path consults the old hand-authored split, fail loudly.
    monkeypatch.setattr(
        worker,
        "plan_expert_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("expert split was used")),
    )

    cfg = {
        "training_strategy": "joint_per_token_v1",
        "checkpoint_id": "checkpoint",
        "checkpoint_ref": "tiny-base",
        "adapter_id": "adapter",
        "quant": "fp32",
        "num_experts": 3,
        "rank": 2,
        "alpha": 4.0,
        "top_k": 2,
        "target_modules": ["q_proj"],
        "corpora": ["corpus-a", "corpus-b"],
        "domains": ["code"],
        # This must be ignored by joint learned routing.
        "expert_plan": [{"not": "a valid expert plan"}],
        "seq_len": 4,
        "micro_batch": 2,
        "grad_accum": 2,
        "max_steps": 1,
        "lr": 1e-2,
        "aux_weight": 0.05,
        "grad_ckpt": False,
        "liger_fused_ce": False,
        "log_every": 1,
        "save_every": None,
    }

    summary = worker.run_mole(cfg, tmp_path / "run")

    assert summary["strategy"] == "joint_per_token_v1"
    assert summary["steps"] == 1
    assert summary["docs_seen"] == FixedBatches.docs_seen
    assert batch_configs == [{"cfg": cfg, "max_steps": 1, "kwargs": {}}]
    assert lock_events == [("check", "checkpoint", "fp32"), ("commit", "checkpoint", "fp32")]

    saved = load_mole_config(adapter_path, expected_base_ref="tiny-base")
    assert saved["config"]["router_type"] == "per_token"
    assert saved["config"]["num_experts"] == 3
    assert saved["meta"]["training_strategy"] == "joint_per_token_v1"
    assert saved["meta"]["global_domain_filter"] == ["code"]
    assert "expert_plan" not in saved["meta"]

    tensors = load_file(str(adapter_path / "mole_weights.safetensors"))
    expert_keys = [key for key in tensors if ".expert_" in key]
    assert len(expert_keys) == 6  # A and B for all three experts in the injected projection.
    assert any(key.endswith("router.gate.weight") for key in tensors)
    assert any(tensors[key].abs().sum().item() > 0 for key in expert_keys if key.endswith("lora_B"))
    gate = next(value for key, value in tensors.items() if key.endswith("router.gate.weight"))
    assert gate.abs().sum().item() > 0
    fresh = TinyTokenLM()
    inject_mole(fresh, MoLEConfig.from_dict(saved["config"]))
    assert apply_mole_weights(fresh, tensors) == 3

    assert len(database_updates) == 1
    sql, args = database_updates[0]
    assert "UPDATE adapters SET status='trained'" in sql
    assert json.loads(args[1]) == ["code"]
    assert json.loads(args[2])["router_arch"] == "shared_per_block_per_token_v1"

    trainable = {name for name, value in loaded["model"].named_parameters() if value.requires_grad}
    assert trainable
    assert all("experts" in name or "router" in name for name in trainable)


def test_mole_worker_strategy_validation_preserves_legacy_default():
    from yylab.workers.train import (
        SEQUENTIAL_EXPERT_STRATEGY,
        _mole_training_strategy,
    )

    assert _mole_training_strategy({}) == SEQUENTIAL_EXPERT_STRATEGY
    try:
        _mole_training_strategy({"training_strategy": "mystery"})
        raise AssertionError("unknown strategies must fail before loading a model")
    except ValueError as exc:
        assert "unknown MoLE training strategy" in str(exc)


def test_soft_lora_worker_strategy_requires_identity_contract(tmp_path):
    from yylab.workers.train import (
        SOFT_LORA_IDENTITY_STRATEGY,
        _mole_training_strategy,
        run_mole,
    )

    assert (
        _mole_training_strategy(
            {"training_strategy": SOFT_LORA_IDENTITY_STRATEGY, "identity_expert": True}
        )
        == SOFT_LORA_IDENTITY_STRATEGY
    )
    try:
        run_mole({"training_strategy": SOFT_LORA_IDENTITY_STRATEGY}, tmp_path)
        raise AssertionError("Soft LoRA must not silently omit its identity route")
    except ValueError as exc:
        assert "identity_expert" in str(exc)
