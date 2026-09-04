from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from yylab.mole.inject import (
    MoLEConfig,
    inject_mole,
    mole_layers,
    set_stage,
    shared_router,
    token_routers,
)


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.q_proj(hidden_states)


class _TinyModel(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width)
        self.embedding = nn.Embedding(32, width)
        self.layers = nn.ModuleList([_Block(width)])

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.layers[0](self.embedding(input_ids))


@pytest.mark.parametrize("router_type", ["per_layer", "per_token", "shared_prompt"])
def test_fp16_compute_contract_covers_experts_and_every_router(router_type: str) -> None:
    model = _TinyModel().to(dtype=torch.float16)
    config = MoLEConfig(
        num_experts=3,
        rank=2,
        alpha=4.0,
        top_k=2,
        target_modules=("q_proj",),
        router_type=router_type,
    )

    inject_mole(model, config, expert_dtype=torch.float16)
    set_stage(model, "both")

    for _, layer in mole_layers(model):
        assert {parameter.dtype for expert in layer.experts for parameter in expert.parameters()} == {
            torch.float16
        }
        if layer.router is not None:
            assert {parameter.dtype for parameter in layer.router.parameters()} == {torch.float16}
    for _, router in token_routers(model):
        assert {parameter.dtype for parameter in router.parameters()} == {torch.float16}
    prompt_router = shared_router(model)
    if prompt_router is not None:
        assert {parameter.dtype for parameter in prompt_router.parameters()} == {torch.float16}

    output = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert output.dtype is torch.float16
    output.float().square().mean().backward()
    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert trainable_gradients
    assert all(gradient is not None for gradient in trainable_gradients)
    assert {gradient.dtype for gradient in trainable_gradients if gradient is not None} == {
        torch.float16
    }


def test_dense_base_rejects_a_different_adapter_compute_dtype() -> None:
    with pytest.raises(ValueError, match="does not match the dense base projection dtype"):
        inject_mole(
            _TinyModel(),
            MoLEConfig(target_modules=("q_proj",)),
            expert_dtype=torch.float16,
        )


def test_quantized_base_requires_an_explicit_verified_compute_dtype() -> None:
    model = _TinyModel()
    projection = model.layers[0].q_proj
    projection.weight = nn.Parameter(
        projection.weight.detach().to(dtype=torch.int8),
        requires_grad=False,
    )

    with pytest.raises(ValueError, match="quantized bases must pass.*explicitly"):
        inject_mole(model, MoLEConfig(target_modules=("q_proj",)))


def test_joint_worker_passes_int8_fp16_compute_to_mole(tmp_path, monkeypatch) -> None:
    from yylab.workers import train as worker

    observed: list[torch.dtype] = []

    class InjectionObserved(RuntimeError):
        pass

    monkeypatch.setattr(
        worker,
        "load_base",
        lambda *_args, **_kwargs: (
            SimpleNamespace(),
            object(),
            False,
            "int8",
            [],
        ),
    )
    monkeypatch.setattr(worker, "prepare_kbit_training", lambda *_args: None)
    monkeypatch.setattr(worker, "enforce_quant_lock", lambda *_args: None)
    monkeypatch.setattr(worker, "log_memory", lambda *_args: None)

    def capture_injection(_model, _config, *, expert_dtype, **_kwargs):
        observed.append(expert_dtype)
        raise InjectionObserved

    monkeypatch.setattr(worker, "inject_mole", capture_injection)

    with pytest.raises(InjectionObserved):
        worker.run_mole(
            {
                "training_strategy": "joint_per_token_v1",
                "checkpoint_id": "checkpoint",
                "checkpoint_ref": "unused",
                "adapter_id": "adapter",
                "quant": "int8",
                "num_experts": 2,
                "rank": 2,
                "top_k": 1,
                "target_modules": ["q_proj"],
            },
            tmp_path,
        )

    assert observed == [torch.float16]


def test_inference_passes_int8_fp16_compute_to_mole(monkeypatch) -> None:
    from yylab.infer.engine import Engine, LoadSpec
    from yylab.models import loading
    from yylab.mole import inject as injection
    from yylab.mole import store

    observed: list[torch.dtype] = []

    class InjectionObserved(RuntimeError):
        pass

    model = SimpleNamespace(config=SimpleNamespace(use_cache=False))
    monkeypatch.setattr(
        loading,
        "load_base",
        lambda *_args, **_kwargs: (model, object(), True, "int8", []),
    )
    config = MoLEConfig(
        num_experts=2,
        rank=2,
        top_k=1,
        target_modules=("q_proj",),
    )
    monkeypatch.setattr(
        store,
        "load_mole_config",
        lambda *_args, **_kwargs: {"config": config.to_dict(), "layers": []},
    )

    def capture_injection(_model, _config, *, expert_dtype, **_kwargs):
        observed.append(expert_dtype)
        raise InjectionObserved

    monkeypatch.setattr(injection, "inject_mole", capture_injection)

    with pytest.raises(InjectionObserved):
        Engine().load(
            LoadSpec(
                ref="unused",
                quant="int8",
                model_family="transformers",
                adapter_path="unused-adapter",
                adapter_arch="mole",
            )
        )

    assert observed == [torch.float16]


def test_performance_benchmark_passes_int8_fp16_compute_to_mole(
    tmp_path, monkeypatch
) -> None:
    from yylab.benchmark import performance
    from yylab.models import loading
    from yylab.mole import inject as injection

    observed: list[torch.dtype] = []

    class InjectionObserved(RuntimeError):
        pass

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    monkeypatch.setattr(
        loading,
        "load_base",
        lambda *_args, **_kwargs: (model, object(), False, "int8", []),
    )
    monkeypatch.setattr(performance, "_prepare_kbit", lambda *_args: None)
    monkeypatch.setattr(
        performance,
        "_gpu_snapshot",
        lambda: {"name": "fixture", "memory_used_gb": 0.0, "memory_total_gb": 1.0},
    )

    def capture_injection(_model, _config, *, expert_dtype, **_kwargs):
        observed.append(expert_dtype)
        raise InjectionObserved

    monkeypatch.setattr(injection, "inject_mole", capture_injection)

    with pytest.raises(InjectionObserved):
        performance.run_training_performance(
            {
                "checkpoint_ref": str(checkpoint),
                "quant": "int8",
                "target_modules": ["q_proj"],
                "require_idle_gpu": False,
                "warmup_steps": 0,
                "measured_steps": 1,
            },
            tmp_path / "output",
        )

    assert observed == [torch.float16]
