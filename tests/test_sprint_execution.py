"""Execution-policy checks that do not allocate CUDA tensors or load large models."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from test_sprint_training import tiny_model
from test_sprint_training import training_fixture as training_fixture
from torch.nn.attention import SDPBackend, sdpa_kernel

from yylab.models.sprint import SprintForCausalLM
from yylab.train import sprint


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class AllocationSentinel:
    def named_parameters(self):
        pytest.fail("invalid execution settings reached model/master allocation")

    def to(self, **kwargs):
        pytest.fail("invalid execution settings attempted a model device/dtype conversion")


def _sdpa_flags():
    cuda = torch.backends.cuda
    return {
        "math": cuda.math_sdp_enabled(), "flash": cuda.flash_sdp_enabled(),
        "efficient": cuda.mem_efficient_sdp_enabled(), "cudnn": cuda.cudnn_sdp_enabled(),
    }


def _engine(backend="math"):
    return sprint.SprintStepEngine(tiny_model(), {
        "attention_backend": backend, "cpu_threads": 1,
        "parameter_dtype": "float32", "optimizer_state_offload": "none", "grad_clip": 0,
    }, "cpu")


def _batch():
    return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])}


@pytest.mark.parametrize("config,match", [
    ({"cpu_threads": 0}, "cpu_threads"),
    ({"cpu_threads": -1}, "cpu_threads"),
    ({"cpu_threads": 1.5}, "cpu_threads"),
    ({"cpu_threads": True}, "cpu_threads"),
    ({"attention_backend": "unsupported"}, "attention_backend"),
    ({"attention_backend": "cudnn"}, "requires CUDA"),
    ({"attention_backend": "efficient"}, "requires CUDA"),
    ({"attention_backend": "flash"}, "requires CUDA"),
    ({"cpu_optimizer_kernel": "unknown", "optimizer_state_offload": "none"}, "cpu_optimizer_kernel"),
    ({"cpu_optimizer_kernel": "compiled", "optimizer_state_offload": "none"}, "requires BF16"),
    ({"cpu_optimizer_kernel": "compiled", "parameter_dtype": "float32"}, "requires BF16"),
])
def test_engine_rejects_invalid_execution_before_model_allocation(config, match):
    with pytest.raises(ValueError, match=match):
        sprint.SprintStepEngine(AllocationSentinel(), config, "cpu")


@pytest.mark.parametrize("update", [
    {"cpu_threads": 0}, {"cpu_threads": 1.25}, {"cpu_threads": True},
    {"attention_backend": "unsupported"},
])
def test_worker_normalizes_execution_settings_before_loading_model(training_fixture, tmp_path, monkeypatch, update):
    cfg, _, _, _ = training_fixture
    monkeypatch.setattr(sprint, "_load_training_inputs", lambda *args: pytest.fail("invalid recipe loaded a model"))
    with pytest.raises(ValueError):
        sprint.run_sprint_training({**cfg, **update}, tmp_path / "invalid-run")


def test_cpu_thread_selection_and_normalized_attention_are_recorded(training_fixture):
    cfg, _, _, _ = training_fixture
    normalized = sprint.normalize_sprint_config({**cfg, "cpu_threads": 2, "attention_backend": "math"})
    assert normalized["cpu_threads"] == 2
    assert normalized["attention_backend"] == "math"
    engine = sprint.SprintStepEngine(tiny_model(), normalized, "cpu")
    assert engine.cpu_threads == torch.get_num_threads() == 2
    assert engine.attention_backend == "math"


def test_math_backend_is_scoped_through_backward_and_evaluation(monkeypatch):
    engine = _engine()
    forward = engine.model.forward
    events = []
    expected = {"math": True, "flash": False, "efficient": False, "cudnn": False}

    def checked_forward(*args, **kwargs):
        assert _sdpa_flags() == expected
        events.append("forward")
        return forward(*args, **kwargs)

    def checked_backward(gradient):
        assert _sdpa_flags() == expected
        events.append("backward")
        return gradient

    monkeypatch.setattr(engine.model, "forward", checked_forward)
    handle = next(engine.model.parameters()).register_hook(checked_backward)
    try:
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]):
            baseline = _sdpa_flags()
            result = engine.train_step([_batch()])
            assert torch.isfinite(result.loss)
            assert _sdpa_flags() == baseline
            evaluation = engine.evaluate([_batch()])
            assert evaluation["eval_supervised_tokens"] == 7
            assert _sdpa_flags() == baseline
            assert engine.model.training
    finally:
        handle.remove()
    assert events == ["forward", "backward", "forward"]


@pytest.mark.parametrize("failure", ["train_forward", "train_backward", "evaluation"])
def test_attention_defaults_restore_after_model_or_backward_failure(monkeypatch, failure):
    engine = _engine()
    expected = {"math": True, "flash": False, "efficient": False, "cudnn": False}

    def fail(*args, **kwargs):
        assert _sdpa_flags() == expected
        raise RuntimeError("deliberate execution failure")

    handle = None
    if failure == "train_backward":
        handle = next(engine.model.parameters()).register_hook(fail)
    else:
        monkeypatch.setattr(engine.model, "forward", fail)
    try:
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]):
            baseline = _sdpa_flags()
            with pytest.raises(RuntimeError, match="deliberate execution failure"):
                if failure == "evaluation":
                    engine.evaluate([_batch()])
                else:
                    engine.train_step([_batch()])
            assert _sdpa_flags() == baseline
            assert engine.model.training
            assert not engine.optimizer.state
    finally:
        if handle is not None:
            handle.remove()


def test_auto_backend_preserves_callers_selection_during_forward(monkeypatch):
    engine = _engine("auto")
    forward = engine.model.forward
    with sdpa_kernel(SDPBackend.MATH):
        baseline = _sdpa_flags()

        def checked_forward(*args, **kwargs):
            assert _sdpa_flags() == baseline
            return forward(*args, **kwargs)

        monkeypatch.setattr(engine.model, "forward", checked_forward)
        engine.train_step([_batch()])
        engine.evaluate([_batch()])
        assert _sdpa_flags() == baseline


def test_resume_rejects_backend_changes_and_migrates_legacy_auto_contract(training_fixture, tmp_path):
    cfg, _, _, _ = training_fixture
    cfg = {**cfg, "max_steps": 2, "save_interval": 1, "eval_interval": 0,
           "cpu_threads": 1, "attention_backend": "auto"}
    full = sprint.run_sprint_training(cfg, tmp_path / "source")
    checkpoint = tmp_path / "source" / "checkpoints" / "step-00000001"
    with pytest.raises(ValueError, match="resume configuration or data differs"):
        sprint.run_sprint_training({"resume_from": str(checkpoint), "attention_backend": "math"},
                                   tmp_path / "changed-backend")
    with pytest.raises(ValueError, match="resume configuration or data differs"):
        sprint.run_sprint_training({"resume_from": str(checkpoint), "offload_bucket_mb": 16},
                                   tmp_path / "changed-buckets")
    with pytest.raises(ValueError, match="resume configuration or data differs"):
        sprint.run_sprint_training({"resume_from": str(checkpoint), "cpu_threads": 2},
                                   tmp_path / "changed-threads")
    torch.set_num_threads(1)

    # Older Sprint runs did not record either execution field. Only the missing
    # attention field needs identity migration; CPU pool sizing is operational.
    state_path = checkpoint / "trainer.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["contract"]["training"].pop("attention_backend")
    state["config"].pop("attention_backend")
    state["config"].pop("cpu_threads")
    state["contract"]["training"].pop("cpu_optimizer_kernel")
    state["contract"]["training"].pop("offload_bucket_mb")
    state["contract"]["training"].pop("cpu_threads")
    state["config"].pop("cpu_optimizer_kernel")
    torch.save(state, state_path)
    resumed = sprint.run_sprint_training({"resume_from": str(checkpoint)}, tmp_path / "legacy-resume")
    assert resumed["runtime"]["attention_backend"] == "auto"
    assert resumed["runtime"]["cpu_threads"] == 1
    assert resumed["runtime"]["cpu_optimizer_kernel"] == "native"
    assert resumed["new_steps"] == 1
    assert resumed["tokens"] == full["tokens"]
    expected = SprintForCausalLM.from_pretrained(Path(full["checkpoint_path"])).state_dict()
    actual = SprintForCausalLM.from_pretrained(Path(resumed["checkpoint_path"])).state_dict()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="resume configuration or data differs"):
        sprint.run_sprint_training({"resume_from": str(checkpoint), "attention_backend": "math"},
                                   tmp_path / "changed-legacy-backend")
