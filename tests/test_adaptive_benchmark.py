from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yylab.benchmark.backend import (
    YyLabBenchmarkBackend,
    _adaptive_generation_phase,
)


class _RecordingRuntime:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.prepared_inputs: list[torch.Tensor] = []
        self.kv_reserves: list[int] = []
        self.decoding = False
        self.phase = 0
        self.lane_loads = 0
        self.h2d_transfers = 0
        self.h2d_bytes = 0

    def prepare_request(
        self,
        input_ids,
        attention_mask,
        *,
        kv_reserve_bytes: int = 0,
    ) -> dict[str, object]:
        assert not self.decoding
        self.events.append("prepare")
        self.phase += 1
        self.prepared_inputs.append(input_ids.detach().clone())
        self.kv_reserves.append(int(kv_reserve_bytes))
        # Residency transfers are allowed only in this pre-decode step.
        self.lane_loads += 1
        self.h2d_transfers += 1
        self.h2d_bytes += 16
        return self.telemetry()

    def logits_processor(self):
        runtime = self

        class _Processor:
            def __call__(self, input_ids, scores):
                assert runtime.decoding
                runtime.events.append("entropy")
                return scores

        return _Processor()

    def snapshot(self):
        return SimpleNamespace(
            lane_loads=self.lane_loads,
            h2d_transfers=self.h2d_transfers,
            h2d_bytes=self.h2d_bytes,
        )

    def begin_decode(self) -> None:
        assert not self.decoding
        assert self.events and self.events[-1] == "prepare"
        self.decoding = True
        self.events.append("begin")

    def end_decode(self) -> None:
        assert self.decoding
        self.events.append("end")
        self.decoding = False

    def telemetry(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "active_expert_count": 2,
            "resident_expert_count": 4,
            "h2d_transfers": self.h2d_transfers,
        }


class _OneTokenModel:
    device = torch.device("cpu")
    config = SimpleNamespace(max_position_embeddings=128)

    def __init__(self, *, runtime: _RecordingRuntime | None = None) -> None:
        self.runtime = runtime
        self.calls: list[dict[str, object]] = []

    def modules(self):
        return ()

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        processors = kwargs.get("logits_processor", ())
        scores = torch.zeros((1, 8))
        for processor in processors:
            scores = processor(kwargs["input_ids"], scores)
        return torch.cat((kwargs["input_ids"], torch.tensor([[7, 2]])), dim=1)


class _SimpleTokenizer:
    eos_token_id = 2
    model_max_length = 128

    def apply_chat_template(self, _messages, **_kwargs):
        return "prompt"

    def __call__(self, text, *, return_tensors, **kwargs):
        assert text == "prompt"
        assert return_tensors == "pt"
        assert not kwargs
        return {
            "input_ids": torch.tensor([[10, 11]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "answer" if 7 in [int(value) for value in token_ids] else ""


def _backend_with_runtime(policy: str) -> tuple[YyLabBenchmarkBackend, _RecordingRuntime, _OneTokenModel]:
    runtime = _RecordingRuntime()
    model = _OneTokenModel(runtime=runtime)
    engine = SimpleNamespace(
        tokenizer=_SimpleTokenizer(),
        model=model,
        cuda=False,
        adaptive_runtime=runtime,
        is_loaded=lambda: True,
    )
    backend = YyLabBenchmarkBackend(
        checkpoint_ref="fixture",
        quant="bf16",
        adapter_path="adapter",
        adapter_arch="mole",
    )
    backend.engine = engine
    backend._policy = policy
    return backend, runtime, model


@pytest.mark.parametrize("policy", ["dynamic", "adapter"])
def test_adaptive_policies_prepare_decode_and_emit_routing_receipt(policy: str) -> None:
    backend, runtime, model = _backend_with_runtime(policy)

    response = backend.generate(
        messages=[{"role": "user", "content": "question"}],
        max_new_tokens=2,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        enable_thinking=False,
        cache_mode="dynamic",
        seed=123,
    )

    assert runtime.events == ["prepare", "begin", "entropy", "end"]
    assert len(model.calls[0]["logits_processor"]) == 1
    assert response.routing is not None
    assert response.routing["adaptive"]["active_expert_count"] == 2
    assert response.routing["adaptive"]["generation_phases"] == [
        {
            "phase": 1,
            "active_expert_count": 2,
            "resident_expert_count": 4,
            "h2d_transfers": 1,
        }
    ]


def test_generation_phase_does_not_mutate_kwargs_or_legacy_path() -> None:
    runtime = _RecordingRuntime()
    model = _OneTokenModel(runtime=runtime)
    kwargs: dict[str, object] = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }

    _adaptive_generation_phase(model, kwargs, runtime)

    assert "logits_processor" not in kwargs
    legacy_model = _OneTokenModel()
    _generated, telemetry = _adaptive_generation_phase(legacy_model, kwargs, None)
    assert telemetry is None
    assert "logits_processor" not in legacy_model.calls[0]


def test_generation_phase_rejects_expert_transfer_during_decode() -> None:
    runtime = _RecordingRuntime()

    class _TransferringModel(_OneTokenModel):
        def generate(self, **kwargs):
            assert runtime.decoding
            runtime.h2d_transfers += 1
            return super().generate(**kwargs)

    model = _TransferringModel(runtime=runtime)

    with pytest.raises(RuntimeError, match="transferred expert weights during decode"):
        _adaptive_generation_phase(
            model,
            {"input_ids": torch.tensor([[1, 2]])},
            runtime,
        )

    assert not runtime.decoding


def test_adaptive_thinking_continuation_prepares_each_generation_phase() -> None:
    open_id = 100
    close_id = 101

    class _ThinkingTokenizer(_SimpleTokenizer):
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {
                "<think>": [open_id],
                "<think>\n": [open_id, 90],
                "</think>": [close_id],
            }[text]

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            pieces = {41: "reason-a", 42: "reason-b", 7: "answer", 2: ""}
            return " ".join(pieces[int(value)] for value in token_ids).strip()

    runtime = _RecordingRuntime()

    class _ThinkingModel(_OneTokenModel):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            scores = torch.zeros((1, 8))
            for processor in kwargs.get("logits_processor", ()):
                scores = processor(kwargs["input_ids"], scores)
            suffix = [41, 42] if len(self.calls) == 1 else [7, 2]
            return torch.cat((kwargs["input_ids"], torch.tensor([suffix])), dim=1)

    model = _ThinkingModel(runtime=runtime)
    backend = YyLabBenchmarkBackend(
        checkpoint_ref="fixture",
        quant="bf16",
        adapter_path="adapter",
        adapter_arch="mole",
    )
    backend.engine = SimpleNamespace(
        tokenizer=_ThinkingTokenizer(),
        model=model,
        cuda=False,
        adaptive_runtime=runtime,
        is_loaded=lambda: True,
    )
    backend._policy = "dynamic"

    response = backend.generate(
        messages=[{"role": "user", "content": "question"}],
        max_new_tokens=2,
        max_thinking_tokens=2,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        enable_thinking=True,
        cache_mode="dynamic",
        seed=123,
    )

    assert runtime.events == [
        "prepare",
        "begin",
        "entropy",
        "end",
        "prepare",
        "begin",
        "entropy",
        "end",
    ]
    assert [int(ids.shape[1]) for ids in runtime.prepared_inputs] == [4, 2]
    assert response.routing is not None
    phases = response.routing["adaptive"]["generation_phases"]
    assert [phase["phase"] for phase in phases] == [1, 2]
    assert response.text.endswith("answer")


def test_forced_expert_policy_rejects_adaptive_v4_before_touching_meta_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = YyLabBenchmarkBackend(
        checkpoint_ref="fixture",
        quant="bf16",
        adapter_path="adapter",
        adapter_arch="mole",
    )
    backend.engine = SimpleNamespace(model=object(), adaptive_runtime=object())
    backend._stack = "adapter"
    touched = False

    def _unexpected_set_active_expert(_model, _expert_index):
        nonlocal touched
        touched = True
        raise AssertionError("adaptive meta-backed lanes must not be accessed")

    monkeypatch.setattr("yylab.mole.inject.set_active_expert", _unexpected_set_active_expert)

    with pytest.raises(ValueError, match="unavailable for adaptive MoLE v4"):
        backend.configure_policy("expert:1")

    assert not touched
