from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from yylab.models.spalmer import (
    SpalmerCausalLMFacade,
    SpalmerTokenizerFacade,
    is_spalmer_checkpoint,
    load_spalmer_runtime,
    resolve_spalmer_checkpoint,
    scan_spalmer_checkpoint,
)


def _checkpoint_payload() -> dict:
    return {
        "format": "spalmer.prototype.checkpoint",
        "format_version": 5,
        "model_parameter_dtype": "bfloat16",
        "model_config": {
            "vocab_size": 32,
            "d_model": 8,
            "n_layers": 4,
            "tokenizer_version": 1,
            "tokenizer_fingerprint": "fixture",
        },
        "kda_config": {"hidden_size": 8, "num_heads": 2},
        "mla_config": {"hidden_size": 8, "num_heads": 2},
        "experts_config": {
            "num_experts": 200,
            "active_experts": 2,
            "expert_weight_format": "mxfp4",
            "expert_activation_format": "mxfp8",
        },
        "directional_config": None,
        "atxy_config": None,
        "tokenizer_binding": {"kind": "spalmer_vocab", "vocab_size": 32},
        "model_state": {
            "first.weight": torch.zeros(3, 4, dtype=torch.bfloat16),
            "second.weight": torch.zeros(2, 4, dtype=torch.bfloat16),
        },
        "metadata": {"sequence_length": 2048, "kind": "code"},
    }


def _write_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(), path)
    return path


def test_spalmer_checkpoint_resolves_direct_file_or_canonical_directory(tmp_path: Path) -> None:
    direct = _write_checkpoint(tmp_path / "prototype.pt")
    canonical = _write_checkpoint(tmp_path / "run" / "spalmer.pt")

    assert resolve_spalmer_checkpoint(direct) == direct.resolve()
    assert resolve_spalmer_checkpoint(canonical.parent) == canonical.resolve()

    with pytest.raises(FileNotFoundError, match="SPALMER checkpoint not found"):
        resolve_spalmer_checkpoint(tmp_path / "missing")
    wrong_suffix = tmp_path / "weights.bin"
    wrong_suffix.write_bytes(b"fixture")
    with pytest.raises(ValueError, match=r"\.pt bundle"):
        resolve_spalmer_checkpoint(wrong_suffix)


def test_spalmer_checkpoint_scan_is_model_free_and_returns_registry_fields(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "spalmer.pt")

    result = scan_spalmer_checkpoint(checkpoint)

    assert result["family"] == result["model_family"] == "spalmer"
    assert result["arch"] == "SPALMERCausalLM"
    assert result["parameter_count"] == 20
    assert result["params_b"] == 0.00000002
    assert result["hidden_size"] == 8
    assert result["num_layers"] == 4
    assert result["trained_context_length"] == 2048
    assert result["base_cfg"]["model_type"] == "spalmer"
    assert result["base_cfg"]["max_position_embeddings"] == 2048
    assert result["metadata"] == {"sequence_length": 2048, "kind": "code"}
    assert result["experts_config"]["expert_weight_format"] == "mxfp4"
    assert result["checkpoint_ref"] == str(checkpoint.resolve())
    assert result["path"] == str(checkpoint.resolve().parent)
    assert is_spalmer_checkpoint(checkpoint)


def test_spalmer_scan_reads_nested_training_context_metadata(tmp_path: Path) -> None:
    payload = _checkpoint_payload()
    payload["metadata"] = {"training_config": {"sequence_length": 1536}}
    checkpoint = tmp_path / "spalmer.pt"
    torch.save(payload, checkpoint)

    result = scan_spalmer_checkpoint(checkpoint)

    assert result["trained_context_length"] == 1536
    assert result["base_cfg"]["max_position_embeddings"] == 1536


def test_spalmer_scan_prefers_planned_parameters_over_state_buffers(tmp_path: Path) -> None:
    payload = _checkpoint_payload()
    payload["metadata"] = {
        "plan": {"parameters": {"total": 19}},
        "training_config": {"sequence_length": 512},
    }
    checkpoint = tmp_path / "spalmer.pt"
    torch.save(payload, checkpoint)

    result = scan_spalmer_checkpoint(checkpoint)

    assert result["parameter_count"] == 19
    assert result["params_b"] == 19 / 1_000_000_000
    assert result["state_tensor_count"] == 20
    assert result["parameter_count_source"] == "metadata.plan.parameters.total"


def test_spalmer_detection_rejects_an_unrelated_pt_payload(tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated.pt"
    torch.save({"model_state": {"weight": torch.ones(1)}}, unrelated)

    assert not is_spalmer_checkpoint(unrelated)
    with pytest.raises(ValueError, match="not a SPALMER checkpoint"):
        scan_spalmer_checkpoint(unrelated)


class _GenericTokenizer:
    vocab_size = 260
    special_tokens = SimpleNamespace(bos_id=256, eos_id=257, eod_id=258, pad_id=259)

    def __init__(self) -> None:
        self.decode_errors: str | None = None

    def encode(self, text, *, add_special_tokens, kind):
        values = [ord(character) for character in text]
        assert kind in {"mixed", "prose", "code"}
        return [256, *values, 257] if add_special_tokens else values

    def decode(self, values, *, skip_special_tokens, errors):
        self.decode_errors = errors
        specials = {256, 257, 258, 259}
        return "".join(chr(value) for value in values if not skip_special_tokens or value not in specials)


def test_generic_tokenizer_facade_supplies_the_hf_subset() -> None:
    backend = _GenericTokenizer()
    tokenizer = SpalmerTokenizerFacade(backend, model_max_length=4096, default_kind="code")

    assert tokenizer.encode("ab") == [97, 98]
    encoded = tokenizer(["ab", "c"], return_tensors="pt", padding=True)
    assert encoded["input_ids"].tolist() == [[97, 98], [99, 259]]
    assert encoded["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert tokenizer.decode(torch.tensor([97, 98, 257]), skip_special_tokens=True) == "ab"
    assert backend.decode_errors == "replace"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert prompt == "### User\nhello\n\n### Assistant\n"
    with pytest.raises(TypeError, match="thinking protocol"):
        tokenizer.apply_chat_template([], enable_thinking=True)


class _FakeSpalmerModel(nn.Module):
    def __init__(self, next_tokens: list[int] | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        self.config = SimpleNamespace(vocab_size=8, d_model=4, n_layers=4)
        self.next_tokens = list(next_tokens or [2])
        self.calls = 0
        self.end_requests = 0
        self.residency = SimpleNamespace(
            config=SimpleNamespace(
                num_experts=200,
                min_active_experts=2,
                max_active_experts=20,
                resident_cap=20,
            )
        )

    def forward(self, input_ids, **_kwargs):
        token = self.next_tokens[min(self.calls, len(self.next_tokens) - 1)]
        self.calls += 1
        logits = torch.full(
            (1, input_ids.shape[1], self.config.vocab_size),
            -10.0,
            device=input_ids.device,
        )
        logits[:, -1, token] = 10.0
        return SimpleNamespace(
            logits=logits,
            token_mixer_states=(f"token-{self.calls}",),
            channel_mixer_states=(f"channel-{self.calls}",),
        )

    def end_residency_request(self) -> None:
        self.end_requests += 1


def _facade(
    model: nn.Module,
    *,
    tokenizer: object | None = None,
    dynamic: bool = False,
) -> SpalmerCausalLMFacade:
    tokenizer = tokenizer or SimpleNamespace(eos_token_id=2)
    return SpalmerCausalLMFacade(
        model,
        tokenizer,
        checkpoint_path=Path("fixture.pt"),
        metadata={},
        dynamic_residency=dynamic,
        max_context_tokens=64,
        cuda_available=False,
    )


def _fake_spalmer_module(monkeypatch, model, tokenizer, metadata) -> None:
    package = types.ModuleType("spalmer")
    package.load_checkpoint = lambda path, map_location: (model, tokenizer, metadata)
    monkeypatch.setitem(sys.modules, "spalmer", package)


def test_load_spalmer_runtime_is_lazy_and_wraps_a_generic_tokenizer(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "spalmer.pt"
    checkpoint.write_bytes(b"loader is mocked")
    model = _FakeSpalmerModel()
    tokenizer = _GenericTokenizer()
    tokenizer.identity = SimpleNamespace(backend="spalmer-rpd-v2")
    _fake_spalmer_module(
        monkeypatch,
        model,
        tokenizer,
        {"sequence_length": 3072, "kind": "code"},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    runtime = load_spalmer_runtime(checkpoint)

    assert isinstance(runtime, SpalmerCausalLMFacade)
    assert isinstance(runtime.tokenizer, SpalmerTokenizerFacade)
    assert runtime.tokenizer.default_kind == "code"
    assert runtime.device.type == "cpu"
    assert next(runtime.model.parameters()).dtype == torch.float32
    assert runtime.config.max_position_embeddings == 3072
    assert runtime.dynamic_residency is True
    assert any("NoPE" in note for note in runtime.notes)
    assert model.calls == 0


def test_load_spalmer_runtime_reads_nested_training_context(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "spalmer.pt"
    checkpoint.write_bytes(b"loader is mocked")
    model = _FakeSpalmerModel()
    tokenizer = _GenericTokenizer()
    tokenizer.identity = SimpleNamespace(backend="spalmer-rpd-v2")
    _fake_spalmer_module(
        monkeypatch,
        model,
        tokenizer,
        {"training_config": {"sequence_length": 1792}},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    runtime = load_spalmer_runtime(checkpoint)

    assert runtime.config.max_position_embeddings == 1792
    assert runtime.tokenizer.model_max_length == 1792
    assert model.calls == 0


def test_load_spalmer_runtime_unwraps_an_hf_tokenizer_adapter(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "spalmer.pt"
    checkpoint.write_bytes(b"loader is mocked")
    model = _FakeSpalmerModel()
    hf_tokenizer = SimpleNamespace(eos_token="<eos>", eos_token_id=2, pad_token=None)
    adapter = SimpleNamespace(
        identity=SimpleNamespace(backend="huggingface"),
        tokenizer=hf_tokenizer,
    )
    _fake_spalmer_module(monkeypatch, model, adapter, {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    runtime = load_spalmer_runtime(checkpoint, dynamic_residency=False, max_context_tokens=512)

    assert runtime.tokenizer is hf_tokenizer
    assert hf_tokenizer.pad_token == "<eos>"
    assert runtime.config.max_position_embeddings == 512
    assert runtime.dynamic_residency is False
    assert model.calls == 0


class _Streamer:
    def __init__(self) -> None:
        self.values: list[list[int]] = []
        self.ended = 0

    def put(self, value: torch.Tensor) -> None:
        self.values.append(value.reshape(-1).tolist())

    def end(self) -> None:
        self.ended += 1


def test_facade_generation_streams_and_stops_at_eos_with_recurrent_state() -> None:
    model = _FakeSpalmerModel([4, 2, 7])
    runtime = _facade(model)
    streamer = _Streamer()

    generated = runtime.generate(
        torch.tensor([[1, 3]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        max_new_tokens=5,
        do_sample=False,
        eos_token_id=2,
        streamer=streamer,
    )

    assert generated.tolist() == [[1, 3, 4, 2]]
    assert model.calls == 2
    assert streamer.values == [[1, 3], [4], [2]]
    assert streamer.ended == 1
    assert model.training is True


def test_facade_generation_honors_external_stop_without_another_decode() -> None:
    model = _FakeSpalmerModel([4, 5, 6])
    runtime = _facade(model)

    generated = runtime.generate(
        torch.tensor([[1]]),
        max_new_tokens=5,
        stopping_criteria=lambda ids, _scores: ids.shape[1] >= 2,
    )

    assert generated.tolist() == [[1, 4]]
    assert model.calls == 1


def test_facade_forward_can_expose_the_final_hidden_states() -> None:
    class HiddenModel(_FakeSpalmerModel):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = nn.Linear(4, self.config.vocab_size, bias=False)

        def forward(self, input_ids, **_kwargs):
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
            logits = self.lm_head(hidden)
            return SimpleNamespace(logits=logits, loss=None)

    runtime = _facade(HiddenModel())

    output = runtime(torch.tensor([[1, 2]]), output_hidden_states=True, use_cache=False)

    assert output.logits.shape == (1, 2, 8)
    assert len(output.hidden_states) == 1
    assert output.hidden_states[-1].shape == (1, 2, 4)


def test_dynamic_generation_records_receipt_and_always_restores_request(monkeypatch) -> None:
    model = _FakeSpalmerModel()
    runtime = _facade(model, dynamic=True)
    initial = model(torch.tensor([[1]]))
    model.calls = 0
    decision = SimpleNamespace(
        output=initial,
        resident_ids=(3, 8, 13, 21),
        active_experts=4,
        effective_nll=1.25,
        predictive_entropy=1.5,
        average_surprise=1.1,
        trace=((2, 1.5), (4, 1.25)),
        expansions=((13, 21),),
    )
    experts = types.ModuleType("spalmer.experts")
    experts.choose_inference_residency = lambda selected_model, prompt: decision
    package = types.ModuleType("spalmer")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "spalmer", package)
    monkeypatch.setitem(sys.modules, "spalmer.experts", experts)

    generated = runtime.generate(torch.tensor([[1]]), max_new_tokens=2, eos_token_id=2)

    assert generated.tolist() == [[1, 2]]
    assert model.calls == 0
    assert model.end_requests == 1
    telemetry = runtime.routing_telemetry()
    assert telemetry is not None
    assert telemetry["routing_mode"] == "adaptive_surprise"
    assert telemetry["resident_experts"] == [3, 8, 13, 21]
    assert telemetry["active_experts"] == 4
    assert telemetry["expansions"] == 1


def test_dynamic_generation_publishes_telemetry_before_decode(monkeypatch) -> None:
    model = _FakeSpalmerModel([4, 2])
    runtime = _facade(model, dynamic=True)
    initial = model(torch.tensor([[1]]))
    model.calls = 0
    decision = SimpleNamespace(
        output=initial,
        resident_ids=(0, 1),
        active_experts=2,
        effective_nll=None,
        predictive_entropy=1.5,
        average_surprise=1.0,
        trace=((2, 1.5),),
        expansions=(),
    )
    experts = types.ModuleType("spalmer.experts")
    experts.choose_inference_residency = lambda _model, _prompt: decision
    package = types.ModuleType("spalmer")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "spalmer", package)
    monkeypatch.setitem(sys.modules, "spalmer.experts", experts)

    class TelemetryStop:
        def __call__(self, _ids, _scores):
            assert runtime.routing_telemetry()["active_experts"] == 2
            return True

    runtime.generate(
        torch.tensor([[1]]),
        max_new_tokens=2,
        eos_token_id=7,
        stopping_criteria=TelemetryStop(),
    )

    assert model.end_requests == 1


def test_facade_rejects_batch_padding_and_hf_cache_objects_before_forward() -> None:
    model = _FakeSpalmerModel()
    runtime = _facade(model)

    with pytest.raises(ValueError, match="one non-empty input sequence"):
        runtime.generate(torch.ones((2, 2), dtype=torch.long))
    with pytest.raises(ValueError, match="does not accept padded"):
        runtime.generate(
            torch.ones((1, 2), dtype=torch.long),
            attention_mask=torch.tensor([[1, 0]]),
        )
    with pytest.raises(ValueError, match="unsupported HF cache options"):
        runtime.generate(torch.ones((1, 2), dtype=torch.long), past_key_values=object())
    assert model.calls == 0
