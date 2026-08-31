from __future__ import annotations

import threading
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from yylab.infer.engine import Engine, LoadSpec, _adaptive_kv_reserve_bytes
from yylab.mole.adaptive import AdaptiveOccupancyConfig
from yylab.mole.inject import MoLEConfig, inject_mole, mole_layers, token_routers
from yylab.mole.runtime import AdaptiveExpertRuntime
from yylab.mole.store import LazyExpertStore, save_mole


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.q_proj(hidden_states))


class _ToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, max_position_embeddings=64)
        self.embed = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([_ToyBlock()])
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.forward_use_cache: list[bool | None] = []
        self.forward_logits_to_keep: list[object] = []

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **_kwargs,
    ):
        del attention_mask
        self.forward_use_cache.append(use_cache)
        self.forward_logits_to_keep.append(logits_to_keep)
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        if isinstance(logits_to_keep, torch.Tensor):
            hidden = hidden.index_select(1, logits_to_keep)
        elif logits_to_keep:
            hidden = hidden[:, -int(logits_to_keep) :]
        return SimpleNamespace(logits=self.lm_head(hidden))


def _adaptive_config(
    *,
    prefill: bool,
    baseline_nll: float | None = None,
    min_marginal_nll_gain_nats: float = 0.0,
) -> MoLEConfig:
    return MoLEConfig(
        num_experts=3,
        rank=2,
        alpha=4.0,
        top_k=2,
        target_modules=("q_proj",),
        router_type="per_token",
        adaptive=AdaptiveOccupancyConfig(
            enabled=True,
            surprise_ema_warmup=1,
            baseline_nll=baseline_nll,
            prefill_nll_probe=prefill,
            max_recomputations=1,
            min_marginal_nll_gain_nats=min_marginal_nll_gain_nats,
            vram_reserve_bytes=0,
            pin_host_memory=False,
        ),
    )


def _runtime(
    tmp_path: Path,
    *,
    prefill: bool = False,
    baseline_nll: float | None = None,
    min_marginal_nll_gain_nats: float = 0.0,
    resident_limit: int | None = None,
) -> tuple[AdaptiveExpertRuntime, _ToyLM]:
    torch.manual_seed(7)
    config = _adaptive_config(
        prefill=prefill,
        baseline_nll=baseline_nll,
        min_marginal_nll_gain_nats=min_marginal_nll_gain_nats,
    )
    source = _ToyLM()
    inject_mole(source, config, expert_dtype=torch.float32, expert_device="cpu")
    artifact = tmp_path / "adaptive"
    save_mole(artifact, source, config, base_ref="toy")

    target = _ToyLM()
    inject_mole(target, config, expert_dtype=torch.float32, expert_device="meta")
    runtime = AdaptiveExpertRuntime(
        target,
        config,
        LazyExpertStore(artifact, expected_base_ref="toy"),
        device="cpu",
        prompt_probe_tokens=4,
        resident_limit=resident_limit,
    )
    target.eval()
    return runtime, target


def _prompt() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    return ids, torch.ones_like(ids)


def test_request_residency_materializes_only_selected_lanes(tmp_path: Path) -> None:
    runtime, model = _runtime(tmp_path, resident_limit=2)
    input_ids, attention_mask = _prompt()

    telemetry = runtime.prepare_request(input_ids, attention_mask)

    assert runtime.resident_experts == (0, 1)
    assert telemetry["resident_expert_count"] == 2
    assert telemetry["active_expert_ids"] == [0, 1]
    for _name, layer in mole_layers(model):
        assert not layer.experts[0].lora_A.is_meta
        assert not layer.experts[1].lora_A.is_meta
        assert layer.experts[2].lora_A.is_meta
    for _name, router in token_routers(model):
        assert router.adaptive_expert_pool == (0, 1)

    runtime.begin_decode()
    with pytest.raises(RuntimeError, match="fixed after decoding"):
        runtime.establish_residency()
    runtime.end_decode()


def test_resident_capacity_reserves_safety_headroom_plus_kv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime, _model = _runtime(tmp_path)
    lane_bytes = runtime.store.bytes_per_expert
    runtime.device = torch.device("cuda")
    runtime.config.adaptive.vram_reserve_bytes = lane_bytes
    runtime._kv_reserve_bytes = lane_bytes
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (4 * lane_bytes, 100 * lane_bytes),
    )

    # Two lanes fit only when both independent reserves are subtracted.
    # Using max(safety, KV) here would incorrectly admit all three lanes.
    assert runtime._resident_capacity() == 2


def test_decode_entropy_changes_next_k_without_loading_weights(tmp_path: Path) -> None:
    runtime, _model = _runtime(tmp_path, resident_limit=3)
    input_ids, attention_mask = _prompt()
    runtime.prepare_request(input_ids, attention_mask)
    before = runtime.snapshot()
    assert len(before.active_experts) == 2

    runtime.begin_decode()
    processor = runtime.logits_processor()
    low_entropy = torch.tensor([[10.0, -10.0]])
    high_entropy = torch.tensor([[0.0, 0.0]])
    assert processor(input_ids, low_entropy) is low_entropy
    assert processor(input_ids, high_entropy) is high_entropy
    after = runtime.snapshot()

    assert len(after.active_experts) == 3
    assert after.lane_loads == before.lane_loads
    assert after.h2d_transfers == before.h2d_transfers
    assert after.h2d_bytes == before.h2d_bytes
    runtime.end_decode()


def test_prompt_nll_recomputes_without_populating_kv_cache(tmp_path: Path) -> None:
    runtime, model = _runtime(
        tmp_path,
        prefill=True,
        baseline_nll=0.0,
        resident_limit=3,
    )
    input_ids, attention_mask = _prompt()
    events: list[dict[str, object]] = []

    telemetry = runtime.prepare_request(
        input_ids,
        attention_mask,
        telemetry_callback=events.append,
    )

    # One full-E routing probe, one prompt-specific baseline, and one
    # candidate expansion evaluation. Every pass is teacher-forced.
    assert telemetry["prompt_probe_count"] == 3
    assert telemetry["recomputations"] == 1
    assert telemetry["active_experts"] == 3
    assert model.forward_use_cache == [False, False, False]
    assert all(isinstance(value, torch.Tensor) for value in model.forward_logits_to_keep)
    resident_event = next(event for event in events if event["phase"] == "resident")
    assert resident_event["resident_expert_count"] == 2
    assert resident_event["lane_loads"] == 2


def test_prompt_predicted_nll_reranks_physical_residency(tmp_path: Path) -> None:
    runtime, model = _runtime(
        tmp_path,
        prefill=True,
        baseline_nll=100.0,
        resident_limit=2,
    )
    for _name, router in token_routers(model):
        with torch.no_grad():
            router.gate.weight.zero_()
            assert router.surprise_bias is not None
            router.surprise_bias.copy_(torch.tensor([3.0, 2.0, -3.0]))
    input_ids, attention_mask = _prompt()

    telemetry = runtime.prepare_request(input_ids, attention_mask)

    assert runtime.resident_experts == (2, 1)
    assert telemetry["request_expert_order"] == [2, 1, 0]
    assert telemetry["residency_reranked"] is True
    assert telemetry["predicted_nll_by_expert"][2] < telemetry["predicted_nll_by_expert"][1]
    assert telemetry["predicted_nll_by_expert"][1] < telemetry["predicted_nll_by_expert"][0]
    assert telemetry["lane_loads"] == 3
    for _name, layer in mole_layers(model):
        assert layer.experts[0].lora_A.is_meta
        assert not layer.experts[1].lora_A.is_meta
        assert not layer.experts[2].lora_A.is_meta


@pytest.mark.parametrize(
    ("candidate_nll", "minimum_gain", "reason"),
    [
        (3.0, 0.0, "worsened NLL"),
        (1.95, 0.1, "below the configured knee"),
    ],
)
def test_prompt_expansion_rolls_back_unhelpful_candidate(
    tmp_path: Path,
    candidate_nll: float,
    minimum_gain: float,
    reason: str,
) -> None:
    runtime, _model = _runtime(
        tmp_path,
        prefill=True,
        baseline_nll=0.0,
        min_marginal_nll_gain_nats=minimum_gain,
        resident_limit=3,
    )
    probes = iter((1.0, 2.0, candidate_nll))

    def fake_prompt_nll(_self, _input_ids, _attention_mask):
        return next(probes)

    runtime.prompt_nll = MethodType(fake_prompt_nll, runtime)
    input_ids, attention_mask = _prompt()

    telemetry = runtime.prepare_request(input_ids, attention_mask)

    assert telemetry["active_experts"] == 2
    assert telemetry["active_expert_ids"] == [0, 1]
    assert telemetry["recomputations"] == 1
    assert reason in telemetry["stop_reason"]


def test_decode_retry_restores_controller_checkpoint_without_transfers(tmp_path: Path) -> None:
    runtime, _model = _runtime(tmp_path, resident_limit=3)
    input_ids, attention_mask = _prompt()
    runtime.prepare_request(input_ids, attention_mask)
    runtime.begin_decode()
    before = runtime.snapshot()

    runtime.observe_entropy(torch.tensor([[10.0, -10.0]]))
    runtime.observe_entropy(torch.tensor([[0.0, 0.0]]))
    assert len(runtime.active_experts) == 3
    runtime.restart_decode_attempt(reason="test cache retry")
    after = runtime.snapshot()
    telemetry = runtime.telemetry()

    assert len(after.active_experts) == 2
    assert after.lane_loads == before.lane_loads
    assert after.h2d_transfers == before.h2d_transfers
    assert after.h2d_bytes == before.h2d_bytes
    assert telemetry["surprise"] is None
    assert telemetry["decode_attempts"] == 2
    assert telemetry["cache_retries"] == 1
    runtime.end_decode()


def test_engine_loads_v4_router_without_materializing_expert_lanes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import yylab.models.loading as loading_module

    config = _adaptive_config(prefill=False)
    source = _ToyLM()
    inject_mole(source, config, expert_dtype=torch.float32, expert_device="cpu")
    artifact = tmp_path / "engine-adaptive"
    save_mole(artifact, source, config, base_ref="toy")

    def fake_load_base(_ref, _quant, **_options):
        return _ToyLM(), _EngineTokenizer(), False, "bf16", []

    monkeypatch.setattr(loading_module, "load_base", fake_load_base)
    engine = Engine()

    notes = engine.load(
        LoadSpec(
            ref="toy",
            quant="bf16",
            adapter_path=str(artifact),
            adapter_arch="mole",
        )
    )

    assert engine.adaptive_runtime is not None
    assert engine.adaptive_runtime.resident_experts == ()
    assert engine.routing_telemetry() == engine.adaptive_telemetry()
    assert any("Adaptive MoLE v4" in note for note in notes)
    for _name, layer in mole_layers(engine.model):
        assert all(expert.lora_A.is_meta for expert in layer.experts)
    engine.unload()


class _EngineTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del messages, tokenize, add_generation_prompt
        return "prompt"

    def __call__(self, _text, return_tensors=None):
        assert return_tensors == "pt"
        ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_engine_serializes_generation_calls() -> None:
    engine = Engine()
    engine.model = SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(max_position_embeddings=32),
    )
    engine.tokenizer = _EngineTokenizer()
    engine.spec = LoadSpec(ref="toy", quant="bf16", settings={"max_context_tokens": 32})
    active = 0
    peak = 0
    counter_lock = threading.Lock()

    def fake_stream(_self, _kwargs):
        nonlocal active, peak
        with counter_lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.04)
            yield "ok"
        finally:
            with counter_lock:
                active -= 1

    engine._stream = MethodType(fake_stream, engine)
    barrier = threading.Barrier(3)
    outputs: list[str] = []

    def generate() -> None:
        barrier.wait()
        outputs.append("".join(engine.generate_stream([{"role": "user", "content": "hi"}])))

    threads = [threading.Thread(target=generate) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert outputs == ["ok", "ok"]
    assert peak == 1


def test_kv_reserve_tracks_cache_location_and_quantization() -> None:
    config = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=8,
    )

    dynamic = _adaptive_kv_reserve_bytes(
        config,
        {"cache_implementation": "dynamic"},
        prompt_tokens=10,
        answer_tokens=20,
        cuda=True,
    )
    quantized = _adaptive_kv_reserve_bytes(
        config,
        {"cache_implementation": "quantized", "kv_quant": "int8"},
        prompt_tokens=10,
        answer_tokens=20,
        cuda=True,
    )
    offloaded = _adaptive_kv_reserve_bytes(
        config,
        {"cache_implementation": "offloaded"},
        prompt_tokens=1000,
        answer_tokens=1000,
        cuda=True,
    )

    assert dynamic == 32 * 30
    assert quantized == 16 * 30
    assert offloaded == 32 * 256
    assert (
        _adaptive_kv_reserve_bytes(
            config,
            {"cache_implementation": "dynamic"},
            prompt_tokens=10,
            answer_tokens=20,
            cuda=False,
        )
        == 0
    )


def test_engine_cache_fallback_restarts_adaptive_decode_state() -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.reserve: int | None = None
            self.begins = 0
            self.retries = 0
            self.ends = 0

        def prepare_request(self, _input_ids, _attention_mask, **kwargs) -> None:
            self.reserve = kwargs["kv_reserve_bytes"]

        def logits_processor(self):
            return lambda _input_ids, scores: scores

        def begin_decode(self) -> None:
            self.begins += 1

        def restart_decode_attempt(self, *, reason: str) -> None:
            assert "dynamic cache" in reason
            self.retries += 1

        def end_decode(self) -> None:
            self.ends += 1

        def telemetry(self) -> dict[str, object]:
            return {"cache_retries": self.retries}

    engine = Engine()
    engine.model = SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(
            max_position_embeddings=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=8,
        ),
    )
    engine.tokenizer = _EngineTokenizer()
    engine.spec = LoadSpec(
        ref="toy",
        quant="bf16",
        settings={"max_context_tokens": 32, "cache_implementation": "static"},
    )
    engine.cuda = True
    runtime = FakeRuntime()
    engine.adaptive_runtime = runtime
    attempts = 0
    attempt_kwargs: list[dict] = []

    def fake_stream(_self, kwargs):
        nonlocal attempts
        attempts += 1
        attempt_kwargs.append(dict(kwargs))
        if attempts == 1:
            raise RuntimeError("cache setup failed")
        yield "ok"

    engine._stream = MethodType(fake_stream, engine)

    output = "".join(
        engine.generate_stream(
            [{"role": "user", "content": "hi"}],
            max_new_tokens=5,
        )
    )

    assert output == "ok"
    assert attempts == 2
    assert runtime.reserve == 32 * 8
    assert runtime.begins == 1
    assert runtime.retries == 1
    assert runtime.ends == 1
    assert attempt_kwargs[0]["cache_implementation"] == "static"
    assert "cache_implementation" not in attempt_kwargs[1]
    assert "max_cache_len" not in attempt_kwargs[1]
    assert "prefill_chunk_size" not in attempt_kwargs[1]
