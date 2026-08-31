from __future__ import annotations

from yylab.mole.adaptive import AdaptiveOccupancyConfig
from yylab.ui.pages.chat import _engine_routing_telemetry, _routing_telemetry_text
from yylab.ui.pages.train_mole import (
    ADAPTIVE_ROUTING_MODE,
    FIXED_ROUTING_MODE,
    _adaptive_ui_config,
    _estimate_adaptive_architecture,
    _routing_run_fields,
)


def test_adaptive_ui_values_serialize_to_runtime_contract() -> None:
    payload = _adaptive_ui_config(vram_reserve_gb=1.5, pin_host_memory=False)
    config = AdaptiveOccupancyConfig.from_dict(payload)

    assert config.enabled is True
    assert config.target_total_multiplier == 1.25
    assert config.expert_footprint_fraction == 0.005
    assert config.min_active_fraction == 0.01
    assert config.max_active_fraction == 0.10
    assert config.minimum_experts == 2
    assert config.maximum_experts(100) == 20
    assert config.surprise_ema_decay == 0.95
    assert config.surprise_ema_warmup == 32
    assert config.expand_margin_nats == 0.0
    assert config.contract_margin_nats == 0.05
    assert config.expand_patience == 1
    assert config.contract_patience == 8
    assert config.min_marginal_nll_gain_nats == 0.01
    assert config.max_recomputations == 4
    assert config.vram_reserve_bytes == round(1.5 * 2**30)
    assert config.pin_host_memory is False


def test_routing_run_fields_leave_legacy_shape_unchanged() -> None:
    assert _routing_run_fields(FIXED_ROUTING_MODE) == {"training_strategy": "joint_per_token_v1"}


def test_routing_run_fields_select_adaptive_strategy_and_validated_payload() -> None:
    fields = _routing_run_fields(ADAPTIVE_ROUTING_MODE, _adaptive_ui_config())

    assert fields["training_strategy"] == "joint_adaptive_surprise_v1"
    assert fields["adaptive"]["enabled"] is True
    assert fields["adaptive"]["score_semantics"] == "predicted_nll"
    assert AdaptiveOccupancyConfig.from_dict(fields["adaptive"]).minimum_experts == 2


def test_adaptive_architecture_estimate_derives_read_only_e_r_and_k() -> None:
    estimate = _estimate_adaptive_architecture(
        {
            "params_b": 0.6,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_layers": 28,
            "num_attention_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        _adaptive_ui_config(),
    )

    assert estimate is not None
    assert estimate["num_experts"] >= 2
    assert estimate["rank"] >= 1
    assert estimate["top_k"] == 2
    assert 2 <= estimate["maximum_active_experts"] <= 20
    assert estimate["expert_parameter_count"] > 0


def test_adaptive_telemetry_formatter_reports_native_controller_values() -> None:
    text = _routing_telemetry_text(
        {
            "routing_mode": "adaptive_surprise",
            "active_experts": 4,
            "minimum_experts": 2,
            "maximum_experts": 20,
            "active_fraction": 0.02,
            "mean_active_experts": 3.25,
            "peak_active_experts": 6,
            "signal_kind": "entropy",
            "surprise": 1.25,
            "surprise_ema": 1.1,
            "expansions": 3,
            "contractions": 1,
            "recomputations": 2,
            "resident_experts": [1, 3, 5, 7],
            "resident_bytes": 8 * 2**20,
            "lane_loads": 4,
            "h2d_bytes": 8 * 2**20,
            "stop_reason": "marginal NLL gain below threshold",
        }
    )

    assert "Adaptive experts 4 (2–20)" in text
    assert "occupancy 2.00%" in text
    assert "entropy 1.250 nats (EMA 1.100)" in text
    assert "expand 3 / contract 1 / redo 2" in text
    assert "resident 4 (8.0 MiB)" in text
    assert "lane loads 4 / H2D 8.0 MiB" in text
    assert "marginal NLL gain below threshold" in text


def test_chat_telemetry_reader_safely_handles_optional_engine_api() -> None:
    class NoTelemetry:
        pass

    class BrokenTelemetry:
        def routing_telemetry(self):
            raise RuntimeError("diagnostic failure")

    class NestedTelemetry:
        def routing_telemetry(self):
            return {"adaptive": {"routing_mode": "adaptive_surprise", "active_experts": 2}}

    assert _engine_routing_telemetry(NoTelemetry()) is None
    assert _engine_routing_telemetry(BrokenTelemetry()) is None
    assert _engine_routing_telemetry(NestedTelemetry()) == {
        "routing_mode": "adaptive_surprise",
        "active_experts": 2,
    }
    assert _routing_telemetry_text(None) == ""
