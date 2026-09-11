from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction

import pytest

from yylab.train import sprint_shapes as shapes


def dimensions(candidate):
    config = candidate["model_config"]
    return config["hidden_size"], config["num_hidden_layers"], config["intermediate_size"]


@pytest.mark.parametrize("vocab,width,ffn,layers,total", [
    (8192, 512, 1792, 2, 9_964_032),
    (8192, 256, 1408, 8, 9_965_824),
    (8192, 1024, 3008, 4, 49_816_576),
    (8192, 2304, 6976, 9, 499_329_792),
    (8192, 2816, 10112, 11, 998_505_728),
])
def test_exact_tied_parameter_breakdown(vocab, width, ffn, layers, total):
    counts = shapes.sprint_parameter_breakdown(vocab, width, ffn, layers)
    assert counts["total"] == total
    assert sum(counts[key] for key in ("embeddings", "attention", "ffn", "norms")) == total
    assert counts["embeddings"] == vocab * width
    assert counts["norms"] == (2 * layers + 1) * width


def test_swiglu_counts_the_third_projection_and_preserves_other_components():
    gelu = shapes.sprint_parameter_breakdown(31, 128, 256, 2)
    gated = shapes.sprint_parameter_breakdown(31, 128, 256, 2, mlp_type="swiglu")
    assert gated["ffn"] == 3 * gelu["ffn"] // 2
    assert gated["total"] - gelu["total"] == 2 * 128 * 256


@pytest.mark.parametrize("target", [10_000_000, 50_000_000, 500_000_000, 1_000_000_000])
def test_candidates_meet_budget_fraction_and_alignment_constraints_without_duplicates(target):
    candidates = shapes.enumerate_sprint_shapes(target)
    assert candidates
    assert len({dimensions(candidate) for candidate in candidates}) == len(candidates)
    for candidate in candidates:
        width, layers, ffn = dimensions(candidate)
        config = candidate["model_config"]
        assert target * Fraction(98, 100) <= candidate["parameters"] <= target
        assert width % 128 == ffn % 64 == 0
        assert width // config["num_attention_heads"] == 128
        assert config["tie_word_embeddings"] is True
        assert candidate["parameter_shortfall"] == target - candidate["parameters"]
        assert candidate["parameter_shortfall_fraction"] <= 0.02
        assert candidate["ffn_fraction"] == candidate["parameter_breakdown"]["ffn"] / candidate["parameters"]
        assert all(match["absolute_error"] <= 0.05 for match in candidate["matched_ffn_fractions"])
        assert candidate["parameter_breakdown"] == shapes.sprint_parameter_breakdown(8192, width, ffn, layers)


def test_ten_million_search_includes_shallow_and_deeper_alternatives():
    candidates = shapes.enumerate_sprint_shapes(10_000_000)
    dimensions_seen = {dimensions(candidate) for candidate in candidates}
    assert (512, 2, 1792) in dimensions_seen
    assert (384, 4, 1408) in dimensions_seen
    assert (256, 8, 1408) in dimensions_seen


def test_fifty_million_search_includes_wide_shallow_and_attention_heavier_alternatives():
    candidates = shapes.enumerate_sprint_shapes(50_000_000)
    dimensions_seen = {dimensions(candidate) for candidate in candidates}
    assert (1536, 2, 3008) in dimensions_seen
    assert (1280, 4, 1280) in dimensions_seen
    assert (768, 4, 5568) in dimensions_seen


def test_strict_exact_fraction_and_total_budget_are_not_relaxed():
    counts = shapes.sprint_parameter_breakdown(8192, 256, 1408, 8)
    candidates = shapes.enumerate_sprint_shapes(
        counts["total"], depths=[8], ffn_fractions=[Fraction(counts["ffn"], counts["total"])],
        tolerance=0, ffn_fraction_tolerance=0, max_width=256,
    )
    assert [dimensions(candidate) for candidate in candidates] == [(256, 8, 1408)]
    assert candidates[0]["parameter_shortfall"] == 0
    assert candidates[0]["matched_ffn_fractions"][0]["absolute_error"] == 0


def test_unreachable_tied_vocabulary_floor_returns_no_smaller_fallback():
    assert shapes.enumerate_sprint_shapes(100_000, vocab_size=8192) == []
    assert shapes.enumerate_sprint_shapes(10_000_001, tolerance=0) == []


def test_all_fp4_friendly_alignment_and_unique_fraction_requests():
    candidates = shapes.enumerate_sprint_shapes(
        50_000_000, ffn_fractions=[0.375, 0.375, 0.5], depths=[4, 2, 4],
        width_alignment=256, ffn_alignment=128,
    )
    assert candidates
    for candidate in candidates:
        width, layers, ffn = dimensions(candidate)
        assert width % 256 == ffn % 128 == 0
        assert layers in {2, 4}
        matched = [item["requested"] for item in candidate["matched_ffn_fractions"]]
        assert len(matched) == len(set(matched))


@pytest.mark.parametrize("updates", [
    {"target_params": 0}, {"target_params": True}, {"vocab_size": 0},
    {"ffn_fractions": []}, {"ffn_fractions": "0.5"}, {"ffn_fractions": [0]},
    {"ffn_fractions": [1]}, {"ffn_fractions": [float("nan")]}, {"depths": []},
    {"depths": [0]}, {"depths": [True]}, {"head_dim": 127},
    {"width_alignment": 64}, {"ffn_alignment": 32}, {"max_width": 0},
    {"tolerance": -0.01}, {"tolerance": 1}, {"ffn_fraction_tolerance": float("inf")},
    {"mlp_type": "unknown"},
])
def test_invalid_requests_fail_before_search(updates):
    with pytest.raises(ValueError):
        shapes.enumerate_sprint_shapes(**{"target_params": 10_000_000, **updates})


def test_search_work_is_explicitly_bounded():
    with pytest.raises(ValueError, match="too broad"):
        shapes.enumerate_sprint_shapes(10**12, depths=list(range(1, 129)), max_width=65_536)


def test_cli_only_writes_json_candidates_and_does_not_import_torch(tmp_path):
    output = tmp_path / "shapes.json"
    command = [sys.executable, "-m", "yylab.train.sprint_shapes", "--target-params", "10000000",
               "--depths", "2,8", "--output", str(output)]
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    report = json.loads(output.read_text("utf-8"))
    assert report["status"] == "ok" and report["candidate_count"] == len(report["candidates"])
    assert json.loads(process.stdout)["candidates"] == report["candidate_count"]
    assert set(tmp_path.iterdir()) == {output}
    subprocess.run([sys.executable, "-c", "import sys; import yylab.train.sprint_shapes; assert 'torch' not in sys.modules"],
                   capture_output=True, text=True, check=True)


def test_candidate_configs_agree_with_sprint_configuration_counter():
    from yylab.models.sprint import SprintConfig, estimate_parameter_count

    for candidate in shapes.enumerate_sprint_shapes(10_000_000, depths=[2, 8]):
        config = SprintConfig(**candidate["model_config"])
        assert config.validate() is config
        assert estimate_parameter_count(config) == candidate["parameters"]
