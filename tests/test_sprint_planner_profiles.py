"""Shape planning checks allocate configuration objects, never model weights."""

import pytest

from yylab.models.sprint import estimate_parameter_count, plan_sprint_model


def _shape(config):
    return (config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
            config.intermediate_size, estimate_parameter_count(config))


@pytest.mark.parametrize("target,expected", [
    (10_000_000, (512, 2, 4, 1792, 9_964_032)),
    (50_000_000, (1024, 4, 8, 3008, 49_816_576)),
    (100_000_000, (1280, 5, 10, 4416, 99_792_640)),
    (500_000_000, (2304, 9, 18, 6976, 499_329_792)),
    (1_000_000_000, (2816, 11, 22, 10112, 998_505_728)),
])
def test_throughput_default_shapes_are_aligned_and_fit_budget(target, expected):
    config = plan_sprint_model(target, 8192)
    assert _shape(config) == expected
    assert _shape(plan_sprint_model(target, 8192, profile="throughput")) == expected
    assert config.hidden_size % 256 == 0
    assert config.head_dim == 128
    assert config.intermediate_size % 64 == 0
    assert 0.98 * target <= estimate_parameter_count(config) <= target


@pytest.mark.parametrize("target,expected", [
    (1_000_000, (96, 2, 1, 352, 995_808)),
    (10_000_000, (320, 6, 5, 1280, 9_998_400)),
    (49_999_999, (640, 10, 10, 2208, 49_902_720)),
    (50_000_000, (640, 10, 10, 2208, 49_902_720)),
    (100_000_000, (768, 13, 12, 3152, 99_922_176)),
    (500_000_000, (1408, 21, 22, 5440, 499_821_696)),
    (1_000_000_000, (1728, 27, 27, 7104, 999_625_536)),
])
def test_balanced_preserves_legacy_shapes_and_small_defaults(target, expected):
    assert _shape(plan_sprint_model(target, 8192, profile="balanced")) == expected
    if target < 10_000_000:
        assert _shape(plan_sprint_model(target, 8192)) == expected


@pytest.mark.parametrize("target", [10_000_000, 50_000_000, 173_000_000, 500_000_000, 1_000_000_000])
@pytest.mark.parametrize("vocabulary", [257, 32768, 128000])
@pytest.mark.parametrize("mlp,tied", [("gelu", True), ("swiglu", False)])
def test_throughput_uses_actual_vocabulary_and_architecture_options(target, vocabulary, mlp, tied):
    config = plan_sprint_model(target, vocabulary, max_position_embeddings=4096,
                               mlp_type=mlp, tie_word_embeddings=tied)
    assert config.vocab_size == vocabulary
    assert config.max_position_embeddings == 4096
    assert config.mlp_type == mlp
    assert config.tie_word_embeddings is tied
    if (target, vocabulary, mlp, tied) == (10_000_000, 128000, "swiglu", False):
        # Untied embeddings alone exclude widths >=48. Even width32 with
        # the planner's maximum 64 layers and 6x FFN cannot reach98% of10M.
        largest_allowed = 2 * vocabulary * 32 + 64 * (4 * 32**2 + 3 * 32 * 192 + 2 * 32) + 32
        assert largest_allowed < 0.98 * target
        assert config.to_dict() == plan_sprint_model(
            target, vocabulary, max_position_embeddings=4096, mlp_type=mlp,
            tie_word_embeddings=tied, profile="balanced").to_dict()
        assert estimate_parameter_count(config) <= target
    else:
        assert 0.98 * target <= estimate_parameter_count(config) <= target


@pytest.mark.parametrize("target", [1_000_000, 9_999_999])
def test_defaults_below_ten_million_keep_balanced_proportions(target):
    assert plan_sprint_model(target, 8192).to_dict() == plan_sprint_model(
        target, 8192, profile="balanced").to_dict()


@pytest.mark.parametrize("target", [10_000_000, 25_000_000, 49_999_999])
def test_small_throughput_profile_preserves_budget_alignment_and_loss_default(target):
    config = plan_sprint_model(target, 8192)
    assert config.num_hidden_layers >= 2
    assert config.hidden_size % 256 == config.intermediate_size % 64 == 0
    assert config.head_dim == 128
    assert config.loss_backend == "auto"
    assert 0.98 * target <= estimate_parameter_count(config) <= target


def test_vocabulary_floor_falls_back_to_balanced_without_oversizing():
    options = dict(tie_word_embeddings=False, mlp_type="swiglu", max_position_embeddings=4096)
    throughput = plan_sprint_model(50_000_000, 128000, **options)
    balanced = plan_sprint_model(50_000_000, 128000, profile="balanced", **options)
    # Even the smallest 256-wide untied embedding exceeds this whole budget.
    assert throughput.hidden_size < 256
    assert throughput.to_dict() == balanced.to_dict()


def test_invalid_profile_is_rejected_even_for_small_models():
    with pytest.raises(ValueError, match="profile"):
        plan_sprint_model(1_000_000, 8192, profile="unknown")
