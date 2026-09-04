from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from yylab import db
from yylab.artifacts import checkpoint_dir
from yylab.train.base_config import (
    TRAINING_RECIPE_PRESETS,
    ScratchModelSpec,
    build_hf_config,
    estimate_parameter_count,
    fit_layers_to_parameter_target,
    fit_scratch_to_parameter_target,
    normalize_training_token_budget,
    projected_training_memory_gb,
    resolve_training_token_budget,
    tokenizer_embedding_parameter_floor,
)
from yylab.train.low_precision import (
    LOW_PRECISION_METADATA_KEY,
    NATIVE_NVFP4_BACKEND,
    QATLinear,
    apply_qat,
    apply_saved_qat,
    native_nvfp4_status,
    qat_metadata,
)
from yylab.train.mxfp8 import NATIVE_MXFP8_BACKEND


def _tokenizer_dir(tmp_path: Path) -> Path:
    raw = Tokenizer(models.WordLevel(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    raw.train_from_iterator(
        ["alpha beta gamma delta epsilon", "one two three four five"],
        trainers.WordLevelTrainer(
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    out = tmp_path / "tokenizer"
    tokenizer.save_pretrained(out)
    return out


def _tiny_spec(family: str = "qwen3") -> ScratchModelSpec:
    return ScratchModelSpec(
        family=family,
        vocab_size=64,
        vocab_padding_multiple=1,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_theta=10_000,
        tie_word_embeddings=True,
    )


def test_nvfp4_sparse_training_preset_resolves_concrete_axes():
    preset = TRAINING_RECIPE_PRESETS["nvfp4_paired_4to8"]
    assert preset == {
        "parameter_dtype": "bf16",
        "qat": "nvfp4",
        "qat_backend": "native",
        "qat_group_size": 16,
        "activation_qat": "nvfp4",
        "sparsity": "paired_4:8",
        "sparsity_backend": "compressed",
    }


def test_sparse_mxfp8_cpu_offload_preset_is_an_explicit_hardware_contract():
    preset = TRAINING_RECIPE_PRESETS["nvfp4_sparse_mxfp8_cpu_offload"]

    assert preset["qat"] == "nvfp4"
    assert preset["qat_backend"] == "native"
    assert preset["sparsity"] == "paired_4:8"
    assert preset["sparsity_backend"] == "compressed"
    assert preset["backward_precision"] == "mxfp8"
    assert preset["backward_backend"] == "native"
    assert preset["optimizer_state"] == "cpu_offload"
    assert preset["optimizer_impl"] == "cpu_offload_adamw"
    assert "master_weight_storage" not in preset


def test_cpu_offload_projection_counts_fp32_master_and_moments():
    parameters = 1024
    gpu_gb, cpu_gb = projected_training_memory_gb(
        parameters,
        parameter_dtype="bf16",
        optimizer_state="cpu_offload",
    )

    assert gpu_gb * 2**30 == parameters * 4
    assert cpu_gb * 2**30 == parameters * 12


@pytest.mark.parametrize("family", ["qwen3", "llama", "bitnet"])
def test_parameter_estimate_matches_hf_model_exactly(family):
    class TokenizerStub:
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2

        def __len__(self):
            return 64

    spec = _tiny_spec(family)
    config = build_hf_config(spec, TokenizerStub())
    model = AutoModelForCausalLM.from_config(config)
    assert estimate_parameter_count(spec) == sum(p.numel() for p in model.parameters())


def test_parameter_target_solver_reports_achievable_shape():
    spec = _tiny_spec()
    per_layer = estimate_parameter_count(replace(spec, num_hidden_layers=2)) - estimate_parameter_count(
        replace(spec, num_hidden_layers=1)
    )
    target = estimate_parameter_count(replace(spec, num_hidden_layers=7)) + per_layer // 4
    fitted = fit_layers_to_parameter_target(spec, target)
    assert fitted.num_hidden_layers == 7
    assert estimate_parameter_count(fitted) != target


def test_flexible_parameter_target_solver_reaches_one_million_with_compact_vocab():
    spec = ScratchModelSpec(vocab_size=0, vocab_padding_multiple=64)

    fitted = fit_scratch_to_parameter_target(
        spec,
        1_000_000,
        tokenizer_vocab_size=2048,
    )
    actual = estimate_parameter_count(fitted, tokenizer_vocab_size=2048)

    fitted.validate(tokenizer_vocab_size=2048)
    assert abs(actual - 1_000_000) / 1_000_000 < 0.01
    assert (
        fitted.hidden_size,
        fitted.intermediate_size,
        fitted.num_hidden_layers,
    ) != (spec.hidden_size, spec.intermediate_size, spec.num_hidden_layers)


@pytest.mark.parametrize("target", [100_000, 500_000, 3_000_000, 12_000_000, 27_000_000])
def test_flexible_parameter_target_solver_supports_continuous_small_targets(target):
    spec = ScratchModelSpec(vocab_size=0, vocab_padding_multiple=64)

    fitted = fit_scratch_to_parameter_target(
        spec,
        target,
        tokenizer_vocab_size=2048,
    )
    actual = estimate_parameter_count(fitted, tokenizer_vocab_size=2048)

    fitted.validate(tokenizer_vocab_size=2048)
    assert abs(actual - target) / target < 0.001


def test_flexible_parameter_target_solver_keeps_nvfp4_alignment():
    fitted = fit_scratch_to_parameter_target(
        ScratchModelSpec(vocab_size=0, vocab_padding_multiple=64),
        3_000_000,
        tokenizer_vocab_size=2048,
        qat_mode="nvfp4",
    )

    fitted.validate(tokenizer_vocab_size=2048, qat_mode="nvfp4")
    assert fitted.hidden_size % 32 == 0
    assert fitted.intermediate_size % 32 == 0
    assert fitted.head_dim % 16 == 0


def test_parameter_target_solver_enforces_large_tokenizer_embedding_floor():
    spec = ScratchModelSpec(vocab_size=0, vocab_padding_multiple=64)

    assert tokenizer_embedding_parameter_floor(spec, 151_680) == 1_213_440
    with pytest.raises(ValueError, match=r"tokenizer embedding floor 1,213,440"):
        fit_scratch_to_parameter_target(
            spec,
            1_000_000,
            tokenizer_vocab_size=151_680,
        )


def test_training_token_budget_rounds_up_and_overrides_step_sources():
    budget = resolve_training_token_budget(
        requested_tokens=1_000_000,
        seq_len=512,
        micro_batch=8,
        grad_accum=2,
        max_steps=999,
        estimated_steps=777,
    )

    assert budget == {
        "requested_training_tokens": 1_000_000,
        "tokens_per_optimizer_step": 8192,
        "max_steps": 123,
        "scheduled_training_tokens": 1_007_616,
        "token_budget_rounding": 7616,
        "step_cap_source": "token_budget",
    }


def test_training_budget_preserves_legacy_step_and_epoch_resolution():
    explicit = resolve_training_token_budget(
        seq_len=128,
        micro_batch=2,
        grad_accum=4,
        max_steps=7,
        estimated_steps=11,
    )
    epoch_derived = resolve_training_token_budget(
        seq_len=128,
        micro_batch=2,
        grad_accum=4,
        estimated_steps=11,
    )

    assert explicit["max_steps"] == 7
    assert explicit["step_cap_source"] == "max_steps"
    assert explicit["scheduled_training_tokens"] == 7168
    assert epoch_derived["max_steps"] == 11
    assert epoch_derived["step_cap_source"] == "epochs"


def test_replayed_config_rederives_stale_step_cap_from_token_budget():
    normalized = normalize_training_token_budget(
        {
            "requested_training_tokens": 1000,
            "seq_len": 128,
            "micro_batch": 2,
            "grad_accum": 2,
            "max_steps": 999,
        }
    )

    assert normalized["max_steps"] == 2
    assert normalized["scheduled_training_tokens"] == 1024
    assert normalized["step_cap_source"] == "token_budget"


def test_scratch_validation_separates_native_and_training_context():
    spec = _tiny_spec()
    with pytest.raises(ValueError, match="exceeds native model context"):
        spec.validate(tokenizer_vocab_size=64, train_sequence_length=256)
    with pytest.raises(ValueError, match="unsupported QAT target"):
        spec.validate(tokenizer_vocab_size=64, qat_mode="ternary")
    with pytest.raises(ValueError, match="dense training only"):
        replace(spec, family="bitnet").validate(tokenizer_vocab_size=64, qat_mode="nvfp4")


@pytest.mark.parametrize("qat_format", ["int4", "ternary"])
def test_low_precision_formats_without_native_kernels_are_rejected(qat_format):
    with pytest.raises(ValueError, match="no native training backend"):
        qat_metadata(qat_format)


def test_emulated_backend_requests_are_rejected_before_model_mutation():
    with pytest.raises(ValueError, match="native execution only"):
        qat_metadata("nvfp4", backend="emulated")
    with pytest.raises(ValueError, match="MXFP8 backward backend"):
        qat_metadata("nvfp4", backward_backend="emulated")
    with pytest.raises(ValueError, match="structured-sparsity backend"):
        qat_metadata("nvfp4", sparsity_backend="dense_masked")
    with pytest.raises(ValueError, match="requires activation='nvfp4'"):
        qat_metadata("nvfp4", activation="int8")
    with pytest.raises(ValueError, match="requires activation='nvfp4'"):
        qat_metadata("nvfp4", activation="none")
    with pytest.raises(ValueError, match="activation quantization requires"):
        qat_metadata("none", activation="nvfp4")
    with pytest.raises(ValueError, match="compressed sparsity backend requires"):
        qat_metadata("none", sparsity_backend="compressed")
    with pytest.raises(ValueError, match="native backward backend requires"):
        qat_metadata("none", backward_backend="native")
    with pytest.raises(ValueError, match="exact block size of 16"):
        qat_metadata("nvfp4", group_size=32)


def test_auto_backend_never_falls_back_when_native_nvfp4_is_unavailable(monkeypatch):
    from yylab.train import low_precision

    monkeypatch.setattr(
        low_precision,
        "native_nvfp4_status",
        lambda _device=None: (False, "test native gap"),
    )
    with pytest.raises(RuntimeError, match="test native gap"):
        qat_metadata("nvfp4", backend="auto")


def test_runtime_mxfp8_resolution_keeps_exact_native_provider(monkeypatch):
    from yylab.train import low_precision, mxfp8

    target = SimpleNamespace(
        weight=SimpleNamespace(
            is_cuda=True,
            dtype=torch.bfloat16,
            device=torch.device("cuda:3"),
        )
    )
    monkeypatch.setattr(
        mxfp8,
        "native_mxfp8_status",
        lambda _device=None: (True, "native provider smoke passed"),
    )

    backend, note = low_precision._resolve_backward_runtime_backend(
        [("projection", target)],
        precision="mxfp8",
        recipe_backend=NATIVE_MXFP8_BACKEND,
    )

    assert backend == NATIVE_MXFP8_BACKEND
    assert note == "native provider smoke passed"


def _native_qat_metadata_for_worker(**overrides):
    metadata = {
        "version": 4,
        "format": "nvfp4",
        "backend": NATIVE_NVFP4_BACKEND,
        "backend_request": "native",
        "group_size": 16,
        "activation": "nvfp4",
        "parameter_dtype": "bf16",
        "native_low_bit_gemm": True,
        "backward_precision": "bf16",
        "backward_backend": "none",
        "backward_backend_request": "auto",
        "native_low_bit_backward": False,
        "sparsity": "none",
        "sparsity_backend": "none",
        "sparsity_backend_request": "auto",
        "native_sparse_gemm": False,
    }
    metadata.update(overrides)
    return metadata


def test_continued_base_applies_requested_verified_qat_and_records_resolution(
    monkeypatch,
):
    from yylab.workers import train as worker

    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    model.config = SimpleNamespace()
    metadata = _native_qat_metadata_for_worker()
    applied: list[tuple[object, dict]] = []
    monkeypatch.setattr(
        worker,
        "_requested_qat_metadata",
        lambda *_args, **_kwargs: dict(metadata),
    )

    def apply(model_arg, metadata_arg):
        applied.append((model_arg, metadata_arg))
        setattr(model_arg.config, LOW_PRECISION_METADATA_KEY, dict(metadata_arg))
        return 1

    monkeypatch.setattr(worker, "apply_qat", apply)

    effective, note = worker._configure_continued_low_precision(
        model,
        {"parameter_dtype": "bf16", "qat": "nvfp4"},
        parameter_dtype="bf16",
        inherit_checkpoint_low_precision=False,
    )

    assert applied == [(model, metadata)]
    assert effective["qat"] == "nvfp4"
    assert effective["resolved_low_precision"] == metadata
    assert effective["inherit_checkpoint_low_precision"] is False
    assert "applied requested verified native nvfp4" in str(note)


def test_continued_base_rejects_conflicting_requested_and_inherited_qat(monkeypatch):
    from yylab.workers import train as worker

    inherited = _native_qat_metadata_for_worker()
    requested = _native_qat_metadata_for_worker(
        backward_precision="mxfp8",
        backward_backend=NATIVE_MXFP8_BACKEND,
        backward_backend_request="native",
        native_low_bit_backward=True,
    )
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model.config = SimpleNamespace(**{LOW_PRECISION_METADATA_KEY: inherited})
    monkeypatch.setattr(
        worker,
        "_requested_qat_metadata",
        lambda *_args, **_kwargs: requested,
    )

    with pytest.raises(ValueError, match="conflicts.*backward_precision"):
        worker._configure_continued_low_precision(
            model,
            {"parameter_dtype": "bf16", "qat": "nvfp4"},
            parameter_dtype="bf16",
            inherit_checkpoint_low_precision=True,
        )


def test_continued_base_inheritance_records_the_effective_checkpoint_recipe(
    monkeypatch,
):
    from yylab.workers import train as worker

    inherited = _native_qat_metadata_for_worker()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model.config = SimpleNamespace(**{LOW_PRECISION_METADATA_KEY: inherited})
    monkeypatch.setattr(
        worker,
        "_requested_qat_metadata",
        lambda *_args, **_kwargs: None,
    )

    effective, note = worker._configure_continued_low_precision(
        model,
        {"parameter_dtype": "bf16", "qat": "none"},
        parameter_dtype="bf16",
        inherit_checkpoint_low_precision=True,
    )

    assert effective["qat"] == "nvfp4"
    assert effective["resolved_low_precision"] == inherited
    assert effective["inherit_checkpoint_low_precision"] is True
    assert "retained" in str(note)


def test_continued_base_rejects_missing_inherited_qat_recipe(monkeypatch):
    from yylab.workers import train as worker

    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model.config = SimpleNamespace()
    monkeypatch.setattr(
        worker,
        "_requested_qat_metadata",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match="has no native low-precision recipe"):
        worker._configure_continued_low_precision(
            model,
            {"parameter_dtype": "bf16", "qat": "none"},
            parameter_dtype="bf16",
            inherit_checkpoint_low_precision=True,
        )


def test_saved_emulated_qat_recipe_fails_closed(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    model.config = SimpleNamespace(
        **{
            LOW_PRECISION_METADATA_KEY: {
                "version": 3,
                "format": "nvfp4",
                "backend": "yylab_ste_emulation",
                "sparsity": "none",
                "sparsity_backend": "none",
                "backward_precision": "bf16",
                "backward_backend": "none",
            }
        }
    )
    from yylab.train import low_precision

    monkeypatch.setattr(
        low_precision,
        "native_nvfp4_status",
        lambda _device=None: (True, "native backend available in test"),
    )

    with pytest.raises(RuntimeError, match="is not native"):
        apply_saved_qat(model)
    assert type(model[0]) is torch.nn.Linear


def test_base_loader_never_substitutes_dense_cpu_for_requested_quantization(monkeypatch):
    import transformers

    from yylab.models import loading

    tokenizer = SimpleNamespace(pad_token="<pad>", eos_token="<eos>")
    loaded: dict = {}

    def load_model(*_args, **kwargs):
        loaded.update(kwargs)
        return SimpleNamespace(config=SimpleNamespace(use_cache=False))

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="NF4 bitsandbytes loading is unavailable.*CUDA is unavailable"):
        loading.load_base("unused", "nf4")
    with pytest.raises(RuntimeError, match="FP16 dense model execution is unavailable.*CPU"):
        loading.load_base("unused", "fp16")
    with pytest.raises(RuntimeError, match="BF16 dense model execution is unavailable.*CPU"):
        loading.load_base("unused", "bf16")
    with pytest.raises(ValueError, match="unsupported base precision"):
        loading.load_base("unused", "made-up-precision")

    loading.load_base("unused", "fp32")
    assert loaded["dtype"] is torch.float32

    loaded.clear()
    legacy_recipe = {
        "format": "nvfp4",
        "backend": "yylab_ste_emulation",
        "sparsity": "none",
        "sparsity_backend": "none",
        "backward_precision": "bf16",
        "backward_backend": "none",
    }
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(yylab_low_precision=legacy_recipe),
    )
    from yylab.models import precision_capabilities

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        precision_capabilities,
        "require_model_precision",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="is not native"):
        loading.load_base("unused", "bf16")
    assert not loaded

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        transformers,
        "BitsAndBytesConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    loading.load_base("unused", "int8")
    assert loaded["dtype"] is torch.float16
    assert loaded["device_map"] == {"": 0}


@pytest.mark.parametrize(
    "quantization_config",
    [
        {"quant_method": "bitnet", "linear_class": "autobitlinear"},
        {"quant_method": "gptq"},
    ],
)
def test_checkpoint_embedded_quantization_must_use_a_verified_yylab_provider(
    quantization_config,
):
    from yylab.models.loading import reject_unverified_checkpoint_quantization

    with pytest.raises(ValueError, match="checkpoint-embedded quantization"):
        reject_unverified_checkpoint_quantization(
            SimpleNamespace(quantization_config=quantization_config)
        )

    reject_unverified_checkpoint_quantization(SimpleNamespace())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_saved_native_qat_recipe_fails_for_non_bf16_cuda(dtype):
    native, reason = native_nvfp4_status()
    if not native:
        pytest.skip(reason)

    metadata = qat_metadata("nvfp4", backend="native", parameter_dtype="bf16")
    assert metadata is not None
    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    model.config = SimpleNamespace(**{LOW_PRECISION_METADATA_KEY: dict(metadata)})
    model = model.to(device="cuda", dtype=dtype)

    with pytest.raises(RuntimeError, match="not BF16"):
        apply_saved_qat(model)
    assert type(model[0]) is torch.nn.Linear


def test_direct_native_qat_application_remains_strict(monkeypatch):
    from yylab.train import low_precision

    metadata = {
        "version": 4,
        "format": "nvfp4",
        "backend": NATIVE_NVFP4_BACKEND,
        "group_size": 16,
        "activation": "nvfp4",
        "parameter_dtype": "bf16",
        "sparsity": "none",
        "sparsity_backend": "none",
        "backward_precision": "bf16",
        "backward_backend": "none",
    }
    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    model.config = SimpleNamespace()
    monkeypatch.setattr(
        low_precision,
        "native_nvfp4_status",
        lambda _device=None: (False, "native backend unavailable in test"),
    )

    with pytest.raises(RuntimeError, match="native NVFP4 QAT runtime is unavailable"):
        apply_qat(model, metadata)
    assert type(model[0]) is torch.nn.Linear


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("activation", "none"),
        ("group_size", 32),
        ("parameter_dtype", "fp16"),
    ],
)
def test_saved_native_qat_requires_exact_w4a4_metadata(monkeypatch, field, value):
    from yylab.train import low_precision

    metadata = {
        "version": 4,
        "format": "nvfp4",
        "backend": NATIVE_NVFP4_BACKEND,
        "group_size": 16,
        "activation": "nvfp4",
        "parameter_dtype": "bf16",
        "sparsity": "none",
        "sparsity_backend": "none",
        "backward_precision": "bf16",
        "backward_backend": "none",
    }
    metadata[field] = value
    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    model.config = SimpleNamespace()
    monkeypatch.setattr(
        low_precision,
        "native_nvfp4_status",
        lambda _device=None: (True, "native backend available in test"),
    )

    with pytest.raises(RuntimeError, match="exact native W4A4 contract"):
        apply_qat(model, metadata)
    assert type(model[0]) is torch.nn.Linear


def test_native_nvfp4_paired_4_8_forward_backward_when_available():
    native, reason = native_nvfp4_status()
    if not native:
        pytest.skip(reason)

    metadata = qat_metadata(
        "nvfp4",
        sparsity="paired_4:8",
        backend="native",
    )
    assert metadata is not None
    assert metadata["native_low_bit_gemm"] is True

    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    model.config = SimpleNamespace()
    model = model.to(device="cuda", dtype=torch.bfloat16)
    assert apply_qat(model, metadata) == 1

    value = torch.randn(
        128,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = model(value)
    output.float().square().mean().backward()

    assert torch.isfinite(output).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert model[0].weight.grad is not None
    assert torch.isfinite(model[0].weight.grad).all()


def test_compressed_nvfp4_forward_native_mxfp8_backward_when_available():
    from yylab.train.low_precision import native_sparse_nvfp4_status
    from yylab.train.mxfp8 import native_mxfp8_status

    sparse, sparse_reason = native_sparse_nvfp4_status()
    mxfp8, mxfp8_reason = native_mxfp8_status()
    if not sparse or not mxfp8:
        pytest.skip(f"sparse={sparse_reason}; mxfp8={mxfp8_reason}")

    metadata = qat_metadata(
        "nvfp4",
        sparsity="paired_4:8",
        backend="native",
        sparsity_backend="compressed",
        backward_precision="mxfp8",
        backward_backend="native",
    )
    assert metadata is not None
    assert metadata["native_sparse_gemm"] is True
    assert metadata["native_low_bit_backward"] is True

    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=True))
    model.config = SimpleNamespace()
    model = model.to(device="cuda", dtype=torch.bfloat16)
    assert apply_qat(model, metadata) == 1
    assert model[0].qat_sparsity_backend == "cusparselt_compressed_nvfp4"
    assert model[0].qat_backward_backend == NATIVE_MXFP8_BACKEND

    value = torch.randn(
        3,
        47,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = model(value)
    output.float().square().mean().backward()
    torch.cuda.synchronize()

    assert output.shape == (3, 47, 128)
    assert torch.isfinite(output).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert model[0].weight.grad is not None
    assert torch.isfinite(model[0].weight.grad).all()


def test_recommended_low_precision_recipe_completes_real_qwen_optimizer_step():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from yylab.train.loop import CPUOffloadAdamW
    from yylab.train.low_precision import native_sparse_nvfp4_status
    from yylab.train.mxfp8 import native_mxfp8_status

    sparse, sparse_reason = native_sparse_nvfp4_status()
    mxfp8, mxfp8_reason = native_mxfp8_status()
    if not sparse or not mxfp8:
        pytest.skip(f"sparse={sparse_reason}; mxfp8={mxfp8_reason}")

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        use_cache=False,
    )
    model = Qwen3ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    metadata = qat_metadata(
        "nvfp4",
        backend="native",
        sparsity="paired_4:8",
        sparsity_backend="compressed",
        backward_precision="mxfp8",
        backward_backend="native",
    )
    assert metadata is not None
    apply_qat(model, metadata)
    optimizer = CPUOffloadAdamW(
        model.parameters(),
        lr=1e-3,
    )
    input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")

    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    summary = optimizer.master_state_summary()

    assert torch.isfinite(loss)
    assert summary["format"] == "fp32"
    assert summary["moment_format"] == "fp32"
    assert float(summary["promoted_fraction"]) == 0.0
    assert all(torch.isfinite(parameter.float()).all() for parameter in model.parameters())


class _OneBatch:
    docs_seen = 1
    tokens_seen = 8

    def __iter__(self):
        return iter([{"input_ids": [[4, 5, 6, 7, 8, 9, 4, 5]]}])

    def data_stats(self):
        return {
            "docs_seen": 1,
            "rendered_tokens": 8,
            "tokens": {
                "rendered_content_tokens": 8,
                "inserted_document_boundary_tokens": 0,
                "rendered_token_positions": 8,
                "covered_rendered_positions": 8,
                "unforwarded_rendered_positions": 0,
                "forwarded_real_inputs": 8,
                "forwarded_model_slots": 8,
                "padding_inputs": 0,
                "packing_replay_inputs": 0,
                "model_target_slots": 7,
                "optimizer_supervised_targets": 7,
                "ignored_target_slots": 0,
            },
            "per_corpus": {},
        }


def test_run_base_scratch_end_to_end_and_reload(tmp_path, monkeypatch):
    from yylab.models import loading
    from yylab.workers import train as worker

    tokenizer_dir = _tokenizer_dir(tmp_path)
    out_id = db.new_id("ckpt")
    db.execute(
        "INSERT INTO checkpoints(id, name, created_at, kind, source, ref) VALUES(?,?,?,?,?,?)",
        (out_id, "tiny-scratch", db.now_iso(), "pending", "pending", ""),
    )
    db.execute(
        "INSERT OR REPLACE INTO corpora(id,name,created_at,updated_at,status,source_path,stats_json) "
        "VALUES(?,?,?,?,?,?,?)",
        (
            "unused-by-test",
            "unused-by-test",
            db.now_iso(),
            db.now_iso(),
            "ready",
            "",
            json.dumps(
                {
                    "docs_written": 1,
                    "chars_written": 80,
                    "domains": {"language": {"docs": 1, "chars": 80}},
                }
            ),
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(worker, "build_batches", lambda *_args, **_kwargs: _OneBatch())
    config = {
        "initialization": {
            "mode": "scratch",
            "tokenizer_ref": str(tokenizer_dir),
        },
        "method": "full",
        "architecture": _tiny_spec().to_dict(),
        "precision": {
            "parameter_dtype": "fp32",
            "qat": "none",
        },
        "optimizer": {
            "implementation": "torch_adamw",
            "state": "auto",
            "lr": 1e-3,
            "weight_decay": 0.01,
            "beta1": 0.8,
            "beta2": 0.95,
            "eps": 1e-7,
        },
        "scheduler": {"name": "linear", "warmup_steps": 0},
        "corpora": ["unused-by-test"],
        "domains": None,
        "seq_len": 8,
        "epochs": 1,
        "micro_batch": 1,
        "grad_accum": 1,
        "max_steps": 1,
        "grad_clip": 0.5,
        "grad_ckpt": False,
        "liger_fused_ce": False,
        "liger_full_kernels": False,
        "cudnn_sdpa": False,
        "allow_tf32": False,
        "torch_compile": False,
        "seed": 123,
        "log_every": 1,
        "save_every": None,
        "out_checkpoint_id": out_id,
    }

    summary = worker.run_base(config, tmp_path / "run")
    out = checkpoint_dir(out_id)
    assert summary["initialization"] == "scratch"
    assert summary["qat"] == "none"
    assert summary["result"]["steps"] == 1
    assert (out / "model.safetensors").is_file()
    assert (out / "yylab_recipe.json").is_file()
    assert (out / "tokenized_view_manifest.json").is_file()
    assert (tmp_path / "run" / "tokenized_view_manifest.json").is_file()
    tokenized_view = json.loads((out / "tokenized_view_manifest.json").read_text("utf-8"))
    assert tokenized_view["kind"] == "yylab_tokenized_training_view"
    assert all(tokenized_view["reconciliation"].values())
    assert summary["data_provenance"]["training"] == tokenized_view
    saved_config = json.loads((out / "config.json").read_text("utf-8"))
    assert "yylab_low_precision" not in saved_config
    assert saved_config["max_position_embeddings"] == 128
    assert saved_config["yylab_training_recipe"]["objective"] == "causal_lm"
    assert saved_config["yylab_training_recipe"]["precision"]["qat"] == "none"

    model, tokenizer, cuda, quant, notes = loading.load_base(str(out), "fp32")
    assert not cuda
    assert quant == "fp32"
    assert len(tokenizer) > 0
    assert not any(isinstance(module, QATLinear) for module in model.modules())

    dense_model, _, _, _, dense_notes = loading.load_base(
        str(out),
        "fp32",
        restore_saved_qat=False,
    )
    assert not any(isinstance(module, QATLinear) for module in dense_model.modules())
    assert not hasattr(dense_model.config, "yylab_low_precision")
    assert not any("without restoring inherited QAT" in note for note in dense_notes)
    row = db.q1("SELECT * FROM checkpoints WHERE id=?", (out_id,))
    metadata = json.loads(row["meta_json"])
    assert row["kind"] == "base"
    assert metadata["initialization"] == "scratch"
    assert metadata["precision"]["qat"] == "none"


def test_failed_base_cleanup_removes_only_pending_output():
    from yylab.workers.train import mark_training_artifacts_failed

    pending = db.new_id("ckpt")
    complete = db.new_id("ckpt")
    for checkpoint_id, kind in ((pending, "pending"), (complete, "base")):
        db.execute(
            "INSERT INTO checkpoints(id, name, created_at, kind, source, ref) VALUES(?,?,?,?,?,?)",
            (checkpoint_id, checkpoint_id, db.now_iso(), kind, "local", "x"),
        )
    mark_training_artifacts_failed("base", {"out_checkpoint_id": pending})
    mark_training_artifacts_failed("base", {"out_checkpoint_id": complete})
    assert db.q1("SELECT id FROM checkpoints WHERE id=?", (pending,)) is None
    assert db.q1("SELECT id FROM checkpoints WHERE id=?", (complete,)) is not None


def test_cpu_offload_optimizer_honors_all_numerical_controls():
    from yylab.train.loop import CPUOffloadAdamW, make_optimizer

    model = torch.nn.Linear(4, 3)
    optimizer = make_optimizer(
        model,
        2e-4,
        weight_decay=0.1,
        cuda=False,
        optimizer_state="cpu_offload",
        betas=(0.8, 0.95),
        eps=1e-6,
    )
    assert isinstance(optimizer, CPUOffloadAdamW)
    assert optimizer.param_groups[0]["lr"] == 2e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.95)
    assert optimizer.param_groups[0]["eps"] == 1e-6
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert all(
        state["master_param"].device.type == "cpu"
        and state["master_param"].dtype == torch.float32
        for state in optimizer.state.values()
    )
    assert optimizer.master_state_summary()["format"] == "fp32"


@pytest.mark.parametrize(
    "optimizer",
    [
        {"master_weight_storage": "fp8_e4m3_potentiated"},
        {"master_weight_storage": "fp8_e4m3_adaptive"},
        {"adaptive_max_promoted_fraction": 0.1},
    ],
)
def test_base_worker_rejects_retired_optimizer_master_settings(optimizer):
    from yylab.workers.train import _make_base_optimizer

    with pytest.raises(ValueError, match="not executable"):
        _make_base_optimizer(torch.nn.Linear(2, 2), {"optimizer": optimizer}, cuda=False)


def test_fp32_cpu_offload_optimizer_state_resumes_exactly_on_cpu():
    from yylab.train.loop import CPUOffloadAdamW

    first_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    first = CPUOffloadAdamW([first_parameter], lr=1e-3)
    first_parameter.grad = torch.tensor([0.25, -0.5])
    first.step()

    second_parameter = torch.nn.Parameter(first_parameter.detach().clone())
    second = CPUOffloadAdamW([second_parameter], lr=1e-3)
    second.load_state_dict(first.state_dict())
    assert all(
        tensor.device.type == "cpu" and tensor.dtype == torch.float32
        for state in second.state.values()
        for tensor in state.values()
        if isinstance(tensor, torch.Tensor)
    )

    gradient = torch.tensor([-0.125, 0.75])
    first_parameter.grad = gradient.clone()
    second_parameter.grad = gradient.clone()
    first.step()
    second.step()
    assert torch.equal(first_parameter, second_parameter)


def test_cpu_offload_optimizer_rejects_legacy_fp8_state():
    from yylab.train.loop import CPUOffloadAdamW

    optimizer = CPUOffloadAdamW([torch.nn.Parameter(torch.ones(2))])
    state = optimizer.state_dict()
    state["param_groups"][0]["master_weight_storage"] = "fp8_e4m3_adaptive"
    with pytest.raises(ValueError, match="not executable"):
        optimizer.load_state_dict(state)


def test_qat_base_auto_selects_high_precision_lora_instead_of_qlora():
    from yylab.ui.pages.train_mole import _auto_method_quant, _checkpoint_qat_format

    checkpoint = {"meta_json": json.dumps({"precision": {"qat": "int4"}})}
    assert _checkpoint_qat_format(checkpoint) == "int4"
    assert _auto_method_quant(checkpoint, "nf4") == ("lora", "bf16")


def test_qat_base_auto_rejects_incoherent_dense_lock():
    from yylab.ui.pages.train_mole import _auto_method_quant

    checkpoint = {
        "meta_json": json.dumps({"precision": {"qat": "nvfp4"}}),
        "quant_locked": "fp32",
    }
    with pytest.raises(ValueError, match="saved native QAT requires BF16"):
        _auto_method_quant(checkpoint, "bf16")


def test_base_ui_recipe_builder_keeps_independent_training_axes():
    from yylab.ui.pages.train_base import _base_run_config

    config = _base_run_config(
        initialization={"mode": "scratch", "tokenizer_ref": "tokenizer"},
        method="full",
        architecture=_tiny_spec().to_dict(),
        precision={"parameter_dtype": "bf16", "qat": "nvfp4"},
        optimizer={"implementation": "cpu_offload_adamw", "state": "cpu_offload", "lr": 3e-4},
        scheduler={"name": "cosine", "warmup_ratio": 0.03},
        corpora=["corpus-z", "corpus-a"],
        domains=["code"],
        seq_len=128,
        epochs=2,
        micro_batch=2,
        grad_accum=8,
        max_steps=99,
        seed=7,
        grad_clip=0.5,
        grad_ckpt=True,
        log_every=3,
        save_every=20,
        runtime={"torch_compile": True, "compile_mode": "reduce-overhead"},
        objective="chat_sft",
        splits=["train", "unspecified"],
        target_parameters=12_345_678,
    )
    assert config["initialization"]["mode"] == "scratch"
    assert config["objective"] == "chat_sft"
    assert config["architecture"]["max_position_embeddings"] == 128
    assert config["precision"]["qat"] == "nvfp4"
    assert config["optimizer"]["state"] == "cpu_offload"
    assert config["scheduler"]["warmup_ratio"] == 0.03
    assert config["corpora"] == ["corpus-z", "corpus-a"]
    assert config["corpus_mix"] == "round_robin"
    assert config["splits"] == ["train", "unspecified"]
    assert config["seq_len"] == 128
    assert config["max_steps"] == 99
    assert config["step_cap_source"] == "max_steps"
    assert config["tokens_per_optimizer_step"] == 2048
    assert config["scheduled_training_tokens"] == 202_752
    assert config["save_every"] == 20
    assert config["torch_compile"] is True
    assert config["shuffle_buffer"] == 8192
    assert config["shuffle_buffer_bytes"] == 256 * 2**20
    assert config["target_parameters"] == 12_345_678


def test_base_ui_recipe_builder_persists_token_budget_rounding():
    from yylab.ui.pages.train_base import _base_run_config

    config = _base_run_config(
        initialization={"mode": "scratch", "tokenizer_ref": "tokenizer"},
        method="full",
        architecture=_tiny_spec().to_dict(),
        precision={"parameter_dtype": "bf16", "qat": "none"},
        optimizer={"implementation": "torch_adamw", "state": "gpu", "lr": 1e-4},
        scheduler={"name": "cosine", "warmup_ratio": 0.03},
        corpora=["corpus"],
        domains=None,
        seq_len=512,
        epochs=3,
        micro_batch=8,
        grad_accum=2,
        max_steps=999,
        seed=7,
        grad_clip=1.0,
        grad_ckpt=True,
        log_every=1,
        save_every=None,
        runtime={"max_steps": 17, "scheduled_training_tokens": 17},
        requested_training_tokens=1_000_000,
        step_cap_source="token_budget",
    )

    assert config["requested_training_tokens"] == 1_000_000
    assert config["max_steps"] == 123
    assert config["scheduled_training_tokens"] == 1_007_616
    assert config["token_budget_rounding"] == 7616


def test_compact_preset_is_honest_about_its_required_vocabulary():
    from yylab.ui.pages.train_base import SCRATCH_PRESETS, _tokenizer_vocab_hint

    preset = SCRATCH_PRESETS["qwen_1m_compact"]
    assert preset["compact_vocab_size"] == 2048
    assert preset["vocab_size"] == 0
    assert preset["hidden_size"] == 64
    assert preset["num_hidden_layers"] == 18
    preset_spec = ScratchModelSpec(
        **{
            key: value
            for key, value in preset.items()
            if key not in {"target_params_m", "compact_tokenizer", "compact_vocab_size"}
        }
    )
    assert (
        abs(estimate_parameter_count(preset_spec, tokenizer_vocab_size=2048) - 1_000_000)
        / 1_000_000
        < 0.03
    )
    assert preset_spec.resolved_vocab_size(151_680) == 151_680
    with pytest.raises(ValueError, match="tokenizer embedding floor"):
        fit_scratch_to_parameter_target(
            preset_spec,
            1_000_000,
            tokenizer_vocab_size=151_680,
        )
    assert _tokenizer_vocab_hint({"ref": "missing", "meta_json": "{}"}, 2048) == 2048


def test_small_parameter_target_names_preserve_requested_precision():
    from yylab.ui.pages.train_base import _parameter_target_slug, _resolve_parameter_target

    assert _parameter_target_slug(100_000) == "0.1m"
    assert _parameter_target_slug(500_000) == "0.5m"
    assert _parameter_target_slug(3_000_000) == "3m"
    assert _parameter_target_slug(12_345_000) == "12.345m"
    assert _parameter_target_slug(123_457) == "0.123457m"
    assert _resolve_parameter_target(0.1) == 100_000
    assert _resolve_parameter_target(0.1, 123_457) == 123_457


def test_scratch_initialization_supports_existing_and_compact_tokenizers():
    from yylab.ui.pages.train_base import (
        TOKENIZER_SOURCE_COMPACT,
        TOKENIZER_SOURCE_EXISTING,
        _scratch_initialization,
    )

    assert _scratch_initialization(
        TOKENIZER_SOURCE_EXISTING,
        tokenizer_ref="tokenizer-ref",
    ) == {"mode": "scratch", "tokenizer_ref": "tokenizer-ref"}
    compact_initialization = _scratch_initialization(
        TOKENIZER_SOURCE_COMPACT,
        compact_tokenizer={
            "vocab_size": 2048,
            "min_frequency": 3,
            "max_documents": 5000,
            "lowercase": True,
        },
    )
    assert compact_initialization["mode"] == "scratch"
    assert compact_initialization["compact_tokenizer"] | {
        "max_training_bytes": 0,
        "special_token_profile": "chat",
        "custom_special_tokens": (),
    } == compact_initialization["compact_tokenizer"]
    assert {
        key: compact_initialization["compact_tokenizer"][key]
        for key in ("vocab_size", "min_frequency", "max_documents", "lowercase")
    } == {
        "vocab_size": 2048,
        "min_frequency": 3,
        "max_documents": 5000,
        "lowercase": True,
    }
    with pytest.raises(ValueError, match="requires a tokenizer reference"):
        _scratch_initialization(TOKENIZER_SOURCE_EXISTING)


def test_scratch_worker_trains_inline_compact_tokenizer_from_selected_data(
    tmp_path, monkeypatch
):
    from yylab.workers import train as train_worker

    calls = {}

    def fake_docs(corpora, domains, **kwargs):
        calls.update(corpora=corpora, domains=domains, **kwargs)
        yield {"text": "system alpha user beta assistant gamma"}
        yield {"text": "code math language delta epsilon"}

    monkeypatch.setattr(train_worker, "iter_corpus_docs", fake_docs)
    monkeypatch.setattr(train_worker, "checkpoint_dir", lambda _checkpoint_id: tmp_path)
    spec = replace(
        _tiny_spec(),
        vocab_size=0,
        vocab_padding_multiple=1,
    )
    model, tokenizer, cuda, quant, notes = train_worker._create_scratch_base(
        {
            "initialization": {
                "mode": "scratch",
                "compact_tokenizer": {
                    "vocab_size": 264,
                    "min_frequency": 1,
                    "max_documents": 10,
                    "lowercase": False,
                    "special_token_profile": "chat",
                },
            },
            "out_checkpoint_id": "checkpoint",
            "corpora": ["corpus-b", "corpus-a"],
            "domains": ["code"],
            "splits": ["train", "unspecified"],
            "corpus_mix": "round_robin",
            "objective": "chat_sft",
            "architecture": spec.to_dict(),
            "precision": {"parameter_dtype": "fp32", "qat": "none"},
            "seq_len": 64,
            "seed": 7,
        }
    )

    assert calls["corpora"] == ["corpus-b", "corpus-a"]
    assert calls["domains"] == ["code"]
    assert calls["splits"] == ["train", "unspecified"]
    assert calls["shuffle_buffer"] == 1
    assert (tmp_path / "tokenizer-source" / "tokenizer.json").is_file()
    assert len(tokenizer) == 264
    assert model.config.vocab_size == 264
    assert cuda is torch.cuda.is_available()
    assert quant == "fp32"
    assert any("trained inline chat byte-BPE tokenizer" in note for note in notes)


def test_base_ui_normalizes_legacy_single_corpus_and_preserves_multi_select_order():
    from yylab.ui.pages.train_base import _ordered_corpora

    assert _ordered_corpora("corpus-legacy") == ["corpus-legacy"]
    assert _ordered_corpora(["corpus-b", "corpus-a", "corpus-b"]) == [
        "corpus-b",
        "corpus-a",
    ]
    assert _ordered_corpora(None) == []
