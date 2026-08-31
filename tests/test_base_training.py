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
)
from yylab.train.low_precision import (
    EMULATED_QAT_BACKEND,
    LOW_PRECISION_METADATA_KEY,
    NATIVE_NVFP4_BACKEND,
    QATLinear,
    apply_qat,
    apply_saved_qat,
    fake_quant_nvfp4,
    fake_structured_sparsity,
    native_nvfp4_status,
    qat_metadata,
    structured_sparsity_mask,
)


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
        "activation_qat": "none",
        "sparsity": "paired_4:8",
    }


def test_sparse_mxfp8_adaptive_preset_is_an_explicit_hardware_contract():
    preset = TRAINING_RECIPE_PRESETS["nvfp4_sparse_mxfp8_adaptive"]

    assert preset["qat"] == "nvfp4"
    assert preset["qat_backend"] == "native"
    assert preset["sparsity"] == "paired_4:8"
    assert preset["sparsity_backend"] == "compressed"
    assert preset["backward_precision"] == "mxfp8"
    assert preset["backward_backend"] == "native"
    assert preset["master_weight_storage"] == "fp8_e4m3_adaptive"
    assert preset["adaptive_max_promoted_fraction"] == 0.10


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


def test_scratch_validation_separates_native_and_training_context():
    spec = _tiny_spec()
    with pytest.raises(ValueError, match="exceeds native model context"):
        spec.validate(tokenizer_vocab_size=64, train_sequence_length=256)
    with pytest.raises(ValueError, match="requires the BitNet"):
        spec.validate(tokenizer_vocab_size=64, qat_mode="ternary")
    replace(spec, family="bitnet").validate(tokenizer_vocab_size=64, qat_mode="ternary")


@pytest.mark.parametrize(
    ("family", "qat_format", "activation"),
    [
        ("llama", "int4", "int8"),
        ("llama", "nvfp4", "nvfp4"),
        ("bitnet", "ternary", "int8"),
    ],
)
def test_qat_formats_propagate_gradients(family, qat_format, activation):
    class TokenizerStub:
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2

        def __len__(self):
            return 64

    model = AutoModelForCausalLM.from_config(build_hf_config(_tiny_spec(family), TokenizerStub()))
    metadata = qat_metadata(qat_format, group_size=16, activation=activation)
    count = apply_qat(model, metadata)
    ids = torch.randint(0, 64, (1, 8))
    model(input_ids=ids, labels=ids).loss.backward()
    assert count > 0
    assert any(isinstance(module, QATLinear) for module in model.modules())
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert metadata["training_state_is_low_bit"] is False


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nvfp4_emulation_matches_torchao_two_level_format(dtype):
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    torch.manual_seed(19)
    value = (torch.randn(19, 128, dtype=dtype) * 0.03).requires_grad_()
    expected_scale = per_tensor_amax_to_scale(value.detach().abs().amax())
    expected = NVFP4Tensor.to_nvfp4(
        value.detach().contiguous(),
        per_tensor_scale=expected_scale,
    ).dequantize(dtype)

    actual = fake_quant_nvfp4(value)
    assert torch.equal(actual.detach(), expected)
    actual.float().sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))


def test_nvfp4_emulation_preserves_exact_zero_and_requires_native_group_size():
    value = torch.zeros(3, 32, dtype=torch.bfloat16, requires_grad=True)
    quantized = fake_quant_nvfp4(value)

    assert torch.equal(quantized, value)
    assert torch.isfinite(quantized).all()
    with pytest.raises(ValueError, match="group size of 16"):
        fake_quant_nvfp4(value, group_size=32)


def test_saved_native_qat_recipe_falls_back_without_rewriting_recipe(monkeypatch):
    from yylab.train import low_precision

    metadata = qat_metadata("nvfp4", backend="emulated")
    assert metadata is not None
    metadata["backend"] = NATIVE_NVFP4_BACKEND
    metadata["backend_request"] = "native"
    metadata["native_low_bit_gemm"] = True
    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    model.config = SimpleNamespace(**{LOW_PRECISION_METADATA_KEY: dict(metadata)})
    monkeypatch.setattr(
        low_precision,
        "native_nvfp4_status",
        lambda: (True, "native backend available in test"),
    )

    count, runtime = apply_saved_qat(model)

    assert count == 1
    assert runtime is not None
    assert runtime["backend"] == NATIVE_NVFP4_BACKEND
    assert runtime["runtime_backend"] == EMULATED_QAT_BACKEND
    assert runtime["native_low_bit_gemm"] is False
    assert model.config.yylab_low_precision["backend"] == NATIVE_NVFP4_BACKEND
    assert model[0].qat_recipe_backend == NATIVE_NVFP4_BACKEND
    assert model[0].qat_backend == EMULATED_QAT_BACKEND
    assert torch.isfinite(model(torch.randn(3, 128))).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_saved_native_qat_recipe_falls_back_for_non_bf16_cuda(dtype):
    native, reason = native_nvfp4_status()
    if not native:
        pytest.skip(reason)

    metadata = qat_metadata("nvfp4", backend="native", parameter_dtype="bf16")
    assert metadata is not None
    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    model.config = SimpleNamespace(**{LOW_PRECISION_METADATA_KEY: dict(metadata)})
    model = model.to(device="cuda", dtype=dtype)

    count, runtime = apply_saved_qat(model)

    assert count == 1
    assert runtime is not None
    assert runtime["runtime_backend"] == EMULATED_QAT_BACKEND
    assert model[0].qat_recipe_backend == NATIVE_NVFP4_BACKEND
    assert model[0].qat_backend == EMULATED_QAT_BACKEND
    value = torch.randn(3, 128, device="cuda", dtype=dtype)
    assert torch.isfinite(model(value)).all()


def test_direct_native_qat_application_remains_strict(monkeypatch):
    from yylab.train import low_precision

    metadata = qat_metadata("nvfp4", backend="emulated")
    assert metadata is not None
    metadata["backend"] = NATIVE_NVFP4_BACKEND
    metadata["backend_request"] = "native"
    metadata["native_low_bit_gemm"] = True
    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    model.config = SimpleNamespace()
    monkeypatch.setattr(
        low_precision,
        "native_nvfp4_status",
        lambda: (False, "native backend unavailable in test"),
    )

    with pytest.raises(RuntimeError, match="native NVFP4 QAT runtime is unavailable"):
        apply_qat(model, metadata)
    assert type(model[0]) is torch.nn.Linear


def test_paired_4_8_mask_is_deterministic_and_keeps_exactly_two_pairs():
    value = torch.arange(1, 33, dtype=torch.float32).reshape(2, 16)

    mask = structured_sparsity_mask(value, "paired_4:8")
    repeated = structured_sparsity_mask(value, "paired_4:8")

    assert torch.equal(mask, repeated)
    assert mask.shape == value.shape
    blocks = mask.to(torch.bool).reshape(-1, 8)
    pairs = blocks.reshape(-1, 4, 2)
    assert torch.equal(pairs[..., 0], pairs[..., 1])
    assert torch.all(pairs[..., 0].sum(dim=-1) == 2)
    assert mask.count_nonzero().item() * 2 == mask.numel()


def test_paired_4_8_fake_sparsity_uses_straight_through_gradients():
    value = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8).requires_grad_()
    gradient = torch.linspace(0.25, 4.0, value.numel()).reshape_as(value)

    sparse = fake_structured_sparsity(value, "paired_4:8")
    (sparse * gradient).sum().backward()

    assert value.grad is not None
    assert torch.equal(value.grad, gradient)
    assert sparse.count_nonzero().item() * 2 == sparse.numel()


def test_nvfp4_paired_4_8_metadata_and_modules_preserve_sparsity_contract():
    metadata = qat_metadata(
        "nvfp4",
        sparsity="paired_4:8",
        backend="emulated",
    )

    assert metadata is not None
    assert metadata["sparsity"] == "paired_4:8"
    assert metadata["sparsity_density"] == 0.5
    assert metadata["native_sparse_gemm"] is False
    assert metadata["compressed_sparse_checkpoint"] is False
    assert metadata["backend"] == "yylab_ste_emulation"

    model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False))
    model.config = SimpleNamespace()
    assert apply_qat(model, metadata) == 1
    qat_modules = [module for module in model.modules() if isinstance(module, QATLinear)]
    assert qat_modules
    assert all(module.qat_sparsity == "paired_4:8" for module in qat_modules)


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
    assert model[0].qat_backward_backend == "native"

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
    from yylab.train.potentiated import FP8_ADAPTIVE_MASTER

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
        master_weight_storage=FP8_ADAPTIVE_MASTER,
        adaptive_max_promoted_fraction=0.10,
    )
    input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")

    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    summary = optimizer.master_state_summary()

    assert torch.isfinite(loss)
    assert summary["format"] == FP8_ADAPTIVE_MASTER
    assert summary["moment_format"] == "fp32"
    assert float(summary["promoted_fraction"]) <= 0.10
    assert all(torch.isfinite(parameter.float()).all() for parameter in model.parameters())


class _OneBatch:
    docs_seen = 1
    tokens_seen = 8

    def __iter__(self):
        return iter([{"input_ids": [[4, 5, 6, 7, 8, 9, 4, 5]]}])


def test_run_base_scratch_end_to_end_and_reload(tmp_path, monkeypatch):
    from yylab.models import loading
    from yylab.workers import train as worker

    tokenizer_dir = _tokenizer_dir(tmp_path)
    out_id = db.new_id("ckpt")
    db.execute(
        "INSERT INTO checkpoints(id, name, created_at, kind, source, ref) VALUES(?,?,?,?,?,?)",
        (out_id, "tiny-scratch", db.now_iso(), "pending", "pending", ""),
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
            "qat": "int4",
            "qat_group_size": 16,
            "activation_qat": "int8",
        },
        "optimizer": {
            "implementation": "torch_adamw",
            "state": "gpu",
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
    assert summary["qat"] == "int4"
    assert summary["result"]["steps"] == 1
    assert (out / "model.safetensors").is_file()
    assert (out / "yylab_recipe.json").is_file()
    saved_config = json.loads((out / "config.json").read_text("utf-8"))
    assert saved_config["yylab_low_precision"]["format"] == "int4"
    assert saved_config["max_position_embeddings"] == 128

    model, tokenizer, cuda, quant, notes = loading.load_base(str(out), "fp32")
    assert not cuda
    assert quant == "fp32"
    assert len(tokenizer) > 0
    assert any(isinstance(module, QATLinear) for module in model.modules())
    assert any("restored int4 QAT" in note for note in notes)
    row = db.q1("SELECT * FROM checkpoints WHERE id=?", (out_id,))
    metadata = json.loads(row["meta_json"])
    assert row["kind"] == "base"
    assert metadata["initialization"] == "scratch"
    assert metadata["precision"]["qat"] == "int4"


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


def test_qat_base_auto_selects_high_precision_lora_instead_of_qlora():
    from yylab.ui.pages.train_mole import _auto_method_quant, _checkpoint_qat_format

    checkpoint = {"meta_json": json.dumps({"precision": {"qat": "int4"}})}
    assert _checkpoint_qat_format(checkpoint) == "int4"
    assert _auto_method_quant(checkpoint, "nf4") == ("lora", "bf16")


def test_base_ui_recipe_builder_keeps_independent_training_axes():
    from yylab.ui.pages.train_base import _base_run_config

    config = _base_run_config(
        initialization={"mode": "scratch", "tokenizer_ref": "tokenizer"},
        method="full",
        architecture=_tiny_spec().to_dict(),
        precision={"parameter_dtype": "bf16", "qat": "ternary"},
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
    )
    assert config["initialization"]["mode"] == "scratch"
    assert config["architecture"]["max_position_embeddings"] == 128
    assert config["precision"]["qat"] == "ternary"
    assert config["optimizer"]["state"] == "cpu_offload"
    assert config["scheduler"]["warmup_ratio"] == 0.03
    assert config["corpora"] == ["corpus-z", "corpus-a"]
    assert config["corpus_mix"] == "round_robin"
    assert config["seq_len"] == 128
    assert config["save_every"] == 20
    assert config["torch_compile"] is True


def test_base_ui_normalizes_legacy_single_corpus_and_preserves_multi_select_order():
    from yylab.ui.pages.train_base import _ordered_corpora

    assert _ordered_corpora("corpus-legacy") == ["corpus-legacy"]
    assert _ordered_corpora(["corpus-b", "corpus-a", "corpus-b"]) == [
        "corpus-b",
        "corpus-a",
    ]
    assert _ordered_corpora(None) == []
