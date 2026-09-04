from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yylab.models import precision_capabilities as capabilities
from yylab.models.families import hf_compatible_checkpoints


def test_cpu_dense_precision_is_conservatively_fp32_only() -> None:
    status = capabilities.dense_precision_status("cpu", force_refresh=True)

    assert status["bf16"][0] is False
    assert status["fp16"][0] is False
    assert status["fp32"][0] is True
    assert capabilities.available_dense_precisions("cpu") == ("fp32",)
    with pytest.raises(RuntimeError, match="BF16.*unavailable.*CPU"):
        capabilities.require_dense_precision("bf16", "cpu")


def test_dense_precision_uses_the_exact_selected_cuda_device(monkeypatch) -> None:
    observed: list[tuple[str, int | None]] = []

    monkeypatch.setattr(
        capabilities,
        "_resolve_device",
        lambda _device=None: (torch.device("cuda", 3), ""),
    )

    def probe(device_type: str, device_index: int | None):
        observed.append((device_type, device_index))
        return (
            ("bf16", False, "no native bf16"),
            ("fp16", True, "fp16 smoke passed"),
            ("fp32", True, "fp32 smoke passed"),
        )

    monkeypatch.setattr(capabilities, "_probe_dense_device", probe)

    assert capabilities.available_dense_precisions("cuda:3") == ("fp16", "fp32")
    assert observed == [("cuda", 3)]


def test_default_model_execution_device_is_cuda_zero(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert capabilities.model_execution_device() == torch.device("cuda", 0)


def test_generic_checkpoint_filter_excludes_native_spalmer_rows() -> None:
    rows = [
        {"id": "hf", "ref": "org/model", "meta_json": "{}"},
        {
            "id": "spalmer",
            "ref": "C:/models/spalmer.pt",
            "meta_json": '{"model_family":"spalmer"}',
        },
    ]

    assert [row["id"] for row in hf_compatible_checkpoints(rows)] == ["hf"]


def test_hf_loader_rejects_spalmer_before_tokenizer_or_model_load(
    tmp_path, monkeypatch
) -> None:
    import transformers

    from yylab.models.loading import load_base

    checkpoint = tmp_path / "spalmer.pt"
    checkpoint.write_bytes(b"metadata-only test marker")
    called = False

    def tokenizer_load(*_args, **_kwargs):
        nonlocal called
        called = True
        return SimpleNamespace()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_load)

    with pytest.raises(ValueError, match="not Hugging Face.*native SPALMER"):
        load_base(str(checkpoint), "fp32")
    assert called is False


def test_unknown_model_precision_fails_before_any_load(monkeypatch) -> None:
    with pytest.raises(ValueError, match="unsupported base precision"):
        capabilities.require_model_precision("made-up", "cpu")


def test_model_compute_dtype_matches_verified_dense_and_quantized_paths() -> None:
    assert capabilities.model_compute_dtype("bf16") is torch.bfloat16
    assert capabilities.model_compute_dtype("fp16") is torch.float16
    assert capabilities.model_compute_dtype("fp32") is torch.float32
    assert capabilities.model_compute_dtype("nf4") is torch.bfloat16
    assert capabilities.model_compute_dtype("int8") is torch.float16
    with pytest.raises(ValueError, match="unsupported base precision for adapter compute"):
        capabilities.model_compute_dtype("fp8")
