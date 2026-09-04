from __future__ import annotations

import sys
import types
from contextlib import nullcontext

import pytest
import torch

from yylab.models import bitsandbytes_capabilities as capabilities


def test_no_cuda_exposes_no_bitsandbytes_precision(monkeypatch) -> None:
    monkeypatch.setattr(capabilities.torch.cuda, "is_available", lambda: False)

    status = capabilities.bitsandbytes_quantization_status()

    assert capabilities.available_bitsandbytes_quantizations() == ()
    assert set(status) == {"nf4", "int8"}
    assert all(not available for available, _detail in status.values())


def test_available_formats_include_only_successful_operator_smokes(monkeypatch) -> None:
    monkeypatch.setattr(capabilities, "_resolved_cuda_index", lambda _device=None: (0, ""))
    monkeypatch.setattr(
        capabilities,
        "_probe_bitsandbytes_cuda",
        lambda _index: (
            ("nf4", True, "executed"),
            ("int8", False, "operator failed"),
        ),
    )

    assert capabilities.available_bitsandbytes_quantizations() == ("nf4",)
    with pytest.raises(RuntimeError, match="INT8 bitsandbytes loading is unavailable"):
        capabilities.require_bitsandbytes_quantization("int8")


def test_compute_precision_matches_the_real_bitsandbytes_operator_path() -> None:
    assert capabilities.bitsandbytes_compute_precision("nf4") == "bf16"
    assert capabilities.bitsandbytes_compute_precision("int8") == "fp16"
    with pytest.raises(ValueError, match="unknown bitsandbytes quantization"):
        capabilities.bitsandbytes_compute_precision("fp8")


def test_native_compute_gate_rejects_emulated_bf16_on_the_exact_device(
    monkeypatch,
) -> None:
    selected: list[int] = []
    checked: list[str] = []
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
    monkeypatch.setattr(
        capabilities.torch.cuda,
        "device",
        lambda index: selected.append(int(index)) or nullcontext(),
    )
    monkeypatch.setattr(
        capabilities.torch.cuda,
        "get_device_capability",
        lambda device: checked.append(str(device)) or (8, 0),
    )
    monkeypatch.setattr(
        capabilities.torch.cuda,
        "is_bf16_supported",
        lambda *, including_emulation: including_emulation,
    )

    available, detail = capabilities._probe_one("nf4", torch.device("cuda", 1))

    assert not available
    assert "no native BF16" in detail
    assert selected == [1]
    assert checked == ["cuda:1"]


def test_native_int8_compute_gate_checks_exact_device_fp16_capability(monkeypatch) -> None:
    checked: list[str] = []
    monkeypatch.setattr(
        capabilities.torch.cuda,
        "get_device_capability",
        lambda device: checked.append(str(device)) or (5, 2),
    )

    with pytest.raises(RuntimeError, match="no native FP16"):
        capabilities._require_native_compute_dtype(
            torch.device("cuda", 3),
            torch.float16,
        )

    assert checked == ["cuda:3"]
