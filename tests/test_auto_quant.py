import json

import pytest

from yylab.ui.pages.train_mole import _auto_method_quant


def test_auto_method_respects_existing_checkpoint_quant_lock() -> None:
    assert _auto_method_quant({"quant_locked": "nf4"}, "int8") == ("qlora", "nf4")
    assert _auto_method_quant({"quant_locked": "int8"}, "nf4") == ("qlora", "int8")
    assert _auto_method_quant({"quant_locked": "bf16"}, "nf4") == ("lora", "bf16")
    assert _auto_method_quant({"quant_locked": "fp16"}, "nf4") == ("lora", "fp16")
    assert _auto_method_quant({"quant_locked": "fp32"}, "nf4") == ("lora", "fp32")
    assert _auto_method_quant({"quant_locked": ""}, "nf4") == ("auto", "nf4")


def test_auto_method_rejects_unknown_checkpoint_precision_lock() -> None:
    with pytest.raises(ValueError, match="unsupported precision lock 'fp8'"):
        _auto_method_quant({"quant_locked": "fp8"}, "bf16")


def test_auto_method_requires_bf16_for_saved_native_qat() -> None:
    checkpoint = {"meta_json": json.dumps({"precision": {"qat": "nvfp4"}})}
    assert _auto_method_quant(checkpoint, "fp16") == ("lora", "bf16")

    checkpoint["quant_locked"] = "fp16"
    with pytest.raises(ValueError, match="saved native QAT requires BF16"):
        _auto_method_quant(checkpoint, "bf16")


def test_auto_method_rejects_unknown_requested_precision() -> None:
    with pytest.raises(ValueError, match="unsupported requested model precision"):
        _auto_method_quant(None, "fp8")
