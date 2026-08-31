from yylab.ui.pages.train_mole import _auto_method_quant


def test_auto_method_respects_existing_checkpoint_quant_lock() -> None:
    assert _auto_method_quant({"quant_locked": "nf4"}, "int8") == ("qlora", "nf4")
    assert _auto_method_quant({"quant_locked": "int8"}, "nf4") == ("qlora", "int8")
    assert _auto_method_quant({"quant_locked": "bf16"}, "nf4") == ("lora", "bf16")
    assert _auto_method_quant({"quant_locked": ""}, "nf4") == ("auto", "nf4")
