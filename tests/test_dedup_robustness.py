from __future__ import annotations

import warnings

from yylab.ingest.dedup import dedup_fingerprints


def _python(text: str) -> dict:
    return dedup_fingerprints(text, mode="code", logical_path="src/module.py", language="python")


def test_python_source_with_null_bytes_gets_generic_structure_without_raising():
    """CPython 3.12's C tokenizer raises SystemError on NUL bytes; never let it escape."""

    fingerprints = _python("x = 1\x00\nprint(x)\n")
    assert fingerprints["parse_status"] == "invalid"
    assert "control_characters" in fingerprints["reason_codes"]
    assert any(code.startswith("python_parse_error:") for code in fingerprints["reason_codes"])
    assert fingerprints["token_count"] > 0
    assert fingerprints["structural_sha256"]


def test_deeply_nested_python_source_falls_back_instead_of_recursing_forever():
    fingerprints = _python("x = " + "(" * 20_000 + "1" + ")" * 20_000 + "\n")
    assert fingerprints["parse_status"] == "invalid"
    assert any(
        code.startswith(("python_parse_error:", "python_parse_failed:", "python_structure_failed:"))
        for code in fingerprints["reason_codes"]
    )


def test_valid_python_is_still_fingerprinted_structurally():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = _python("def add(left, right):\n    return left + right\n")
        renamed = _python("def plus(a, b):\n    return a + b\n")
    assert first["parse_status"] == renamed["parse_status"] == "valid"
    assert first["structural_sha256"] == renamed["structural_sha256"]
