from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

from yylab.ingest import dedup
from yylab.ingest.dedup import dedup_fingerprints

CODE_WITH_BAD_ESCAPES = 'pattern = "\\d+\\s*"\nother = "path\\_name"\n'


def test_python_structure_of_corpus_code_does_not_emit_syntax_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")  # even a permissive global policy must not leak them
        fingerprints = dedup_fingerprints(
            CODE_WITH_BAD_ESCAPES,
            mode="code",
            logical_path="third_party/module.py",
            language="python",
        )
    assert fingerprints["parse_status"] == "valid"
    assert not [w for w in caught if issubclass(w.category, SyntaxWarning)]


def test_worker_process_stays_silent_on_bad_escapes(tmp_path):
    """The filter must hold in a fresh interpreter, which is what workers are."""

    root = Path(dedup.__file__).resolve().parents[2]
    script = (
        "from yylab.ingest.dedup import dedup_fingerprints\n"
        f"dedup_fingerprints({CODE_WITH_BAD_ESCAPES!r}, mode='code', logical_path='m.py', language='python')\n"
        "print('done')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(root), "YYLAB_HOME": str(tmp_path / "home")},
    )
    assert result.returncode == 0, result.stderr
    assert "SyntaxWarning" not in result.stderr
