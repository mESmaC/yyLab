from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from yylab.ingest.adapters.loader import walk_files


def _relative_assets(root: Path, **kwargs: float) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in walk_files(root, **kwargs)
    }


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for this regression test")
def test_walk_files_includes_gitignored_dataset_assets(tmp_path: Path) -> None:
    root = tmp_path / "corpus-repository"
    root.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(root)],
        check=True,
        capture_output=True,
    )
    (root / ".gitignore").write_text("dataset/\n", encoding="utf-8")
    dataset = root / "dataset"
    dataset.mkdir()
    (dataset / "records.jsonl").write_text(
        '{"text":"ignored by Git, but valid corpus data"}\n',
        encoding="utf-8",
    )

    assets = _relative_assets(root)

    assert "dataset/records.jsonl" in assets
    assert not any(path == ".git" or path.startswith(".git/") for path in assets)


def test_walk_files_returns_assets_larger_than_legacy_size_limit(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    oversized = root / "large.jsonl"
    oversized.write_bytes(b"x" * 4096)

    assets = _relative_assets(root, max_file_mb=0.0001)

    assert oversized.name in assets
