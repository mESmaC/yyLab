from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

HF_MODEL_FAMILY = "hf"
SPALMER_MODEL_FAMILY = "spalmer"


def checkpoint_family(checkpoint: Mapping[str, Any] | None) -> str:
    """Return the explicit model family recorded on a checkpoint row.

    Older yyLab rows predate model-family metadata and are therefore Hugging
    Face checkpoints.  File-layout detection is deliberately only a fallback
    for caller-owned local references; registry metadata remains authoritative.
    """

    if not checkpoint:
        return HF_MODEL_FAMILY
    direct = str(checkpoint.get("model_family") or "").strip().casefold()
    if direct:
        return _normalize_family(direct)
    try:
        metadata = json.loads(str(checkpoint.get("meta_json") or "{}"))
    except (TypeError, ValueError):
        metadata = {}
    if isinstance(metadata, Mapping):
        declared = str(
            metadata.get("model_family")
            or metadata.get("family")
            or metadata.get("architecture_type")
            or ""
        ).strip().casefold()
        if declared:
            return _normalize_family(declared)
    return family_from_reference(checkpoint.get("ref") or checkpoint.get("path"))


def family_from_reference(reference: object) -> str:
    """Recognize a local SPALMER bundle without importing model code."""

    raw = str(reference or "").strip()
    if not raw:
        return HF_MODEL_FAMILY
    path = Path(raw).expanduser()
    if path.is_dir() and (path / "spalmer.pt").is_file():
        return SPALMER_MODEL_FAMILY
    if path.is_file() and path.suffix.casefold() == ".pt":
        # A .pt file still has to pass the SPALMER format check at load time.
        return SPALMER_MODEL_FAMILY
    return HF_MODEL_FAMILY


def is_spalmer_checkpoint(checkpoint: Mapping[str, Any] | None) -> bool:
    return checkpoint_family(checkpoint) == SPALMER_MODEL_FAMILY


def hf_compatible_checkpoints(
    checkpoints: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Filter generic Hugging Face workflows away from native-only families."""

    return [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint_family(checkpoint) == HF_MODEL_FAMILY
    ]


def _normalize_family(value: str) -> str:
    if value in {"spalmer", "spalmer_v1"}:
        return SPALMER_MODEL_FAMILY
    return HF_MODEL_FAMILY


__all__ = [
    "HF_MODEL_FAMILY",
    "SPALMER_MODEL_FAMILY",
    "checkpoint_family",
    "family_from_reference",
    "hf_compatible_checkpoints",
    "is_spalmer_checkpoint",
]
