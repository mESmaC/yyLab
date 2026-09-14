"""Bounded, read-only presentation of Sprint's saved generation checks."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MAX_HEALTH_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_HEALTH_OUTPUTS = 16
STATUS_LABELS = {
    "repetition_detected": "Generated text is repeating",
    "no_repetition_detected": "No repetition detected in these short samples",
    "insufficient_length": "Samples are too short to judge repetition",
    "nonfinite_logits": "Generated text check failed",
}
QUALITY_LIMITATION = "Short samples help reveal problems. Passing this check does not establish model quality."


def _bounded_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > MAX_HEALTH_RECEIPT_BYTES:
            return None
        with path.open("rb") as stream:
            raw = stream.read(MAX_HEALTH_RECEIPT_BYTES + 1)
        if len(raw) > MAX_HEALTH_RECEIPT_BYTES:
            return None
        result = json.loads(raw)
    except (OSError, ValueError):
        return None
    return result if isinstance(result, dict) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def _count(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number >= 0 and number.is_integer() else None


def sprint_health_status(value: Any) -> str:
    return STATUS_LABELS.get(str(value or ""), "")


def _gradient_text(progress: Mapping[str, Any]) -> str:
    norm = _number(progress.get("gradient_norm"))
    if norm is None:
        return ""
    parts = [f"Gradient norm before clipping: {norm:.3f}"]
    mean = _number(progress.get("gradient_norm_mean"))
    if mean is not None:
        parts.append(f"mean {mean:.3f}")
    fraction, count = _number(progress.get("gradient_clip_fraction")), _count(progress.get("gradient_norm_steps"))
    if fraction is not None and 0 <= fraction <= 1 and count:
        parts.append(f"clipped {fraction:.0%} of the last {count:,} updates")
    return " · ".join(parts)


def sprint_health_evidence(run_path: str | Path, *, fallback: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Show completed checks for running, stopped, and completed runs alike.

    Receipt filenames come from the saved step, not arbitrary paths embedded
    in progress metadata. Text is returned literally for plain-label rendering.
    """

    root = Path(run_path).resolve()
    saved = dict(fallback) if isinstance(fallback, Mapping) else {}
    progress = _bounded_json(root / "progress.json") or _bounded_json(root / "summary.json") or saved
    step = _count(progress.get("generation_eval_step"))
    status = str(progress.get("generation_eval_status") or "")
    summary = progress.get("generation_eval_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    parts = [f"Check at step {step:,}"] if step is not None else []
    total, flagged, skipped = (_count(summary.get(name)) for name in
                               ("outputs", "repetition_flagged_outputs", "skipped_outputs"))
    if total is not None and flagged is not None:
        parts.append(f"{flagged} of {total} samples flagged for repetition")
    if skipped:
        parts.append(f"{skipped} skipped")
    result = {
        "status": status,
        "status_text": sprint_health_status(status) or ("Generated text check saved" if step is not None
                                                         else "No generated text check has completed yet"),
        "warning": status in {"repetition_detected", "nonfinite_logits"},
        "step": step, "details": " · ".join(parts), "gradient_text": _gradient_text(progress),
        "limitations": QUALITY_LIMITATION, "samples": [], "receipt_path": None, "receipt_note": "",
    }
    if step is None:
        return result
    path = root / "generation-eval" / f"step-{step:08d}.json"
    receipt = _bounded_json(path) if path.resolve().is_relative_to(root) else None
    if receipt is None or receipt.get("step") != step or not isinstance(receipt.get("outputs"), list):
        result["receipt_note"] = "Saved sample details are unavailable; the last check status is shown above."
        return result
    result["receipt_path"] = str(path)
    for index, output in enumerate(receipt["outputs"][:MAX_HEALTH_OUTPUTS]):
        if not isinstance(output, Mapping):
            continue
        repetition = output.get("repetition")
        repetition = repetition if isinstance(repetition, Mapping) else {}
        sample_status = (
            "Repetition detected" if repetition.get("flagged") else
            "Skipped" if output.get("status") == "skipped" else
            "Check failed" if output.get("status") == "failed" else
            "Too short to judge" if not repetition.get("sufficient_length") else
            "No repetition detected"
        )
        result["samples"].append({
            "name": str(output.get("name") or f"Sample {index + 1}"),
            "mode": {"greedy": "Greedy", "sample": "Sampled"}.get(str(output.get("mode")), ""),
            "status": sample_status, "prompt": str(output.get("prompt") or ""),
            "text": str(output.get("generated_text") or ""),
            "stop_reason": str(output.get("stop_reason") or ""),
        })
    if len(receipt["outputs"]) > MAX_HEALTH_OUTPUTS:
        result["receipt_note"] = "Showing the first 16 samples; download check details for the complete record."
    return result
