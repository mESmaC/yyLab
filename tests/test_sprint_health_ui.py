from __future__ import annotations

import json

import pytest

from yylab.ui.sprint_health import MAX_HEALTH_RECEIPT_BYTES, sprint_health_evidence, sprint_health_status


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _progress(status="repetition_detected"):
    return {"stage": "complete", "generation_eval_step": 500, "generation_eval_status": status,
            "generation_eval_summary": {"outputs": 2, "repetition_flagged_outputs": 1, "skipped_outputs": 0},
            "gradient_norm": 1.23, "gradient_norm_mean": 1.11,
            "gradient_clip_fraction": 0.75, "gradient_norm_steps": 20}


def _receipt():
    return {"step": 500, "outputs": [
        {"name": "prose", "mode": "greedy", "status": "complete", "prompt": "The morning ",
         "generated_text": "  rain fell.\nrain fell.\n<script>literal sample</script>", "stop_reason": "max_new_tokens",
         "repetition": {"flagged": True, "sufficient_length": True}},
        {"name": "prose", "mode": "sample", "status": "complete", "prompt": "The morning ",
         "generated_text": "rain stopped.", "repetition": {"flagged": False, "sufficient_length": False}},
    ]}


def test_completed_run_keeps_warning_real_samples_and_gradient_window(tmp_path):
    _save(tmp_path / "progress.json", _progress())
    _save(tmp_path / "generation-eval" / "step-00000500.json", _receipt())
    evidence = sprint_health_evidence(tmp_path)
    assert evidence["warning"]
    assert evidence["status_text"] == "Generated text is repeating"
    assert evidence["details"] == "Check at step 500 · 1 of 2 samples flagged for repetition"
    assert "before clipping: 1.230" in evidence["gradient_text"]
    assert "mean 1.110" in evidence["gradient_text"]
    assert "clipped 75% of the last 20 updates" in evidence["gradient_text"]
    assert evidence["samples"][0]["text"] == _receipt()["outputs"][0]["generated_text"]
    assert evidence["samples"][0]["prompt"] == "The morning "
    assert evidence["samples"][0]["status"] == "Repetition detected"
    assert evidence["samples"][1]["status"] == "Too short to judge"
    assert evidence["receipt_path"].endswith("step-00000500.json")
    assert "does not establish model quality" in evidence["limitations"]


@pytest.mark.parametrize("status,fragment,warning", [
    ("no_repetition_detected", "these short samples", False),
    ("insufficient_length", "too short", False),
    ("nonfinite_logits", "failed", True),
])
def test_status_labels_do_not_claim_model_acceptance(tmp_path, status, fragment, warning):
    evidence = sprint_health_evidence(tmp_path, fallback=_progress(status))
    assert fragment in evidence["status_text"]
    assert evidence["warning"] is warning
    assert "approved" not in evidence["status_text"]
    assert "ready" not in evidence["status_text"]
    assert sprint_health_status(status) == evidence["status_text"]


def test_missing_or_bad_receipt_retains_status_without_reading_arbitrary_metadata_path(tmp_path):
    secret = tmp_path.parent / "private-evaluation.json"
    _save(secret, _receipt())
    progress = {**_progress(), "generation_eval_path": str(secret)}
    _save(tmp_path / "progress.json", progress)
    evidence = sprint_health_evidence(tmp_path)
    assert evidence["warning"] and evidence["samples"] == []
    assert evidence["receipt_path"] is None
    assert "unavailable" in evidence["receipt_note"]
    _save(tmp_path / "generation-eval" / "step-00000500.json", {**_receipt(), "step": 400})
    assert sprint_health_evidence(tmp_path)["samples"] == []


def test_oversize_receipt_is_bounded_and_older_summary_can_supply_status(tmp_path):
    _save(tmp_path / "summary.json", _progress())
    _save(tmp_path / "generation-eval" / "step-00000500.json",
          {"step": 500, "outputs": [], "extra": "x" * MAX_HEALTH_RECEIPT_BYTES})
    evidence = sprint_health_evidence(tmp_path)
    assert evidence["step"] == 500
    assert evidence["warning"] and evidence["samples"] == []
    assert evidence["receipt_path"] is None


def test_receipt_preview_limit_preserves_full_download(tmp_path):
    _save(tmp_path / "progress.json", _progress())
    _save(tmp_path / "generation-eval" / "step-00000500.json", {"step": 500, "outputs": _receipt()["outputs"] * 10})
    evidence = sprint_health_evidence(tmp_path)
    assert len(evidence["samples"]) == 16
    assert "first 16" in evidence["receipt_note"]
    assert evidence["receipt_path"] is not None


def test_unstarted_or_malformed_health_metadata_remains_nonblocking(tmp_path):
    evidence = sprint_health_evidence(tmp_path, fallback=[])
    assert "No generated text check" in evidence["status_text"]
    _save(tmp_path / "progress.json", {"generation_eval_step": True, "generation_eval_summary": [],
                                      "gradient_norm": float("nan")})
    evidence = sprint_health_evidence(tmp_path)
    assert evidence["samples"] == [] and evidence["gradient_text"] == ""
    assert evidence["step"] is None
