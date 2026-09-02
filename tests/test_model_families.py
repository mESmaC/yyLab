from __future__ import annotations

import json

from yylab.models.families import checkpoint_family, family_from_reference


def test_legacy_checkpoint_defaults_to_hf(tmp_path):
    folder = tmp_path / "hf"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")

    assert checkpoint_family({"ref": str(folder), "meta_json": "{}"}) == "hf"


def test_explicit_spalmer_metadata_is_authoritative(tmp_path):
    folder = tmp_path / "artifact"
    folder.mkdir()

    row = {
        "ref": str(folder),
        "meta_json": json.dumps({"model_family": "spalmer"}),
    }
    assert checkpoint_family(row) == "spalmer"


def test_spalmer_bundle_layout_is_recognized(tmp_path):
    folder = tmp_path / "bundle"
    folder.mkdir()
    checkpoint = folder / "spalmer.pt"
    checkpoint.write_bytes(b"format validation happens at load time")

    assert family_from_reference(folder) == "spalmer"
    assert family_from_reference(checkpoint) == "spalmer"


def test_unsupported_spalmer_like_layouts_are_not_detected(tmp_path):
    manifest_only = tmp_path / "manifest-only"
    manifest_only.mkdir()
    (manifest_only / "spalmer_manifest.json").write_text("{}", encoding="utf-8")
    legacy_suffix = tmp_path / "spalmer.pth"
    legacy_suffix.write_bytes(b"unsupported")

    assert family_from_reference(manifest_only) == "hf"
    assert family_from_reference(legacy_suffix) == "hf"
