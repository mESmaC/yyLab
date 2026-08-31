from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import load_file

from yylab.ingest.ranker import (
    CONTINUOUS_FEATURES,
    FORMAT_VERSION,
    RankerPersistenceError,
    RankerTrainingError,
    Stage1Ranker,
    deterministic_validation_split,
    featurize,
    grouped_cross_validation,
    load_ranker_if_available,
    train_ranker,
)


def _records() -> list[dict]:
    records = []
    for index in range(24):
        group = f"corpus-{index % 6}"
        records.append(
            {
                "text": (
                    f"This carefully edited explanation {index} uses evidence, complete sentences, and a "
                    "specific example to teach readers how a reliable technical system works. The argument "
                    "is organized clearly and gives enough context to remain useful during model training."
                ),
                "verdict": "good",
                "domain": "science",
                "lang": "en",
                "quality": 1.0,
                "corpus_id": group,
            }
        )
        records.append(
            {
                "text": (f"BUY NOW {index} !!! https://spam.invalid/click " * 24),
                "verdict": "bad",
                "domain": "web",
                "lang": "en",
                "quality": 0.05,
                "quarantine_reasons": ["url_heavy", "uppercase_heavy", "bigram_repetition"],
                "corpus_id": group,
            }
        )
    return records


def test_featurize_is_fixed_deterministic_and_accepts_quality_report():
    text = "LINE ONE!!!\nLINE ONE!!!\n\nLINE ONE!!!\nLINE ONE!!! https://example.test"
    report = {"score": 0.25, "reasons": ["duplicate_lines", "url_heavy"]}

    first = featurize(text, report)
    second = featurize(text, report)

    assert tuple(first) == CONTINUOUS_FEATURES
    assert first == second
    assert all(np.isfinite(value) for value in first.values())
    assert first["quality_score"] == 0.25
    assert first["quality_reason_count"] == 2.0
    assert first["duplicate_line_char_fraction"] > 0.0
    assert first["url_word_fraction"] > 0.0


def test_train_ranker_accepts_iterable_and_learns_usefulness_order():
    ranker = train_ranker((record for record in _records()), seed=17)
    good, bad = _records()[:2]

    good_score = ranker.score_doc(good)
    bad_score = ranker.score_doc(bad)

    assert good_score["rank_score"] > 0.9
    assert bad_score["rank_score"] < 0.1
    assert good_score["rank_score"] > bad_score["rank_score"]
    assert 0.0 <= good_score["uncertainty"] <= 1.0
    assert good_score["model_version"] == ranker.model_version
    assert ranker.predict(good["text"], domain="science", lang="en", quality_report=1.0)
    assert ranker.metrics["holdout"]["available"] is True
    assert ranker.metrics["holdout"]["strategy"] == "stratified_group"


def test_training_is_deterministic():
    first = train_ranker(_records(), seed=9)
    second = train_ranker(_records(), seed=9)

    assert first.model_hash == second.model_hash
    assert first.metrics == second.metrics
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.continuous_mean, second.continuous_mean)
    np.testing.assert_array_equal(first.continuous_scale, second.continuous_scale)


def test_grouped_cross_validation_scores_every_document_without_leaking_groups():
    metrics = grouped_cross_validation(_records(), l2=0.1)

    assert metrics["strategy"] == "leave_one_group_out"
    assert metrics["groups"] == 6
    assert metrics["overall"]["count"] == len(_records())
    assert metrics["overall"]["roc_auc"] > 0.95
    assert sum(fold["count"] for fold in metrics["folds"]) == len(_records())


def test_score_docs_returns_queue_metadata_without_mutating_documents():
    ranker = train_ranker(_records(), validation_fraction=0)
    documents = [dict(record) for record in _records()[:4]]
    originals = [dict(document) for document in documents]

    scores = ranker.score_docs(documents)

    assert documents == originals
    assert len(scores) == len(documents)
    assert all(set(score) == {"rank_score", "uncertainty", "model_version"} for score in scores)
    assert all(0.0 <= float(score["rank_score"]) <= 1.0 for score in scores)


def test_partial_stored_quality_does_not_become_a_missingness_feature():
    ranker = train_ranker(_records(), validation_fraction=0)
    document = {
        "text": "A complete and carefully edited technical explanation with useful supporting detail. " * 4,
        "domain": "science",
        "lang": "en",
    }

    baseline = ranker.score_doc(document)
    partial = ranker.score_doc({**document, "quality": None, "quarantine_reasons": []})
    stale = ranker.score_doc({**document, "quality": 0.01})

    assert partial == baseline
    assert stale == baseline


def test_training_rejects_single_class_and_insufficient_labels():
    one_class = [
        {"text": f"Useful document number {index} with enough prose.", "verdict": "good"}
        for index in range(8)
    ]
    too_few = [
        {"text": f"Document number {index} with enough prose.", "label": index % 2}
        for index in range(6)
    ]
    imbalanced = [
        {"text": f"Document number {index} with enough prose.", "label": int(index > 0)}
        for index in range(8)
    ]

    with pytest.raises(RankerTrainingError, match="single class"):
        train_ranker(one_class)
    with pytest.raises(RankerTrainingError, match="at least 8"):
        train_ranker(too_few)
    with pytest.raises(RankerTrainingError, match="each class"):
        train_ranker(imbalanced)


def test_grouped_split_is_deterministic_stratified_and_has_no_group_leakage():
    labels = [value for _ in range(8) for value in (0, 1)]
    groups = [f"corpus-{index}" for index in range(8) for _ in range(2)]

    first = deterministic_validation_split(labels, groups=groups, validation_fraction=0.25, seed=42)
    second = deterministic_validation_split(labels, groups=groups, validation_fraction=0.25, seed=42)

    assert first == second
    assert first.strategy == "stratified_group"
    train_groups = {groups[index] for index in first.train_indices}
    validation_groups = {groups[index] for index in first.validation_indices}
    assert train_groups.isdisjoint(validation_groups)
    assert {labels[index] for index in first.train_indices} == {0, 1}
    assert {labels[index] for index in first.validation_indices} == {0, 1}


def test_grouped_split_never_silently_falls_back_to_document_split():
    labels = [0, 0, 1, 1]
    groups = ["all-bad", "all-bad", "all-good", "all-good"]

    with pytest.raises(RankerTrainingError, match="no grouped holdout"):
        deterministic_validation_split(labels, groups=groups, validation_fraction=0.5)


def test_save_load_round_trip_includes_versioned_schema_scaler_metrics_and_hash(tmp_path):
    ranker = train_ranker(_records(), seed=3)
    weights_path, metadata_path = ranker.save(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tensors = load_file(str(weights_path))

    assert metadata["format_version"] == FORMAT_VERSION
    assert metadata["model_hash"] == ranker.model_hash
    assert metadata["training_config"]["training_data_hash"] == ranker.training_config["training_data_hash"]
    assert len(metadata["training_config"]["training_data_hash"]) == 64
    assert metadata["feature_schema"]["continuous"] == list(CONTINUOUS_FEATURES)
    assert metadata["feature_schema"]["categorical"] == ["domain", "lang"]
    assert set(metadata["scaler"]) == {"mean", "scale"}
    assert set(metadata["categories"]) == {"domain", "lang"}
    assert "holdout" in metadata["metrics"]
    assert int(tensors["format_version"][0]) == FORMAT_VERSION

    loaded = Stage1Ranker.load(tmp_path)
    document = _records()[0]
    assert loaded.model_hash == ranker.model_hash
    assert loaded.model_version == ranker.model_version
    assert loaded.score_doc(document) == ranker.score_doc(document)


def test_load_validates_hash_and_optional_loader_only_ignores_absence(tmp_path):
    assert load_ranker_if_available(tmp_path) is None
    ranker = train_ranker(_records(), validation_fraction=0)
    _, metadata_path = ranker.save(tmp_path)
    assert load_ranker_if_available(tmp_path) is not None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_hash"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RankerPersistenceError, match="model hash"):
        load_ranker_if_available(tmp_path)


def test_partial_artifact_is_reported_as_corrupt_not_missing(tmp_path):
    (tmp_path / "ranker.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RankerPersistenceError, match="incomplete"):
        load_ranker_if_available(tmp_path)
