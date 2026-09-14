"""Strict complete-row cleanup must not become a fuzzy content filter."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from yylab.train.sprint_row_filter import (
    SAMPLE_LIMIT,
    VERSION,
    QuotaRowFilter,
    normalize_row_filter,
    redundant_eightgram_fraction,
)


def test_low_information_rejects_constant_and_dna_but_keeps_distinct_and_moderate_repetition(tmp_path):
    distinct = np.arange(512, dtype="<u2")
    moderate = np.tile(np.arange(128, dtype="<u2"), 4)
    dna = np.tile(np.array([11, 12, 13, 14], dtype="<u2"), 128)
    rows = np.stack([distinct, np.zeros(512, dtype="<u2"), dna, moderate])
    assert redundant_eightgram_fraction(moderate) < 0.9
    with QuotaRowFilter(tmp_path / "state.db") as filtering:
        assert filtering.filter_batch(rows, source="a", row_offset=0).tolist() == [True, False, False, True]
        stats = filtering.source_receipt("a")
        assert stats["input_rows"] == 4
        assert stats["retained_rows"] == stats["rejected_rows"] == 2
        assert stats["retained_token_slots"] == stats["rejected_token_slots"] == 1024
        assert stats["reason_counts"] == {"low_information_8gram": 2, "exact_duplicate_row": 0}
        assert stats["rejected_samples"][0]["original_pool_row"] == 1
        assert stats["rejected_samples"][0]["sha256"] == hashlib.sha256(rows[1].tobytes()).hexdigest()


@pytest.mark.parametrize("length", [8, 128, 256, 512, 1024, 2048])
def test_exact_row_dedup_is_global_and_length_independent(tmp_path, length):
    a = np.arange(length, dtype="<u2")
    b = a + 1
    with QuotaRowFilter(tmp_path / "state.db") as filtering:
        assert filtering.filter_batch(np.stack([a, b]), source="a", row_offset=0).tolist() == [True, True]
        assert filtering.filter_batch(np.stack([b, a]), source="b", row_offset=0).tolist() == [False, False]
        stats = filtering.source_receipt("b")
        assert stats["reason_counts"]["exact_duplicate_row"] == 2
        assert stats["rejected_samples"][0]["duplicate_of"] == {"source": "a", "original_pool_row": 1}


def test_rows_under_256_are_exempt_from_low_information_gate(tmp_path):
    with QuotaRowFilter(tmp_path / "state.db") as filtering:
        assert filtering.filter_batch(np.zeros((1, 128), dtype="<u2"), source="a", row_offset=0).tolist() == [True]


def test_numerical_word_problem_variants_are_not_fuzzy_deduplicated(tmp_path):
    # The same tokenized explanatory structure with one changed quantity is a
    # different exact row. The gate deliberately does not normalize numbers.
    rows = np.tile(np.arange(512, dtype="<u2"), (3, 1))
    rows[1, 100] = 900
    rows[2, 100] = 901
    with QuotaRowFilter(tmp_path / "state.db") as filtering:
        assert filtering.filter_batch(rows, source="math", row_offset=0).tolist() == [True, True, True]


def test_receipts_are_bounded_and_chunk_offsets_are_checked(tmp_path):
    with QuotaRowFilter(tmp_path / "state.db") as filtering:
        filtering.filter_batch(np.zeros((20, 256), dtype="<u2"), source="a", row_offset=0)
        filtering.filter_batch(np.zeros((20, 256), dtype="<u2"), source="a", row_offset=20)
        stats = filtering.source_receipt("a")
        assert stats["input_rows"] == stats["rejected_rows"] == 40
        assert len(stats["rejected_samples"]) == SAMPLE_LIMIT
        with pytest.raises(ValueError, match="contiguous"):
            filtering.filter_batch(np.zeros((1, 256), dtype="<u2"), source="a", row_offset=0)


def test_exact_eightgram_fraction_matches_python_tuple_reference():
    for length in (8, 128, 256, 512):
        row = np.random.default_rng(length).integers(0, 3, size=length, dtype=np.uint16)
        tuples = [tuple(row[i:i + 8]) for i in range(length - 7)]
        assert redundant_eightgram_fraction(row) == (len(tuples) - len(set(tuples))) / len(tuples)


def test_policy_defaults_disabled_and_version_validation():
    assert normalize_row_filter(None) is None
    assert normalize_row_filter({"enabled": False}) is None
    assert normalize_row_filter({"enabled": True}) == {"enabled": True, "version": VERSION}
    for value in (True, {}, {"enabled": 1}, {"enabled": True, "version": "future"}, {"enabled": True, "threshold": 0.5}):
        with pytest.raises(ValueError):
            normalize_row_filter(value)


def test_state_database_must_be_new(tmp_path):
    path = tmp_path / "state.db"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="must be new"):
        QuotaRowFilter(path)


def test_filter_does_not_compare_bytes_across_different_token_shapes(tmp_path):
    with QuotaRowFilter(tmp_path / "state.db") as filtering:
        filtering.filter_batch(np.arange(128, dtype="<u2")[None, :], source="a", row_offset=0)
        with pytest.raises(ValueError, match="remain constant"):
            filtering.filter_batch(np.arange(256, dtype="<u2")[None, :], source="b", row_offset=0)
        with pytest.raises(ValueError, match="remain constant"):
            filtering.filter_batch(np.arange(128, dtype="<u4")[None, :], source="c", row_offset=0)
