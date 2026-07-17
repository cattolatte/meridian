"""Tests for error attribution and PubMedQA scoring."""

from __future__ import annotations

import pytest

from meridian.eval.attribution import UNATTRIBUTED, attribute_failure, attribution_study
from meridian.eval.pubmedqa import pubmedqa_accuracy


def test_attributes_to_earliest_fixing_stage() -> None:
    # Both retrieval and generation oracles fix it -> blame retrieval (earlier).
    assert attribute_failure({"retrieval": True, "generation": True}) == "retrieval"
    assert attribute_failure({"generation": True}) == "generation"
    assert attribute_failure({"verification": True}) == "verification"


def test_unattributed_when_no_oracle_fixes() -> None:
    assert attribute_failure({"retrieval": False, "generation": False}) == UNATTRIBUTED
    assert attribute_failure({}) == UNATTRIBUTED


def test_attribution_study_counts_and_fractions() -> None:
    cases = [
        {"retrieval": True},
        {"retrieval": True},
        {"generation": True},
        {},  # unattributed
    ]
    study = attribution_study(cases)
    assert study.total == 4
    assert study.counts["retrieval"] == 2
    assert study.counts["generation"] == 1
    assert study.counts[UNATTRIBUTED] == 1
    assert study.fraction("retrieval") == 0.5


def test_pubmedqa_accuracy() -> None:
    gold = {"a": "yes", "b": "no", "c": "maybe"}
    predictions = {"a": "yes", "b": "yes", "c": "MAYBE"}  # b wrong, c case-insensitive
    assert pubmedqa_accuracy(predictions, gold) == pytest.approx(2 / 3)


def test_pubmedqa_missing_prediction_is_wrong() -> None:
    assert pubmedqa_accuracy({}, {"a": "yes"}) == 0.0


def test_pubmedqa_empty_gold_rejected() -> None:
    with pytest.raises(ValueError):
        pubmedqa_accuracy({"a": "yes"}, {})
