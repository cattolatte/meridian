"""Tests for building frozen eval splits from PubMedQA."""

from __future__ import annotations

import pytest

from meridian.eval.pubmedqa import build_eval_set_from_pubmedqa, split_dev_test

_PQA = {
    "18784090": {"QUESTION": "Does intensive glucose control reduce mortality?"},
    "30862451": {"QUESTION": "Is metformin better than sulfonylureas?"},
    "28199843": {"QUESTION": "Do beta-blockers help after MI?"},
    "10000004": {"QUESTION": "Does adjuvant chemotherapy improve survival?"},
}


def test_build_maps_each_entry_to_its_source_pmid() -> None:
    eval_set = build_eval_set_from_pubmedqa(_PQA, name="pqa")
    assert len(eval_set) == 4
    by_id = {q.query_id: q for q in eval_set.queries}
    assert by_id["18784090"].relevant_pmids == frozenset({"18784090"})
    assert by_id["30862451"].question == "Is metformin better than sulfonylureas?"


def test_split_is_disjoint_and_complete() -> None:
    eval_set = build_eval_set_from_pubmedqa(_PQA, name="pqa")
    dev, test = split_dev_test(eval_set)
    dev_ids = {q.query_id for q in dev.queries}
    test_ids = {q.query_id for q in test.queries}
    assert dev_ids.isdisjoint(test_ids)
    assert dev_ids | test_ids == set(_PQA)


def test_split_is_deterministic() -> None:
    eval_set = build_eval_set_from_pubmedqa(_PQA, name="pqa")
    a = split_dev_test(eval_set)
    b = split_dev_test(eval_set)
    assert [q.query_id for q in a[0].queries] == [q.query_id for q in b[0].queries]


def test_split_names() -> None:
    dev, test = split_dev_test(build_eval_set_from_pubmedqa(_PQA, name="pqa"))
    assert dev.name == "pqa-dev"
    assert test.name == "pqa-test"


def test_invalid_dev_fraction_rejected() -> None:
    eval_set = build_eval_set_from_pubmedqa(_PQA, name="pqa")
    with pytest.raises(ValueError):
        split_dev_test(eval_set, dev_fraction=1.0)
