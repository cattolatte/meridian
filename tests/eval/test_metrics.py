"""Tests for the binary-relevance retrieval metrics."""

from __future__ import annotations

import math

import pytest

from meridian.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k

_RANKED = ["a", "b", "c", "d", "e"]
_RELEVANT = frozenset({"b", "d"})


def test_recall_at_k() -> None:
    assert recall_at_k(_RANKED, _RELEVANT, 1) == 0.0  # 'a' not relevant
    assert recall_at_k(_RANKED, _RELEVANT, 2) == 0.5  # found 'b'
    assert recall_at_k(_RANKED, _RELEVANT, 4) == 1.0  # found 'b' and 'd'


def test_mrr_at_k_uses_first_relevant() -> None:
    assert mrr_at_k(_RANKED, _RELEVANT, 10) == 0.5  # first relevant at rank 2


def test_mrr_zero_when_no_relevant_in_cutoff() -> None:
    assert mrr_at_k(_RANKED, frozenset({"e"}), 3) == 0.0


def test_ndcg_perfect_ranking_is_one() -> None:
    ranked = ["b", "d", "a", "c"]
    assert math.isclose(ndcg_at_k(ranked, _RELEVANT, 10), 1.0)


def test_ndcg_matches_hand_computation() -> None:
    # relevant at ranks 2 and 4; ideal would place both at ranks 1 and 2.
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert math.isclose(ndcg_at_k(_RANKED, _RELEVANT, 10), dcg / idcg)


def test_metrics_reject_empty_relevant() -> None:
    empty: frozenset[str] = frozenset()
    with pytest.raises(ValueError):
        recall_at_k(_RANKED, empty, 5)
    with pytest.raises(ValueError):
        mrr_at_k(_RANKED, empty, 5)
    with pytest.raises(ValueError):
        ndcg_at_k(_RANKED, empty, 5)
