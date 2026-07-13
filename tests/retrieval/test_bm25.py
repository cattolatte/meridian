"""Tests for the from-scratch BM25 index."""

from __future__ import annotations

import math

from meridian.retrieval.bm25 import BM25Index


def _toy_index(k1: float = 1.5, b: float = 0.75) -> BM25Index:
    corpus = [
        ("d1", ["heart", "failure", "study"]),
        ("d2", ["diabetes", "and", "heart", "disease"]),
        ("d3", ["cancer", "immunotherapy"]),
        ("d4", ["heart", "heart", "heart"]),  # term-frequency saturation case
    ]
    return BM25Index.build(corpus, k1=k1, b=b)


def test_len_is_document_count() -> None:
    assert len(_toy_index()) == 4


def test_idf_is_higher_for_rarer_terms() -> None:
    index = _toy_index()
    # "cancer" occurs in 1 doc, "heart" in 3 -> cancer has higher idf.
    assert index.idf["cancer"] > index.idf["heart"]


def test_idf_matches_formula() -> None:
    index = _toy_index()
    n = len(index)
    df_heart = 3
    expected = math.log(1.0 + (n - df_heart + 0.5) / (df_heart + 0.5))
    assert index.idf["heart"] == expected


def test_search_returns_only_matching_docs() -> None:
    index = _toy_index()
    hits = index.search(["cancer"], k=10)
    assert [doc_id for doc_id, _ in hits] == ["d3"]


def test_unknown_term_returns_nothing() -> None:
    assert _toy_index().search(["nonexistent"], k=5) == []


def test_top_k_limit() -> None:
    index = _toy_index()
    hits = index.search(["heart"], k=2)
    assert len(hits) == 2


def test_ranking_is_deterministic_and_score_sorted() -> None:
    index = _toy_index()
    hits = index.search(["heart", "disease"], k=10)
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
    # Determinism: repeated queries give identical order.
    assert index.search(["heart", "disease"], k=10) == hits


def test_bm25_score_matches_hand_computation() -> None:
    # Single-term query against a single-doc corpus; check the closed form.
    index = BM25Index.build([("d1", ["heart", "heart", "study"])], k1=1.5, b=0.75)
    ((doc_id, score),) = index.search(["heart"], k=1)
    n = 1
    df = 1
    idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
    tf, dl, avgdl, k1, b = 2, 3, 3.0, 1.5, 0.75
    denom = tf + k1 * (1.0 - b + b * dl / avgdl)
    expected = idf * tf * (k1 + 1.0) / denom
    assert doc_id == "d1"
    assert math.isclose(score, expected)


def test_b_zero_removes_length_normalization() -> None:
    index = _toy_index()
    # With b=0, a longer doc is not penalized for length.
    hits_b0 = dict(index.search(["heart"], k=10, b=0.0))
    hits_b1 = dict(index.search(["heart"], k=10, b=1.0))
    assert hits_b0 != hits_b1  # tuning b changes scores without a rebuild
