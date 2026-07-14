"""Tests for RRF hybrid retrieval."""

from __future__ import annotations

import pytest

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.retrieval.hybrid import HybridRetriever
from meridian.retrieval.pipeline import RetrievalHit, Retriever

_DOCS = [Document(pmid=str(i), title=f"t{i}", abstract="a") for i in range(1, 6)]


class _FixedRetriever:
    """A retriever that returns a fixed ranking of PMIDs (score = -rank)."""

    def __init__(self, order: list[str]) -> None:
        self._order = order

    def retrieve(self, query: str, *, k: int = 10) -> list[RetrievalHit]:
        return [RetrievalHit(pmid=p, score=-i) for i, p in enumerate(self._order[:k])]


def _store() -> SqliteDocumentStore:
    store = SqliteDocumentStore(":memory:")
    store.add_many(_DOCS)
    return store


def test_requires_a_component() -> None:
    with _store() as store, pytest.raises(ValueError):
        HybridRetriever([], store)


def test_satisfies_protocol() -> None:
    with _store() as store:
        hybrid = HybridRetriever([_FixedRetriever(["1"])], store)
        assert isinstance(hybrid, Retriever)


def test_rrf_rewards_agreement() -> None:
    # "3" is ranked highly by both retrievers; "1" and "5" only by one each.
    a = _FixedRetriever(["1", "3", "2"])
    b = _FixedRetriever(["5", "3", "4"])
    with _store() as store:
        hybrid = HybridRetriever([a, b], store, k_rrf=60)
        top = hybrid.retrieve("q", k=1)[0]
        assert top.pmid == "3"


def test_rrf_score_matches_formula() -> None:
    a = _FixedRetriever(["1", "2"])
    b = _FixedRetriever(["2", "1"])
    with _store() as store:
        hybrid = HybridRetriever([a, b], store, k_rrf=60)
        hits = {h.pmid: h.score for h in hybrid.retrieve("q", k=2)}
        # Each of "1","2" appears at ranks 1 and 2 across the two retrievers.
        expected = 1.0 / (60 + 1) + 1.0 / (60 + 2)
        assert hits["1"] == pytest.approx(expected)
        assert hits["2"] == pytest.approx(expected)


def test_attaches_documents_and_is_deterministic() -> None:
    a = _FixedRetriever(["2", "1"])
    with _store() as store:
        hybrid = HybridRetriever([a], store)
        hits = hybrid.retrieve("q", k=2)
        assert all(h.document is not None for h in hits)
        assert hybrid.retrieve("q", k=2) == hits
