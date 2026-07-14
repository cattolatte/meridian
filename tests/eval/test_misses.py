"""Tests for retrieval-miss sampling."""

from __future__ import annotations

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.eval.misses import sample_misses
from meridian.eval.qrels import EvalQuery, EvalSet
from meridian.retrieval.pipeline import BM25Retriever

_DOCS = [
    Document(pmid="1", title="Heart failure", abstract="beta blockers reduce mortality"),
    Document(pmid="2", title="Melanoma", abstract="checkpoint immunotherapy responses"),
]


def _retriever() -> BM25Retriever:
    store = SqliteDocumentStore(":memory:")
    store.add_many(_DOCS)
    return BM25Retriever.from_store(store)


def test_hit_is_not_a_miss() -> None:
    eval_set = EvalSet("dev", (EvalQuery("q1", "beta blockers mortality", frozenset({"1"})),))
    assert sample_misses(_retriever(), eval_set, k=2) == []


def test_miss_is_recorded_with_retrieved() -> None:
    # Relevant doc is "1" (heart failure), but the query is about melanoma.
    eval_set = EvalSet("dev", (EvalQuery("q1", "melanoma immunotherapy", frozenset({"1"})),))
    misses = sample_misses(_retriever(), eval_set, k=1)
    assert len(misses) == 1
    assert misses[0].query_id == "q1"
    assert misses[0].relevant_pmids == ("1",)
    assert "2" in misses[0].retrieved_pmids  # what was returned instead


def test_queries_without_relevant_are_skipped() -> None:
    eval_set = EvalSet("dev", (EvalQuery("q1", "anything", frozenset()),))
    assert sample_misses(_retriever(), eval_set) == []
