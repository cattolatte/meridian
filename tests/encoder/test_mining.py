"""Tests for hard-negative mining."""

from __future__ import annotations

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.encoder.mining import mine_hard_negatives
from meridian.retrieval.pipeline import BM25Retriever

_DOCS = [
    Document(pmid="1", title="Heart failure", abstract="beta blockers reduce mortality"),
    Document(pmid="2", title="Heart failure guidelines", abstract="mortality and heart outcomes"),
    Document(pmid="3", title="Diabetes", abstract="metformin cardiovascular outcomes"),
    Document(pmid="4", title="Melanoma", abstract="checkpoint immunotherapy responses"),
]


def _store() -> SqliteDocumentStore:
    store = SqliteDocumentStore(":memory:")
    store.add_many(_DOCS)
    return store


def test_negatives_exclude_positive() -> None:
    with _store() as store:
        retriever = BM25Retriever.from_store(store)
        triples = mine_hard_negatives(
            [retriever], store, [("heart failure mortality", "1")], num_negatives=3
        )
        assert len(triples) == 1
        anchor, positive, negatives = triples[0]
        assert anchor == "heart failure mortality"
        assert positive == _DOCS[0].chunk_text()
        # The positive (pmid 1) is never a negative.
        assert _DOCS[0].chunk_text() not in negatives
        assert len(negatives) <= 3


def test_respects_num_negatives() -> None:
    with _store() as store:
        retriever = BM25Retriever.from_store(store)
        triples = mine_hard_negatives(
            [retriever], store, [("heart failure mortality", "1")], num_negatives=1, pool=10
        )
        assert len(triples[0][2]) == 1


def test_missing_positive_is_skipped() -> None:
    with _store() as store:
        retriever = BM25Retriever.from_store(store)
        triples = mine_hard_negatives([retriever], store, [("q", "does-not-exist")])
        assert triples == []


def test_pools_across_retrievers_without_duplicates() -> None:
    with _store() as store:
        r1 = BM25Retriever.from_store(store)
        r2 = BM25Retriever.from_store(store)
        triples = mine_hard_negatives(
            [r1, r2], store, [("heart failure mortality", "1")], num_negatives=4
        )
        negatives = triples[0][2]
        assert len(negatives) == len(set(negatives))  # deduplicated across retrievers
