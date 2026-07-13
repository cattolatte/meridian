"""Tests for the BM25 retriever over the document store."""

from __future__ import annotations

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.retrieval.pipeline import BM25Retriever, Retriever

_DOCS = [
    Document(pmid="1", title="Heart failure", abstract="beta blockers reduce mortality"),
    Document(pmid="2", title="Diabetes", abstract="metformin and cardiovascular outcomes"),
    Document(pmid="3", title="Melanoma", abstract="checkpoint immunotherapy responses"),
]


def _store() -> SqliteDocumentStore:
    store = SqliteDocumentStore(":memory:")
    store.add_many(_DOCS)
    return store


def test_retriever_satisfies_protocol() -> None:
    with _store() as store:
        retriever = BM25Retriever.from_store(store)
        assert isinstance(retriever, Retriever)


def test_retrieve_returns_relevant_document_first() -> None:
    with _store() as store:
        retriever = BM25Retriever.from_store(store)
        hits = retriever.retrieve("immunotherapy for melanoma", k=3)
        assert hits[0].pmid == "3"
        assert hits[0].document is not None
        assert hits[0].document.title == "Melanoma"


def test_retrieve_respects_k() -> None:
    with _store() as store:
        retriever = BM25Retriever.from_store(store)
        assert len(retriever.retrieve("heart", k=1)) == 1


def test_index_covers_whole_corpus() -> None:
    with _store() as store:
        retriever = BM25Retriever.from_store(store)
        assert len(retriever.index) == 3
