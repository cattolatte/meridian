"""Retriever interface and the BM25 retriever over the document store.

:class:`Retriever` is the contract the CLI and eval harness depend on; swapping in
the dense retriever (Phase 3) or hybrid fusion (Phase 5) is a matter of providing
another implementation. :class:`BM25Retriever` builds a :class:`BM25Index` from the
stored corpus and resolves hits back to full :class:`Document` records.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from meridian.corpus.records import Document
from meridian.corpus.store import DocumentStore
from meridian.retrieval.analyzer import simple_analyzer
from meridian.retrieval.bm25 import DEFAULT_B, DEFAULT_K1, BM25Index

Analyzer = Callable[[str], Sequence[str]]


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One retrieved passage: its PMID, score, and (optionally) the document."""

    pmid: str
    score: float
    document: Document | None = None


@runtime_checkable
class Retriever(Protocol):
    """Returns the top-``k`` passages for a query, best first."""

    def retrieve(self, query: str, *, k: int = 10) -> list[RetrievalHit]: ...


class BM25Retriever:
    """A :class:`Retriever` backed by a from-scratch BM25 index over the store."""

    def __init__(
        self,
        index: BM25Index,
        store: DocumentStore,
        *,
        analyze: Analyzer = simple_analyzer,
    ) -> None:
        self._index = index
        self._store = store
        self._analyze = analyze

    @classmethod
    def from_store(
        cls,
        store: DocumentStore,
        *,
        analyze: Analyzer = simple_analyzer,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> BM25Retriever:
        """Build a BM25 retriever by indexing every document's chunk text."""
        corpus = (
            (document.pmid, analyze(document.chunk_text())) for document in store.iter_documents()
        )
        index = BM25Index.build(corpus, k1=k1, b=b)
        return cls(index, store, analyze=analyze)

    @property
    def index(self) -> BM25Index:
        return self._index

    def retrieve(self, query: str, *, k: int = 10) -> list[RetrievalHit]:
        """Return the top-``k`` hits for ``query`` with their documents attached."""
        terms = self._analyze(query)
        ranked = self._index.search(terms, k=k)
        return [
            RetrievalHit(pmid=pmid, score=score, document=self._store.get(pmid))
            for pmid, score in ranked
        ]
