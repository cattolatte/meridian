"""Dense bi-encoder retrieval over the brute-force embedding index.

A :class:`DenseRetriever` encodes the query with the same Polaris ``TextEmbedder``
used to embed the corpus, searches the :class:`EmbeddingIndex`, and resolves hits back
to :class:`Document` records. It implements the same :class:`Retriever` protocol as the
BM25 baseline, so the CLI and eval harness treat them interchangeably (RAG.md §4.2).
"""

from __future__ import annotations

from polaris.models import TextEmbedder
from polaris.tokenizers import BPETokenizer

from meridian.corpus.store import DocumentStore
from meridian.encoder.embed import embed_documents, encode_texts
from meridian.retrieval.ann.base import VectorIndex
from meridian.retrieval.embedding_index import EmbeddingIndex
from meridian.retrieval.pipeline import RetrievalHit


class DenseRetriever:
    """A :class:`~meridian.retrieval.pipeline.Retriever` backed by dense embeddings."""

    def __init__(
        self,
        embedder: TextEmbedder,
        tokenizer: BPETokenizer,
        index: VectorIndex,
        store: DocumentStore,
        *,
        max_length: int = 256,
    ) -> None:
        self._embedder = embedder
        self._tokenizer = tokenizer
        self._index = index
        self._store = store
        self._max_length = max_length

    @classmethod
    def from_store(
        cls,
        embedder: TextEmbedder,
        tokenizer: BPETokenizer,
        store: DocumentStore,
        *,
        max_length: int = 256,
        batch_size: int = 32,
    ) -> DenseRetriever:
        """Embed the whole corpus into a fresh index and return a retriever."""
        pmids, vectors = embed_documents(
            embedder,
            tokenizer,
            store.iter_documents(),
            max_length=max_length,
            batch_size=batch_size,
        )
        index = EmbeddingIndex.build(pmids, vectors)
        return cls(embedder, tokenizer, index, store, max_length=max_length)

    @property
    def index(self) -> VectorIndex:
        return self._index

    def retrieve(self, query: str, *, k: int = 10) -> list[RetrievalHit]:
        """Return the top-``k`` hits for ``query`` with their documents attached."""
        query_vec = encode_texts(
            self._embedder, self._tokenizer, [query], max_length=self._max_length
        )[0]
        ranked = self._index.search(query_vec, k=k)
        return [
            RetrievalHit(pmid=pmid, score=score, document=self._store.get(pmid))
            for pmid, score in ranked
        ]
