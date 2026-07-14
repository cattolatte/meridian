"""Retrieval: the from-scratch BM25 baseline and the retriever interface.

Phase 2 provides the honest lexical baseline (:class:`~meridian.retrieval.bm25.BM25Index`)
that every later neural retrieval claim is measured against, plus the
:class:`~meridian.retrieval.pipeline.Retriever` protocol the CLI and eval harness
depend on.
"""

from meridian.retrieval.analyzer import simple_analyzer
from meridian.retrieval.bm25 import BM25Index
from meridian.retrieval.dense import DenseRetriever
from meridian.retrieval.embedding_index import EmbeddingIndex
from meridian.retrieval.pipeline import BM25Retriever, RetrievalHit, Retriever

__all__ = [
    "BM25Index",
    "BM25Retriever",
    "DenseRetriever",
    "EmbeddingIndex",
    "RetrievalHit",
    "Retriever",
    "simple_analyzer",
]
