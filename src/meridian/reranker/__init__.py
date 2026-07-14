"""Cross-encoder reranker (Phase 6).

A Polaris ``SentencePairClassifier`` (``num_classes=1``) that jointly encodes a
(query, passage) pair and scores its relevance, initialized from the shared Stage-0
MLM trunk (ADR-0004). Used by :class:`~meridian.retrieval.rerank.RerankingRetriever`
to re-order the top candidates from a base retriever.
"""

from meridian.reranker.artifact import (
    RerankerConfig,
    build_reranker,
    load_reranker,
    save_reranker,
)
from meridian.reranker.data import make_pair_samples, pairs_from_triples
from meridian.reranker.training import train_reranker

__all__ = [
    "RerankerConfig",
    "build_reranker",
    "load_reranker",
    "make_pair_samples",
    "pairs_from_triples",
    "save_reranker",
    "train_reranker",
]
