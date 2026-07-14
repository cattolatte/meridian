"""Approximate nearest-neighbour indexes, built from scratch (Phase 4).

Sub-linear vector search — IVF (inverted file over k-means cells) and HNSW (a
navigable small-world graph) — behind the same :class:`VectorIndex` interface as the
brute-force :class:`~meridian.retrieval.embedding_index.EmbeddingIndex`, which is the
exact ground truth their recall is measured against. No external ANN library is used.
"""

from meridian.retrieval.ann.base import VectorIndex
from meridian.retrieval.ann.hnsw import HNSWIndex
from meridian.retrieval.ann.ivf import IVFIndex
from meridian.retrieval.ann.kmeans import kmeans

__all__ = ["HNSWIndex", "IVFIndex", "VectorIndex", "kmeans"]
