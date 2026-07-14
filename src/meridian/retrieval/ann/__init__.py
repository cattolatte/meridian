"""Approximate nearest-neighbour indexes, built from scratch (Phase 4).

Sub-linear vector search — IVF (inverted file over k-means cells) and HNSW (a
navigable small-world graph) — behind the same :class:`VectorIndex` interface as the
brute-force :class:`~meridian.retrieval.embedding_index.EmbeddingIndex`, which is the
exact ground truth their recall is measured against. No external ANN library is used.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from meridian.retrieval.ann.base import VectorIndex
from meridian.retrieval.ann.hnsw import HNSWIndex
from meridian.retrieval.ann.ivf import IVFIndex
from meridian.retrieval.ann.kmeans import kmeans

__all__ = ["HNSWIndex", "IVFIndex", "VectorIndex", "build_ann_index", "kmeans"]

ANN_KINDS = ("ivf", "hnsw")


def build_ann_index(
    kind: str,
    pmids: list[str],
    vectors: npt.NDArray[np.float32],
    *,
    nlist: int | None = None,
    nprobe: int = 8,
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 64,
    seed: int = 0,
) -> VectorIndex:
    """Build the named ANN index over ``(pmids, vectors)``.

    ``nlist`` defaults to ``round(sqrt(N))`` cells (a common IVF rule of thumb).
    Raises :class:`ValueError` for an unknown ``kind``.
    """
    if kind == "ivf":
        cells = nlist if nlist is not None else max(1, round(len(pmids) ** 0.5))
        return IVFIndex.build(pmids, vectors, nlist=cells, nprobe=nprobe, seed=seed)
    if kind == "hnsw":
        return HNSWIndex.build(
            pmids, vectors, m=m, ef_construction=ef_construction, ef_search=ef_search, seed=seed
        )
    raise ValueError(f"unknown ANN index kind: {kind!r} (expected one of {ANN_KINDS})")
