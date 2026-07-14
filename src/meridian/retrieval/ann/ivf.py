"""IVF (inverted file) index — from-scratch coarse-quantization ANN.

Build: k-means partitions the corpus into ``nlist`` cells; each vector joins the
inverted list of its nearest centroid. Search: score the query against all centroids,
take the ``nprobe`` nearest cells, and exactly rank the vectors they contain. With
``nprobe == nlist`` this searches every vector and recovers brute-force recall — the
correctness anchor. Ranking ties break by ascending PMID.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from meridian.retrieval.ann.kmeans import kmeans

Array = npt.NDArray[np.float32]


@dataclass(frozen=True)
class IVFIndex:
    """An IVF index over corpus embeddings."""

    pmids: tuple[str, ...]
    vectors: Array  # (N, D)
    centroids: Array  # (nlist, D)
    lists: tuple[tuple[int, ...], ...]  # per-cell vector indices
    nprobe: int = 1

    def __len__(self) -> int:
        return len(self.pmids)

    @property
    def nlist(self) -> int:
        return len(self.centroids)

    @classmethod
    def build(
        cls,
        pmids: list[str],
        vectors: Array,
        *,
        nlist: int,
        nprobe: int = 1,
        seed: int = 0,
        n_iters: int = 25,
    ) -> IVFIndex:
        """Build an IVF index with ``nlist`` k-means cells."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        centroids, assignments = kmeans(vectors, nlist, seed=seed, n_iters=n_iters)
        lists: list[list[int]] = [[] for _ in range(nlist)]
        for index, cell in enumerate(assignments):
            lists[int(cell)].append(index)
        return cls(
            pmids=tuple(pmids),
            vectors=vectors,
            centroids=centroids,
            lists=tuple(tuple(cell) for cell in lists),
            nprobe=nprobe,
        )

    def search(
        self, query: npt.NDArray[np.float32], *, k: int = 10, nprobe: int | None = None
    ) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(pmid, score)`` pairs, probing the nearest cells."""
        if len(self.pmids) == 0:
            return []
        query = query.astype(np.float32)
        probes = min(nprobe if nprobe is not None else self.nprobe, self.nlist)

        centroid_scores = self.centroids @ query
        nearest_cells = np.argsort(-centroid_scores, kind="stable")[:probes]
        candidates = [idx for cell in nearest_cells for idx in self.lists[int(cell)]]
        if not candidates:
            return []

        candidate_idx = np.asarray(candidates, dtype=np.intp)
        scores = self.vectors[candidate_idx] @ query
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-float(scores[i]), self.pmids[candidates[i]]),
        )
        return [(self.pmids[candidates[i]], float(scores[i])) for i in order[:k]]
