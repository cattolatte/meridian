"""HNSW index — a from-scratch hierarchical navigable small-world graph.

A layered proximity graph (Malkov & Yashunin): higher layers are sparse express
lanes, layer 0 holds every vector. Search greedily descends from the top entry point,
then explores layer 0 with an ``ef_search``-sized frontier. Build is incremental and
seeded (levels drawn from a geometric distribution), so a rebuild is deterministic.

Similarity is the dot product on the L2-normalized embeddings; internally we minimize
distance ``-dot`` and return the dot product as the score. No external ANN library.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float32]


class HNSWIndex:
    """A navigable small-world graph index over corpus embeddings."""

    def __init__(
        self,
        pmids: list[str],
        vectors: Array,
        *,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> None:
        self.pmids = tuple(pmids)
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._max_conn0 = 2 * m
        self._level_scale = 1.0 / math.log(m) if m > 1 else 1.0
        self._neighbors: list[list[list[int]]] = []  # [node][layer] -> neighbor ids
        self._levels: list[int] = []
        self._entry: int | None = None
        self._max_level = 0

    def __len__(self) -> int:
        return len(self.pmids)

    @classmethod
    def build(
        cls,
        pmids: list[str],
        vectors: Array,
        *,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 0,
    ) -> HNSWIndex:
        """Build the graph by inserting vectors in order with seeded levels."""
        index = cls(pmids, vectors, m=m, ef_construction=ef_construction, ef_search=ef_search)
        rng = np.random.default_rng(seed)
        for node in range(len(index.pmids)):
            level = int(-math.log(max(float(rng.random()), 1e-12)) * index._level_scale)
            index._insert(node, level)
        return index

    def _dist(self, query: Array, node: int) -> float:
        return -float(query @ self.vectors[node])

    def _search_layer(
        self, query: Array, entry_points: list[int], ef: int, layer: int
    ) -> list[tuple[float, int]]:
        """Return up to ``ef`` closest ``(dist, node)`` reachable on ``layer``."""
        visited = set(entry_points)
        candidates: list[tuple[float, int]] = []  # min-heap by distance
        results: list[tuple[float, int]] = []  # max-heap: (-dist, node)
        for ep in entry_points:
            d = self._dist(query, ep)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            dist_c, node_c = heapq.heappop(candidates)
            if dist_c > -results[0][0]:  # nearest candidate worse than farthest kept
                break
            for neighbor in self._neighbors[node_c][layer]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = self._dist(query, neighbor)
                if len(results) < ef or d < -results[0][0]:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)
        return [(-neg_d, node) for neg_d, node in results]

    def _insert(self, node: int, level: int) -> None:
        self._neighbors.append([[] for _ in range(level + 1)])
        self._levels.append(level)
        query = self.vectors[node]

        if self._entry is None:
            self._entry = node
            self._max_level = level
            return

        entry_points = [self._entry]
        for layer in range(self._max_level, level, -1):  # greedy descent to level+1
            found = self._search_layer(query, entry_points, 1, layer)
            entry_points = [min(found)[1]]

        for layer in range(min(level, self._max_level), -1, -1):
            found = self._search_layer(query, entry_points, self.ef_construction, layer)
            selected = [n for _, n in sorted(found)[: self.m]]
            for neighbor in selected:
                self._neighbors[node][layer].append(neighbor)
                self._neighbors[neighbor][layer].append(node)
            max_conn = self._max_conn0 if layer == 0 else self.m
            for neighbor in selected:
                self._prune(neighbor, layer, max_conn)
            entry_points = [n for _, n in found]

        if level > self._max_level:
            self._entry = node
            self._max_level = level

    def _prune(self, node: int, layer: int, max_conn: int) -> None:
        connections = self._neighbors[node][layer]
        if len(connections) <= max_conn:
            return
        node_vec = self.vectors[node]
        self._neighbors[node][layer] = sorted(
            connections, key=lambda other: self._dist(node_vec, other)
        )[:max_conn]

    def search(
        self, query: npt.NDArray[np.float32], *, k: int = 10, ef_search: int | None = None
    ) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(pmid, score)`` pairs, best first."""
        if self._entry is None:
            return []
        query = query.astype(np.float32)
        entry_points = [self._entry]
        for layer in range(self._max_level, 0, -1):
            found = self._search_layer(query, entry_points, 1, layer)
            entry_points = [min(found)[1]]

        ef = max(ef_search if ef_search is not None else self.ef_search, k)
        found = self._search_layer(query, entry_points, ef, 0)
        ranked = sorted(found, key=lambda item: (item[0], self.pmids[item[1]]))
        return [(self.pmids[node], -dist) for dist, node in ranked[:k]]
