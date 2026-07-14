"""Seeded Lloyd k-means (NumPy) — learns the IVF cell centroids.

Deterministic given a seed: random init from data points, fixed Lloyd iterations,
empty clusters re-seeded to a random point. Uses squared Euclidean distance, which on
the L2-normalized embeddings orders points the same way as cosine.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float32]


def _assign(vectors: Array, centroids: Array) -> npt.NDArray[np.intp]:
    """Assign each vector to its nearest centroid (squared-Euclidean argmin)."""
    # ||v - c||^2 = ||v||^2 - 2 v·c + ||c||^2; the ||v||^2 term is constant per row.
    cross = vectors @ centroids.T  # (N, K)
    centroid_sq = np.einsum("kd,kd->k", centroids, centroids)  # (K,)
    distances = centroid_sq[None, :] - 2.0 * cross
    return np.asarray(np.argmin(distances, axis=1), dtype=np.intp)


def _kmeans_plus_plus_init(vectors: Array, n_clusters: int, rng: np.random.Generator) -> Array:
    """Choose initial centroids by k-means++ (D^2 sampling) for good separation."""
    n_vectors = len(vectors)
    chosen = [int(rng.integers(n_vectors))]
    closest_sq = np.sum((vectors - vectors[chosen[0]]) ** 2, axis=1)
    for _ in range(1, n_clusters):
        total = float(closest_sq.sum())
        if total <= 0.0:  # remaining points coincide with a centroid; pick uniformly
            nxt = int(rng.integers(n_vectors))
        else:
            nxt = int(rng.choice(n_vectors, p=closest_sq / total))
        chosen.append(nxt)
        closest_sq = np.minimum(closest_sq, np.sum((vectors - vectors[nxt]) ** 2, axis=1))
    return vectors[chosen].astype(np.float32)


def kmeans(
    vectors: Array,
    n_clusters: int,
    *,
    seed: int = 0,
    n_iters: int = 25,
) -> tuple[Array, npt.NDArray[np.intp]]:
    """Cluster ``vectors`` into ``n_clusters``; return ``(centroids, assignments)``.

    Raises :class:`ValueError` if ``n_clusters`` is not in ``[1, len(vectors)]``.
    """
    n_vectors = len(vectors)
    if not 1 <= n_clusters <= n_vectors:
        raise ValueError("n_clusters must be between 1 and the number of vectors")

    rng = np.random.default_rng(seed)
    centroids = _kmeans_plus_plus_init(vectors, n_clusters, rng)
    assignments = _assign(vectors, centroids)

    for _ in range(n_iters):
        new_centroids = centroids.copy()
        for cluster in range(n_clusters):
            members = vectors[assignments == cluster]
            if len(members):
                new_centroids[cluster] = members.mean(axis=0)
            else:
                new_centroids[cluster] = vectors[rng.integers(n_vectors)]
        new_assignments = _assign(vectors, new_centroids)
        centroids = new_centroids
        if np.array_equal(new_assignments, assignments):
            assignments = new_assignments
            break
        assignments = new_assignments

    return centroids.astype(np.float32), assignments
