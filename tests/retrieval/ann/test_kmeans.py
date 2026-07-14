"""Tests for seeded Lloyd k-means."""

from __future__ import annotations

import numpy as np
import pytest

from meridian.retrieval.ann.kmeans import kmeans


def _clustered(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array([[5.0, 5.0], [-5.0, -5.0], [5.0, -5.0]], dtype=np.float32)
    points = np.concatenate([c + 0.1 * rng.standard_normal((20, 2)) for c in centers])
    return points.astype(np.float32)


def test_assignments_partition_all_points() -> None:
    vectors = _clustered()
    _, assignments = kmeans(vectors, 3, seed=0)
    assert assignments.shape == (len(vectors),)
    assert set(assignments.tolist()) <= {0, 1, 2}


def test_recovers_three_clusters() -> None:
    vectors = _clustered()
    _, assignments = kmeans(vectors, 3, seed=0)
    # Each of the three true groups of 20 lands in a single cluster.
    for start in (0, 20, 40):
        group = assignments[start : start + 20]
        assert len(set(group.tolist())) == 1


def test_n_clusters_equals_points_is_zero_error() -> None:
    vectors = _clustered()
    centroids, assignments = kmeans(vectors, len(vectors), seed=0)
    # Every point is its own centroid -> each vector coincides with its centroid.
    assert np.allclose(centroids[assignments], vectors, atol=1e-5)


def test_determinism() -> None:
    vectors = _clustered()
    a = kmeans(vectors, 3, seed=7)
    b = kmeans(vectors, 3, seed=7)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_invalid_n_clusters_rejected() -> None:
    vectors = _clustered()
    with pytest.raises(ValueError):
        kmeans(vectors, 0)
    with pytest.raises(ValueError):
        kmeans(vectors, len(vectors) + 1)
