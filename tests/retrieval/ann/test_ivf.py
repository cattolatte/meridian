"""Tests for the IVF index, anchored against brute-force ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from meridian.retrieval.ann.ivf import IVFIndex
from meridian.retrieval.embedding_index import EmbeddingIndex


def _corpus(n: int = 60, dim: int = 8, seed: int = 0) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # normalize (cosine)
    pmids = [str(i) for i in range(n)]
    return pmids, vectors


def _recall_at_k(approx: list[tuple[str, float]], truth: list[tuple[str, float]], k: int) -> float:
    truth_ids = {pmid for pmid, _ in truth[:k]}
    approx_ids = {pmid for pmid, _ in approx[:k]}
    return len(truth_ids & approx_ids) / k


def test_len_and_nlist() -> None:
    pmids, vectors = _corpus()
    index = IVFIndex.build(pmids, vectors, nlist=6, seed=0)
    assert len(index) == 60
    assert index.nlist == 6


def test_nprobe_equals_nlist_matches_brute_force() -> None:
    pmids, vectors = _corpus()
    brute = EmbeddingIndex.build(pmids, vectors)
    ivf = IVFIndex.build(pmids, vectors, nlist=6, seed=0)
    for q in range(5):
        query = vectors[q]
        assert ivf.search(query, k=10, nprobe=6) == brute.search(query, k=10)


def test_recall_increases_with_nprobe() -> None:
    pmids, vectors = _corpus(n=120, seed=1)
    brute = EmbeddingIndex.build(pmids, vectors)
    ivf = IVFIndex.build(pmids, vectors, nlist=12, seed=0)
    queries = vectors[:20]

    def mean_recall(nprobe: int) -> float:
        return float(
            np.mean(
                [
                    _recall_at_k(ivf.search(q, k=10, nprobe=nprobe), brute.search(q, k=10), 10)
                    for q in queries
                ]
            )
        )

    assert mean_recall(1) <= mean_recall(4) <= mean_recall(12)
    assert mean_recall(12) == 1.0  # probing all cells is exact


def test_determinism() -> None:
    pmids, vectors = _corpus()
    a = IVFIndex.build(pmids, vectors, nlist=6, seed=0)
    b = IVFIndex.build(pmids, vectors, nlist=6, seed=0)
    q = vectors[0]
    assert a.search(q, k=5, nprobe=2) == b.search(q, k=5, nprobe=2)


def test_empty_corpus_cannot_build() -> None:
    # nlist must be in [1, N], so an empty corpus is rejected by the k-means guard.
    with pytest.raises(ValueError):
        IVFIndex.build([], np.zeros((0, 4), dtype=np.float32), nlist=1)
