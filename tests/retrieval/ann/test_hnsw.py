"""Property-based tests for the HNSW index against brute-force ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from meridian.retrieval.ann.base import VectorIndex
from meridian.retrieval.ann.hnsw import HNSWIndex
from meridian.retrieval.embedding_index import EmbeddingIndex


def _corpus(n: int = 200, dim: int = 16, seed: int = 0) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return [str(i) for i in range(n)], vectors


def _mean_recall(index: HNSWIndex, brute: EmbeddingIndex, queries: np.ndarray, k: int) -> float:
    recalls = []
    for q in queries:
        truth = {pmid for pmid, _ in brute.search(q, k=k)}
        approx = {pmid for pmid, _ in index.search(q, k=k)}
        recalls.append(len(truth & approx) / k)
    return float(np.mean(recalls))


def test_satisfies_vector_index_protocol() -> None:
    pmids, vectors = _corpus(n=10)
    assert isinstance(HNSWIndex.build(pmids, vectors, seed=0), VectorIndex)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recall_floor_vs_brute_force(seed: int) -> None:
    pmids, vectors = _corpus(n=200, seed=seed)
    brute = EmbeddingIndex.build(pmids, vectors)
    hnsw = HNSWIndex.build(pmids, vectors, m=16, ef_construction=100, ef_search=50, seed=seed)
    recall = _mean_recall(hnsw, brute, vectors[:25], k=10)
    assert recall >= 0.90  # graph search recovers almost all true neighbours


def test_finds_self_as_nearest() -> None:
    pmids, vectors = _corpus(n=120, seed=3)
    hnsw = HNSWIndex.build(pmids, vectors, seed=0)
    for node in range(0, 120, 10):
        top_pmid, _ = hnsw.search(vectors[node], k=1)[0]
        assert top_pmid == str(node)


def test_determinism() -> None:
    pmids, vectors = _corpus(n=80, seed=4)
    a = HNSWIndex.build(pmids, vectors, seed=0)
    b = HNSWIndex.build(pmids, vectors, seed=0)
    q = vectors[0]
    assert a.search(q, k=5) == b.search(q, k=5)


def test_single_element() -> None:
    hnsw = HNSWIndex.build(["only"], np.array([[1.0, 0.0]], dtype=np.float32), seed=0)
    assert len(hnsw) == 1
    assert hnsw.search(np.array([1.0, 0.0], dtype=np.float32), k=5) == [("only", 1.0)]


def test_k_larger_than_corpus() -> None:
    pmids, vectors = _corpus(n=5, seed=5)
    hnsw = HNSWIndex.build(pmids, vectors, seed=0)
    assert len(hnsw.search(vectors[0], k=20)) == 5
