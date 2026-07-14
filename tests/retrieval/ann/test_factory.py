"""Tests for the ANN index dispatcher."""

from __future__ import annotations

import numpy as np
import pytest

from meridian.retrieval.ann import build_ann_index
from meridian.retrieval.ann.hnsw import HNSWIndex
from meridian.retrieval.ann.ivf import IVFIndex


def _corpus(n: int = 64, dim: int = 8, seed: int = 0) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return [str(i) for i in range(n)], vectors


def test_build_ivf() -> None:
    pmids, vectors = _corpus()
    index = build_ann_index("ivf", pmids, vectors)
    assert isinstance(index, IVFIndex)
    assert index.nlist == 8  # round(sqrt(64))
    assert len(index.search(vectors[0], k=5)) == 5


def test_build_hnsw() -> None:
    pmids, vectors = _corpus()
    index = build_ann_index("hnsw", pmids, vectors)
    assert isinstance(index, HNSWIndex)
    assert len(index.search(vectors[0], k=5)) == 5


def test_unknown_kind_rejected() -> None:
    pmids, vectors = _corpus(n=4)
    with pytest.raises(ValueError):
        build_ann_index("annoy", pmids, vectors)
