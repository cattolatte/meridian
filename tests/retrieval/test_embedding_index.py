"""Tests for the brute-force embedding index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from meridian.retrieval.embedding_index import EmbeddingIndex


def _index() -> EmbeddingIndex:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    return EmbeddingIndex.build(["a", "b", "c"], vectors)


def test_len_and_dim() -> None:
    index = _index()
    assert len(index) == 3
    assert index.dim == 2


def test_search_returns_nearest_first() -> None:
    index = _index()
    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), k=3)
    assert [pmid for pmid, _ in hits] == ["a", "c", "b"]


def test_search_top_k_limit() -> None:
    assert len(_index().search(np.array([1.0, 0.0], dtype=np.float32), k=1)) == 1


def test_search_ties_break_by_pmid() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    index = EmbeddingIndex.build(["zeta", "alpha"], vectors)
    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert [pmid for pmid, _ in hits] == ["alpha", "zeta"]


def test_empty_index_returns_nothing() -> None:
    index = EmbeddingIndex.build([], np.zeros((0, 4), dtype=np.float32))
    assert index.search(np.zeros(4, dtype=np.float32), k=5) == []
    assert len(index) == 0


def test_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        EmbeddingIndex(pmids=("a",), vectors=np.zeros((2, 3), dtype=np.float32))


def test_save_load_roundtrip(tmp_path: Path) -> None:
    index = _index()
    index.save(tmp_path / "idx")
    reloaded = EmbeddingIndex.load(tmp_path / "idx")
    assert reloaded.pmids == index.pmids
    q = np.array([0.0, 1.0], dtype=np.float32)
    assert reloaded.search(q, k=3) == index.search(q, k=3)
