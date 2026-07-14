"""The vector-index interface every backend satisfies.

``search`` returns ``(pmid, score)`` pairs best-first, identical to the brute-force
:class:`~meridian.retrieval.embedding_index.EmbeddingIndex`, so
:class:`~meridian.retrieval.dense.DenseRetriever` treats brute force, IVF, and HNSW
interchangeably (RAG.md §4.1).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class VectorIndex(Protocol):
    """A searchable index over corpus embedding vectors."""

    def __len__(self) -> int: ...

    def search(self, query: npt.NDArray[np.float32], *, k: int = 10) -> list[tuple[str, float]]: ...
