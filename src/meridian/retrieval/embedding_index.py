"""Brute-force embedding index — the dense-retrieval ground truth.

Holds the corpus embedding matrix (memory-mapped float32 shards) and a PMID map, and
answers nearest-neighbour queries by an exact dot product. Because the embedder emits
L2-normalized vectors (``TextEmbedder(normalize=True)``), the dot product is cosine
similarity. This exact search is the ground truth the Phase-4 ANN indexes (IVF/HNSW)
are measured against, so it deliberately does no approximation.

Ranking is deterministic: ties break by ascending PMID.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

_VECTORS_FILE = "embeddings.npy"
_PMIDS_FILE = "pmids.json"


@dataclass(frozen=True)
class EmbeddingIndex:
    """A corpus embedding matrix plus its PMID map, with exact top-k search."""

    pmids: tuple[str, ...]
    vectors: npt.NDArray[np.float32]  # shape (N, D), one row per PMID

    def __post_init__(self) -> None:
        if len(self.pmids) != self.vectors.shape[0]:
            raise ValueError("pmids and vectors must have the same length")

    def __len__(self) -> int:
        return len(self.pmids)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if len(self.pmids) else 0

    @classmethod
    def build(cls, pmids: list[str], vectors: npt.NDArray[np.float32]) -> EmbeddingIndex:
        """Build an index from parallel PMIDs and an ``(N, D)`` float32 matrix."""
        return cls(pmids=tuple(pmids), vectors=np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query: npt.NDArray[np.float32], *, k: int = 10) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(pmid, score)`` pairs by dot product, best first.

        ``query`` is a 1-D ``(D,)`` vector (normalized, to make the score a cosine).
        Ties break by ascending PMID for determinism.
        """
        if len(self.pmids) == 0:
            return []
        scores = self.vectors @ query.astype(np.float32)  # (N,)
        order = sorted(range(len(self.pmids)), key=lambda i: (-float(scores[i]), self.pmids[i]))
        return [(self.pmids[i], float(scores[i])) for i in order[:k]]

    def save(self, directory: str | Path) -> None:
        """Write the index to ``directory`` as a ``.npy`` matrix + a PMID JSON list."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / _VECTORS_FILE, self.vectors)
        (directory / _PMIDS_FILE).write_text(json.dumps(list(self.pmids)))

    @classmethod
    def load(cls, directory: str | Path) -> EmbeddingIndex:
        """Load an index, memory-mapping the embedding matrix (no full RAM copy)."""
        directory = Path(directory)
        vectors = np.load(directory / _VECTORS_FILE, mmap_mode="r")
        pmids = json.loads((directory / _PMIDS_FILE).read_text())
        return cls(pmids=tuple(pmids), vectors=vectors)
