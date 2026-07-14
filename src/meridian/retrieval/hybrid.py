"""Hybrid retrieval by Reciprocal Rank Fusion (RRF).

BM25 scores and dense cosine scores are not comparable, so hybrid retrieval fuses
*ranks*, not scores: ``score(d) = Σ_r 1 / (k_rrf + rank_r(d))`` over the component
retrievers (standard ``k_rrf = 60``). This needs no score normalization and is robust
to the two backends' different score scales (RAG.md §4.2). The fused score replaces the
component scores in the returned hits.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from meridian.corpus.store import DocumentStore
from meridian.retrieval.pipeline import RetrievalHit, Retriever

DEFAULT_K_RRF = 60


class HybridRetriever:
    """Fuse several retrievers' rankings with RRF, behind the Retriever protocol."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        store: DocumentStore,
        *,
        k_rrf: int = DEFAULT_K_RRF,
        depth: int = 100,
    ) -> None:
        if not retrievers:
            raise ValueError("hybrid retrieval needs at least one component retriever")
        self._retrievers = retrievers
        self._store = store
        self._k_rrf = k_rrf
        self._depth = depth

    def retrieve(self, query: str, *, k: int = 10) -> list[RetrievalHit]:
        """Return the top-``k`` hits by fused reciprocal rank, best first."""
        fused: dict[str, float] = defaultdict(float)
        for retriever in self._retrievers:
            for rank, hit in enumerate(retriever.retrieve(query, k=self._depth), start=1):
                fused[hit.pmid] += 1.0 / (self._k_rrf + rank)

        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return [
            RetrievalHit(pmid=pmid, score=score, document=self._store.get(pmid))
            for pmid, score in ranked[:k]
        ]
