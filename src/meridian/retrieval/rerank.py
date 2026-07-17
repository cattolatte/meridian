"""Reranking stage — cross-encoder precision over a base retriever's candidates.

Takes the top-N candidates from any base :class:`Retriever`, scores each
``(query, passage)`` pair jointly with the cross-encoder (full cross-attention, unlike
the bi-encoder), and returns the top-``k`` by that score. This is where top-of-ranking
precision comes from — and where the latency is (RAG.md §4.2). It composes with BM25,
dense, or hybrid as the base, and is one config flag away from off.
"""

from __future__ import annotations

import torch
from polaris.collation import collate_pairs
from polaris.models import SentencePairClassifier
from polaris.tokenizers import BPETokenizer

from meridian.corpus.store import DocumentStore
from meridian.reranker.data import make_pair_samples
from meridian.retrieval.pipeline import RetrievalHit, Retriever


class RerankingRetriever:
    """Rerank a base retriever's top candidates with the cross-encoder."""

    def __init__(
        self,
        base: Retriever,
        model: SentencePairClassifier,
        tokenizer: BPETokenizer,
        store: DocumentStore,
        *,
        candidates: int = 100,
        max_length: int = 256,
        base_weight: float = 0.0,
        rrf_k: int = 60,
    ) -> None:
        vocab = tokenizer.vocabulary
        if vocab.cls_id is None or vocab.sep_id is None:
            raise ValueError("reranker tokenizer must define <cls> and <sep> tokens")
        self._base = base
        self._model = model
        self._tokenizer = tokenizer
        self._store = store
        self._candidates = candidates
        self._max_length = max_length
        self._base_weight = base_weight
        self._rrf_k = rrf_k
        self._pad_id = vocab.pad_id or 0
        self._cls_id = vocab.cls_id
        self._sep_id = vocab.sep_id

    def retrieve(self, query: str, *, k: int = 10) -> list[RetrievalHit]:
        """Retrieve top-``candidates`` from the base, rerank, and return top-``k``.

        With ``base_weight == 0`` (default) this is a pure cross-encoder rerank. With
        ``base_weight > 0`` the reranker's ranking is *fused* with the base ranking by
        reciprocal-rank fusion — ``base_weight/(rrf_k + base_rank) + 1/(rrf_k +
        rerank_rank)`` — so a reranker that is *weaker* than its base (e.g. a
        from-scratch model that overfits limited data) can no longer scramble a strong
        base ranking; it degrades gracefully toward the base instead of below random.
        Ties break by base rank, then PMID, so the fallback is the base order, never an
        arbitrary PMID sort.
        """
        hits = [hit for hit in self._base.retrieve(query, k=self._candidates) if hit.document]
        if not hits:
            return []

        samples = make_pair_samples(
            [(query, hit.document.chunk_text(), 0) for hit in hits if hit.document],
            self._tokenizer,
        )
        batch = collate_pairs(
            samples,
            pad_id=self._pad_id,
            cls_id=self._cls_id,
            sep_id=self._sep_id,
            max_length=self._max_length,
        )
        self._model.eval()
        with torch.no_grad():
            scores = self._model(batch).squeeze(-1).tolist()

        base_rank = {hit.pmid: rank for rank, hit in enumerate(hits)}
        score_by_pmid = {hit.pmid: score for hit, score in zip(hits, scores, strict=True)}
        rerank_order = sorted(
            hits, key=lambda hit: (-score_by_pmid[hit.pmid], base_rank[hit.pmid], hit.pmid)
        )
        rerank_rank = {hit.pmid: rank for rank, hit in enumerate(rerank_order)}

        def fused_score(hit: RetrievalHit) -> float:
            rr = 1.0 / (self._rrf_k + rerank_rank[hit.pmid])
            if self._base_weight == 0.0:
                return rr
            return self._base_weight / (self._rrf_k + base_rank[hit.pmid]) + rr

        ranked = sorted(hits, key=lambda hit: (-fused_score(hit), base_rank[hit.pmid], hit.pmid))
        return [
            RetrievalHit(pmid=hit.pmid, score=float(score_by_pmid[hit.pmid]), document=hit.document)
            for hit in ranked[:k]
        ]
