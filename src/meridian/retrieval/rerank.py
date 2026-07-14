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
        self._pad_id = vocab.pad_id or 0
        self._cls_id = vocab.cls_id
        self._sep_id = vocab.sep_id

    def retrieve(self, query: str, *, k: int = 10) -> list[RetrievalHit]:
        """Retrieve top-``candidates`` from the base, rerank, and return top-``k``."""
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

        ranked = sorted(zip(hits, scores, strict=True), key=lambda item: (-item[1], item[0].pmid))
        return [
            RetrievalHit(pmid=hit.pmid, score=float(score), document=hit.document)
            for hit, score in ranked[:k]
        ]
