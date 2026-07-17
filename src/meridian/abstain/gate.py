"""Gate 1 — retrieval confidence.

If the retriever is not confident, the system abstains before it ever generates
(RAG.md §4.2). Confidence combines the top passage's score with the *margin* — the gap
between the top and the k-th score — which separates "one clearly-relevant passage"
from "a flat, uncertain ranking". Thresholds are tuned on the dev split (risk-coverage;
see :mod:`meridian.abstain.calibration`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from meridian.retrieval.pipeline import RetrievalHit


@dataclass(frozen=True, slots=True)
class RetrievalConfidence:
    """Confidence signals derived from a ranked hit list."""

    top_score: float
    margin: float  # top-1 score minus the k-th score


def retrieval_confidence(hits: Sequence[RetrievalHit], *, margin_k: int = 5) -> RetrievalConfidence:
    """Compute the top score and the top-1 vs top-``margin_k`` margin."""
    if not hits:
        return RetrievalConfidence(top_score=0.0, margin=0.0)
    top = hits[0].score
    tail_index = min(len(hits), margin_k) - 1
    return RetrievalConfidence(top_score=top, margin=top - hits[tail_index].score)


@dataclass(frozen=True, slots=True)
class RetrievalGate:
    """Answer only when retrieval confidence clears both thresholds."""

    min_score: float = 0.0
    min_margin: float = 0.0
    margin_k: int = 5

    def confidence(self, hits: Sequence[RetrievalHit]) -> RetrievalConfidence:
        return retrieval_confidence(hits, margin_k=self.margin_k)

    def answerable(self, hits: Sequence[RetrievalHit]) -> bool:
        """True iff the top score and margin both clear their thresholds."""
        confidence = self.confidence(hits)
        return confidence.top_score >= self.min_score and confidence.margin >= self.min_margin
