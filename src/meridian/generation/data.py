"""Build grounded-SFT examples from retrieved passages (Phase 7, ADR-0006).

A grounded example is ``(question, passages, answer)`` where ``passages`` is a list of
``(passage_id, text)`` and ``answer`` is the cited answer string — or ``None`` to train
an abstention. Passages come from the retriever's (reranked) hits; the actual PQA-derived
answers with aligned ``[n]`` citations are built by the Phase 7 SFT data pipeline
(citation alignment + hand-audit), which is a separate, data-heavy step.
"""

from __future__ import annotations

from collections.abc import Sequence

from meridian.retrieval.pipeline import RetrievalHit

# (question, passages[(id, text)], answer | None)
GroundedExample = tuple[str, list[tuple[str, str]], str | None]


def passages_from_hits(hits: Sequence[RetrievalHit]) -> list[tuple[str, str]]:
    """Turn retrieval hits into ``(pmid, chunk_text)`` passages for the prompt."""
    return [(hit.pmid, hit.document.chunk_text()) for hit in hits if hit.document is not None]


def grounded_example(
    question: str,
    hits: Sequence[RetrievalHit],
    answer: str | None,
) -> GroundedExample:
    """Assemble one grounded SFT example (``answer=None`` trains an abstention)."""
    return question, passages_from_hits(hits), answer
