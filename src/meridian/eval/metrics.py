"""Binary-relevance retrieval metrics (RAG.md §7).

Each metric scores one ranked list of PMIDs against a set of relevant PMIDs. The
harness averages them over an eval set. Relevance is binary (a PMID is relevant or
not), matching the PubMedQA-derived qrels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _require_relevant(relevant: frozenset[str]) -> None:
    if not relevant:
        raise ValueError("metric is undefined for a query with no relevant documents")


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of relevant documents found in the top ``k``."""
    _require_relevant(relevant)
    top_k = set(ranked[:k])
    return len(relevant & top_k) / len(relevant)


def mrr_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int = 10) -> float:
    """Reciprocal rank of the first relevant document within the top ``k`` (else 0)."""
    _require_relevant(relevant)
    for position, pmid in enumerate(ranked[:k], start=1):
        if pmid in relevant:
            return 1.0 / position
    return 0.0


def _dcg(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    return sum(
        1.0 / math.log2(position + 1)
        for position, pmid in enumerate(ranked[:k], start=1)
        if pmid in relevant
    )


def ndcg_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int = 10) -> float:
    """Normalized discounted cumulative gain over the top ``k`` (binary relevance)."""
    _require_relevant(relevant)
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(ranked, relevant, k) / ideal_dcg
