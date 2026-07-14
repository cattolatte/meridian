"""Sample retrieval misses for the failure taxonomy (Phase 5 → Phase 11).

A *miss* is a dev query whose relevant PMID(s) the retriever fails to return within a
cutoff ``k``. :func:`sample_misses` collects them with what *was* retrieved, so they
can be hand-categorized (vocabulary gap / granularity / annotation noise) in
``docs/design/failure-taxonomy.md`` and later attributed in the Phase 11 study.
"""

from __future__ import annotations

from dataclasses import dataclass

from meridian.eval.qrels import EvalSet
from meridian.retrieval.pipeline import Retriever


@dataclass(frozen=True, slots=True)
class MissRecord:
    """A query the retriever missed at the cutoff, with what it returned instead."""

    query_id: str
    question: str
    relevant_pmids: tuple[str, ...]
    retrieved_pmids: tuple[str, ...]


def sample_misses(retriever: Retriever, eval_set: EvalSet, *, k: int = 20) -> list[MissRecord]:
    """Return the queries whose relevant PMIDs are all absent from the top ``k``.

    Queries with no relevant PMIDs are skipped (nothing to miss).
    """
    misses: list[MissRecord] = []
    for query in eval_set.queries:
        if not query.relevant_pmids:
            continue
        retrieved = [hit.pmid for hit in retriever.retrieve(query.question, k=k)]
        if not (set(retrieved) & query.relevant_pmids):
            misses.append(
                MissRecord(
                    query_id=query.query_id,
                    question=query.question,
                    relevant_pmids=tuple(sorted(query.relevant_pmids)),
                    retrieved_pmids=tuple(retrieved),
                )
            )
    return misses
