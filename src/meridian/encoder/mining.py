"""Hard-negative mining for retriever v2 (Phase 5).

For each training query with a known positive PMID, the passages a *current* retriever
ranks highest — excluding the gold passage — are the hardest negatives. Pooling BM25
and dense retrievers surfaces both lexical and semantic confusables. The resulting
``(anchor, positive, [negatives])`` triples feed the Phase-3 contrastive machinery
(``make_contrastive_samples_with_negatives`` → ``train_retriever``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from meridian.corpus.store import DocumentStore
from meridian.retrieval.pipeline import Retriever


def mine_hard_negatives(
    retrievers: Sequence[Retriever],
    store: DocumentStore,
    queries: Iterable[tuple[str, str]],
    *,
    num_negatives: int = 4,
    pool: int = 20,
) -> list[tuple[str, str, list[str]]]:
    """Mine hard negatives for each ``(query_text, positive_pmid)`` example.

    Returns ``(anchor, positive, negatives)`` text triples. Each query's negatives are
    the highest-ranked passages (pooled across ``retrievers``, deduplicated, positive
    excluded), capped at ``num_negatives``. Queries whose positive is not in the store,
    or that yield no negatives, are skipped.
    """
    triples: list[tuple[str, str, list[str]]] = []
    for query_text, positive_pmid in queries:
        positive = store.get(positive_pmid)
        if positive is None:
            continue

        negative_texts: list[str] = []
        seen: set[str] = {positive_pmid}
        for retriever in retrievers:
            for hit in retriever.retrieve(query_text, k=pool):
                if hit.pmid in seen or hit.document is None:
                    continue
                seen.add(hit.pmid)
                negative_texts.append(hit.document.chunk_text())
                if len(negative_texts) >= num_negatives:
                    break
            if len(negative_texts) >= num_negatives:
                break

        if negative_texts:
            triples.append((query_text, positive.chunk_text(), negative_texts))
    return triples
