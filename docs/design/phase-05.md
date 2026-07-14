# Phase 5 — Hard negatives, retriever v2, hybrid retrieval (design doc)

- **Release:** v0.4.0
- **Goal:** the dense-retrieval *quality* phase — mine hard negatives, retrain,
  fuse with BM25, and diagnose the remaining misses.

## Module layout

```
encoder/mining.py       mine_hard_negatives(): for each (query, positive_pmid),
                        pool the top passages from one or more retrievers, drop the
                        positive, and keep the hardest non-relevant as negatives ->
                        (anchor, positive, [negatives]) triples (Phase-3 machinery
                        already tokenizes and trains on these).
retrieval/hybrid.py     HybridRetriever: reciprocal rank fusion (RRF) of any set of
                        retrievers (BM25 + dense), behind the Retriever protocol.
eval/misses.py          sample_misses(): dev queries whose relevant PMID is missed at
                        cutoff k — the raw material for the failure taxonomy (Phase 11).
docs/design/failure-taxonomy.md   the committed taxonomy (vocabulary gap / granularity
                        / annotation noise), filled from a hand-audit of dev misses.
```

## Key decisions

- **Hard negatives = top-retrieved non-positives.** For a training query, the passages
  a *current* retriever ranks highest but that are not the gold passage are the hardest
  negatives (ADR-0004 deferred these from Phase 3). Pooling BM25 **and** dense catches
  both lexical and semantic confusability.
- **Retrain, don't re-architect.** Retriever v2 reuses the Phase-3 `train_retriever`
  with `(anchor, positive, negatives)` samples; the ablation is simply
  `{in-batch only}` vs `{+ hard negatives}` on the same embedder init.
- **RRF, not score mixing.** BM25 and cosine scores are not comparable, so hybrid
  retrieval fuses *ranks*: `score(d) = Σ_r 1/(k_rrf + rank_r(d))` (standard `k_rrf=60`).
  This needs no score normalization and is robust — the reason it is the default hybrid.
- **Honesty gate.** Retriever v2 must be ≥ v1 on **all** frozen-dev metrics to be
  adopted; if not, that is reported and diagnosed (RAG.md §7).

## Testing strategy (offline-only)

- Mining: negatives exclude the positive; pooled/deduped across retrievers; respects
  `num_negatives`; skips queries whose positive is absent.
- RRF: hand-computed fusion on two toy rankings; a document ranked well by both
  retrievers outranks one ranked well by only one; protocol conformance; determinism.
- Misses: returns exactly the queries missed at k; empty when all are hit.

## Environment constraint

The retriever-v2 delta and the hybrid row need the real corpus + trained embedder +
frozen dev split (real training run). The mining, fusion, and miss-sampling logic are
verified offline on synthetic fixtures; the BENCHMARKS rows and the taxonomy's audited
counts come from the real run — no number written from memory.

## Exit criteria tracking

| Criterion | Status |
|---|---|
| Retriever v2 ≥ v1 on all frozen-dev metrics | mining + retrain mechanism done (`mine_hard_negatives` → `train_retriever`); delta pending real run |
| Hybrid measured | RRF done + wired (`meridian ask --retriever hybrid`, `evaluate.py`); row pending real run |
| Failure taxonomy committed | **done** — [failure-taxonomy.md](failure-taxonomy.md) scaffold + `sample_misses`; audited counts pending real dev misses |
| ≥ 90% coverage held | maintained (97%) |

## Remaining user-triggered step

Mining, RRF fusion, and miss-sampling are complete and verified offline. The real
numbers need the trained retriever + frozen dev split:

1. `mine_hard_negatives([bm25, dense], store, train_pairs)` → retrain with
   `train_retriever` → retriever v2; compare to v1 on frozen dev (must be ≥ on all
   metrics to adopt — honesty gate).
2. `scripts/evaluate.py --retriever hybrid …` → the hybrid row in `BENCHMARKS.md`.
3. `sample_misses(retriever, dev, k=20)` → hand-audit 50 misses → fill the taxonomy
   counts.
