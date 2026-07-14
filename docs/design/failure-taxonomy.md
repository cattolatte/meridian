# Retrieval failure taxonomy (Phase 5 → Phase 11)

A hand-audit of dev-split retrieval misses, categorized so the Phase 11 error-attribution
study knows *why* retrieval fails, not just *how often*. **Frozen dev split only** — the
test split is never inspected before Phase 11.

## How the misses are sampled

```python
from meridian.eval import sample_misses, load_frozen_split
misses = sample_misses(retriever, load_frozen_split("benchmarks/splits/dev.json", CHECKSUM), k=20)
```

Each `MissRecord` carries the question, the relevant PMID(s), and what was retrieved
instead — enough to categorize by reading the query against the top passages. The audit
target is **50 dev misses** (plan.md Phase 5).

## Categories

| Category | Definition | Typical fix |
|---|---|---|
| **Vocabulary gap** | Query and gold passage use different surface terms for the same concept (synonyms, abbreviations, drug brand vs generic). Lexical retrieval (BM25) fails; dense should help. | dense retrieval; domain-adapted embeddings (Stage B) |
| **Granularity mismatch** | The answer is a sub-part of a longer passage, or the query is broader/narrower than the chunk unit (ADR-0002). | reranking; full-text chunking (Phase 13 stretch) |
| **Annotation noise** | The "relevant" PMID is a weak/incorrect label (PubMedQA mapping artifact); the retrieved passages are actually reasonable. | none — excluded from the honest denominator, counted separately |

## Audited counts — pending real dev misses

Filled from the hand-audit once the real corpus + trained retriever exist. No counts
are written from memory (claims hygiene).

| Category | Count (of 50) | Example query id | Notes |
|---|---|---|---|
| Vocabulary gap | TBD | TBD | |
| Granularity mismatch | TBD | TBD | |
| Annotation noise | TBD | TBD | |

## Feeds

- **Phase 5:** motivates hard-negative mining (semantic confusables) and hybrid RRF
  (lexical + semantic coverage).
- **Phase 11:** the attribution pie separates retrieval failures (these categories)
  from rerank / generation / verification failures.
