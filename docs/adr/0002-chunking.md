# ADR-0002: Chunking policy — one title+abstract chunk per document

- **Status:** Accepted
- **Date:** 2026-07-13

## Context

Retrieval and generation operate over *chunks*, not raw documents. The chunk unit
determines index granularity, embedding semantics, and citation resolution. PubMed
records in the Phase 1 corpus are (title, abstract) pairs; abstracts are naturally
bounded, single-topic passages of roughly 150–350 words. We must fix the chunking
unit before the document store schema and the tokenizer are built, because both
downstream stages assume it.

## Decision

1. **Chunk unit = `title + "\n\n" + abstract`, exactly one chunk per document.**
   The title is prepended because it carries high-signal terms (drug names,
   conditions) often absent from the abstract body, and improves both lexical
   (BM25) and dense retrieval.
2. **Chunk identity = PMID.** Because there is one chunk per document, the PMID is
   the citation unit end-to-end. This keeps citations (RAG.md §3) trivially
   resolvable and matches how PubMedQA relevance is annotated (per-PMID).
3. **No sub-abstract splitting at this stage.** Abstracts fit comfortably within the
   encoder context budget (ADR-0001 model sizes), so splitting would only fragment
   context and complicate citation without measured benefit.

## Alternatives considered

- **Sentence- or paragraph-level chunks:** finer retrieval granularity, but PubMed
  abstracts are already short; splitting raises index size and citation-alignment
  cost for no demonstrated recall gain at this corpus size.
- **Sliding-window chunks:** designed for long full-text articles; irrelevant while
  the corpus is abstracts only.

## Full-text stretch criteria (when this ADR is superseded)

Structure-aware chunking of full-text open-access articles (a Phase 13 stretch
track) is adopted only when **all** of the following hold, and only via a
superseding ADR:

1. The abstract-only pipeline is complete through v1.0 with measured baselines.
2. A measured retrieval ceiling is traced to missing full-text content in the
   Phase 11 error-attribution study (i.e., relevant evidence exists only in article
   bodies, not abstracts).
3. Open-access full text is available under terms cleared in `license-review.md`.

Until then, the corpus is abstracts and the chunk is the document.

## Consequences

- The document store carries one indexable text field per record (composed at read
  time from title + abstract); no separate chunk table is needed in Phase 1.
- Citations are PMIDs throughout; the verifier (Phase 8) checks answer sentences
  against title+abstract spans.
- Adopting full text later is an additive, ADR-gated change, not a rewrite.
