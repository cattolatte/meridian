# ADR-0001: Project scope — corpus domains, corpus size, model budget, GPU budget

- **Status:** Accepted
- **Date:** 2026-07-13

## Context

Meridian is a from-scratch grounded RAG engine over biomedical literature, built
part-time on laptop-scale (Apple MPS) hardware with optional rented CUDA. The
biggest project risk is scope creep (four trained models + a from-scratch ANN
index + serving). Scope must be frozen before any data or ML work so that every
later phase optimizes inside fixed boundaries rather than renegotiating them.

## Decision

1. **Corpus domains:** cardiology, endocrinology, and oncology, selected by MeSH
   term filters over the PubMed baseline (exact MeSH lists fixed in Phase 1
   alongside the ingest filter).
2. **Initial corpus size:** ~200K abstracts (≈ 60M tokens). Scaling to 1M+ is a
   post-1.0 stretch track, not a v1.0 goal.
3. **Model size budget:**
   - Encoders (bi-encoder retriever, cross-encoder reranker, NLI verifier):
     ~10–30M parameters each, MPS-trainable, all Polaris.
   - Generator: base Zenith model ~30M (MPS-trainable); upgrade to ~125M only if
     the rented-GPU budget allows after core phases are funded.
4. **Rented-GPU budget cap: $150 total** for the life of the project through
   v1.0. Spending is recorded per run; when the cap is hit, all remaining
   training stays on MPS at the smaller model sizes.
5. **Package name:** `meridian-rag` (verified available on PyPI 2026-07-13;
   fallbacks `meridian-nlp`, `meridian-qa` remain unused).

## Alternatives considered

- **Single-domain corpus (e.g., cardiology only):** smaller and simpler, but
  three related domains give more realistic vocabulary diversity and a more
  credible retrieval story at a corpus size that still fits on a laptop.
- **Full PubMed baseline (~36M records):** infeasible for laptop indexing and
  embedding within budget; deferred to the stretch scale-up track.
- **Larger generator from the start (125M+):** blows the GPU budget before the
  system exists end-to-end; the design deliberately makes generator quality
  additive (extractive fallback), not load-bearing.

## Consequences

- Phase 1 ingest filters by the three domains' MeSH terms; corpus statistics
  will verify the ~200K target is met and report the actual count.
- All training configs must fit the model-size budget; any exception requires a
  superseding ADR.
- GPU spend is tracked in the phase design docs; the $150 cap is a hard stop.
