# Model Card — Meridian (placeholder)

> **Placeholder.** Meridian is at Phase 0 (scaffolding); no models are trained yet.
> This card is filled in as components ship, and every quantitative field is
> populated **only** from the committed evaluation harness — never estimated.
> See [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md).

## Intended use

- **Intended:** a research literature assistant that answers biomedical questions
  strictly from retrieved, cited PubMed abstracts, or abstains.
- **Not intended:** medical advice, diagnosis, or treatment decisions. Every
  surface carries: *"Research literature assistant. Not medical advice."*

## Scope

- **Corpus:** filtered PubMed abstracts (cardiology, endocrinology, oncology per
  [ADR-0001](docs/adr/0001-scope.md)); target ~200K abstracts.
- **Components (all trained in-house):** BPE tokenizer, dense bi-encoder retriever,
  cross-encoder reranker, NLI faithfulness verifier (Polaris), grounded generator
  (Zenith). No external NLP models or embedding APIs.

## Components

| Component | Framework | Size | Status |
|---|---|---|---|
| BPE tokenizer | Polaris | — | TBD (Phase 1) |
| Bi-encoder retriever | Polaris | ~10–30M | TBD (Phase 3) |
| Cross-encoder reranker | Polaris | ~10–30M | TBD (Phase 6) |
| Answerability gate | Polaris | shares reranker | TBD (Phase 9) |
| NLI faithfulness verifier | Polaris | ~10–30M | TBD (Phase 8) |
| Grounded generator | Zenith | ~30–125M | TBD (Phase 7) |

## Evaluation

TBD — populated from the harness (retrieval, faithfulness, calibration, latency).

## Limitations & risks

- Small generator; quality is additive, not load-bearing (extractive fallback).
- Laptop-scale corpus and training; not a large-model competitor.
- Verifier reliability on biomedical text is measured and hand-audited (Phase 8),
  not assumed. Low-confidence outputs abstain rather than guess.

## Ethical considerations

Not for clinical use. Answers reflect only what the retrieved literature states,
with citations; the system is designed to abstain rather than hallucinate.
