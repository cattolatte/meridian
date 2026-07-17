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

Every component's training/inference pipeline is **implemented and tested offline**;
each is trained on real data by a committed, seeded script (not yet run at scale).

| Component | Framework | Size | Status |
|---|---|---|---|
| BPE tokenizer | Polaris | — | implemented (mixed-corpus BPE, versioned artifact) |
| Bi-encoder retriever | Polaris | ~10–30M | implemented (MLM → contrastive; brute/IVF/HNSW) |
| Cross-encoder reranker | Polaris | ~10–30M | implemented (pointwise BCE pair scorer) |
| Answerability gate | Polaris | shares pair head | implemented (2-class) |
| NLI faithfulness verifier | Polaris | ~10–30M | implemented (3-class entailment) |
| Grounded generator | Zenith | ~30–125M | implemented (LoRA SFT, citation-constrained decoding, abstain) |

"Implemented" = the model, training, and inference run end-to-end offline on a tiny
sample; documented **quality numbers require the real training runs** (deferred).

## Evaluation

Populated from the harness (retrieval, faithfulness, calibration, latency) once the real
corpus is ingested and the components are trained. Every cell is reproducible from a
committed script; no number is estimated. See [BENCHMARKS.md](benchmarks/BENCHMARKS.md).

## Limitations & risks

- Small generator; quality is additive, not load-bearing (extractive fallback).
- Laptop-scale corpus and training; not a large-model competitor.
- Verifier reliability on biomedical text is measured and hand-audited (Phase 8),
  not assumed. Low-confidence outputs abstain rather than guess.

## Ethical considerations

Not for clinical use. Answers reflect only what the retrieved literature states,
with citations; the system is designed to abstain rather than hallucinate.
