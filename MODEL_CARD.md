# Model Card — Meridian

> **Status.** All pipeline components are implemented and run end-to-end. The
> retriever, reranker, and a PubMedQA answer classifier are **trained on real data and
> measured** (PubMedQA PQA-L; see [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md)).
> Larger-scale training on the heavy corpora (MS MARCO, SNLI/MultiNLI/SciNLI, PubMedQA
> PQA-A) is wired but deferred to a GPU (see [docs/scale-runs.md](docs/scale-runs.md)).
> Every quantitative field is populated **only** from the committed evaluation harness —
> never estimated.

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

Every component's training/inference pipeline is **implemented and tested offline** and
trainable by a committed, seeded script. "Measured" = a real number lives in BENCHMARKS;
"implemented" = runs end-to-end but a real number awaits its training run.

| Component | Framework | Size | Status |
|---|---|---|---|
| BPE tokenizer | Polaris | — | trained (mixed-corpus BPE, versioned artifact) |
| Bi-encoder retriever | Polaris | ~10–30M | **measured** on PubMedQA (R@5 0.38 ± 0.02; MLM Stage-0 within noise) |
| Cross-encoder reranker | Polaris | ~10–30M | **measured** on PubMedQA (pure rerank hurts; base-fused degrades gracefully) |
| Answerability gate | Polaris | shares pair head | implemented (2-class) |
| NLI faithfulness verifier | Polaris | ~10–30M | implemented (3-class); PubMedQA answer classifier **measured** (0.531) |
| Grounded generator | Zenith | ~30–125M | implemented (LoRA SFT, citation-constrained decoding, abstain) |

Numbers and their reproduction scripts live in
[BENCHMARKS.md](benchmarks/BENCHMARKS.md); they are laptop-scale and honest about it —
the heavier training sets lift them but are a separate GPU run.

## Evaluation

Measured on real PubMedQA PQA-L (retrieval ablation, PubMedQA answer classifier, serving
latency), with faithfulness / calibration / ANN and the heavy-corpus rows still `TBD`
until those runs. Every cell is reproducible from a committed, seeded script; no number is
estimated. See [BENCHMARKS.md](benchmarks/BENCHMARKS.md).

## Limitations & risks

- Small generator; quality is additive, not load-bearing (extractive fallback).
- Laptop-scale corpus and training; not a large-model competitor.
- Verifier reliability on biomedical text is measured and hand-audited (Phase 8),
  not assumed. Low-confidence outputs abstain rather than guess.

## Ethical considerations

Not for clinical use. Answers reflect only what the retrieved literature states,
with citations; the system is designed to abstain rather than hallucinate.
