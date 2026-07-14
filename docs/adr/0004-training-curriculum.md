# ADR-0004: Dense retriever training curriculum

- **Status:** Accepted
- **Date:** 2026-07-14

## Context

Phase 3 trains the dense bi-encoder retriever (a Polaris `TextEmbedder`, consumed
from `polaris-nlp==1.2.0`). A from-scratch, randomly-initialized bi-encoder trained
contrastively at laptop scale underperforms badly — the representation has to be
learned twice (language, then retrieval) from a small signal. We must fix the
training curriculum before writing the training code, and decide what is ablated so
every claim has an honest denominator (baselines before models).

Polaris 1.2.0 provides exactly the primitives this needs: MLM pretraining
(`polaris.pretraining.pretrain`), `TextEmbedder`, `MaskedLanguageModel.transfer_encoder_to`
for trunk transfer, `collate_contrastive`, `info_nce_loss`, and `train_contrastive`.

## Decision

Train in three stages, each initialized from the previous:

1. **Stage 0 — MLM pretraining.** Masked-language-model-pretrain the encoder trunk on
   the Phase-1 PubMed corpus using Polaris's existing `pretrain(...)` pipeline. This
   is what makes a from-scratch bi-encoder viable, and it showcases Polaris's flagship
   pretraining feature. **The Stage-0 checkpoint is kept and reused** to initialize the
   cross-encoder reranker (Phase 6) and the NLI verifier (Phase 8) — one pretrained
   trunk, four downstream heads.
2. **Stage A — general contrastive.** Initialize a `TextEmbedder` from the Stage-0
   trunk (`transfer_encoder_to`) and train it contrastively on **MS MARCO** triples
   (`train_contrastive` + InfoNCE, in-batch negatives) for general retrieval competence
   on ordinary English.
3. **Stage B — domain adaptation.** Continue training on self-mined **title↔abstract**
   pairs plus **PQA-A question↔abstract** pairs, adapting the general retriever to
   biomedical text.

**Ablations (on the frozen dev split):**
- `{random-init vs Stage-0 init}` — the direct sequel to Polaris's pretraining-ablation
  story; expected to show MLM pretraining is load-bearing.
- `{Stage A vs Stage A+B}` — does domain adaptation pay rent?

**Honesty clause:** if dense retrieval loses to BM25 on the frozen dev metrics, the
number is published anyway with a diagnosis, and iterated on in Phase 5
(hard-negative mining, hybrid RRF). BM25 is the denominator (Phase 2), not a rival to
be beaten quietly.

## Alternatives considered

- **Skip Stage 0 (contrastive from random init).** Rejected: known to underperform at
  this scale; the ablation will document exactly how much.
- **Skip Stage A (domain-only).** Rejected: MS MARCO teaches general query↔passage
  matching that a ~200K-abstract corpus alone cannot; the tokenizer's 30% general mix
  (ADR-0003) exists precisely so Stage A isn't crippled.
- **In-batch negatives only, forever.** Accepted *for Phase 3*; hard-negative mining is
  the explicit subject of Phase 5, so it is deliberately deferred (the InfoNCE path
  already supports explicit negatives when Phase 5 arrives).

## Consequences

- Meridian wraps Polaris's training entry points; it does **not** reimplement training.
- Configs are seeded and reproducible; the Stage-0 checkpoint is a versioned artifact
  with its own metadata (corpus checksum, hyperparameters).
- Real training requires the ingested corpus, an MS MARCO sample, and MPS/CUDA compute
  — networked/heavy steps the user runs. The pipeline and search path are verified
  offline on tiny synthetic data with small models; the ablation table's real numbers
  are filled from the harness, never estimated (claims hygiene).
- Retriever quality metrics land in `benchmarks/BENCHMARKS.md` as the dense row beside
  the BM25 row.
