# ADR-0003: Tokenizer training corpus mix and vocabulary size

- **Status:** Accepted
- **Date:** 2026-07-13

## Context

Every Meridian encoder and the generator share a single BPE vocabulary trained here
in Phase 1 (RAG.md §4.3). Two choices must be fixed before training: **what text the
tokenizer learns on**, and **how large the vocabulary is**.

The tension: a biomedical-only tokenizer minimizes fertility (tokens per word) on
PubMed text — good for the retriever and verifier — but over-fragments general
English. That matters because the retriever and reranker are trained in Phase 3/6
**Stage A on MS MARCO** (general web passages) before domain adaptation. A tokenizer
that shatters ordinary English into many subwords would cripple Stage-A training and
inflate sequence lengths on exactly the data used to teach general retrieval
competence. The plan flags this risk explicitly (plan.md Phase 1, Task 5).

## Decision

1. **Mixed training corpus, ≈ 70% biomedical / 30% general English**, sampled by
   token count (not document count) so the ratio reflects what the merge-learner
   actually sees.
   - **Biomedical portion:** title+abstract chunks (ADR-0002) from the Phase 1
     PubMed corpus.
   - **General portion:** a sample of **MS MARCO passages** — the same general-text
     distribution the retriever/reranker train on in Phase 3/6 Stage A. Tying the
     tokenizer's general half to MS MARCO directly mitigates the Stage-A fertility
     risk, and adds no new dataset dependency (MS MARCO is already on the roadmap;
     license cleared in `license-review.md`).
   - The mix ratio is a config parameter; 70/30 is the default and the reported
     configuration.
2. **Vocabulary-size sweep: 16K and 32K.** Both are trained and compared by
   fertility on held-out biomedical vs general text; the smaller size that stays
   within a small fertility margin of the larger is adopted as the default, and the
   choice plus the fertility table is recorded in `benchmarks/corpus.md`.
3. **Special tokens reserved at training time:** `<pad>` and `<unk>`. The MLM
   `<mask>` token and any task tokens are a Phase 3 concern and are introduced by a
   superseding note there (they must not silently shift ids of an already-trained
   vocabulary).
4. **Fertility is the selection metric**, defined as mean subword tokens per
   whitespace word, measured separately on a biomedical held-out sample and a
   general held-out sample. Reported to three decimals with the sample sizes.

## Alternatives considered

- **Biomedical-only (100% PubMed):** lowest biomedical fertility, but degrades
  Stage-A MS MARCO training — rejected for the reason above.
- **General-only or 50/50:** wastes vocabulary budget on general English at the
  expense of the domain the system actually serves.
- **Single fixed vocab size (e.g., 32K):** foregoes the fertility/size trade-off
  evidence that the sweep produces cheaply; a sweep is standard tokenizer practice
  and provides an honest, reported basis for the default.
- **Wikipedia/other web sample for the general portion:** a reasonable general
  source, but MS MARCO is already required downstream and aligning the two removes a
  dependency and directly targets the Stage-A risk.

## Consequences

- Phase 1 ingestion must be able to sample both PubMed chunks and a MS MARCO passage
  sample for tokenizer training; the trainer takes both corpora as iterables with a
  configurable token-count ratio.
- The tokenizer artifact records its training config (mix ratio, vocab size, source
  sample sizes, corpus checksum) so the vocabulary is reproducible and versioned.
- If Stage-A training later shows fertility problems despite the mix, the remedy is a
  new ADR adjusting the ratio or vocab size — not an ad-hoc retrain.
