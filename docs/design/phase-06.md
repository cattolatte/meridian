# Phase 6 — Cross-encoder reranker (design doc)

- **Release:** v0.5.0
- **Goal:** precision at the top of the ranking, where the generator reads.
- **Upstream:** consumes `polaris-nlp==1.4.0` — `SentencePairClassifier`
  (`num_classes=1` for a ranking score, `pooling="cls"`, `.encoder` for Stage-0
  transfer), `collate_pairs` (`[CLS] a [SEP] b [SEP]`), and `train_bpe` special-token
  reservation (§5).

## Module layout

```
tokenization/training.py   now reserves <mask>/<cls>/<sep> so one vocabulary serves
                           the MLM, embedder, and pair models (supersedes ADR-0003's
                           deferral — Polaris 1.3.0 §5 enables it).
reranker/
  artifact.py   RerankerConfig + build_reranker (SentencePairClassifier) +
                versioned save/load (arch config + weights).
  data.py       make_pair_samples(): (query, passage, label) -> (Encoding, Encoding,
                int) triples; from MS MARCO (q, pos, neg) -> pos=1 / neg=0.
  training.py   train_reranker(): collate_pairs + pointwise BCE over the num_classes=1
                score; initialized from the Stage-0 trunk (transfer_encoder_to).
retrieval/rerank.py   RerankingRetriever: wrap a base Retriever, take its top-N, rerank
                the (query, passage) pairs with the cross-encoder, return top-k.
                Same Retriever protocol — composes with BM25/dense/hybrid.
```

## Key decisions

- **Pointwise BCE on a single logit.** `SentencePairClassifier(num_classes=1)` emits one
  relevance logit; train it with `(query, positive)=1` / `(query, negative)=0` and
  `BCEWithLogitsLoss`. Simple, standard, and enough for top-k reranking; pairwise/listwise
  is a later refinement if measured to help.
- **One trunk, reused.** The reranker is initialized from the same Stage-0 MLM checkpoint
  as the embedder (ADR-0004), then domain-adapted on mined PubMed pairs.
- **Rerank is a composable stage, config-gated.** `RerankingRetriever` wraps *any* base
  retriever (retrieve top-N → rerank → top-k). `meridian ask --rerank` turns it on; the
  base retriever and candidate depth are configurable. Latency lives here — the rerank
  is the hog, measured honestly.
- **Honesty gate.** Adopt only if nDCG@10 improves by a measured margin; the disable flag
  keeps the un-reranked path one config away.

## Testing strategy (offline-only)

- Tokenizer: reserves all five special tokens; `<cls>`/`<sep>`/`<mask>` ids present.
- Reranker: pointwise BCE loss decreases on separable pairs; determinism; artifact
  round-trip reproduces scores; transfer copies the Stage-0 trunk.
- RerankingRetriever: reorders a base ranking by the model's scores; returns ≤ k; caps at
  the candidate pool; protocol conformance; documents attached.

## Environment constraint

The nDCG@10 lift and the latency delta need the real corpus + trained reranker + frozen
dev split. The pair model, training, and rerank stage are verified offline on synthetic
fixtures; the reranker row and latency numbers in `BENCHMARKS.md` come from the real run
— none written from memory.

## Exit criteria tracking

| Criterion | Status |
|---|---|
| Reranker improves nDCG@10 by a measured margin | mechanism; delta pending real run |
| Config flag to disable it | planned (Milestone 3: `--rerank`) |
| Latency documented | planned (rerank-stage benchmark) |
| ≥ 90% coverage held | maintained |
