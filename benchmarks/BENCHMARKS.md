# Benchmarks

> **Claims hygiene (house rule):** a number appears here only if it is reproduced
> by a committed, seeded script, with its MLflow run ID referenced. No number is
> estimated in prose. Frozen dev/test splits are created in Phase 2 and never
> touched afterward.

**Status:** All 12 build phases are implemented. Retrieval is **measured on real
PubMedQA data** (below); the remaining rows need the heavier training sets (MS MARCO,
SNLI/SciNLI, PQA-A) or the multi-GB domain-filtered PubMed baseline, and stay `TBD`
until those runs — never estimated in prose.

## Reproduction

Each result row references the script and MLflow run that produced it:

```
scripts/<benchmark>.py   →   MLflow run ID   →   row below
```

(Scripts arrive with their phases; this section lists them as they land.)

## Retrieval

### PubMedQA PQA-L retrieval benchmark (real, seed-averaged ablation)

A self-contained retrieval task from PubMedQA PQA-L: **1000 real abstracts**, each question
retrieving its own source abstract. **Clean 3-way split — no leakage**: train 590 (trains
the dense retriever + reranker), dev 239 (reported below), test held out. Main table
reproduced by `scripts/ablate_stage0.py`; the seed-variance study by
`scripts/variance_dense.py` (both take `--pqal data/pubmedqa/ori_pqal.json`).

**Main comparison** (dev n=239, seed 0):

| Config | R@5 | R@20 | R@100 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| **BM25** | **0.987** | 0.987 | 0.996 | **0.969** | **0.974** |
| Dense (from-scratch, supervised contrastive) | 0.360 | 0.506 | 0.757 | 0.276 | 0.314 |
| BM25 + cross-encoder rerank (pure) | 0.029 | 0.117 | 0.996 | 0.017 | 0.030 |
| BM25 + rerank (base-fused, graceful) | 0.983 | 0.987 | 0.996 | 0.651 | 0.736 |

Dense R@5 varies 0.36–0.42 across seeds; the seed-0 run is shown above and the
distribution below. (Hybrid RRF sits between BM25 and dense — R@5 0.774 in the earlier
`campaign_pubmedqa.py` run; it cannot beat a near-perfect BM25 here.)

**Stage-0 ablation** — does MLM pretraining help retrieval? (4 seeds, `variance_dense.py`):

| Dense trunk init | R@5 (mean ± std) | R@20 (mean ± std) |
|---|---|---|
| random-init | 0.382 ± 0.023 | 0.542 ± 0.025 |
| MLM Stage-0 (2 epochs) | 0.371 ± 0.022 | 0.548 ± 0.025 |

**Findings — measured, not spun (RAG.md §9):**

1. **BM25 wins on this lexically-easy task (0.987).** Questions reuse their abstract's
   vocabulary, so lexical retrieval is near-perfect. A small from-scratch dense model on 590
   pairs does not close that gap — and shouldn't be expected to at this scale.
2. **Supervised pairs are the robust lever; MLM Stage-0 is not — and we publish the ablation
   that overturns our own hypothesis.** Switching from self-supervised (abstract-half) pairs
   to 590 supervised question→abstract pairs lifts the from-scratch bi-encoder from R@5 ~0.01
   (`train_pubmedqa_dense.py`) to **0.38 ± 0.02** — a ~38× gain that is real and robust.
   Adding MLM Stage-0 pretraining on top changes **nothing measurable**: 0.371 ± 0.022 vs
   0.382 ± 0.023 over four seeds, a difference well inside one standard deviation (and
   directionally slightly *negative*). An earlier draft credited the gain to an MLM
   "curriculum"; a clean, seed-averaged comparison shows the *supervised data*, not Stage-0
   pretraining, does the work at this scale. Reporting this correction is the point.
3. **Pure cross-encoder reranking overfits and destroys a strong base — the fusion fix makes
   it safe.** A from-scratch reranker trained on 590 mined pairs separates positives from
   negatives on *train* but learns nothing transferable: on *dev* the gold passage falls from
   BM25 rank 1 to a median of ~54, so applied purely it drags R@5 to **0.029** (below
   random). The reranker now fuses with the base ranking by reciprocal-rank fusion
   (`base_weight`), which recovers R@5 to **0.983** — graceful degradation instead of
   catastrophe. Reranking stays off by default; a cross-encoder needs MS MARCO-scale
   relevance data to actually beat BM25 here.

### Domain-filtered PubMed corpus (~200K, ADR-0001) — pending baseline download

Metrics on the frozen dev split of the three-domain (cardiology/endocrinology/oncology)
corpus. All `TBD` until the multi-GB PubMed baseline is downloaded and ingested.

| Config | R@5 | R@20 | R@100 | MRR@10 | nDCG@10 | Run ID |
|---|---|---|---|---|---|---|
| BM25 (Phase 2) | TBD | TBD | TBD | TBD | TBD | TBD |
| Dense (Phase 3) | TBD | TBD | TBD | TBD | TBD | TBD |
| Retriever v2 (+hard neg, Phase 5) | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid RRF (Phase 5) | TBD | TBD | TBD | TBD | TBD | TBD |
| Dense + rerank (Phase 6) | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid + rerank (Phase 6) | TBD | TBD | TBD | TBD | TBD | TBD |

## ANN index quality (Phase 4)

Recall@10 vs brute-force ground truth, as a recall / latency / memory trade-off.
Reproduce with `scripts/benchmark_ann.py` (default: synthetic; `--embedding-index <dir>`
for the real corpus). Default backend chosen in [ADR-0005](../docs/adr/0005-default-index.md);
trade-off curve: [`figures/ann_tradeoff.png`](figures/ann_tradeoff.png).

| Index | Recall@10 | P50 latency | RAM | Run ID |
|---|---|---|---|---|
| Brute force | 1.000 | TBD | TBD | TBD |
| IVF | TBD | TBD | TBD | TBD |
| HNSW | TBD | TBD | TBD | TBD |

The three backends are implemented and benchmarked offline on synthetic vectors
(N=2000, dim=64): HNSW reaches recall@10 ≈ 0.999 at efSearch=16 below brute-force
latency; IVF trades recall for latency across nprobe. Real-corpus rows are filled from
the Phase-3 embeddings (no number written from memory).

## Faithfulness (Phase 8)

Computed by `meridian.verify.verify_grounded_answer` (NLI entailment of each cited
sentence). Real values come from the trained verifier over real generated answers.

| Metric | Value | Run ID |
|---|---|---|
| Citation precision | TBD | TBD |
| Citation recall | TBD | TBD |
| Hallucination rate | TBD | TBD |
| Verifier–human agreement | TBD | TBD |

## Calibration / abstention (Phase 9)

Risk-coverage trade-off (error rate vs fraction answered), with the operating point at
~80% coverage (ADR-0007). Reproduce with `scripts/benchmark_calibration.py` (default:
synthetic calibrated distribution; `--records` for the real dev gates). Curve:
[`figures/risk_coverage.png`](figures/risk_coverage.png). Real thresholds and the
off-domain abstain rate come from the trained gates on the frozen dev split.

## End-to-end (Phase 11)

PubMedQA PQA-L labels (n=1000): yes 552 / no 338 / maybe 110 → **majority-class baseline
0.552**. A from-scratch 3-class classifier (Polaris pair head over question+context,
96-dim, trained on 808 examples, evaluated on 192) reaches **0.531** — reproduce with
`scripts/train_pubmedqa_classifier.py`. Neither Stage-0 pretraining nor class weighting
lets it beat the majority baseline (below).

| Metric | Value | Run |
|---|---|---|
| PubMedQA majority-class baseline (eval split) | 0.547 | build_pubmedqa.py |
| PubMedQA accuracy (from-scratch 3-class, MLM-pretrained) | **0.531** | train_pubmedqa_classifier.py |
| — variant: no Stage-0 pretraining | 0.495 | `... --no-pretrain` |
| — variant: inverse-frequency class weights | 0.411 | `... --class-weights` |
| PubMedQA accuracy (full pipeline, trained at scale) | TBD | — |
| Answer coverage @ operating point | TBD | — |

**Honest finding (below baseline; two levers tried, neither helps).** The from-scratch
classifier reaches 0.531 and does **not** beat the 0.547 majority baseline. Two things were
tried and reported, not cherry-picked:

- **Stage-0 MLM pretraining**: 0.495 → 0.531 in a single run. Read this cautiously — the
  seed-averaged retrieval ablation above shows MLM's effect is *within noise*, so a
  single-run +3.6 pts on the classifier is not strong evidence of a real gain.
- **Class weighting** (up-weight the 11% *maybe* class 3.2×): 0.531 → **0.411**. It *hurts*
  — a model too weak to separate the classes trades overall accuracy for minority recall.

We deliberately do **not** keep tuning weights toward 0.547: the eval split is frozen and
fitting it would violate claims hygiene. 800 examples and a tiny model are far too little
for a reasoning task where published results (~68–78%) use large *pretrained* models. This
is the project's honest scope (RAG.md §2): a laptop-scale, from-scratch system is **not
accuracy-competitive with large models**, and does not pretend to be. Its value is the
*engineering and safety design* — grounded, citation-constrained generation; NLI
verification; calibrated abstention — not a headline accuracy number. The path to a real
number is the full training sets (MS MARCO / SNLI / PQA-A), which the pipeline implements
but has not run at scale.

## Systems / latency (Phase 10)

Measured over 527 real PubMedQA dev queries on the 1000-abstract corpus
(`scripts/benchmark_latency.py`). Search = BM25 top-k; Answer = extractive answerer
(retrieve + sentence scoring). Embed / rerank / generate / verify latencies are added
once those models are trained.

| Stage | P50 | P95 | n |
|---|---|---|---|
| Search (BM25) | 0.84 ms | 1.36 ms | 527 |
| Answer (extractive) | 1.12 ms | 1.54 ms | 527 |
| Embed / Rerank / Generate / Verify | TBD | TBD | — |
