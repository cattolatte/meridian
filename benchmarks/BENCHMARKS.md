# Benchmarks

> **Claims hygiene (house rule):** a number appears here only if it is reproduced
> by a committed, seeded script, with its MLflow run ID referenced. No number is
> estimated in prose. Frozen dev/test splits are created in Phase 2 and never
> touched afterward.

**Status:** All 12 build phases are implemented, and every section below carries **real
measured numbers** — retrieval, ANN, NLI verifier quality, faithfulness, calibration, the
PubMedQA end-to-end classifier, and per-stage latency. Three things are explicitly **not
run** rather than estimated, and say so where they appear: the ~200K domain-filtered PubMed
corpus (needs the multi-GB baseline), the generator-based pipeline and its `generate`
latency (needs the Phase-7 Zenith generator), and verifier–human agreement (needs human
annotation). No number here is estimated in prose.

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

Chart: [`figures/retrieval_ablation.png`](figures/retrieval_ablation.png).

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

### Domain-filtered PubMed corpus (~200K, ADR-0001) — not run

This larger three-domain (cardiology/endocrinology/oncology) corpus requires the multi-GB
PubMed baseline download + ingest, which has **not been run**. It is not estimated here;
the measured retrieval numbers above come from the real 1000-abstract PubMedQA corpus.
`scripts/ingest.py` builds this corpus when the baseline is available.

## ANN index quality (Phase 4)

Recall@10 vs brute-force ground truth on the **real corpus** (N=1000 abstracts, dim=256,
100 queries), as a recall / latency / memory trade-off. Reproduce with
`uv run python scripts/benchmark_ann.py --embedding-index <index dir> --k 10`. Default
backend chosen in [ADR-0005](../docs/adr/0005-default-index.md); trade-off curve:
[`figures/ann_tradeoff.png`](figures/ann_tradeoff.png).

| Index | Recall@10 | Mean query latency | Memory |
|---|---|---|---|
| Brute force (exact) | 1.000 | 0.282 ms | 1.0 MB (vectors) |
| IVF (nprobe=16) | 0.975 | 0.215 ms | +1.1 MB |
| **HNSW (efSearch=16)** | **0.996** | **0.262 ms** | **+0.5 MB** |
| HNSW (efSearch=32) | 1.000 | 0.403 ms | +0.5 MB |

**Finding.** HNSW dominates on this corpus: 0.996 recall *below* brute-force latency, at
half the index overhead of IVF, and exact recall (1.000) by efSearch=32. IVF trades recall
steeply for latency (0.413 at nprobe=1 → 0.975 at nprobe=16). At N=1000 the absolute
latencies are all sub-millisecond, so the ANN win is structural, not yet load-bearing.

## NLI verifier quality (Phase 8)

3-class NLI accuracy on the held-out **SNLI dev** split (n=5000; chance = 0.333), trained
from scratch on SNLI+MultiNLI with `scripts/train_verifier.py --eval-nli ... --eval-every 1`
(best-epoch checkpoint). Every row is a real run on Apple MPS.

| Config | Tokenizer | Data (pairs) | LR | SNLI dev accuracy |
|---|---|---|---|---|
| 96-dim, 2 layers | biomedical, vocab 1.5k | 10k | 1e-3 | 0.415 |
| 256-dim, 4 layers | biomedical, vocab 1.5k | 100k | 1e-3 | 0.464 |
| 256-dim, 4 layers | **English, vocab 8k** | 100k | 1e-3 | 0.524 |
| 256-dim, 4 layers | English | 300k | 1e-3 | 0.583 |
| 256-dim, 4 layers | English | 600k | 1e-3 | 0.668 |
| 256-dim, 4 layers | English | 942k (full) | 1e-3 | 0.742 |
| 384-dim, 6 layers | English | 942k (full) | 1e-3 | 0.607 ⚠️ |
| **384-dim, 6 layers** | **English** | **942k (full)** | **3e-4** | **0.783** |

Chart: [`figures/verifier_progression.png`](figures/verifier_progression.png).

**Findings — three levers, measured in order:**

1. **Tokenizer (+6 pts).** A vocab-1.5k tokenizer trained ~70% on biomedical text shreds
   everyday SNLI English. Rebuilding it on the NLI corpus itself (vocab 8k, general
   fertility 1.34) lifted 0.464 → 0.524 with nothing else changed.
2. **Data (+22 pts).** The dominant lever: 100k → 942k pairs took 0.524 → 0.742 at fixed
   capacity, still climbing at the end.
3. **Learning rate (+17.6 pts) — capacity is worthless without it.** The identical
   384×6 model scored **0.607 at lr 1e-3** (dev accuracy oscillating, final train loss
   0.893) and **0.783 at lr 3e-4** (smooth monotone climb, train loss 0.521). A deeper
   model fitting the *training* set worse is an optimization failure, not a data or
   capacity limit. Per-epoch eval with best-checkpointing (`--eval-every`) is what caught
   the peak instead of shipping an overfit final epoch.

At 0.783 this is a genuinely competent from-scratch NLI model (chance 0.333); large
pretrained models reach ~0.90+ on SNLI, which needs pretraining scale outside this
project's laptop-scale scope (RAG.md §2).

## Faithfulness (Phase 8)

`meridian.verify.verify_grounded_answer` runs the trained NLI verifier over each cited
sentence of a real answer. Measured over all **527 PubMedQA dev queries** answered by the
extractive answerer, verified by the SNLI+MultiNLI verifier (dev accuracy 0.7828):
`uv run python scripts/benchmark_faithfulness.py --db <store> --split <dev split>
--verifier <dir> --tokenizer <tokenizer>`.

| Metric | Value | Run |
|---|---|---|
| Citation precision | 0.751 | benchmark_faithfulness.py |
| Citation recall | 0.750 | benchmark_faithfulness.py |
| Hallucination rate | 0.437 | benchmark_faithfulness.py |
| Fully-grounded answers | 0.000 | benchmark_faithfulness.py |
| Verifier–human agreement | *not measured* — requires human annotation (not performed) | — |

**Honest finding — the verifier does not transfer to biomedical text.** Extractive answers
quote their cited abstract **verbatim**, so a perfect verifier would label every sentence
ENTAILMENT and report a 0.000 hallucination rate. It reports **0.437**. That number is a
*verifier error rate on out-of-domain text*, not a generator hallucination rate: a verifier
trained on SNLI/MultiNLI (everyday English: "a man playing guitar") misjudges dense
biomedical prose. This is exactly the gap ADR-0004 anticipates with **SciNLI domain
adaptation**, which has not been run. Until it is, the Gate-3 faithfulness check would
reject a large share of correctly-grounded answers — the fail-safe ladder degrades toward
extractive/abstain, which is safe but conservative.

## Calibration / abstention (Phase 9)

Risk-coverage trade-off (error rate vs fraction answered) from the **real Gate-1 retrieval
confidence** (top-1 vs top-5 BM25 margin) over the frozen PubMedQA dev split (n=527), with
the operating point at ~80% coverage (ADR-0007). Curve:
[`figures/risk_coverage.png`](figures/risk_coverage.png). Reproduce with
`uv run python scripts/benchmark_calibration.py --db <store> --split <dev split>`.

| Metric | Value | Run |
|---|---|---|
| Coverage at the operating point | 0.801 | benchmark_calibration.py |
| Error rate among answered, at that point | **0.000** | benchmark_calibration.py |
| Answer coverage (extractive answerer, dev) | 1.000 | benchmark_faithfulness.py |

**Finding.** Gate 1 is well-ordered on this corpus: at 80% coverage the retrieval error
rate is **zero** — every question the gate is confident about has its gold abstract in the
top-5. That follows from BM25's 0.987 R@5 (few errors, and they are the low-confidence
ones). The extractive answerer abstains on none of the 527 dev queries, so coverage is
1.000 before Gate 1 is applied; the gate is what buys the selective-answering trade-off.

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
| Answer coverage @ operating point | 0.801 (err 0.000) | benchmark_calibration.py |
| PubMedQA accuracy (full pipeline w/ generator) | *not run* — needs the Phase-7 Zenith generator (untrained) | — |

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

Per-stage wall-clock over real PubMedQA dev queries on the 1000-abstract corpus, **CPU**,
with the trained embedder / reranker / verifier loaded (`scripts/benchmark_latency.py`
with `--embedder --reranker --verifier --tokenizer`). Search = BM25 top-k; Answer =
extractive answerer; Embed = query encode; Rerank = cross-encoder over 100 candidates;
Verify = NLI over the cited sentences.

| Stage | P50 | P95 | n |
|---|---|---|---|
| Search (BM25) | 1.33 ms | 2.07 ms | 200 |
| Answer (extractive) | 1.22 ms | 1.84 ms | 200 |
| Embed (bi-encoder, 256-dim) | 2.56 ms | 3.43 ms | 200 |
| Verify (NLI over cited sentences) | 31.61 ms | 42.79 ms | 200 |
| **Rerank (cross-encoder, 100 candidates)** | **421.13 ms** | **510.12 ms** | 200 |
| Generate | *not run* — needs the Phase-7 Zenith generator (untrained) | — | — |

Chart: [`figures/latency_breakdown.png`](figures/latency_breakdown.png).

**Finding — reranking is the latency budget, exactly as designed (RAG.md §4.2).** The
cross-encoder costs **~320× BM25** (421 ms vs 1.33 ms) because it runs full cross-attention
over 100 query-passage pairs, while the bi-encoder embed is ~2.6 ms (it encodes the query
once and reuses a precomputed index). Verification adds ~32 ms. Combined with the retrieval
ablation — where reranking *hurt* quality on this corpus — the honest conclusion is that
the rerank stage is off by default here: it is both the most expensive stage and, at this
training scale, a negative-value one. An earlier 527-query CPU run measured search 0.84 ms
/ answer 1.12 ms; the numbers above are a fresh 200-query run with all models resident.
