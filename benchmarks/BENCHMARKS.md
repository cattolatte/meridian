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

### PubMedQA PQA-L retrieval benchmark (real, full ablation)

A self-contained retrieval task from PubMedQA PQA-L: **1000 real abstracts**, each question
retrieving its own source abstract. **Clean 3-way split — no leakage**: train 590 (trains
the dense retriever + reranker), dev 239 (reported below), test held out. The dense
retriever and reranker are trained by the **actual ADR-0004 curriculum** (MLM Stage-0
pretraining → supervised contrastive → hard-negative reranking). One command reproduces the
whole table: `uv run python scripts/campaign_pubmedqa.py --pqal data/pubmedqa/ori_pqal.json --out data/pubmedqa/campaign.json`.

| Config | R@5 | R@20 | R@100 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| **BM25** | **0.987** | 0.987 | 0.996 | **0.969** | **0.974** |
| Dense (MLM-pretrained + supervised contrastive) | 0.460 | 0.636 | 0.787 | 0.391 | 0.427 |
| Hybrid RRF (BM25 + dense) | 0.774 | 0.987 | 0.996 | 0.689 | 0.725 |
| BM25 + cross-encoder rerank | 0.025 | 0.134 | 0.996 | 0.014 | 0.023 |
| Hybrid + rerank | 0.021 | 0.121 | 0.996 | 0.012 | 0.019 |

**Findings — measured, not spun (RAG.md §9):**

1. **The training curriculum works (≈46× dense lift).** A from-scratch bi-encoder,
   MLM-pretrained on the corpus then contrastively fine-tuned on 590 real
   question→abstract pairs, reaches **Recall@5 0.46 / Recall@100 0.79** — versus ~0.01 for a
   naive random-init, self-supervised baseline (`scripts/train_pubmedqa_dense.py`). That is
   the ADR-0004 story (Stage-0 pretraining is what makes a from-scratch bi-encoder viable),
   measured end-to-end.
2. **BM25 wins on this lexically-easy task.** Questions reuse their abstract's vocabulary,
   so lexical retrieval is near-perfect (0.987). 590 pairs and a small from-scratch dense
   model don't close the gap — and shouldn't be expected to at this scale.
3. **Reranking and hybrid don't help here — reported anyway.** BM25's top-5 is already
   0.987, so a small from-scratch reranker only degrades it, and RRF with the weaker dense
   ranking drags Recall@5 down (0.987 → 0.774). At larger corpus scale, with vocabulary
   mismatch, and with the full training sets (MS MARCO + hard negatives), these stages earn
   their keep; on *this* benchmark they don't, and hiding that would betray the whole point.

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
96-dim, trained on 808 examples, evaluated on 192) reaches **0.531** when it runs the same
ADR-0004 Stage-0 MLM pretraining as the retriever — reproduce with
`scripts/train_pubmedqa_classifier.py`.

| Metric | Value | Run |
|---|---|---|
| PubMedQA majority-class baseline (eval split) | 0.547 | build_pubmedqa.py |
| PubMedQA accuracy (from-scratch 3-class, MLM-pretrained) | **0.531** | train_pubmedqa_classifier.py |
| — ablation: same model, no Stage-0 pretraining | 0.495 | `... --no-pretrain` |
| PubMedQA accuracy (full pipeline, trained at scale) | TBD | — |
| Answer coverage @ operating point | TBD | — |

**Honest finding (pretraining helps, still short of baseline).** Stage-0 MLM pretraining
lifts the from-scratch classifier from **0.495 → 0.531** (+3.6 pts) — the *same* curriculum
that gives the retriever its ~46× gain also helps the classifier, a consistent result. But
it still does **not** beat the 0.547 majority baseline: 800 examples and a tiny model are
far too little for a reasoning task where published results (~68–78%) use large *pretrained*
models. This is the project's honest scope (RAG.md §2): a laptop-scale, from-scratch system
is **not accuracy-competitive with large models**, and does not pretend to be. Its value is
the *engineering and safety design* — grounded, citation-constrained generation; NLI
verification; calibrated abstention — not a headline accuracy number. The path to a real
number is this curriculum plus the full training sets, which the pipeline implements but has
not run at scale.

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
