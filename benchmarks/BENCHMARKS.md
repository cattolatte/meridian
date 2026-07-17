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

### PubMedQA PQA-L context corpus (measured, real)

A self-contained retrieval task built from PubMedQA PQA-L: **1000 real abstracts**, each
question retrieving its own source abstract. Frozen **dev** split = 527 queries (checksum
`09b3befa…`). Reproduce from scratch:

```bash
curl -sL https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json \
    -o data/pubmedqa/ori_pqal.json
uv run python scripts/build_pubmedqa.py --pqal data/pubmedqa/ori_pqal.json \
    --db data/pubmedqa.sqlite --out-dir data/pubmedqa
uv run python scripts/evaluate.py --db data/pubmedqa.sqlite \
    --split data/pubmedqa/pqal_dev.json --checksum 09b3befa4b897c53be47594fbc07160b3196af1e29d608bfd79b95787d96cc98 \
    --out data/pubmedqa/bm25-dev.json
```

| Config | R@5 | R@20 | R@100 | MRR@10 | nDCG@10 | n |
|---|---|---|---|---|---|---|
| **BM25** | **0.987** | **0.990** | **0.996** | **0.969** | **0.973** | 527 |
| Dense (self-supervised, untrained trunk) | 0.010 | 0.057 | 0.256 | 0.006 | 0.009 | 527 |

**Honest finding (dense ≪ BM25).** BM25 is a very strong lexical baseline here — the
questions reuse their source abstract's vocabulary. The dense row is a deliberately weak
configuration: a tiny (32-dim, 1-layer) **random-initialized** bi-encoder trained for 1
epoch on **self-supervised** abstract-half pairs only — no Stage-0 MLM pretraining, no MS
MARCO / PQA-A. As ADR-0004 predicts, that underperforms badly. **Diagnosis:** the gap is
the missing MLM pretraining (Stage 0) and a real contrastive training set (Stage A/B) —
the exact curriculum the pipeline implements but has not yet been run at scale.
Reproduce: `scripts/train_pubmedqa_dense.py`. This is the honest-benchmark culture (RAG.md
§9): the number is published and diagnosed, not hidden. Hybrid/reranker rows follow once
the pair-model training sets are added.

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
64-dim, trained on 808 examples, evaluated on 192) reaches **0.495** — reproduce with
`scripts/train_pubmedqa_classifier.py`.

| Metric | Value | Run |
|---|---|---|
| PubMedQA majority-class baseline (eval split) | 0.547 | build_pubmedqa.py |
| PubMedQA accuracy (from-scratch 3-class classifier) | **0.495** | train_pubmedqa_classifier.py |
| PubMedQA accuracy (full pipeline, trained at scale) | TBD | — |
| Answer coverage @ operating point | TBD | — |

**Honest finding (below baseline).** The from-scratch classifier does **not** beat the
majority baseline — 800 examples and a tiny randomly-initialized model are far too little
for a reasoning task where published results (~68–78%) use large *pretrained* models. This
is the project's honest scope (RAG.md §2): a laptop-scale, from-scratch system is **not
accuracy-competitive with large models**, and does not pretend to be. Its value is the
*engineering and safety design* — grounded, citation-constrained generation; NLI
verification; calibrated abstention — not a headline accuracy number. The path to a real
number is MLM pretraining + the full training sets, which the pipeline implements but has
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
