# Benchmarks

> **Claims hygiene (house rule):** a number appears here only if it is reproduced
> by a committed, seeded script, with its MLflow run ID referenced. No number is
> estimated in prose. Frozen dev/test splits are created in Phase 2 and never
> touched afterward.

**Status:** Phase 0 — no measurements yet. Tables below are scaffolds; every cell
is filled from the evaluation harness as the corresponding phase ships.

## Reproduction

Each result row references the script and MLflow run that produced it:

```
scripts/<benchmark>.py   →   MLflow run ID   →   row below
```

(Scripts arrive with their phases; this section lists them as they land.)

## Retrieval

Metrics: Recall@{5,20,100}, MRR@10, nDCG@10 on the frozen **dev** split.

Reproduce a row (once the real corpus + PubMedQA split exist):

```bash
uv run python scripts/evaluate.py --db data/corpus.sqlite \
    --split benchmarks/splits/dev.json --checksum <registered> \
    --out benchmarks/results/bm25-dev.json
```

| Config | R@5 | R@20 | R@100 | MRR@10 | nDCG@10 | Run ID |
|---|---|---|---|---|---|---|
| BM25 (Phase 2) | TBD | TBD | TBD | TBD | TBD | TBD |
| Dense (Phase 3) | TBD | TBD | TBD | TBD | TBD | TBD |
| Retriever v2 (+hard neg, Phase 5) | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid RRF (Phase 5) | TBD | TBD | TBD | TBD | TBD | TBD |
| Dense + rerank (Phase 6) | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid + rerank (Phase 6) | TBD | TBD | TBD | TBD | TBD | TBD |

The harness and BM25 baseline are implemented and verified offline on the committed
sample corpus/split; the real row is filled once the PubMed corpus and PubMedQA
splits are downloaded (no number is written from memory).

The dense retriever (Phase 3) is implemented and runs end-to-end
(`scripts/train_retriever.py` → `scripts/embed_corpus.py` →
`scripts/evaluate.py --retriever dense`). Its row, and the ADR-0004 ablations
(`--random-init` vs Stage-0; Stage A vs A+B), are filled from a real training run.

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

| Metric | Value | Run ID |
|---|---|---|
| Citation precision | TBD | TBD |
| Citation recall | TBD | TBD |
| Hallucination rate | TBD | TBD |
| Verifier–human agreement | TBD | TBD |

## End-to-end (Phase 11)

| Metric | Value | Run ID |
|---|---|---|
| PubMedQA accuracy (yes/no/maybe) | TBD | TBD |
| Answer coverage @ operating point | TBD | TBD |

## Systems / latency (Phase 10)

| Stage | P50 | P95 | Run ID |
|---|---|---|---|
| Embed | TBD | TBD | TBD |
| Search | TBD | TBD | TBD |
| Rerank | TBD | TBD | TBD |
| Generate | TBD | TBD | TBD |
| Verify | TBD | TBD | TBD |
