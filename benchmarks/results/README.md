# Measured benchmark results

Raw JSON emitted by the benchmark scripts, committed so every number in
[`../BENCHMARKS.md`](../BENCHMARKS.md) and every chart in [`../figures/`](../figures)
is reproducible from data rather than hand-transcribed (claims hygiene, house rule #5).

| File | Produced by |
|---|---|
| `retrieval_ablation.json` | `scripts/ablate_stage0.py` — BM25 / dense / reranker (pure + base-fused) on PubMedQA dev |
| `stage0_variance.json` | `scripts/variance_dense.py` — 4-seed MLM-vs-random-init dense comparison |
| `faithfulness.json` | `scripts/benchmark_faithfulness.py` — citation precision/recall, hallucination rate over 527 dev queries |
| `verifier_progression.json` | `scripts/train_verifier.py --eval-nli --eval-every` runs (SNLI dev) + per-stage P50 latency |

Regenerate the summary charts from these files with:

```bash
uv run python scripts/plot_summary.py
```

The ANN and risk-coverage figures are written directly by their own benchmarks
(`scripts/benchmark_ann.py`, `scripts/benchmark_calibration.py`).
