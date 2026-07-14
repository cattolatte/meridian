# ADR-0005: Default ANN index backend

- **Status:** Accepted (default provisional pending the real-corpus benchmark)
- **Date:** 2026-07-14

## Context

Phase 4 implements three interchangeable search backends behind `VectorIndex`:
brute-force (exact, Phase 3), IVF (k-means inverted file), and HNSW (navigable
small-world graph). We must choose the default backend for dense retrieval, from
measured recall/latency curves rather than reputation (RAG.md §7). The exit criterion
is recall@10 ≥ 0.95 versus the brute-force ground truth at an acceptable latency.

`scripts/benchmark_ann.py` produces the trade-off curve; see
[`benchmarks/figures/ann_tradeoff.png`](../../benchmarks/figures/ann_tradeoff.png).

## Measured trade-off (synthetic, N=2000, dim=64, seed=0)

| Backend | Operating point | recall@10 | mean latency |
|---|---|---|---|
| Brute force | exact | 1.000 | 0.60 ms |
| IVF | nprobe=16 | 0.773 | 0.26 ms |
| HNSW | efSearch=16 | 0.999 | 0.36 ms |
| HNSW | efSearch=32 | 1.000 | 0.59 ms |

On this **structureless random** data HNSW reaches ≥ 0.999 recall *below* brute-force
latency, while IVF lags — k-means finds no meaningful cells when the data has no
cluster structure. Real sentence embeddings are strongly clustered, so IVF is expected
to do materially better on the real corpus; that is exactly what the real-corpus run
will measure.

## Decision

1. **Default backend: HNSW.** It meets the recall target at the lowest latency across
   operating points and degrades gracefully via `efSearch`. `meridian ask --ann hnsw`
   / `--retriever dense --ann hnsw`.
2. **Brute force stays the default until a corpus is large enough to need ANN.** At
   small N, exact search is already sub-millisecond and is the honest ground truth;
   `--ann none` remains the safe, exact option and the recall denominator.
3. **IVF is retained** as a second family for the real-corpus comparison (its standing
   may improve markedly on clustered embeddings) and for the systems narrative.
4. **This default is reaffirmed from the real-corpus benchmark** (`--embedding-index`
   pointed at the Phase-3 corpus embeddings) before v1.0; if the real curves disagree,
   a superseding ADR records the change.

## Alternatives considered

- **IVF as default:** rejected on the current evidence (lower recall at comparable
  latency here), but explicitly re-evaluated on real embeddings.
- **Brute force as the permanent default:** fine at laptop scale, but does not scale to
  the stretch-goal corpus sizes; the point of Phase 4 is a measured sub-linear path.

## Consequences

- The CLI/eval default is `--ann none` (exact) until the real benchmark justifies
  switching the shipped default to HNSW; the config flag makes the switch a one-word
  change with no code impact.
- HNSW build cost is paid once per corpus; the graph is rebuilt deterministically from
  the persisted embeddings (seeded), so there is no separate stale-index artifact.
- The real recall/latency/RAM numbers fill the ANN section of `benchmarks/BENCHMARKS.md`
  from the committed benchmark script (claims hygiene).
