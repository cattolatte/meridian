# Phase 4 — ANN index from scratch (design doc)

- **Release:** v0.3.0
- **Goal:** sub-linear vector search with measured recall / latency / memory
  trade-offs, benchmarked against the brute-force ground truth (Phase 3).
- **Governing ADR:** [0005](../adr/0005-default-index.md) (default index choice — written
  from the measured curves).

## Module layout (`src/meridian/retrieval/ann/`)

```
base.py     VectorIndex protocol — search(query, k) + __len__; the seam every
            index (brute force, IVF, HNSW) satisfies so DenseRetriever is
            backend-agnostic.
kmeans.py   Seeded Lloyd k-means (NumPy) used to learn IVF cells.
ivf.py      IVFIndex: nlist inverted lists by nearest centroid; nprobe search.
hnsw.py     HNSWIndex: layered navigable-small-world graph; M / efConstruction /
            efSearch; greedy insert + search.
```

The brute-force `EmbeddingIndex` (Phase 3) already satisfies `VectorIndex`
structurally, so it is the exact ground truth ANN recall is measured against. No
external ANN library is used (house rule).

## Milestones

1. **IVF** — `VectorIndex` protocol, seeded k-means, `IVFIndex` (build + nprobe
   search), recall-vs-brute-force tests.
2. **HNSW** — layered graph build/search, property-based recall-floor tests.
3. **Storage + wiring + benchmark** — memory-mapped index formats, `scripts/build_ann.py`,
   `DenseRetriever`/CLI backend switch, `ADR-0005`, and a recall/latency/RAM benchmark
   (matplotlib figure committed).

## Key decisions

- **One search interface.** `VectorIndex.search(query, *, k)` returns
  `list[(pmid, score)]`, identical to brute force, so swapping backends is config only.
- **Exact rerank inside probed cells.** IVF searches the vectors in the nprobe nearest
  cells exactly; `nprobe = nlist` recovers brute-force recall (a correctness anchor).
- **Deterministic.** k-means, graph construction, and tie-breaking are seeded/stable, so
  a rebuild reproduces the index and its recall.
- **Recall is measured, not assumed.** Every ANN result is scored against the brute-force
  top-k; the exit criterion (recall@10 ≥ 0.95 at an acceptable latency) is checked on the
  real index, or the shortfall is documented (claims hygiene).

## Testing strategy (offline-only)

- k-means: assignments partition the data; centroids are cluster means; determinism;
  empty-cluster handling; `n_clusters == n_points` gives zero-error clustering.
- IVF: `nprobe == nlist` matches brute-force top-k exactly; recall rises with nprobe;
  determinism; small-corpus guards.
- HNSW: property-based — recall@k vs brute force stays above a floor on synthetic
  clustered data across seeds; determinism; single-element and duplicate-vector edges.

## Environment constraint

Recall/latency/RAM on the real ~200K-vector corpus need the trained embedder's corpus
embeddings (Phase 3 real run). All index logic and the recall relationships are verified
offline on synthetic vectors; the benchmark script produces the committed figure and the
ADR-0005 numbers from the real index — none are written from memory.

## Exit criteria tracking

| Criterion | Status |
|---|---|
| ANN recall@10 ≥ 0.95 vs brute force at acceptable latency | mechanism + tests; real point pending Phase-3 embeddings |
| CLI switches index backend by config | planned (Milestone 3) |
| ADR-0005 default from measured curves | planned (Milestone 3) |
| ≥ 90% coverage held | maintained |
