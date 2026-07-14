#!/usr/bin/env python
"""Benchmark ANN indexes: recall@10 vs query latency vs brute force.

Builds the brute-force, IVF, and HNSW indexes over a corpus, measures recall against
the brute-force ground truth and mean query latency at several operating points, and
plots the recall/latency trade-off (matplotlib). Feeds the ADR-0005 default-index
decision.

By default it runs on **synthetic** normalized vectors so the machinery and the shape
of the curves are reproducible offline; point `--embedding-index` at a real
`EmbeddingIndex` directory (from `scripts/embed_corpus.py`) for the real numbers.

Requires the ``benchmark`` extra: ``uv sync --extra benchmark``.

Example
-------
    uv run python scripts/benchmark_ann.py --n 2000 --dim 64 \\
        --out benchmarks/figures/ann_tradeoff.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meridian.retrieval.ann.hnsw import HNSWIndex
from meridian.retrieval.ann.ivf import IVFIndex
from meridian.retrieval.embedding_index import EmbeddingIndex

Array = npt.NDArray[np.float32]


def _synthetic(n: int, dim: int, seed: int) -> tuple[list[str], Array]:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return [str(i) for i in range(n)], vectors


def _truth(brute: EmbeddingIndex, queries: Array, k: int) -> list[set[str]]:
    return [{pmid for pmid, _ in brute.search(q, k=k)} for q in queries]


def _measure(search: object, queries: Array, truth: list[set[str]], k: int) -> tuple[float, float]:
    """Return (mean recall@k, mean latency in ms) for a search callable."""
    hits = 0
    start = time.perf_counter()
    for q, gold in zip(queries, truth, strict=True):
        got = {pmid for pmid, _ in search(q)}  # type: ignore[operator]
        hits += len(got & gold)
    elapsed = time.perf_counter() - start
    recall = hits / (k * len(queries))
    latency_ms = 1000.0 * elapsed / len(queries)
    return recall, latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding-index", type=Path, help="use a real EmbeddingIndex dir")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/figures/ann_tradeoff.png"))
    args = parser.parse_args()

    if args.embedding_index is not None:
        brute = EmbeddingIndex.load(args.embedding_index)
        pmids, vectors = list(brute.pmids), np.asarray(brute.vectors, dtype=np.float32)
        source = "real corpus"
    else:
        pmids, vectors = _synthetic(args.n, args.dim, args.seed)
        brute = EmbeddingIndex.build(pmids, vectors)
        source = "synthetic"

    rng = np.random.default_rng(args.seed + 1)
    queries = vectors[rng.choice(len(pmids), size=min(args.queries, len(pmids)), replace=False)]
    truth = _truth(brute, queries, args.k)

    brute_recall, brute_latency = _measure(
        lambda q: brute.search(q, k=args.k), queries, truth, args.k
    )
    print(f"brute      recall={brute_recall:.3f} latency={brute_latency:.3f} ms")

    ivf = IVFIndex.build(pmids, vectors, nlist=max(1, round(len(pmids) ** 0.5)), seed=args.seed)
    ivf_points = []
    for nprobe in (1, 2, 4, 8, 16):
        recall, latency = _measure(
            lambda q, p=nprobe: ivf.search(q, k=args.k, nprobe=p), queries, truth, args.k
        )
        ivf_points.append((latency, recall))
        print(f"ivf   np={nprobe:<3} recall={recall:.3f} latency={latency:.3f} ms")

    hnsw = HNSWIndex.build(pmids, vectors, seed=args.seed)
    hnsw_points = []
    for ef in (16, 32, 64, 128):
        recall, latency = _measure(
            lambda q, e=ef: hnsw.search(q, k=args.k, ef_search=e), queries, truth, args.k
        )
        hnsw_points.append((latency, recall))
        print(f"hnsw  ef={ef:<4} recall={recall:.3f} latency={latency:.3f} ms")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="brute force (exact)")
    ax.plot(*zip(*ivf_points, strict=True), "o-", label="IVF (nprobe sweep)")
    ax.plot(*zip(*hnsw_points, strict=True), "s-", label="HNSW (efSearch sweep)")
    ax.set_xscale("log")
    ax.set_xlabel("mean query latency (ms, log scale)")
    ax.set_ylabel(f"recall@{args.k} vs brute force")
    ax.set_title(f"ANN recall/latency trade-off ({source}, N={len(pmids)}, dim={vectors.shape[1]})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote figure -> {args.out}")


if __name__ == "__main__":
    main()
