#!/usr/bin/env python
"""Plot the summary figures from the committed measured results (claims hygiene).

Reads the JSON written by the benchmark scripts (``benchmarks/results/``) and renders
three figures into ``benchmarks/figures/``:

* ``retrieval_ablation.png`` -- BM25 vs from-scratch dense vs reranker (pure / base-fused)
* ``verifier_progression.png`` -- SNLI dev accuracy across the levers that moved it
* ``latency_breakdown.png`` -- per-stage P50 latency (log scale; rerank dominates)

Every value comes from a committed result file, so the figures are reproducible and never
hand-drawn. Requires the ``benchmark`` extra.

    uv run python scripts/plot_summary.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ACCENT = "#2b6cb0"
_MUTED = "#a0aec0"
_WARN = "#c53030"
_GOOD = "#2f855a"


def _retrieval_figure(results: Path, out: Path) -> None:
    data = json.loads((results / "retrieval_ablation.json").read_text())
    rows = [
        ("BM25", data["bm25"]["recall@5"], _ACCENT),
        ("Dense\n(from scratch)", data["dense_random_init"]["recall@5"], _MUTED),
        ("BM25 + rerank\n(pure)", data["bm25_rerank_pure"]["recall@5"], _WARN),
        ("BM25 + rerank\n(base-fused)", data["bm25_rerank_fused"]["recall@5"], _GOOD),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows])
    for bar, (_label, value, _c) in zip(bars, rows, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Recall@5")
    ax.set_title(f"Retrieval on PubMedQA dev (n={data['n_dev']}) — measured")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def _verifier_figure(results: Path, out: Path) -> None:
    data = json.loads((results / "verifier_progression.json").read_text())
    runs = data["runs"]
    labels = [r["label"] for r in runs]
    values = [r["accuracy"] for r in runs]
    colors = [_WARN if r["lever"] == "bad LR" else _ACCENT for r in runs]
    colors[values.index(max(values))] = _GOOD

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars = ax.bar(range(len(runs)), values, color=colors)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center")
    ax.axhline(1 / 3, color="gray", linestyle="--", linewidth=1, label="chance (0.333)")
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("SNLI dev accuracy (n=5000)")
    ax.set_title("From-scratch NLI verifier: what actually moved the number")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def _latency_figure(results: Path, out: Path) -> None:
    data = json.loads((results / "verifier_progression.json").read_text())
    stages = data["latency_ms_p50"]
    order = sorted(stages.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in order]
    values = [v for _, v in order]
    colors = [_WARN if v > 100 else _ACCENT for v in values]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(labels, values, color=colors)
    for bar, value in zip(bars, values, strict=True):
        ax.text(value * 1.15, bar.get_y() + bar.get_height() / 2, f"{value:.2f} ms", va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.5, max(values) * 4)
    ax.set_xlabel("P50 latency (ms, log scale)")
    ax.set_title("Per-stage serving latency — reranking is the budget")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--figures", type=Path, default=Path("benchmarks/figures"))
    args = parser.parse_args()

    args.figures.mkdir(parents=True, exist_ok=True)
    _retrieval_figure(args.results, args.figures / "retrieval_ablation.png")
    _verifier_figure(args.results, args.figures / "verifier_progression.png")
    _latency_figure(args.results, args.figures / "latency_breakdown.png")


if __name__ == "__main__":
    main()
