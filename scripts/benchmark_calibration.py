#!/usr/bin/env python
"""Plot a risk-coverage curve for selective answering (Phase 9).

Given per-question ``(correct, confidence)`` records, plot error rate vs coverage and
mark the operating point (ADR-0007). Runs on a labelled JSON file, or on a synthetic,
seeded distribution (a well-calibrated retriever: confident answers are mostly right) so
the figure and mechanism are reproducible offline. Real curves use the trained gates on
the frozen dev split.

Requires the ``benchmark`` extra.

    uv run python scripts/benchmark_calibration.py --out benchmarks/figures/risk_coverage.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meridian.abstain.calibration import operating_point, risk_coverage_curve


def _synthetic(n: int, seed: int) -> list[tuple[bool, float]]:
    rng = np.random.default_rng(seed)
    confidence = rng.uniform(0.0, 1.0, size=n)
    # A calibrated model: P(correct) rises with confidence.
    correct = rng.uniform(0.0, 1.0, size=n) < (0.3 + 0.65 * confidence)
    return [(bool(c), float(s)) for c, s in zip(correct, confidence, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, help="JSON list of [correct, confidence]")
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/figures/risk_coverage.png"))
    args = parser.parse_args()

    if args.records is not None:
        raw = json.loads(args.records.read_text())
        records = [(bool(c), float(s)) for c, s in raw]
        source = "real"
    else:
        records = _synthetic(args.n, args.seed)
        source = "synthetic"

    curve = risk_coverage_curve(records)
    point = operating_point(records, target_coverage=args.target_coverage)
    coverage = [p.coverage for p in curve]
    error = [p.error_rate for p in curve]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(coverage, error, "-", label="risk-coverage")
    ax.scatter(
        [point.coverage],
        [point.error_rate],
        color="crimson",
        zorder=5,
        label=f"operating point (~{args.target_coverage:.0%} coverage, err {point.error_rate:.2f})",
    )
    ax.set_xlabel("coverage (fraction of questions answered)")
    ax.set_ylabel("error rate among answered")
    ax.set_title(f"Selective answering: risk-coverage ({source}, n={len(records)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"operating point: coverage={point.coverage:.3f} error={point.error_rate:.3f}")
    print(f"wrote figure -> {args.out}")


if __name__ == "__main__":
    main()
