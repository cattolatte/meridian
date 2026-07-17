"""Risk-coverage analysis for selective answering (Phase 9).

Given per-question ``(correct, confidence)`` records, sweep the confidence threshold to
trace error rate vs the fraction of questions answered. The operating point (ADR-0007)
is chosen from this curve — e.g. minimize error at ~80% coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    """One point on the risk-coverage curve."""

    threshold: float
    coverage: float  # fraction of questions answered
    error_rate: float  # fraction wrong among answered


def risk_coverage_curve(records: Sequence[tuple[bool, float]]) -> list[RiskCoveragePoint]:
    """Trace the risk-coverage curve from ``(correct, confidence)`` records.

    Answering the most-confident questions first, each point reports the coverage and
    the error rate when the threshold admits that prefix. Raises :class:`ValueError` on
    an empty input.
    """
    if not records:
        raise ValueError("risk-coverage needs at least one record")

    ordered = sorted(records, key=lambda item: item[1], reverse=True)
    points: list[RiskCoveragePoint] = []
    wrong = 0
    total = len(ordered)
    for answered, (correct, confidence) in enumerate(ordered, start=1):
        if not correct:
            wrong += 1
        points.append(
            RiskCoveragePoint(
                threshold=confidence,
                coverage=answered / total,
                error_rate=wrong / answered,
            )
        )
    return points


def operating_point(
    records: Sequence[tuple[bool, float]], *, target_coverage: float = 0.8
) -> RiskCoveragePoint:
    """Return the curve point whose coverage is closest to ``target_coverage``."""
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    curve = risk_coverage_curve(records)
    return min(curve, key=lambda point: abs(point.coverage - target_coverage))
