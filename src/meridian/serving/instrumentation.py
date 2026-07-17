"""Per-stage latency instrumentation (embed / search / rerank / generate / verify).

A tiny timer that records wall-clock per named stage, so ``/metrics`` can report P50/P95
per stage under load (RAG.md §7) — the earned serving numbers, not inherited claims.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager


class StageTimer:
    """Accumulates per-stage durations across requests."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._samples[name].append((time.perf_counter() - start) * 1000.0)

    def percentiles(self) -> dict[str, dict[str, float]]:
        """Return ``{stage: {p50, p95, count}}`` in milliseconds."""
        result: dict[str, dict[str, float]] = {}
        for name, samples in self._samples.items():
            ordered = sorted(samples)
            result[name] = {
                "p50": _percentile(ordered, 0.50),
                "p95": _percentile(ordered, 0.95),
                "count": float(len(ordered)),
            }
        return result


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[index]
