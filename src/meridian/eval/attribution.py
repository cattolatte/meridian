"""Error attribution via oracle-component substitution (Phase 11).

For every wrong end-to-end answer, substitute an oracle for each pipeline stage in turn
(gold passages for retrieval/rerank, gold answer for generation, a perfect verifier) and
see which substitution *fixes* the answer. The failure is attributed to the **earliest**
stage whose oracle fixes it — the stage that first broke the chain. Aggregated over the
test split, this yields the attribution breakdown (retrieval / rerank / generation /
verification) that is the project's research centerpiece (RAG.md §7).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# Pipeline order: a failure is blamed on the earliest stage an oracle repairs.
STAGES: tuple[str, ...] = ("retrieval", "rerank", "generation", "verification")
UNATTRIBUTED = "unattributed"


def attribute_failure(oracle_fixes: Mapping[str, bool]) -> str:
    """Attribute one wrong answer to the earliest stage whose oracle fixes it.

    ``oracle_fixes`` maps a stage name to whether substituting its oracle produces a
    correct answer. Returns the stage, or ``"unattributed"`` if no oracle fixes it
    (the failure is elsewhere or compound).
    """
    for stage in STAGES:
        if oracle_fixes.get(stage, False):
            return stage
    return UNATTRIBUTED


@dataclass(frozen=True, slots=True)
class AttributionStudy:
    """Aggregate attribution counts and their fractions."""

    counts: dict[str, int]
    total: int

    def fraction(self, stage: str) -> float:
        return self.counts.get(stage, 0) / self.total if self.total else 0.0


def attribution_study(cases: Iterable[Mapping[str, bool]]) -> AttributionStudy:
    """Attribute a collection of wrong answers and tally the stages."""
    counts: dict[str, int] = {stage: 0 for stage in (*STAGES, UNATTRIBUTED)}
    total = 0
    for oracle_fixes in cases:
        counts[attribute_failure(oracle_fixes)] += 1
        total += 1
    return AttributionStudy(counts=counts, total=total)
