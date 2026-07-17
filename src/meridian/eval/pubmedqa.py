"""Build frozen eval splits from PubMedQA (Phase 2, Task 2).

PubMedQA (PQA-L) is a JSON object keyed by PubMed id; each entry has a ``QUESTION``
derived from that article. The source article is the relevant document, so each
question maps to a single relevant PMID (its key). This module converts the dataset
into :class:`EvalSet` form and splits it deterministically into dev/test, which are
then frozen and checksum-guarded (:mod:`meridian.eval.splits`).

Deterministic splitting hashes the query id, so dev/test membership is stable across
runs and machines without storing an explicit assignment — a rebuild from the same
PubMedQA file yields byte-identical splits.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from meridian.eval.qrels import EvalQuery, EvalSet

PUBMEDQA_LABELS = ("yes", "no", "maybe")


def load_pubmedqa(path: str | Path) -> dict[str, Any]:
    """Load a PubMedQA JSON file (id -> entry)."""
    data: dict[str, Any] = json.loads(Path(path).read_text())
    return data


def build_eval_set_from_pubmedqa(data: dict[str, Any], *, name: str) -> EvalSet:
    """Map each PubMedQA entry to an :class:`EvalQuery` (relevant PMID = its key)."""
    queries = tuple(
        EvalQuery(
            query_id=pubid,
            question=entry["QUESTION"],
            relevant_pmids=frozenset({pubid}),
        )
        for pubid, entry in sorted(data.items())
    )
    return EvalSet(name=name, queries=queries)


def _bucket(query_id: str, *, salt: str, buckets: int = 100) -> int:
    digest = hashlib.sha256(f"{salt}:{query_id}".encode()).hexdigest()
    return int(digest, 16) % buckets


def split_dev_test(
    eval_set: EvalSet,
    *,
    dev_fraction: float = 0.5,
    salt: str = "meridian",
) -> tuple[EvalSet, EvalSet]:
    """Deterministically partition an eval set into (dev, test) by hashed query id.

    A query goes to dev if its hash bucket falls below ``dev_fraction`` of the range.
    Membership is stable and independent of input order.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in (0, 1)")
    threshold = round(dev_fraction * 100)
    dev: list[EvalQuery] = []
    test: list[EvalQuery] = []
    for query in eval_set.queries:
        target = dev if _bucket(query.query_id, salt=salt) < threshold else test
        target.append(query)
    return (
        EvalSet(name=f"{eval_set.name}-dev", queries=tuple(dev)),
        EvalSet(name=f"{eval_set.name}-test", queries=tuple(test)),
    )


def pubmedqa_accuracy(
    predictions: Mapping[str, str],
    gold: Mapping[str, str],
) -> float:
    """Accuracy of yes/no/maybe predictions against gold labels (RAG.md §7 headline).

    Scored over the ids present in ``gold``; a missing or malformed prediction counts as
    wrong. Raises :class:`ValueError` if ``gold`` is empty.
    """
    if not gold:
        raise ValueError("no gold labels to score against")
    correct = sum(
        1 for qid, label in gold.items() if predictions.get(qid, "").lower() == label.lower()
    )
    return correct / len(gold)
