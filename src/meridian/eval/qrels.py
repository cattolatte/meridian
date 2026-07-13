"""Eval-set (qrels) types and JSON I/O.

An :class:`EvalSet` is a named list of :class:`EvalQuery` items, each a question with
its set of relevant PMIDs. This is the input to the eval harness and the on-disk
form of the frozen dev/test splits (built from PubMedQA in Phase 2).

The JSON form is canonical (sorted keys, sorted PMIDs) so a split has a stable
content hash for the checksum guard (:mod:`meridian.eval.splits`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """A single evaluation query: a question and its relevant PMIDs."""

    query_id: str
    question: str
    relevant_pmids: frozenset[str]


@dataclass(frozen=True, slots=True)
class EvalSet:
    """A named collection of evaluation queries."""

    name: str
    queries: tuple[EvalQuery, ...]

    def __len__(self) -> int:
        return len(self.queries)


def to_jsonable(eval_set: EvalSet) -> dict[str, Any]:
    """Return a canonical, JSON-serializable representation of an eval set."""
    return {
        "name": eval_set.name,
        "queries": [
            {
                "query_id": query.query_id,
                "question": query.question,
                "relevant_pmids": sorted(query.relevant_pmids),
            }
            for query in eval_set.queries
        ],
    }


def from_jsonable(payload: dict[str, Any]) -> EvalSet:
    """Reconstruct an eval set from its JSON representation."""
    queries = tuple(
        EvalQuery(
            query_id=item["query_id"],
            question=item["question"],
            relevant_pmids=frozenset(item["relevant_pmids"]),
        )
        for item in payload["queries"]
    )
    return EvalSet(name=payload["name"], queries=queries)


def save_eval_set(eval_set: EvalSet, path: str | Path) -> None:
    """Write an eval set to a canonical JSON file."""
    text = json.dumps(to_jsonable(eval_set), indent=2, sort_keys=True) + "\n"
    Path(path).write_text(text)


def load_eval_set(path: str | Path) -> EvalSet:
    """Load an eval set from a JSON file."""
    return from_jsonable(json.loads(Path(path).read_text()))
