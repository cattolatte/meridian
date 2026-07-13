"""Run a retriever over an eval set and produce reproducible metrics.

The harness retrieves to the deepest requested cutoff once per query, computes the
binary-relevance metrics (RAG.md §7), and averages them over the queries that have
at least one relevant document. Results serialize to canonical JSON; MLflow logging
is optional and lazily imported, so the offline test suite never depends on it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meridian.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from meridian.eval.qrels import EvalSet
from meridian.retrieval.pipeline import Retriever

DEFAULT_K_VALUES = (5, 20, 100)


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Averaged metrics from one evaluation run."""

    eval_set: str
    n_queries: int
    metrics: dict[str, float]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "eval_set": self.eval_set,
            "n_queries": self.n_queries,
            "metrics": self.metrics,
        }


def run_evaluation(
    retriever: Retriever,
    eval_set: EvalSet,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    mrr_k: int = 10,
    ndcg_k: int = 10,
) -> EvalResult:
    """Evaluate ``retriever`` on ``eval_set`` and return averaged metrics.

    Queries with no relevant documents are skipped. Raises :class:`ValueError` if no
    query is scorable.
    """
    depth = max(*k_values, mrr_k, ndcg_k)
    recall_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    scored = 0

    for query in eval_set.queries:
        if not query.relevant_pmids:
            continue
        ranked = [hit.pmid for hit in retriever.retrieve(query.question, k=depth)]
        scored += 1
        for k in k_values:
            recall_sums[k] += recall_at_k(ranked, query.relevant_pmids, k)
        mrr_sum += mrr_at_k(ranked, query.relevant_pmids, mrr_k)
        ndcg_sum += ndcg_at_k(ranked, query.relevant_pmids, ndcg_k)

    if scored == 0:
        raise ValueError("eval set has no queries with relevant documents")

    metrics = {f"recall@{k}": recall_sums[k] / scored for k in k_values}
    metrics[f"mrr@{mrr_k}"] = mrr_sum / scored
    metrics[f"ndcg@{ndcg_k}"] = ndcg_sum / scored
    return EvalResult(eval_set=eval_set.name, n_queries=scored, metrics=metrics)


def write_results(result: EvalResult, path: str | Path) -> None:
    """Write an :class:`EvalResult` to canonical JSON."""
    text = json.dumps(result.to_jsonable(), indent=2, sort_keys=True) + "\n"
    Path(path).write_text(text)


def log_to_mlflow(result: EvalResult, *, experiment: str, run_name: str | None = None) -> None:
    """Log metrics to MLflow (requires the optional ``tracking`` extra)."""
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - exercised without the extra installed
        raise RuntimeError(
            "MLflow is not installed; install the 'tracking' extra to log runs"
        ) from exc
    mlflow.set_experiment(experiment)  # pragma: no cover - requires the tracking extra
    with mlflow.start_run(run_name=run_name):  # pragma: no cover
        mlflow.log_param("eval_set", result.eval_set)  # pragma: no cover
        mlflow.log_param("n_queries", result.n_queries)  # pragma: no cover
        mlflow.log_metrics(result.metrics)  # pragma: no cover
