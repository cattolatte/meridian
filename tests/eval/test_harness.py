"""Tests for the evaluation harness runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.eval.harness import EvalResult, run_evaluation, write_results
from meridian.eval.qrels import EvalQuery, EvalSet
from meridian.retrieval.pipeline import BM25Retriever

_DOCS = [
    Document(pmid="1", title="Heart failure", abstract="beta blockers reduce mortality"),
    Document(pmid="2", title="Diabetes", abstract="metformin cardiovascular outcomes"),
    Document(pmid="3", title="Melanoma", abstract="checkpoint immunotherapy responses"),
]


def _retriever() -> BM25Retriever:
    store = SqliteDocumentStore(":memory:")
    store.add_many(_DOCS)
    return BM25Retriever.from_store(store)


def test_run_evaluation_scores_relevant_first() -> None:
    eval_set = EvalSet(
        name="dev",
        queries=(
            EvalQuery("q1", "beta blockers mortality heart failure", frozenset({"1"})),
            EvalQuery("q2", "metformin cardiovascular diabetes", frozenset({"2"})),
        ),
    )
    result = run_evaluation(_retriever(), eval_set, k_values=(1, 3))
    assert result.n_queries == 2
    assert result.metrics["recall@1"] == 1.0
    assert result.metrics["mrr@10"] == 1.0
    assert result.metrics["ndcg@10"] == 1.0


def test_queries_without_relevant_are_skipped() -> None:
    eval_set = EvalSet(
        name="dev",
        queries=(
            EvalQuery("q1", "beta blockers heart failure", frozenset({"1"})),
            EvalQuery("q2", "no relevant marked", frozenset()),
        ),
    )
    result = run_evaluation(_retriever(), eval_set, k_values=(1,))
    assert result.n_queries == 1


def test_no_scorable_queries_raises() -> None:
    eval_set = EvalSet("dev", (EvalQuery("q1", "q", frozenset()),))
    with pytest.raises(ValueError):
        run_evaluation(_retriever(), eval_set)


def test_write_results(tmp_path: Path) -> None:
    result = EvalResult(eval_set="dev", n_queries=2, metrics={"recall@5": 0.5})
    path = tmp_path / "results.json"
    write_results(result, path)
    loaded = json.loads(path.read_text())
    assert loaded == {"eval_set": "dev", "n_queries": 2, "metrics": {"recall@5": 0.5}}
