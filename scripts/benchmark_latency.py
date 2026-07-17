#!/usr/bin/env python
"""Measure per-stage serving latency over a real query set (Phase 10, RAG.md §7).

Runs retrieval (search) and the extractive answerer (answer) over every question in an
eval split and reports P50/P95 wall-clock per stage via ``StageTimer`` — earned numbers,
not inherited claims. Generate/verify stages are added once those models are trained.

    uv run python scripts/benchmark_latency.py --db data/pubmedqa.sqlite \\
        --split data/pubmedqa/pqal_dev.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from meridian.answer.extractive import answer_extractive
from meridian.corpus.store import SqliteDocumentStore
from meridian.eval.qrels import load_eval_set
from meridian.retrieval.pipeline import BM25Retriever
from meridian.serving.instrumentation import StageTimer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    eval_set = load_eval_set(args.split)
    timer = StageTimer()
    with SqliteDocumentStore(args.db) as store:
        corpus_size = store.count()
        retriever = BM25Retriever.from_store(store)
        for query in eval_set.queries:
            with timer.stage("search"):
                retriever.retrieve(query.question, k=args.k)
            with timer.stage("answer"):
                answer_extractive(retriever, query.question, k_passages=args.k)

    print(f"queries: {len(eval_set)}  corpus: {corpus_size}")
    for stage, stats in timer.percentiles().items():
        n = int(stats["count"])
        print(f"{stage:8s} P50={stats['p50']:.2f} ms  P95={stats['p95']:.2f} ms  n={n}")


if __name__ == "__main__":
    main()
