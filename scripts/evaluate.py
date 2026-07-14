#!/usr/bin/env python
"""Evaluate BM25 retrieval over a frozen split and write JSON results.

Reports Recall@{5,20,100}, MRR@10, nDCG@10 (RAG.md §7). ``k1``/``b`` are search-time
parameters, so tuning them on the dev split needs no re-index; the test split is used
only once, in Phase 11.

Example
-------
    uv run python scripts/evaluate.py --db data/corpus.sqlite \\
        --split benchmarks/splits/dev.json --out benchmarks/results/bm25-dev.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from meridian.corpus.store import SqliteDocumentStore
from meridian.eval.harness import log_to_mlflow, run_evaluation, write_results
from meridian.eval.qrels import load_eval_set
from meridian.eval.splits import load_frozen_split
from meridian.retrieval.factory import build_dense_retriever
from meridian.retrieval.pipeline import BM25Retriever, Retriever


def _require(parser: argparse.ArgumentParser, value: Path | None, flag: str) -> Path:
    if value is None:
        parser.error(f"dense retrieval requires {flag}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="SQLite document store path")
    parser.add_argument("--split", type=Path, required=True, help="eval split JSON")
    parser.add_argument("--checksum", help="expected split checksum (enforces the frozen guard)")
    parser.add_argument("--out", type=Path, help="write JSON results here")
    parser.add_argument(
        "--retriever", choices=("bm25", "dense"), default="bm25", help="retrieval backend"
    )
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--embedder", type=Path, help="embedder artifact dir (dense)")
    parser.add_argument("--tokenizer", type=Path, help="tokenizer artifact path (dense)")
    parser.add_argument("--index", type=Path, help="prebuilt embedding index dir (dense; optional)")
    parser.add_argument(
        "--ann", choices=("none", "ivf", "hnsw"), default="none", help="ANN search backend (dense)"
    )
    parser.add_argument("--mlflow-experiment", help="log metrics to this MLflow experiment")
    args = parser.parse_args()

    eval_set = (
        load_frozen_split(args.split, args.checksum) if args.checksum else load_eval_set(args.split)
    )
    with SqliteDocumentStore(args.db) as store:
        if args.retriever == "bm25":
            retriever: Retriever = BM25Retriever.from_store(store, k1=args.k1, b=args.b)
        else:
            retriever = build_dense_retriever(
                store,
                embedder_dir=_require(parser, args.embedder, "--embedder"),
                tokenizer_path=_require(parser, args.tokenizer, "--tokenizer"),
                index_dir=args.index,
                ann=args.ann,
            )
        result = run_evaluation(retriever, eval_set)

    for name, value in sorted(result.metrics.items()):
        print(f"{name}: {value:.4f}")
    print(f"(n_queries={result.n_queries})")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_results(result, args.out)
        print(f"wrote results: {args.out}")
    if args.mlflow_experiment is not None:
        log_to_mlflow(result, experiment=args.mlflow_experiment)


if __name__ == "__main__":
    main()
