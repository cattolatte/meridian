#!/usr/bin/env python
"""Measure per-stage serving latency over a real query set (Phase 10, RAG.md §7).

Times each pipeline stage over every question in an eval split and reports P50/P95
wall-clock via ``StageTimer`` — earned numbers, not inherited claims. ``search`` and
``answer`` always run; ``embed``, ``rerank``, and ``verify`` are timed when their trained
artifacts are supplied. (``generate`` needs the Phase-7 Zenith generator.)

    uv run python scripts/benchmark_latency.py --db data/pubmedqa.sqlite \\
        --split data/pubmedqa/pqal_dev.json \\
        --embedder artifacts/embedder --tokenizer artifacts/tokenizer.json \\
        --reranker artifacts/reranker --verifier artifacts/verifier
"""

from __future__ import annotations

import argparse
from pathlib import Path

from meridian.answer.extractive import answer_extractive
from meridian.corpus.store import SqliteDocumentStore
from meridian.device import resolve_device
from meridian.encoder.artifact import load_embedder
from meridian.encoder.embed import encode_texts
from meridian.eval.qrels import load_eval_set
from meridian.generation.answerer import GroundedAnswer
from meridian.reranker.artifact import load_reranker
from meridian.retrieval.pipeline import BM25Retriever
from meridian.retrieval.rerank import RerankingRetriever
from meridian.serving.instrumentation import StageTimer
from meridian.tokenization.artifact import load_tokenizer
from meridian.verify.artifact import load_verifier
from meridian.verify.verifier import verify_grounded_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tokenizer", type=Path, help="tokenizer artifact (embed/rerank/verify)")
    parser.add_argument("--embedder", type=Path, help="embedder artifact dir (times `embed`)")
    parser.add_argument("--reranker", type=Path, help="reranker artifact dir (times `rerank`)")
    parser.add_argument("--verifier", type=Path, help="verifier artifact dir (times `verify`)")
    parser.add_argument("--limit", type=int, help="cap queries")
    parser.add_argument("--device", default="cpu", help="auto | cpu | cuda | mps")
    args = parser.parse_args()

    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else None
    if (args.embedder or args.reranker or args.verifier) and tokenizer is None:
        parser.error("--tokenizer is required with --embedder/--reranker/--verifier")

    embedder = load_embedder(args.embedder).to(device) if args.embedder else None
    reranker = load_reranker(args.reranker).to(device) if args.reranker else None
    verifier = load_verifier(args.verifier).to(device) if args.verifier else None

    eval_set = load_eval_set(args.split)
    queries = list(eval_set.queries)[: args.limit] if args.limit else list(eval_set.queries)
    timer = StageTimer()
    with SqliteDocumentStore(args.db) as store:
        corpus_size = store.count()
        retriever = BM25Retriever.from_store(store)
        reranking = (
            RerankingRetriever(retriever, reranker, tokenizer, store, candidates=100)
            if reranker is not None and tokenizer is not None
            else None
        )
        for query in queries:
            with timer.stage("search"):
                retriever.retrieve(query.question, k=args.k)
            with timer.stage("answer"):
                extractive = answer_extractive(retriever, query.question, k_passages=args.k)

            if embedder is not None and tokenizer is not None:
                with timer.stage("embed"):
                    encode_texts(embedder, tokenizer, [query.question], device=device)

            if reranking is not None:
                with timer.stage("rerank"):
                    reranking.retrieve(query.question, k=args.k)

            if verifier is not None and tokenizer is not None and extractive.sentences:
                order = {pmid: i + 1 for i, (pmid, _t) in enumerate(extractive.sources)}
                text = " ".join(
                    f"{s.text} [{order[s.pmid]}]" for s in extractive.sentences if s.pmid in order
                )
                answer = GroundedAnswer(
                    query=query.question,
                    abstained=False,
                    text=text,
                    citations=(),
                    passages=extractive.sources,
                )
                with timer.stage("verify"):
                    verify_grounded_answer(answer, store, verifier, tokenizer)

    print(f"queries: {len(queries)}  corpus: {corpus_size}  device: {device}")
    for stage, stats in timer.percentiles().items():
        n = int(stats["count"])
        print(f"{stage:8s} P50={stats['p50']:.2f} ms  P95={stats['p95']:.2f} ms  n={n}")


if __name__ == "__main__":
    main()
