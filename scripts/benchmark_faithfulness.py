#!/usr/bin/env python
"""Measure faithfulness of cited answers with the trained NLI verifier (Phase 8, RAG.md 7).

Answers every query in an eval split with the extractive answerer, then runs
``verify_grounded_answer`` over each cited sentence and aggregates citation precision,
citation recall, and hallucination rate -- plus the answer coverage (fraction not
abstained) used for the abstention operating point.

**What this measures.** Extractive sentences are quoted *verbatim* from their cited
abstract, so a perfect verifier would mark every one ENTAILMENT. The numbers below are
therefore a direct read on the *verifier's* judgement on real biomedical text, not on a
generator's hallucination rate; a generated-answer run needs the Phase-7 generator.

    uv run python scripts/benchmark_faithfulness.py --db data/pqal.sqlite \\
        --split data/splits/pqal_dev.json --verifier artifacts/verifier \\
        --tokenizer artifacts/tokenizer.json --out data/faithfulness.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from meridian.answer.extractive import answer_extractive
from meridian.corpus.store import SqliteDocumentStore
from meridian.device import resolve_device
from meridian.eval.qrels import load_eval_set
from meridian.generation.answerer import GroundedAnswer
from meridian.retrieval.pipeline import BM25Retriever
from meridian.tokenization.artifact import load_tokenizer
from meridian.verify.artifact import load_verifier
from meridian.verify.verifier import verify_grounded_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True, help="verifier artifact dir")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="write JSON results here")
    parser.add_argument("--k", type=int, default=5, help="passages retrieved per query")
    parser.add_argument("--limit", type=int, help="cap queries (smoke runs)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = parser.parse_args()

    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    verifier = load_verifier(args.verifier).to(device)
    eval_set = load_eval_set(args.split)
    queries = list(eval_set.queries)[: args.limit] if args.limit else list(eval_set.queries)
    print(f"device: {device} | queries: {len(queries)}", flush=True)

    precisions: list[float] = []
    recalls: list[float] = []
    hallucinations: list[float] = []
    grounded_flags: list[bool] = []
    answered = 0

    with SqliteDocumentStore(args.db) as store:
        retriever = BM25Retriever.from_store(store)
        for n, query in enumerate(queries, start=1):
            extractive = answer_extractive(retriever, query.question, k_passages=args.k)
            if extractive.abstained or not extractive.sentences:
                continue
            answered += 1

            # Render the extractive answer as a cited GroundedAnswer so the NLI verifier
            # can check each sentence against the abstract it cites.
            order = {pmid: i + 1 for i, (pmid, _title) in enumerate(extractive.sources)}
            text = " ".join(
                f"{sentence.text} [{order[sentence.pmid]}]"
                for sentence in extractive.sentences
                if sentence.pmid in order
            )
            answer = GroundedAnswer(
                query=query.question,
                abstained=False,
                text=text,
                citations=(),
                passages=extractive.sources,
            )
            report = verify_grounded_answer(answer, store, verifier, tokenizer)
            precisions.append(report.citation_precision)
            recalls.append(report.citation_recall)
            hallucinations.append(report.hallucination_rate)
            grounded_flags.append(report.grounded)
            if n % 50 == 0:
                print(f"  {n}/{len(queries)} queries", flush=True)

    if not precisions:
        parser.error("no answered queries to score")

    results = {
        "queries": len(queries),
        "answered": answered,
        "answer_coverage": answered / len(queries),
        "citation_precision": statistics.mean(precisions),
        "citation_recall": statistics.mean(recalls),
        "hallucination_rate": statistics.mean(hallucinations),
        "grounded_rate": sum(grounded_flags) / len(grounded_flags),
    }
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
