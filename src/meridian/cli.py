"""The ``meridian`` command-line interface.

Phase 2 exposes two subcommands over the document store:

- ``meridian ingest`` — build/rebuild the store from raw PubMed files.
- ``meridian ask`` — answer a question with cited, verbatim extractive passages
  (the Phase 2 vertical slice), or abstain.

Uses only the standard library so the base install stays dependency-light.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from meridian.answer.extractive import answer_extractive, render_answer
from meridian.corpus.ingest import ingest_documents
from meridian.corpus.store import SqliteDocumentStore
from meridian.retrieval.factory import build_retriever

_XML_SUFFIXES = (".xml", ".xml.gz")


def _resolve_inputs(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            files.extend(sorted(p for p in item.iterdir() if p.name.endswith(_XML_SUFFIXES)))
        else:
            files.append(item)
    return files


def _run_ingest(args: argparse.Namespace) -> int:
    files = _resolve_inputs(args.inputs)
    if not files:
        print("no input XML files found")
        return 1
    args.db.parent.mkdir(parents=True, exist_ok=True)
    with SqliteDocumentStore(args.db) as store:
        summary = ingest_documents(files, store, apply_domain_filter=not args.no_domain_filter)
    print(
        f"parsed={summary.parsed} unique={summary.unique} "
        f"in_domain={summary.in_domain} stored={summary.stored}"
    )
    return 0


def _run_ask(args: argparse.Namespace) -> int:
    if not args.db.exists():
        print(f"document store not found: {args.db} (run 'meridian ingest' first)")
        return 1
    with SqliteDocumentStore(args.db) as store:
        if store.count() == 0:
            print("document store is empty (run 'meridian ingest' first)")
            return 1
        try:
            retriever = build_retriever(
                args.retriever,
                store,
                embedder_dir=args.embedder,
                tokenizer_path=args.tokenizer,
                index_dir=args.index,
                ann=args.ann,
                rerank=args.rerank,
                reranker_dir=args.reranker,
            )
        except ValueError as error:
            print(str(error))
            return 1
        answer = answer_extractive(
            retriever, args.question, k_passages=args.passages, max_sentences=args.sentences
        )
        print(render_answer(answer))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meridian", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="build the document store from raw PubMed files")
    ingest.add_argument(
        "inputs", type=Path, nargs="+", help="raw .xml/.xml.gz files or directories"
    )
    ingest.add_argument("--db", type=Path, required=True, help="SQLite document store path")
    ingest.add_argument("--no-domain-filter", action="store_true", help="keep all records")
    ingest.set_defaults(func=_run_ingest)

    ask = subparsers.add_parser("ask", help="answer a question with cited extractive passages")
    ask.add_argument("question", help="the question to answer")
    ask.add_argument("--db", type=Path, required=True, help="SQLite document store path")
    ask.add_argument(
        "--retriever",
        choices=("bm25", "dense", "hybrid"),
        default="bm25",
        help="retrieval backend (hybrid = RRF of BM25 + dense)",
    )
    ask.add_argument("--embedder", type=Path, help="trained embedder artifact dir (dense)")
    ask.add_argument("--tokenizer", type=Path, help="tokenizer artifact path (dense)")
    ask.add_argument("--index", type=Path, help="prebuilt embedding index dir (dense; optional)")
    ask.add_argument(
        "--ann", choices=("none", "ivf", "hnsw"), default="none", help="ANN search backend (dense)"
    )
    ask.add_argument(
        "--rerank", action="store_true", help="rerank candidates with the cross-encoder"
    )
    ask.add_argument("--reranker", type=Path, help="reranker artifact dir (with --rerank)")
    ask.add_argument("--passages", type=int, default=5, help="passages to retrieve")
    ask.add_argument("--sentences", type=int, default=3, help="cited sentences to return")
    ask.set_defaults(func=_run_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``meridian`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
