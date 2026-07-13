#!/usr/bin/env python
"""One-command corpus ingest: raw PubMed files → SQLite store + stats report.

Examples
--------
Rebuild the store from downloaded baseline files and write the report::

    uv run python scripts/ingest.py data/baseline --db data/corpus.sqlite \\
        --report benchmarks/corpus.md

Run the offline demo on the committed sample fixture::

    uv run python scripts/ingest.py examples/sample_pubmed.xml \\
        --db build/sample.sqlite --report build/sample-corpus.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from meridian.corpus.ingest import ingest_documents
from meridian.corpus.stats import compute_stats, render_markdown
from meridian.corpus.store import SqliteDocumentStore

_XML_SUFFIXES = (".xml", ".xml.gz")


def _resolve_inputs(inputs: list[Path]) -> list[Path]:
    """Expand directories to their PubMed XML files; keep explicit files as given."""
    files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            files.extend(sorted(p for p in item.iterdir() if p.name.endswith(_XML_SUFFIXES)))
        else:
            files.append(item)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", type=Path, nargs="+", help="raw .xml/.xml.gz files or directories"
    )
    parser.add_argument("--db", type=Path, required=True, help="SQLite document store path")
    parser.add_argument(
        "--report", type=Path, help="write a corpus statistics Markdown report here"
    )
    parser.add_argument(
        "--no-domain-filter",
        action="store_true",
        help="keep all records instead of filtering to the ADR-0001 domains",
    )
    args = parser.parse_args()

    files = _resolve_inputs(args.inputs)
    if not files:
        parser.error("no input XML files found")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    with SqliteDocumentStore(args.db) as store:
        summary = ingest_documents(files, store, apply_domain_filter=not args.no_domain_filter)
        print(
            f"parsed={summary.parsed} unique={summary.unique} "
            f"in_domain={summary.in_domain} stored={summary.stored}"
        )
        if args.report is not None:
            stats = compute_stats(store.iter_documents())
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(render_markdown(stats, title="Corpus statistics"))
            print(f"wrote report: {args.report}")


if __name__ == "__main__":
    main()
