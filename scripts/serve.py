#!/usr/bin/env python
"""Run the Meridian API over a document store.

    uv run python scripts/serve.py --db data/corpus.sqlite --host 0.0.0.0 --port 8000

Requires the ``serving`` extra: ``uv sync --extra serving``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from meridian.corpus.store import SqliteDocumentStore
from meridian.retrieval.pipeline import BM25Retriever
from meridian.serving.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="SQLite document store path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    store = SqliteDocumentStore(args.db)
    app = create_app(BM25Retriever.from_store(store), store)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
