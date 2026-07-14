#!/usr/bin/env python
"""Embed the corpus with a trained embedder and save a brute-force index.

Loads the document store, tokenizer artifact, and trained embedder, embeds every
document's chunk text, and writes a memory-mapped :class:`EmbeddingIndex` for dense
retrieval.

Example
-------
    uv run python scripts/embed_corpus.py --db data/corpus.sqlite \\
        --tokenizer artifacts/tokenizer.json --embedder artifacts/embedder \\
        --out artifacts/embedding_index
"""

from __future__ import annotations

import argparse
from pathlib import Path

from meridian.corpus.store import SqliteDocumentStore
from meridian.encoder.artifact import load_embedder
from meridian.encoder.embed import embed_documents
from meridian.retrieval.embedding_index import EmbeddingIndex
from meridian.tokenization.artifact import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="SQLite document store path")
    parser.add_argument("--tokenizer", type=Path, required=True, help="tokenizer artifact path")
    parser.add_argument("--embedder", type=Path, required=True, help="embedder artifact dir")
    parser.add_argument("--out", type=Path, required=True, help="output embedding index dir")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    embedder = load_embedder(args.embedder)
    with SqliteDocumentStore(args.db) as store:
        if store.count() == 0:
            parser.error("document store is empty; run scripts/ingest.py first")
        pmids, vectors = embed_documents(
            embedder,
            tokenizer,
            store.iter_documents(),
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    index = EmbeddingIndex.build(pmids, vectors)
    index.save(args.out)
    print(f"embedded {len(index)} documents (dim={index.dim}) -> {args.out}")


if __name__ == "__main__":
    main()
