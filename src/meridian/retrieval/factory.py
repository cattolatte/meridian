"""Build a retriever by name — the seam the CLI and eval harness share.

Both ``meridian ask`` and ``scripts/evaluate.py`` need to construct either the BM25
baseline or the dense retriever from the same options, so that logic lives here once.
Dense retrieval loads a saved tokenizer + embedder artifact and either loads a
prebuilt embedding index or embeds the corpus on the fly.
"""

from __future__ import annotations

from pathlib import Path

from meridian.corpus.store import DocumentStore
from meridian.encoder.artifact import load_embedder
from meridian.retrieval.dense import DenseRetriever
from meridian.retrieval.embedding_index import EmbeddingIndex
from meridian.retrieval.pipeline import BM25Retriever, Retriever
from meridian.tokenization.artifact import load_tokenizer


def build_dense_retriever(
    store: DocumentStore,
    *,
    embedder_dir: str | Path,
    tokenizer_path: str | Path,
    index_dir: str | Path | None = None,
    max_length: int = 256,
) -> DenseRetriever:
    """Load the embedder + tokenizer and build a dense retriever over ``store``."""
    tokenizer = load_tokenizer(tokenizer_path)
    embedder = load_embedder(embedder_dir)
    if index_dir is not None:
        index = EmbeddingIndex.load(index_dir)
        return DenseRetriever(embedder, tokenizer, index, store, max_length=max_length)
    return DenseRetriever.from_store(embedder, tokenizer, store, max_length=max_length)


def build_retriever(
    kind: str,
    store: DocumentStore,
    *,
    embedder_dir: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    index_dir: str | Path | None = None,
) -> Retriever:
    """Build the ``bm25`` or ``dense`` retriever.

    Raises :class:`ValueError` for an unknown ``kind`` or missing dense artifacts.
    """
    if kind == "bm25":
        return BM25Retriever.from_store(store)
    if kind == "dense":
        if embedder_dir is None or tokenizer_path is None:
            raise ValueError("dense retrieval requires --embedder and --tokenizer")
        return build_dense_retriever(
            store,
            embedder_dir=embedder_dir,
            tokenizer_path=tokenizer_path,
            index_dir=index_dir,
        )
    raise ValueError(f"unknown retriever kind: {kind!r}")
