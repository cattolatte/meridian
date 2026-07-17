"""Build a retriever by name — the seam the CLI and eval harness share.

Both ``meridian ask`` and ``scripts/evaluate.py`` need to construct either the BM25
baseline or the dense retriever from the same options, so that logic lives here once.
Dense retrieval loads a saved tokenizer + embedder artifact and either loads a
prebuilt embedding index or embeds the corpus on the fly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from meridian.corpus.store import DocumentStore
from meridian.encoder.artifact import load_embedder
from meridian.encoder.embed import embed_documents
from meridian.reranker.artifact import load_reranker
from meridian.retrieval.ann import build_ann_index
from meridian.retrieval.ann.base import VectorIndex
from meridian.retrieval.dense import DenseRetriever
from meridian.retrieval.embedding_index import EmbeddingIndex
from meridian.retrieval.hybrid import HybridRetriever
from meridian.retrieval.pipeline import BM25Retriever, Retriever
from meridian.retrieval.rerank import RerankingRetriever
from meridian.tokenization.artifact import load_tokenizer


def build_dense_retriever(
    store: DocumentStore,
    *,
    embedder_dir: str | Path,
    tokenizer_path: str | Path,
    index_dir: str | Path | None = None,
    ann: str = "none",
    max_length: int = 256,
) -> DenseRetriever:
    """Load the embedder + tokenizer and build a dense retriever over ``store``.

    ``ann`` selects the search backend: ``"none"`` (exact brute force), ``"ivf"``, or
    ``"hnsw"``. The ANN structure is built deterministically from the corpus vectors
    (loaded from ``index_dir`` if given, else embedded on the fly).
    """
    tokenizer = load_tokenizer(tokenizer_path)
    embedder = load_embedder(embedder_dir)
    if index_dir is not None:
        base = EmbeddingIndex.load(index_dir)
        pmids, vectors = list(base.pmids), np.asarray(base.vectors, dtype=np.float32)
    else:
        pmids, vectors = embed_documents(
            embedder, tokenizer, store.iter_documents(), max_length=max_length
        )
    index: VectorIndex = (
        EmbeddingIndex.build(pmids, vectors)
        if ann == "none"
        else build_ann_index(ann, pmids, vectors)
    )
    return DenseRetriever(embedder, tokenizer, index, store, max_length=max_length)


def build_retriever(
    kind: str,
    store: DocumentStore,
    *,
    embedder_dir: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    index_dir: str | Path | None = None,
    ann: str = "none",
    rerank: bool = False,
    reranker_dir: str | Path | None = None,
    rerank_base_weight: float = 0.0,
) -> Retriever:
    """Build the ``bm25``/``dense``/``hybrid`` retriever, optionally reranked.

    ``rerank_base_weight`` fuses the reranker with the base ranking (reciprocal-rank
    fusion): ``0.0`` is a pure cross-encoder rerank, while a positive value anchors the
    result to the base so a reranker weaker than its base degrades gracefully instead of
    scrambling a strong ranking (see :class:`RerankingRetriever`).

    Raises :class:`ValueError` for an unknown ``kind`` or missing artifacts.
    """
    base = _build_base_retriever(
        kind,
        store,
        embedder_dir=embedder_dir,
        tokenizer_path=tokenizer_path,
        index_dir=index_dir,
        ann=ann,
    )
    if not rerank:
        return base
    if reranker_dir is None or tokenizer_path is None:
        raise ValueError("reranking requires --reranker and --tokenizer")
    return RerankingRetriever(
        base,
        load_reranker(reranker_dir),
        load_tokenizer(tokenizer_path),
        store,
        base_weight=rerank_base_weight,
    )


def _build_base_retriever(
    kind: str,
    store: DocumentStore,
    *,
    embedder_dir: str | Path | None,
    tokenizer_path: str | Path | None,
    index_dir: str | Path | None,
    ann: str,
) -> Retriever:
    if kind == "bm25":
        return BM25Retriever.from_store(store)
    if kind in ("dense", "hybrid"):
        if embedder_dir is None or tokenizer_path is None:
            raise ValueError(f"{kind} retrieval requires --embedder and --tokenizer")
        dense = build_dense_retriever(
            store,
            embedder_dir=embedder_dir,
            tokenizer_path=tokenizer_path,
            index_dir=index_dir,
            ann=ann,
        )
        if kind == "dense":
            return dense
        return HybridRetriever([BM25Retriever.from_store(store), dense], store)
    raise ValueError(f"unknown retriever kind: {kind!r}")
