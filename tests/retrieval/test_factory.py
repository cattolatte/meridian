"""Tests for the retriever factory and dense end-to-end wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from polaris.tokenizers import BPETokenizer

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.encoder.artifact import EmbedderConfig, build_embedder, load_embedder, save_embedder
from meridian.encoder.embed import embed_documents
from meridian.retrieval.dense import DenseRetriever
from meridian.retrieval.embedding_index import EmbeddingIndex
from meridian.retrieval.factory import build_dense_retriever, build_retriever
from meridian.retrieval.hybrid import HybridRetriever
from meridian.retrieval.pipeline import BM25Retriever
from meridian.tokenization.artifact import save_tokenizer
from meridian.tokenization.training import train_tokenizer

_DOCS = [
    Document(pmid="1", title="Heart failure", abstract="beta blockers reduce mortality"),
    Document(pmid="2", title="Diabetes", abstract="metformin cardiovascular outcomes"),
]
_BIO = ["heart failure mortality", "diabetes heart disease"] * 4
_GENERAL = ["a general english passage here"] * 4


def _store() -> SqliteDocumentStore:
    store = SqliteDocumentStore(":memory:")
    store.add_many(_DOCS)
    return store


def _artifacts(tmp_path: Path) -> tuple[Path, Path, BPETokenizer]:
    tok = train_tokenizer(_BIO, _GENERAL, vocab_size=120)
    tok_path = tmp_path / "tokenizer.json"
    save_tokenizer(tok, tok_path)
    config = EmbedderConfig(
        vocab_size=tok.vocabulary.size,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
        max_len=64,
        pad_id=tok.vocabulary.pad_id or 0,
    )
    torch.manual_seed(0)
    emb_dir = tmp_path / "embedder"
    save_embedder(build_embedder(config), config, emb_dir)
    return emb_dir, tok_path, tok


def test_build_bm25() -> None:
    with _store() as store:
        assert isinstance(build_retriever("bm25", store), BM25Retriever)


def test_unknown_kind_rejected() -> None:
    with _store() as store, pytest.raises(ValueError):
        build_retriever("nope", store)


def test_dense_missing_artifacts_rejected() -> None:
    with _store() as store, pytest.raises(ValueError):
        build_retriever("dense", store)


def test_build_dense_from_artifacts(tmp_path: Path) -> None:
    emb_dir, tok_path, _ = _artifacts(tmp_path)
    with _store() as store:
        retriever = build_retriever("dense", store, embedder_dir=emb_dir, tokenizer_path=tok_path)
        assert isinstance(retriever, DenseRetriever)
        hits = retriever.retrieve("heart failure", k=2)
        assert len(hits) == 2


@pytest.mark.parametrize("ann", ["ivf", "hnsw"])
def test_build_dense_with_ann_backend(tmp_path: Path, ann: str) -> None:
    emb_dir, tok_path, _ = _artifacts(tmp_path)
    with _store() as store:
        retriever = build_retriever(
            "dense", store, embedder_dir=emb_dir, tokenizer_path=tok_path, ann=ann
        )
        assert isinstance(retriever, DenseRetriever)
        assert len(retriever.retrieve("diabetes", k=2)) == 2


def test_build_hybrid(tmp_path: Path) -> None:
    emb_dir, tok_path, _ = _artifacts(tmp_path)
    with _store() as store:
        retriever = build_retriever(
            "hybrid", store, embedder_dir=emb_dir, tokenizer_path=tok_path
        )
        assert isinstance(retriever, HybridRetriever)
        assert len(retriever.retrieve("heart failure", k=2)) == 2


def test_hybrid_missing_artifacts_rejected() -> None:
    with _store() as store, pytest.raises(ValueError):
        build_retriever("hybrid", store)


def test_dense_with_prebuilt_index(tmp_path: Path) -> None:
    emb_dir, tok_path, tok = _artifacts(tmp_path)
    with _store() as store:
        # Persist an embedding index once, then load it via the factory.
        embedder = load_embedder(emb_dir)
        pmids, vectors = embed_documents(embedder, tok, list(store.iter_documents()))
        EmbeddingIndex.build(pmids, vectors).save(tmp_path / "idx")

        retriever = build_dense_retriever(
            store, embedder_dir=emb_dir, tokenizer_path=tok_path, index_dir=tmp_path / "idx"
        )
        assert len(retriever.retrieve("diabetes", k=1)) == 1
