"""Tests for the dense bi-encoder retriever (small random embedder)."""

from __future__ import annotations

import torch
from polaris.models import TextEmbedder
from polaris.tokenizers import BPETokenizer

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.retrieval.dense import DenseRetriever
from meridian.retrieval.pipeline import Retriever
from meridian.tokenization import train_tokenizer

_BIO = ["heart failure mortality study", "diabetes heart disease outcomes"] * 4
_GENERAL = ["a general english passage for the mix"] * 4

_DOCS = [
    Document(pmid="1", title="Heart failure", abstract="beta blockers reduce mortality"),
    Document(pmid="2", title="Diabetes", abstract="metformin cardiovascular outcomes"),
    Document(pmid="3", title="Melanoma", abstract="checkpoint immunotherapy responses"),
]


def _tokenizer() -> BPETokenizer:
    return train_tokenizer(_BIO, _GENERAL, vocab_size=120)


def _embedder(tokenizer: BPETokenizer) -> TextEmbedder:
    torch.manual_seed(0)
    return TextEmbedder(
        vocab_size=tokenizer.vocabulary.size,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
        max_len=64,
        pad_id=tokenizer.vocabulary.pad_id or 0,
    )


def _store() -> SqliteDocumentStore:
    store = SqliteDocumentStore(":memory:")
    store.add_many(_DOCS)
    return store


def test_dense_retriever_satisfies_protocol() -> None:
    tok = _tokenizer()
    with _store() as store:
        retriever = DenseRetriever.from_store(_embedder(tok), tok, store)
        assert isinstance(retriever, Retriever)


def test_index_covers_corpus() -> None:
    tok = _tokenizer()
    with _store() as store:
        retriever = DenseRetriever.from_store(_embedder(tok), tok, store)
        assert len(retriever.index) == 3


def test_retrieve_returns_k_hits_with_documents() -> None:
    tok = _tokenizer()
    with _store() as store:
        retriever = DenseRetriever.from_store(_embedder(tok), tok, store)
        hits = retriever.retrieve("heart failure mortality", k=2)
        assert len(hits) == 2
        assert all(hit.document is not None for hit in hits)
        assert {hit.pmid for hit in hits} <= {"1", "2", "3"}


def test_retrieve_is_deterministic() -> None:
    tok = _tokenizer()
    with _store() as store:
        retriever = DenseRetriever.from_store(_embedder(tok), tok, store)
        assert retriever.retrieve("diabetes", k=3) == retriever.retrieve("diabetes", k=3)
