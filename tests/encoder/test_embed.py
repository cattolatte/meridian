"""Tests for corpus embedding with a (small, random) Polaris TextEmbedder."""

from __future__ import annotations

import numpy as np
import torch
from polaris.models import TextEmbedder
from polaris.tokenizers import BPETokenizer

from meridian.corpus.records import Document
from meridian.encoder.embed import embed_documents, encode_texts
from meridian.tokenization import train_tokenizer

_BIO = ["heart failure study mortality", "diabetes and heart disease outcomes"] * 4
_GENERAL = ["a general english passage for the tokenizer mix"] * 4


def _tokenizer() -> BPETokenizer:
    return train_tokenizer(_BIO, _GENERAL, vocab_size=120)


def _embedder(tokenizer: BPETokenizer, dim: int = 16) -> TextEmbedder:
    torch.manual_seed(0)
    pad_id = tokenizer.vocabulary.pad_id or 0
    return TextEmbedder(
        vocab_size=tokenizer.vocabulary.size,
        embed_dim=dim,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
        max_len=64,
        pad_id=pad_id,
    )


def test_encode_texts_shape_and_normalized() -> None:
    tok = _tokenizer()
    emb = _embedder(tok)
    vectors = encode_texts(emb, tok, ["heart failure", "diabetes disease"])
    assert vectors.shape == (2, emb.embedding_dim)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)  # TextEmbedder(normalize=True)


def test_encode_empty_texts() -> None:
    tok = _tokenizer()
    emb = _embedder(tok)
    vectors = encode_texts(emb, tok, [])
    assert vectors.shape == (0, emb.embedding_dim)


def test_encode_is_deterministic() -> None:
    tok = _tokenizer()
    emb = _embedder(tok)
    a = encode_texts(emb, tok, ["heart failure mortality"])
    b = encode_texts(emb, tok, ["heart failure mortality"])
    assert np.array_equal(a, b)


def test_embed_documents_returns_pmids_and_matrix() -> None:
    tok = _tokenizer()
    emb = _embedder(tok)
    docs = [
        Document(pmid="1", title="Heart failure", abstract="beta blockers reduce mortality"),
        Document(pmid="2", title="Diabetes", abstract="metformin and cardiovascular outcomes"),
    ]
    pmids, vectors = embed_documents(emb, tok, docs)
    assert pmids == ["1", "2"]
    assert vectors.shape == (2, emb.embedding_dim)
