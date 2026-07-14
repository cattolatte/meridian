"""Tests for the contrastive training wrapper."""

from __future__ import annotations

import math

import pytest
import torch
from polaris.models import TextEmbedder
from polaris.tokenizers import BPETokenizer

from meridian.encoder.data import make_contrastive_samples
from meridian.encoder.training import train_retriever
from meridian.tokenization import train_tokenizer

_BIO = ["heart failure mortality study", "diabetes heart disease outcomes"] * 4
_GENERAL = ["a general english passage for the mix"] * 4

# Easy positives: anchor and positive share content; distinct topics across pairs.
_PAIRS = [
    ("heart failure mortality", "beta blockers reduce mortality"),
    ("diabetes glucose control", "metformin lowers glucose"),
    ("melanoma immunotherapy", "checkpoint blockade responses"),
    ("hypertension blood pressure", "reduced systolic pressure"),
]


def _tok() -> BPETokenizer:
    return train_tokenizer(_BIO, _GENERAL, vocab_size=160)


def _embedder(tok: BPETokenizer) -> TextEmbedder:
    torch.manual_seed(0)
    return TextEmbedder(
        vocab_size=tok.vocabulary.size,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
        max_len=64,
        pad_id=tok.vocabulary.pad_id or 0,
    )


def test_empty_samples_rejected() -> None:
    tok = _tok()
    with pytest.raises(ValueError):
        train_retriever(_embedder(tok), [], pad_id=tok.vocabulary.pad_id or 0)


def test_returns_one_loss_per_epoch_all_finite() -> None:
    tok = _tok()
    samples = make_contrastive_samples(_PAIRS, tok)
    losses = train_retriever(
        _embedder(tok), samples, pad_id=tok.vocabulary.pad_id or 0, epochs=3, batch_size=2
    )
    assert len(losses) == 3
    assert all(math.isfinite(loss) for loss in losses)


def test_loss_decreases_over_epochs() -> None:
    tok = _tok()
    samples = make_contrastive_samples(_PAIRS, tok)
    losses = train_retriever(
        _embedder(tok),
        samples,
        pad_id=tok.vocabulary.pad_id or 0,
        epochs=15,
        batch_size=4,
        learning_rate=5e-3,
    )
    assert losses[-1] < losses[0]  # the InfoNCE objective is being optimized


def test_training_is_deterministic() -> None:
    tok = _tok()
    samples = make_contrastive_samples(_PAIRS, tok)
    a = train_retriever(
        _embedder(tok), samples, pad_id=tok.vocabulary.pad_id or 0, epochs=3, seed=7
    )
    b = train_retriever(
        _embedder(tok), samples, pad_id=tok.vocabulary.pad_id or 0, epochs=3, seed=7
    )
    assert a == b
