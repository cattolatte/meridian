"""Tests for Stage-0 MLM pretraining and trunk transfer."""

from __future__ import annotations

import pytest
import torch
from polaris.tokenizers import BPETokenizer

from meridian.encoder.artifact import EmbedderConfig, build_embedder
from meridian.encoder.pretrain import build_mlm, initialize_from_mlm, mlm_pretrain
from meridian.tokenization.special_tokens import ensure_mask_token
from meridian.tokenization.training import train_tokenizer

_BIO = ["heart failure mortality study", "diabetes heart disease outcomes"] * 6
_GENERAL = ["a general english passage for the tokenizer mix"] * 6


def _setup() -> tuple[BPETokenizer, int, EmbedderConfig]:
    tok = train_tokenizer(_BIO, _GENERAL, vocab_size=160)
    masked, mask_id = ensure_mask_token(tok)
    config = EmbedderConfig(
        vocab_size=masked.vocabulary.size,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
        max_len=64,
        pad_id=masked.vocabulary.pad_id or 0,
    )
    return masked, mask_id, config


def test_pretrain_returns_metrics_per_epoch() -> None:
    tok, mask_id, config = _setup()
    torch.manual_seed(0)
    mlm = build_mlm(config)
    records = mlm_pretrain(
        mlm,
        _BIO + _GENERAL,
        tok,
        mask_id=mask_id,
        vocab_size=config.vocab_size,
        epochs=2,
        batch_size=4,
    )
    assert len(records) == 2


def test_empty_texts_rejected() -> None:
    tok, mask_id, config = _setup()
    mlm = build_mlm(config)
    with pytest.raises(ValueError):
        mlm_pretrain(mlm, [], tok, mask_id=mask_id, vocab_size=config.vocab_size)


def test_transfer_copies_trunk_into_embedder() -> None:
    tok, mask_id, config = _setup()
    torch.manual_seed(1)
    mlm = build_mlm(config)
    mlm_pretrain(
        mlm, _BIO, tok, mask_id=mask_id, vocab_size=config.vocab_size, epochs=1, batch_size=4
    )

    torch.manual_seed(2)  # different init so the trunks start unequal
    embedder = build_embedder(config)
    initialize_from_mlm(mlm, embedder)

    mlm_trunk = mlm.encoder.state_dict()
    emb_trunk = embedder.encoder.state_dict()
    assert mlm_trunk.keys() == emb_trunk.keys()
    assert all(torch.equal(mlm_trunk[k], emb_trunk[k]) for k in mlm_trunk)
