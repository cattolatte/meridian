"""Tests for the versioned embedder artifact (save/load)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from polaris.tokenizers import BPETokenizer

from meridian.encoder.artifact import (
    EmbedderConfig,
    build_embedder,
    load_embedder,
    save_embedder,
)
from meridian.encoder.embed import encode_texts
from meridian.tokenization import train_tokenizer

_BIO = ["heart failure mortality", "diabetes heart disease"] * 4
_GENERAL = ["a general english passage here"] * 4


def _tok() -> BPETokenizer:
    return train_tokenizer(_BIO, _GENERAL, vocab_size=120)


def _config(tok: BPETokenizer) -> EmbedderConfig:
    return EmbedderConfig(
        vocab_size=tok.vocabulary.size,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
        max_len=64,
        pad_id=tok.vocabulary.pad_id or 0,
    )


def test_save_load_reproduces_embeddings(tmp_path: Path) -> None:
    tok = _tok()
    config = _config(tok)
    torch.manual_seed(0)
    embedder = build_embedder(config)
    before = encode_texts(embedder, tok, ["heart failure mortality"])

    save_embedder(embedder, config, tmp_path / "emb", metadata={"stage": "A"})
    reloaded = load_embedder(tmp_path / "emb")
    after = encode_texts(reloaded, tok, ["heart failure mortality"])

    assert np.allclose(before, after, atol=1e-6)


def test_config_and_metadata_persisted(tmp_path: Path) -> None:
    tok = _tok()
    config = _config(tok)
    torch.manual_seed(0)
    save_embedder(build_embedder(config), config, tmp_path / "emb", metadata={"stage": "A"})
    payload = json.loads((tmp_path / "emb" / "config.json").read_text())
    assert payload["arch"]["embed_dim"] == 16
    assert payload["metadata"] == {"stage": "A"}


def test_unsupported_format_version_rejected(tmp_path: Path) -> None:
    tok = _tok()
    config = _config(tok)
    torch.manual_seed(0)
    save_embedder(build_embedder(config), config, tmp_path / "emb")
    path = tmp_path / "emb" / "config.json"
    data = json.loads(path.read_text())
    data["format_version"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_embedder(tmp_path / "emb")
