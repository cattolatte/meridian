"""Tests for tokenizer artifact save/load and versioning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meridian.tokenization.artifact import (
    FORMAT_VERSION,
    load_artifact,
    load_tokenizer,
    save_tokenizer,
)
from meridian.tokenization.training import train_tokenizer

_BIO = ["metformin reduces cardiovascular mortality", "diabetes and heart disease"] * 4
_GENERAL = ["a general english sentence for the mix"] * 4


def test_save_load_preserves_encoding(tmp_path: Path) -> None:
    tok = train_tokenizer(_BIO, _GENERAL, vocab_size=120)
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tok, path, metadata={"vocab_size": 120, "mix_ratio": 0.7})

    reloaded = load_tokenizer(path)
    text = "metformin reduces cardiovascular mortality"
    assert reloaded.encode(text).ids == tok.encode(text).ids
    assert reloaded.tokenize(text) == tok.tokenize(text)


def test_metadata_roundtrips(tmp_path: Path) -> None:
    tok = train_tokenizer(_BIO, _GENERAL, vocab_size=120)
    path = tmp_path / "tokenizer.json"
    meta = {"vocab_size": 120, "mix_ratio": 0.7, "corpus_checksum": "abc123"}
    save_tokenizer(tok, path, metadata=meta)

    artifact = load_artifact(path)
    assert artifact.metadata == meta
    assert artifact.format_version == FORMAT_VERSION


def test_artifact_is_deterministic_on_disk(tmp_path: Path) -> None:
    tok = train_tokenizer(_BIO, _GENERAL, vocab_size=120)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    save_tokenizer(tok, a, metadata={"k": "v"})
    save_tokenizer(tok, b, metadata={"k": "v"})
    assert a.read_text() == b.read_text()


def test_unsupported_format_version_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    tok = train_tokenizer(_BIO, _GENERAL, vocab_size=120)
    save_tokenizer(tok, path)
    data = json.loads(path.read_text())
    data["format_version"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_artifact(path)
