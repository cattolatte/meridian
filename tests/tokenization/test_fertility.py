"""Tests for the fertility metric."""

from __future__ import annotations

import pytest

from meridian.tokenization.fertility import fertility
from meridian.tokenization.training import train_tokenizer

_CORPUS = ["alpha beta gamma delta", "beta gamma alpha epsilon"] * 4


def test_fertility_is_at_least_one() -> None:
    tok = train_tokenizer(_CORPUS, ["some general text here"], vocab_size=100)
    value = fertility(tok, ["alpha beta gamma"])
    assert value >= 1.0


def test_fertility_ignores_empty_texts() -> None:
    tok = train_tokenizer(_CORPUS, ["some general text here"], vocab_size=100)
    with_blank = fertility(tok, ["alpha beta", "   ", "gamma"])
    without_blank = fertility(tok, ["alpha beta", "gamma"])
    assert with_blank == without_blank


def test_fertility_empty_sample_raises() -> None:
    tok = train_tokenizer(_CORPUS, ["some general text here"], vocab_size=100)
    with pytest.raises(ValueError):
        fertility(tok, ["", "   "])
