"""Tests for adding a mask token without shifting existing ids (ADR-0003)."""

from __future__ import annotations

from meridian.tokenization.special_tokens import ensure_mask_token
from meridian.tokenization.training import train_tokenizer

_BIO = ["heart failure mortality", "diabetes heart disease"] * 4
_GENERAL = ["a general english passage here"] * 4


def test_mask_token_appended_without_renumbering() -> None:
    tok = train_tokenizer(_BIO, _GENERAL, vocab_size=120)
    assert tok.vocabulary.mask_token is None
    original_size = tok.vocabulary.size
    original_ids = tok.encode("heart failure mortality").ids

    masked, mask_id = ensure_mask_token(tok)

    assert masked.vocabulary.mask_token == "<mask>"
    assert mask_id == original_size  # appended at the end
    assert masked.vocabulary.size == original_size + 1
    # Existing tokens keep their ids; normal encoding is unchanged.
    assert masked.encode("heart failure mortality").ids == original_ids


def test_idempotent_when_mask_already_present() -> None:
    tok = train_tokenizer(_BIO, _GENERAL, vocab_size=120)
    masked, mask_id = ensure_mask_token(tok)
    again, again_id = ensure_mask_token(masked)
    assert again_id == mask_id
    assert again is masked  # returned unchanged
