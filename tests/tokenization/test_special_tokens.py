"""Tests for adding a mask token without shifting existing ids (ADR-0003)."""

from __future__ import annotations

from polaris.tokenizers import BPETokenizer, train_bpe

from meridian.tokenization.special_tokens import ensure_mask_token
from meridian.tokenization.training import train_tokenizer

_BIO = ["heart failure mortality", "diabetes heart disease"] * 4
_GENERAL = ["a general english passage here"] * 4


def _maskless_tokenizer() -> BPETokenizer:
    # A tokenizer trained without a mask token, to exercise the append path.
    return train_bpe(
        [text.split() for text in _BIO + _GENERAL],
        vocab_size=120,
        unk_token="<unk>",
        pad_token="<pad>",
    )


def test_mask_token_appended_without_renumbering() -> None:
    tok = _maskless_tokenizer()
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
    # train_tokenizer now reserves <mask>, so ensure_mask_token is a no-op.
    tok = train_tokenizer(_BIO, _GENERAL, vocab_size=120)
    assert tok.vocabulary.mask_token == "<mask>"
    same, mask_id = ensure_mask_token(tok)
    assert same is tok
    assert mask_id == tok.vocabulary.mask_id
