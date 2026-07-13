"""Tests for mixed-corpus BPE training and the vocabulary-size sweep."""

from __future__ import annotations

import pytest

from meridian.tokenization.training import (
    _mixed_word_sequences,
    sweep_vocabulary_sizes,
    train_tokenizer,
)

# Small but non-trivial corpora; repetition gives the merge-learner real frequencies.
_BIO = [
    "metformin reduces cardiovascular mortality in type 2 diabetes",
    "cardiovascular disease and diabetes mellitus outcomes",
    "myocardial infarction risk with glucose lowering therapy",
] * 4
_GENERAL = [
    "the quick brown fox jumps over the lazy dog",
    "a general web passage about everyday english text",
] * 4


def test_mix_ratio_controls_general_word_budget() -> None:
    bio = ["one two three four"]  # 4 words
    general = ["a b", "c d", "e f", "g h"]  # 2 words each
    # ratio 0.5 -> target general words == bio words == 4 -> two general seqs.
    seqs = _mixed_word_sequences(bio, general, 0.5)
    general_selected = seqs[1:]
    assert sum(len(s) for s in general_selected) == 4


def test_mix_ratio_one_uses_no_general() -> None:
    seqs = _mixed_word_sequences(["a b c"], ["x y z"], 1.0)
    assert seqs == [["a", "b", "c"]]


def test_invalid_mix_ratio_rejected() -> None:
    with pytest.raises(ValueError):
        _mixed_word_sequences(["a"], ["b"], 0.0)


def test_empty_biomedical_corpus_rejected() -> None:
    with pytest.raises(ValueError):
        _mixed_word_sequences(["   "], ["b"], 0.7)


def test_train_tokenizer_roundtrips_text() -> None:
    tok = train_tokenizer(_BIO, _GENERAL, vocab_size=120, mix_ratio=0.7)
    assert (
        tok.decode(tok.encode("metformin reduces mortality").ids) == "metformin reduces mortality"
    )


def test_training_is_deterministic() -> None:
    a = train_tokenizer(_BIO, _GENERAL, vocab_size=120)
    b = train_tokenizer(_BIO, _GENERAL, vocab_size=120)
    ids = "cardiovascular diabetes"
    assert a.encode(ids).ids == b.encode(ids).ids


def test_larger_vocab_does_not_increase_fertility() -> None:
    results = sweep_vocabulary_sizes(
        _BIO,
        _GENERAL,
        [80, 200],
        biomedical_eval=_BIO,
        general_eval=_GENERAL,
    )
    assert [r.vocab_size for r in results] == [80, 200]
    # More merges can only keep or reduce fragmentation on the training-like sample.
    assert results[1].biomedical_fertility <= results[0].biomedical_fertility
