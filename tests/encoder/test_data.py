"""Tests for contrastive-sample construction and pair mining."""

from __future__ import annotations

from polaris.tokenizers import BPETokenizer

from meridian.corpus.records import Document
from meridian.encoder.data import (
    make_contrastive_samples,
    make_contrastive_samples_with_negatives,
    mine_title_abstract_pairs,
)
from meridian.tokenization import train_tokenizer

_BIO = ["heart failure mortality", "diabetes heart disease"] * 4
_GENERAL = ["a general english passage here"] * 4


def _tok() -> BPETokenizer:
    return train_tokenizer(_BIO, _GENERAL, vocab_size=120)


def test_make_contrastive_samples_are_encoding_pairs() -> None:
    tok = _tok()
    samples = make_contrastive_samples([("heart failure", "beta blockers")], tok)
    assert len(samples) == 1
    anchor, positive = samples[0]
    assert anchor.ids == tok.encode("heart failure").ids
    assert positive.ids == tok.encode("beta blockers").ids


def test_make_samples_with_negatives() -> None:
    tok = _tok()
    samples = make_contrastive_samples_with_negatives([("q", "pos", ["neg1", "neg2"])], tok)
    anchor, positive, negatives = samples[0]
    assert anchor.ids == tok.encode("q").ids
    assert len(negatives) == 2


def test_mine_title_abstract_pairs_skips_empty() -> None:
    docs = [
        Document(pmid="1", title="Heart failure", abstract="reduces mortality"),
        Document(pmid="2", title="", abstract="no title"),
        Document(pmid="3", title="Diabetes", abstract=""),
    ]
    pairs = mine_title_abstract_pairs(docs)
    assert pairs == [("Heart failure", "reduces mortality")]
