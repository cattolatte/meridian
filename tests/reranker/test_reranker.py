"""Tests for reranker data, training, and artifact."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from polaris.collation import collate_pairs
from polaris.tokenizers import BPETokenizer

from meridian.reranker.artifact import RerankerConfig, build_reranker, load_reranker, save_reranker
from meridian.reranker.data import make_pair_samples, pairs_from_triples
from meridian.reranker.training import train_reranker
from meridian.tokenization.training import train_tokenizer

_BIO = ["heart failure mortality study", "diabetes heart disease outcomes"] * 6
_GENERAL = ["a general english passage for the tokenizer mix"] * 6

# Relevant pairs (label 1) and clearly-irrelevant pairs (label 0).
_EXAMPLES = [
    ("heart failure mortality", "beta blockers reduce mortality", 1),
    ("heart failure mortality", "checkpoint immunotherapy melanoma", 0),
    ("diabetes glucose", "metformin lowers glucose", 1),
    ("diabetes glucose", "beta blockers reduce mortality", 0),
] * 2


def _tok() -> BPETokenizer:
    return train_tokenizer(_BIO, _GENERAL, vocab_size=200)


def _config(tok: BPETokenizer) -> RerankerConfig:
    return RerankerConfig(
        vocab_size=tok.vocabulary.size,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
        max_len=64,
        pad_id=tok.vocabulary.pad_id or 0,
    )


def _ids(tok: BPETokenizer) -> tuple[int, int, int]:
    v = tok.vocabulary
    return (v.pad_id or 0), v.cls_id, v.sep_id


def test_pairs_from_triples_expands() -> None:
    pairs = pairs_from_triples([("q", "pos", "neg")])
    assert pairs == [("q", "pos", 1), ("q", "neg", 0)]


def test_make_pair_samples() -> None:
    tok = _tok()
    samples = make_pair_samples([("q", "p", 1)], tok)
    anchor, passage, label = samples[0]
    assert anchor.ids == tok.encode("q").ids
    assert passage.ids == tok.encode("p").ids
    assert label == 1


def test_collate_pairs_builds_cls_sep_structure() -> None:
    tok = _tok()
    pad_id, cls_id, sep_id = _ids(tok)
    samples = make_pair_samples([("heart", "mortality", 1)], tok)
    batch = collate_pairs(samples, pad_id=pad_id, cls_id=cls_id, sep_id=sep_id)
    assert int(batch.input_ids[0, 0]) == cls_id  # starts with [CLS]
    assert sep_id in batch.input_ids[0].tolist()
    assert set(batch.token_type_ids[0].tolist()) <= {0, 1}


def test_empty_samples_rejected() -> None:
    tok = _tok()
    pad_id, cls_id, sep_id = _ids(tok)
    with pytest.raises(ValueError):
        train_reranker(
            build_reranker(_config(tok)), [], pad_id=pad_id, cls_id=cls_id, sep_id=sep_id
        )


def test_loss_decreases_and_is_finite() -> None:
    tok = _tok()
    pad_id, cls_id, sep_id = _ids(tok)
    torch.manual_seed(0)
    model = build_reranker(_config(tok))
    samples = make_pair_samples(_EXAMPLES, tok)
    losses = train_reranker(
        model,
        samples,
        pad_id=pad_id,
        cls_id=cls_id,
        sep_id=sep_id,
        epochs=15,
        batch_size=4,
        learning_rate=5e-3,
    )
    assert all(math.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]


def test_save_load_reproduces_scores(tmp_path: Path) -> None:
    tok = _tok()
    pad_id, cls_id, sep_id = _ids(tok)
    config = _config(tok)
    torch.manual_seed(0)
    model = build_reranker(config)
    save_reranker(model, config, tmp_path / "rr", metadata={"stage": "A"})
    reloaded = load_reranker(tmp_path / "rr")

    samples = make_pair_samples([("heart failure", "beta blockers mortality", 1)], tok)
    batch = collate_pairs(samples, pad_id=pad_id, cls_id=cls_id, sep_id=sep_id)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(batch), reloaded(batch), atol=1e-6)


def test_unsupported_format_version_rejected(tmp_path: Path) -> None:
    tok = _tok()
    config = _config(tok)
    torch.manual_seed(0)
    save_reranker(build_reranker(config), config, tmp_path / "rr")
    path = tmp_path / "rr" / "config.json"
    data = json.loads(path.read_text())
    data["format_version"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_reranker(tmp_path / "rr")
