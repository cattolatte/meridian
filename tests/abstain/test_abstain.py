"""Tests for Gate 1 (retrieval), Gate 2 (answerability), and risk-coverage."""

from __future__ import annotations

import math

import pytest
import torch
from polaris.tokenizers import BPETokenizer

from meridian.abstain.answerability import (
    AnswerabilityConfig,
    answerable_probability,
    build_answerability,
    make_answerability_samples,
    train_answerability,
)
from meridian.abstain.calibration import operating_point, risk_coverage_curve
from meridian.abstain.gate import RetrievalGate, retrieval_confidence
from meridian.retrieval.pipeline import RetrievalHit
from meridian.tokenization.training import train_tokenizer

_BIO = ["metformin mortality", "melanoma immunotherapy"] * 4
_GENERAL = ["a general english passage here"] * 4


def _hits(scores: list[float]) -> list[RetrievalHit]:
    return [RetrievalHit(pmid=str(i), score=s) for i, s in enumerate(scores)]


# ---- Gate 1: retrieval confidence ------------------------------------------------


def test_confidence_top_and_margin() -> None:
    confidence = retrieval_confidence(_hits([5.0, 3.0, 1.0]), margin_k=3)
    assert confidence.top_score == 5.0
    assert confidence.margin == 4.0  # 5 - 1


def test_confidence_empty_hits() -> None:
    c = retrieval_confidence([])
    assert c.top_score == 0.0 and c.margin == 0.0


def test_gate_answerable_requires_both_thresholds() -> None:
    gate = RetrievalGate(min_score=2.0, min_margin=1.0, margin_k=2)
    assert gate.answerable(_hits([5.0, 3.0])) is True  # score 5>=2, margin 2>=1
    assert gate.answerable(_hits([5.0, 4.5])) is False  # margin 0.5 < 1
    assert gate.answerable(_hits([1.0, 0.0])) is False  # score 1 < 2


# ---- Risk-coverage ---------------------------------------------------------------


def test_risk_coverage_orders_by_confidence() -> None:
    # Most-confident answer is wrong; least-confident is right.
    records = [(False, 0.9), (True, 0.5), (True, 0.1)]
    curve = risk_coverage_curve(records)
    assert curve[0].coverage == pytest.approx(1 / 3)
    assert curve[0].error_rate == 1.0  # first (most confident) is wrong
    assert curve[-1].coverage == 1.0
    assert curve[-1].error_rate == pytest.approx(1 / 3)


def test_operating_point_targets_coverage() -> None:
    records = [(True, 0.9), (True, 0.7), (False, 0.5), (True, 0.3)]
    point = operating_point(records, target_coverage=0.5)
    assert point.coverage == pytest.approx(0.5)


def test_risk_coverage_empty_rejected() -> None:
    with pytest.raises(ValueError):
        risk_coverage_curve([])


# ---- Gate 2: answerability -------------------------------------------------------


def _tok() -> BPETokenizer:
    return train_tokenizer(_BIO, _GENERAL, vocab_size=200)


def test_answerability_training_and_probability() -> None:
    tok = _tok()
    v = tok.vocabulary
    torch.manual_seed(0)
    model = build_answerability(
        AnswerabilityConfig(
            vocab_size=v.size,
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            ff_dim=32,
            max_len=64,
            pad_id=v.pad_id or 0,
        )
    )
    examples = [
        ("does metformin reduce mortality", ["metformin lowers mortality"], 1),
        ("what is the best treatment for my chest pain", ["melanoma immunotherapy"], 0),
    ] * 4
    losses = train_answerability(
        model,
        make_answerability_samples(examples, tok),
        pad_id=v.pad_id or 0,
        cls_id=v.cls_id,
        sep_id=v.sep_id,
        epochs=15,
        batch_size=4,
        learning_rate=5e-3,
    )
    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < losses[0]
    p = answerable_probability(
        model, tok, "does metformin reduce mortality", ["metformin lowers mortality"]
    )
    assert 0.0 <= p <= 1.0


def test_empty_answerability_rejected() -> None:
    tok = _tok()
    v = tok.vocabulary
    model = build_answerability(
        AnswerabilityConfig(
            vocab_size=v.size,
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            ff_dim=32,
            max_len=64,
            pad_id=v.pad_id or 0,
        )
    )
    with pytest.raises(ValueError):
        train_answerability(model, [], pad_id=v.pad_id or 0, cls_id=v.cls_id, sep_id=v.sep_id)
