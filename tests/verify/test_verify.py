"""Tests for the NLI verifier, faithfulness metrics, and the fail-safe ladder."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from polaris.tokenizers import BPETokenizer

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.generation.answerer import GroundedAnswer
from meridian.tokenization.training import train_tokenizer
from meridian.verify.artifact import NLIConfig, build_verifier, load_verifier, save_verifier
from meridian.verify.data import NLILabel, make_nli_samples
from meridian.verify.training import train_verifier
from meridian.verify.verifier import verify_grounded_answer

_BIO = ["metformin lowers mortality", "beta blockers heart failure"] * 4
_GENERAL = ["a general english passage here"] * 4
_DOCS = [
    Document(pmid="10", title="Metformin", abstract="Metformin lowers cardiovascular mortality.")
]


def _tok() -> BPETokenizer:
    return train_tokenizer(_BIO, _GENERAL, vocab_size=200)


def _config(tok: BPETokenizer) -> NLIConfig:
    return NLIConfig(
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


def test_nli_labels() -> None:
    assert (NLILabel.ENTAILMENT, NLILabel.NEUTRAL, NLILabel.CONTRADICTION) == (0, 1, 2)


def test_training_loss_decreases() -> None:
    tok = _tok()
    pad, cls, sep = _ids(tok)
    torch.manual_seed(0)
    model = build_verifier(_config(tok))
    examples = [
        ("metformin lowers mortality", "metformin reduces mortality", int(NLILabel.ENTAILMENT)),
        ("metformin lowers mortality", "metformin raises mortality", int(NLILabel.CONTRADICTION)),
        ("beta blockers help", "melanoma immunotherapy works", int(NLILabel.NEUTRAL)),
    ] * 4
    losses = train_verifier(
        model,
        make_nli_samples(examples, tok),
        pad_id=pad,
        cls_id=cls,
        sep_id=sep,
        epochs=15,
        batch_size=4,
        learning_rate=5e-3,
    )
    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < losses[0]


def test_training_accepts_class_weights() -> None:
    tok = _tok()
    pad, cls, sep = _ids(tok)
    torch.manual_seed(0)
    model = build_verifier(_config(tok))
    examples = [
        ("metformin lowers mortality", "metformin reduces mortality", int(NLILabel.ENTAILMENT)),
        ("metformin lowers mortality", "metformin raises mortality", int(NLILabel.CONTRADICTION)),
        ("beta blockers help", "melanoma immunotherapy works", int(NLILabel.NEUTRAL)),
    ] * 4
    losses = train_verifier(
        model,
        make_nli_samples(examples, tok),
        pad_id=pad,
        cls_id=cls,
        sep_id=sep,
        epochs=10,
        batch_size=4,
        learning_rate=5e-3,
        class_weights=[1.0, 1.0, 3.0],  # up-weight the minority (maybe/neutral) class
    )
    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < losses[0]


def test_training_epoch_callback_runs_each_epoch() -> None:
    tok = _tok()
    pad, cls, sep = _ids(tok)
    torch.manual_seed(0)
    model = build_verifier(_config(tok))
    seen: list[int] = []
    train_verifier(
        model,
        make_nli_samples([("a b", "c d", 0), ("e f", "g h", 2)] * 2, tok),
        pad_id=pad,
        cls_id=cls,
        sep_id=sep,
        epochs=3,
        epoch_callback=lambda epoch, _model: seen.append(epoch),
    )
    assert seen == [0, 1, 2]


def test_artifact_roundtrip(tmp_path: Path) -> None:
    tok = _tok()
    config = _config(tok)
    torch.manual_seed(0)
    model = build_verifier(config)
    save_verifier(model, config, tmp_path / "nli")
    reloaded = load_verifier(tmp_path / "nli")
    pad, cls, sep = _ids(tok)
    from polaris.collation import collate_pairs

    batch = collate_pairs(
        make_nli_samples([("a b c", "d e f", 0)], tok), pad_id=pad, cls_id=cls, sep_id=sep
    )
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(batch), reloaded(batch), atol=1e-6)


def test_unsupported_format_version(tmp_path: Path) -> None:
    tok = _tok()
    config = _config(tok)
    torch.manual_seed(0)
    save_verifier(build_verifier(config), config, tmp_path / "nli")
    path = tmp_path / "nli" / "config.json"
    data = json.loads(path.read_text())
    data["format_version"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_verifier(tmp_path / "nli")


def test_verify_report_metrics() -> None:
    tok = _tok()
    torch.manual_seed(0)
    model = build_verifier(_config(tok))
    # Two content sentences: one cited, one not.
    answer = GroundedAnswer(
        query="q",
        abstained=False,
        text="Metformin lowers mortality [1]. This is an uncited claim.",
        citations=((1, "10", "Metformin"),),
        passages=(("10", "Metformin"),),
    )
    with SqliteDocumentStore(":memory:") as store:
        store.add_many(_DOCS)
        report = verify_grounded_answer(answer, store, model, tok)
    assert len(report.verdicts) == 2
    assert report.citation_recall == 0.5  # one of two sentences cited
    # metrics are in [0, 1]
    for value in (report.citation_precision, report.citation_recall, report.hallucination_rate):
        assert 0.0 <= value <= 1.0


def test_grounded_requires_all_entailed() -> None:
    tok = _tok()
    torch.manual_seed(0)
    model = build_verifier(_config(tok))
    # An uncited sentence can never be "grounded".
    answer = GroundedAnswer(
        query="q",
        abstained=False,
        text="An uncited claim with no citation.",
        citations=(),
        passages=(("10", "Metformin"),),
    )
    with SqliteDocumentStore(":memory:") as store:
        store.add_many(_DOCS)
        report = verify_grounded_answer(answer, store, model, tok)
    assert report.grounded is False  # no citation -> not grounded
