"""Tests for the grounded generator: artifact, data, and constrained answering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from zenith.tokenizers.byte_tokenizer import ByteTokenizer

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.generation.answerer import (
    GroundedAnswer,
    answer_grounded,
    render_grounded_answer,
)
from meridian.generation.artifact import (
    GeneratorConfig,
    build_generator,
    load_generator,
    save_generator,
)
from meridian.generation.data import grounded_example, passages_from_hits
from meridian.retrieval.pipeline import BM25Retriever

_DOCS = [
    Document(pmid="10", title="Metformin", abstract="Metformin lowers cardiovascular mortality."),
    Document(pmid="20", title="Melanoma", abstract="Checkpoint immunotherapy responses."),
]


def _store() -> SqliteDocumentStore:
    store = SqliteDocumentStore(":memory:")
    store.add_many(_DOCS)
    return store


def _config() -> GeneratorConfig:
    # Tiny, no LoRA (simplest artifact round-trip).
    return GeneratorConfig(
        block_size=128, embed_dim=32, num_layers=1, num_heads=2, ff_dim=64, use_lora=False
    )


def test_passages_from_hits() -> None:
    with _store() as store:
        hits = BM25Retriever.from_store(store).retrieve("metformin mortality", k=2)
        passages = passages_from_hits(hits)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in passages)


def test_grounded_example_abstain() -> None:
    question, passages, answer = grounded_example("q", [], None)
    assert question == "q"
    assert answer is None
    assert passages == []


def test_artifact_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = _config()
    model = build_generator(config)
    save_generator(model, config, tmp_path / "gen", metadata={"demo": True})
    reloaded = load_generator(tmp_path / "gen")
    ids = torch.tensor([[1, 2, 3, 4]])
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(ids), reloaded(ids), atol=1e-6)


def test_unsupported_format_version(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = _config()
    save_generator(build_generator(config), config, tmp_path / "gen")
    path = tmp_path / "gen" / "config.json"
    data = json.loads(path.read_text())
    data["format_version"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_generator(tmp_path / "gen")


def test_answer_grounded_produces_valid_citations() -> None:
    # With a random tiny model the prose is gibberish, but citations are constrained
    # to valid passage indices and parsed back to real PMIDs (the mechanism under test).
    torch.manual_seed(0)
    model = build_generator(_config())
    tok = ByteTokenizer()
    with _store() as store:
        retriever = BM25Retriever.from_store(store)
        answer = answer_grounded(
            retriever, model, tok, "metformin mortality", k=2, max_new_tokens=40
        )
        assert isinstance(answer, GroundedAnswer)
        valid_pmids = {pmid for pmid, _ in answer.passages}
        for _n, pmid, _title in answer.citations:
            assert pmid in valid_pmids  # never cites a passage not retrieved


def test_answer_grounded_abstains_when_no_passages() -> None:
    torch.manual_seed(0)
    model = build_generator(_config())
    with _store() as store:
        # A query with no lexical overlap -> BM25 returns nothing -> abstain.
        answer = answer_grounded(
            BM25Retriever.from_store(store), model, ByteTokenizer(), "xyzzy quux", k=2
        )
        assert answer.abstained


def test_render_generated_and_abstain() -> None:
    generated = GroundedAnswer(
        query="q",
        abstained=False,
        text="Metformin lowers mortality [1].",
        citations=((1, "10", "Metformin"),),
        passages=(("10", "Metformin"),),
    )
    text = render_grounded_answer(generated)
    assert "[1]" in text and "GENERATED" in text and "Not medical advice." in text

    abstain = GroundedAnswer("q", True, "", (), (("10", "Metformin"),))
    out = render_grounded_answer(abstain)
    assert "ABSTAIN" in out and "Not medical advice." in out
