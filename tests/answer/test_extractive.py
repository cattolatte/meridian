"""Tests for the extractive answerer v0."""

from __future__ import annotations

from meridian.answer.extractive import ExtractiveAnswer, answer_extractive, render_answer
from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.retrieval.pipeline import BM25Retriever

_DOCS = [
    Document(
        pmid="1",
        title="Beta-blockers after myocardial infarction",
        abstract=(
            "Beta-blockers are prescribed after myocardial infarction. "
            "Treatment reduced all-cause mortality over five years."
        ),
    ),
    Document(
        pmid="2",
        title="Metformin and cardiovascular outcomes",
        abstract="Metformin lowered cardiovascular mortality compared with sulfonylureas.",
    ),
    Document(
        pmid="3",
        title="Checkpoint inhibitors in melanoma",
        abstract="Checkpoint blockade produced durable responses in metastatic melanoma.",
    ),
]


def _retriever() -> BM25Retriever:
    store = SqliteDocumentStore(":memory:")
    store.add_many(_DOCS)
    return BM25Retriever.from_store(store)


def test_answer_cites_relevant_sentence_verbatim() -> None:
    answer = answer_extractive(_retriever(), "beta-blockers mortality myocardial infarction")
    assert not answer.abstained
    top = answer.sentences[0]
    assert top.pmid == "1"
    # Returned verbatim: the sentence is a substring of the source abstract.
    assert top.text in _DOCS[0].abstract
    assert ("1", "Beta-blockers after myocardial infarction") in answer.sources


def test_abstains_when_no_content_overlap() -> None:
    answer = answer_extractive(_retriever(), "quantum chromodynamics lattice gauge theory")
    assert answer.abstained
    assert answer.sentences == ()


def test_render_abstain_shows_nearest_and_banner() -> None:
    # Render an abstention that retrieved nearby passages (nearest populated).
    answer = ExtractiveAnswer(
        query="q",
        abstained=True,
        sentences=(),
        sources=(),
        nearest=(("1", "Beta-blockers after myocardial infarction"),),
    )
    text = render_answer(answer)
    assert "ABSTAIN" in text
    assert "Nearest passages:" in text
    assert "PMID 1" in text
    assert "Not medical advice." in text


def test_max_sentences_is_respected() -> None:
    answer = answer_extractive(_retriever(), "mortality cardiovascular melanoma", max_sentences=2)
    assert len(answer.sentences) <= 2


def test_render_grounded_answer_has_citations_and_banner() -> None:
    answer = answer_extractive(_retriever(), "beta-blockers mortality")
    text = render_answer(answer)
    assert "[1]" in text
    assert "Sources:" in text
    assert "GROUNDED" in text
    assert "Not medical advice." in text


def test_determinism() -> None:
    a = answer_extractive(_retriever(), "cardiovascular mortality")
    b = answer_extractive(_retriever(), "cardiovascular mortality")
    assert a == b
