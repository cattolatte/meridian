"""Tests for corpus statistics computation and rendering."""

from __future__ import annotations

from meridian.corpus.records import Document
from meridian.corpus.stats import compute_stats, render_markdown

_DOCS = [
    Document(
        pmid="1",
        title="t1",
        abstract="one two three",
        year=2010,
        journal="Circulation",
        mesh_terms=("Myocardial Infarction",),
    ),
    Document(
        pmid="2",
        title="t2",
        abstract="four five",
        year=2010,
        journal="Circulation",
        mesh_terms=("Diabetes Mellitus, Type 2", "Cardiomyopathies"),
    ),
    Document(
        pmid="3",
        title="t3",
        abstract="six",
        year=None,
        journal=None,
        mesh_terms=("Malaria",),
    ),
]


def test_counts_and_means() -> None:
    stats = compute_stats(_DOCS)
    assert stats.document_count == 3
    assert stats.total_abstract_words == 6
    assert stats.mean_abstract_words == 2.0


def test_mean_of_empty_corpus_is_zero() -> None:
    stats = compute_stats([])
    assert stats.mean_abstract_words == 0.0


def test_domain_counts_multi_and_out_of_scope() -> None:
    stats = compute_stats(_DOCS)
    assert stats.domain_counts["cardiology"] == 2
    assert stats.domain_counts["endocrinology"] == 1
    assert stats.domain_counts["oncology"] == 0
    assert stats.domain_counts["out_of_scope"] == 1


def test_year_histogram_includes_unknown() -> None:
    stats = compute_stats(_DOCS)
    assert stats.year_histogram == {2010: 2, None: 1}


def test_top_journals() -> None:
    stats = compute_stats(_DOCS)
    assert stats.top_journals == (("Circulation", 2),)


def test_render_markdown_contains_sections() -> None:
    md = render_markdown(compute_stats(_DOCS), title="Sample")
    assert "## Sample" in md
    assert "Documents: 3" in md
    assert "| cardiology | 2 |" in md
    assert "| unknown | 1 |" in md
    assert "| Circulation | 2 |" in md
