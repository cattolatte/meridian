"""Corpus statistics for the Phase 1 report (`benchmarks/corpus.md`).

Reports document count, abstract length, publication-year distribution, MeSH
domain coverage (ADR-0001), and the most common journals — computed by a single
streaming pass over the stored documents.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from meridian.corpus.mesh import DOMAINS, classify_domains
from meridian.corpus.records import Document

_OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Aggregate statistics over a corpus of documents."""

    document_count: int
    total_abstract_words: int
    year_histogram: dict[int | None, int] = field(default_factory=dict)
    domain_counts: dict[str, int] = field(default_factory=dict)
    top_journals: tuple[tuple[str, int], ...] = ()

    @property
    def mean_abstract_words(self) -> float:
        if self.document_count == 0:
            return 0.0
        return self.total_abstract_words / self.document_count


def compute_stats(documents: Iterable[Document], *, top_journals: int = 10) -> CorpusStats:
    """Compute :class:`CorpusStats` in one streaming pass over ``documents``."""
    document_count = 0
    total_abstract_words = 0
    years: Counter[int | None] = Counter()
    domains: Counter[str] = Counter()
    journals: Counter[str] = Counter()

    for document in documents:
        document_count += 1
        total_abstract_words += len(document.abstract.split())
        years[document.year] += 1
        matched = classify_domains(document.mesh_terms)
        if matched:
            for domain in matched:
                domains[domain] += 1
        else:
            domains[_OUT_OF_SCOPE] += 1
        if document.journal:
            journals[document.journal] += 1

    domain_counts = {domain: domains.get(domain, 0) for domain in DOMAINS}
    domain_counts[_OUT_OF_SCOPE] = domains.get(_OUT_OF_SCOPE, 0)

    return CorpusStats(
        document_count=document_count,
        total_abstract_words=total_abstract_words,
        year_histogram=dict(years),
        domain_counts=domain_counts,
        top_journals=tuple(journals.most_common(top_journals)),
    )


def render_markdown(stats: CorpusStats, *, title: str = "Corpus statistics") -> str:
    """Render :class:`CorpusStats` as a Markdown section."""
    lines = [
        f"## {title}",
        "",
        f"- Documents: {stats.document_count}",
        f"- Total abstract words: {stats.total_abstract_words}",
        f"- Mean abstract words: {stats.mean_abstract_words:.1f}",
        "",
        "### Domain coverage (MeSH; a document may match multiple)",
        "",
        "| Domain | Documents |",
        "|---|---|",
    ]
    for domain, count in stats.domain_counts.items():
        lines.append(f"| {domain} | {count} |")

    lines += ["", "### Publication years", "", "| Year | Documents |", "|---|---|"]
    for year in sorted(stats.year_histogram, key=lambda value: (value is None, value)):
        label = "unknown" if year is None else str(year)
        lines.append(f"| {label} | {stats.year_histogram[year]} |")

    if stats.top_journals:
        lines += ["", "### Top journals", "", "| Journal | Documents |", "|---|---|"]
        lines += [f"| {journal} | {count} |" for journal, count in stats.top_journals]

    return "\n".join(lines) + "\n"
