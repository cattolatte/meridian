"""End-to-end corpus ingestion: raw files → normalized, filtered document store.

The pipeline is a single streaming pass (RAG.md §4.1): parse each raw file,
de-duplicate globally across all files, filter to the ADR-0001 domains, and write
to the store. Re-running over the same raw files is idempotent (the store upserts by
PMID), so this is the one-command rebuild the Phase 1 exit criteria require.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from meridian.corpus.dedup import deduplicate
from meridian.corpus.mesh import in_scope
from meridian.corpus.parser import iter_documents
from meridian.corpus.records import Document
from meridian.corpus.store import DocumentStore


@dataclass(slots=True)
class IngestSummary:
    """Counts at each pipeline stage, for logging and the ingest report."""

    parsed: int = 0
    unique: int = 0
    in_domain: int = 0
    stored: int = 0


def _parse_all(paths: Iterable[Path], summary: IngestSummary) -> Iterator[Document]:
    for path in paths:
        for document in iter_documents(path):
            summary.parsed += 1
            yield document


def _count_unique(documents: Iterable[Document], summary: IngestSummary) -> Iterator[Document]:
    for document in documents:
        summary.unique += 1
        yield document


def _filter_domains(
    documents: Iterable[Document],
    summary: IngestSummary,
    apply_filter: bool,
) -> Iterator[Document]:
    for document in documents:
        if not apply_filter or in_scope(document.mesh_terms):
            summary.in_domain += 1
            yield document


def ingest_documents(
    paths: Iterable[Path],
    store: DocumentStore,
    *,
    apply_domain_filter: bool = True,
) -> IngestSummary:
    """Ingest raw PubMed files into ``store``; return per-stage counts.

    Documents are parsed, de-duplicated across all files, optionally filtered to
    the target domains, and upserted into the store.
    """
    summary = IngestSummary()
    parsed = _parse_all(list(paths), summary)
    unique = _count_unique(deduplicate(parsed), summary)
    filtered = _filter_domains(unique, summary, apply_domain_filter)
    summary.stored = store.add_many(filtered)
    return summary
