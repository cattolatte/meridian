"""Corpus ingestion: download, parse, deduplicate, and store PubMed records.

This package implements the Phase 1 offline pipeline (RAG.md §4.1): raw PubMed
baseline XML is parsed into normalized :class:`~meridian.corpus.records.Document`
records, filtered to the ADR-0001 domains, deduplicated, and written to a
:class:`~meridian.corpus.store.DocumentStore`.
"""

from meridian.corpus.records import Document

__all__ = ["Document"]
