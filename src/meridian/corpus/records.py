"""The normalized corpus record.

A :class:`Document` is one PubMed abstract after parsing and cleaning. Per
:doc:`ADR-0002 </adr/0002-chunking>` there is exactly one retrievable chunk per
document, composed by :meth:`Document.chunk_text`, and the PMID is the citation
unit end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Separates title from abstract in the composed chunk (ADR-0002).
CHUNK_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class Document:
    """A single normalized PubMed record.

    Attributes
    ----------
    pmid:
        PubMed identifier; the primary key and the citation unit.
    title:
        Article title (cleaned, whitespace-normalized).
    abstract:
        Abstract text; for structured abstracts, sections are joined into one
        string during parsing.
    year:
        Publication year, or ``None`` if the source record omitted it.
    journal:
        Journal title, or ``None`` if absent.
    mesh_terms:
        MeSH descriptor names attached to the record, in source order.
    """

    pmid: str
    title: str
    abstract: str
    year: int | None = None
    journal: str | None = None
    mesh_terms: tuple[str, ...] = field(default_factory=tuple)

    def chunk_text(self) -> str:
        """Return the single indexable chunk for this document (ADR-0002).

        The title is prepended to the abstract because it carries high-signal
        terms (drug names, conditions) that improve both lexical and dense
        retrieval.
        """
        return f"{self.title}{CHUNK_SEPARATOR}{self.abstract}"
