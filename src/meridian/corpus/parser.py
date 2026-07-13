"""Parse PubMed baseline XML into normalized :class:`Document` records.

The PubMed baseline ships as large ``.xml`` / ``.xml.gz`` files, each a
``<PubmedArticleSet>`` of ``<PubmedArticle>`` elements. Parsing streams over the
file with :func:`xml.etree.ElementTree.iterparse`, clearing each article element
after use so a multi-gigabyte file never loads into memory at once.

This module performs only *record-level* normalization and language/empty
filtering. Domain filtering (:mod:`meridian.corpus.mesh`) and de-duplication
(:mod:`meridian.corpus.dedup`) are applied downstream by the ingest pipeline, so
each concern is independently testable.

A record is yielded only if it has a PMID, a non-empty title, and a non-empty
abstract, and is not explicitly tagged as non-English.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from pathlib import Path
from typing import IO, cast
from xml.etree.ElementTree import Element, iterparse

from meridian.corpus.records import Document

_WHITESPACE = re.compile(r"\s+")
_YEAR = re.compile(r"\b(\d{4})\b")


def _clean(text: str | None) -> str:
    """Collapse whitespace and strip; ``None`` becomes the empty string."""
    if not text:
        return ""
    return _WHITESPACE.sub(" ", text).strip()


def _element_text(element: Element | None) -> str:
    """Return all descendant text of an element (handles inline markup)."""
    if element is None:
        return ""
    return _clean("".join(element.itertext()))


def _extract_abstract(article: Element) -> str:
    """Join all ``<AbstractText>`` sections of a (possibly structured) abstract."""
    sections = [_element_text(node) for node in article.iterfind(".//Abstract/AbstractText")]
    return _clean(" ".join(section for section in sections if section))


def _extract_year(article: Element) -> int | None:
    """Read the publication year from ``<Year>`` or fall back to ``<MedlineDate>``."""
    year_node = article.find(".//Journal/JournalIssue/PubDate/Year")
    if year_node is not None and year_node.text and year_node.text.strip().isdigit():
        return int(year_node.text.strip())
    medline = article.find(".//Journal/JournalIssue/PubDate/MedlineDate")
    if medline is not None and medline.text:
        match = _YEAR.search(medline.text)
        if match is not None:
            return int(match.group(1))
    return None


def _is_english(article: Element) -> bool:
    """Return ``True`` unless the record is explicitly tagged as non-English.

    Records with no ``<Language>`` element are kept (English is the overwhelming
    default and absence is not evidence of another language).
    """
    languages = [node.text.strip().lower() for node in article.iterfind(".//Language") if node.text]
    return not languages or "eng" in languages


def _extract_mesh_terms(citation: Element) -> tuple[str, ...]:
    names = (
        _element_text(node)
        for node in citation.iterfind(".//MeshHeadingList/MeshHeading/DescriptorName")
    )
    return tuple(name for name in names if name)


def _parse_article(citation: Element) -> Document | None:
    """Build a Document from one ``<MedlineCitation>``, or ``None`` if unusable."""
    pmid = _element_text(citation.find("./PMID"))
    if not pmid:
        return None
    article = citation.find("./Article")
    if article is None:
        return None
    if not _is_english(article):
        return None
    title = _element_text(article.find("./ArticleTitle"))
    abstract = _extract_abstract(article)
    if not title or not abstract:
        return None
    journal = _element_text(article.find(".//Journal/Title")) or None
    return Document(
        pmid=pmid,
        title=title,
        abstract=abstract,
        year=_extract_year(article),
        journal=journal,
        mesh_terms=_extract_mesh_terms(citation),
    )


def parse_articles(source: IO[bytes] | str | Path) -> Iterator[Document]:
    """Yield normalized documents from a PubMed XML byte stream or file path.

    Passing/empty records (missing PMID, title, or abstract) and explicitly
    non-English records are silently skipped.
    """
    for _event, element in iterparse(source, events=("end",)):
        if element.tag != "PubmedArticle":
            continue
        citation = element.find("./MedlineCitation")
        if citation is not None:
            document = _parse_article(citation)
            if document is not None:
                yield document
        element.clear()


def iter_documents(path: str | Path) -> Iterator[Document]:
    """Yield documents from a ``.xml`` or ``.xml.gz`` PubMed baseline file."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            yield from parse_articles(cast("IO[bytes]", handle))
    else:
        with path.open("rb") as handle:
            yield from parse_articles(handle)
