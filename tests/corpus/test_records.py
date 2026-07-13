"""Tests for the normalized Document record and its ADR-0002 chunk composition."""

from meridian.corpus.records import CHUNK_SEPARATOR, Document


def test_chunk_text_prepends_title() -> None:
    doc = Document(pmid="1", title="Metformin and the heart", abstract="It helps.")
    assert doc.chunk_text() == f"Metformin and the heart{CHUNK_SEPARATOR}It helps."


def test_defaults_are_empty() -> None:
    doc = Document(pmid="2", title="T", abstract="A")
    assert doc.year is None
    assert doc.journal is None
    assert doc.mesh_terms == ()


def test_document_is_frozen_and_hashable() -> None:
    doc = Document(pmid="3", title="T", abstract="A", mesh_terms=("Neoplasms",))
    assert doc in {doc}  # hashable
    # frozen: attribute assignment is rejected
    try:
        doc.title = "changed"  # type: ignore[misc]
    except AttributeError:
        pass
    else:  # pragma: no cover - only runs if frozen enforcement breaks
        raise AssertionError("Document should be immutable")
