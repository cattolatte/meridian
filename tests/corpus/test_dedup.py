"""Tests for exact-PMID and near-duplicate-title de-duplication."""

from meridian.corpus.dedup import deduplicate, title_fingerprint
from meridian.corpus.records import Document


def _doc(pmid: str, title: str) -> Document:
    return Document(pmid=pmid, title=title, abstract="body")


def test_title_fingerprint_normalizes() -> None:
    assert title_fingerprint("Metformin & the Heart!") == title_fingerprint("metformin the heart")


def test_exact_pmid_duplicate_removed() -> None:
    docs = [_doc("1", "First"), _doc("1", "First again")]
    assert [d.pmid for d in deduplicate(docs)] == ["1"]


def test_near_duplicate_title_removed() -> None:
    docs = [_doc("1", "A Study of X"), _doc("2", "a study of x")]
    kept = list(deduplicate(docs))
    assert [d.pmid for d in kept] == ["1"]


def test_distinct_titles_are_kept() -> None:
    docs = [_doc("1", "Study A"), _doc("2", "Study B")]
    assert [d.pmid for d in deduplicate(docs)] == ["1", "2"]


def test_order_is_preserved_first_wins() -> None:
    docs = [_doc("3", "Same Title"), _doc("1", "same title"), _doc("2", "Other")]
    assert [d.pmid for d in deduplicate(docs)] == ["3", "2"]
