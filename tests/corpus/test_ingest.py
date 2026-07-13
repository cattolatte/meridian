"""End-to-end ingest pipeline tests (parse → dedup → domain filter → store)."""

from __future__ import annotations

from pathlib import Path

from meridian.corpus.ingest import ingest_documents
from meridian.corpus.store import SqliteDocumentStore


def _write_articleset(path: Path, *articles: str) -> None:
    body = "\n".join(articles)
    path.write_text(f"<PubmedArticleSet>{body}</PubmedArticleSet>")


def _article(pmid: str, title: str, mesh: str, *, language: str = "eng") -> str:
    return (
        "<PubmedArticle><MedlineCitation>"
        f'<PMID Version="1">{pmid}</PMID>'
        "<Article><Journal><JournalIssue><PubDate><Year>2015</Year></PubDate>"
        "</JournalIssue><Title>J</Title></Journal>"
        f"<ArticleTitle>{title}</ArticleTitle>"
        "<Abstract><AbstractText>Some findings.</AbstractText></Abstract>"
        f"<Language>{language}</Language></Article>"
        f"<MeshHeadingList><MeshHeading><DescriptorName>{mesh}</DescriptorName>"
        "</MeshHeading></MeshHeadingList>"
        "</MedlineCitation></PubmedArticle>"
    )


def test_ingest_filters_dedups_and_stores(tmp_path: Path) -> None:
    raw = tmp_path / "sample.xml"
    _write_articleset(
        raw,
        _article("1", "Cardiac study", "Myocardial Infarction"),
        _article("2", "Cancer study", "Breast Neoplasms"),
        _article("3", "Off topic", "Malaria"),  # out of domain -> filtered
        _article("1", "Cardiac study dup", "Myocardial Infarction"),  # dup PMID
    )
    with SqliteDocumentStore(":memory:") as store:
        summary = ingest_documents([raw], store)
        assert summary.parsed == 4
        assert summary.unique == 3  # PMID-1 duplicate removed
        assert summary.in_domain == 2  # Malaria filtered
        assert summary.stored == 2
        assert store.count() == 2
        assert store.contains("1") and store.contains("2")
        assert not store.contains("3")


def test_ingest_without_domain_filter_keeps_all_unique(tmp_path: Path) -> None:
    raw = tmp_path / "sample.xml"
    _write_articleset(
        raw,
        _article("1", "Cardiac", "Myocardial Infarction"),
        _article("3", "Off topic", "Malaria"),
    )
    with SqliteDocumentStore(":memory:") as store:
        summary = ingest_documents([raw], store, apply_domain_filter=False)
        assert summary.in_domain == 2
        assert store.count() == 2


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    raw = tmp_path / "sample.xml"
    _write_articleset(raw, _article("1", "Cardiac", "Myocardial Infarction"))
    with SqliteDocumentStore(":memory:") as store:
        ingest_documents([raw], store)
        ingest_documents([raw], store)
        assert store.count() == 1
