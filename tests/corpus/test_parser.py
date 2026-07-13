"""Parser edge-case tests (Phase 1 exit criterion) using synthetic PubMed XML."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from textwrap import dedent

from meridian.corpus.parser import iter_documents, parse_articles
from meridian.corpus.records import Document


def _articleset(*articles: str) -> bytes:
    body = "\n".join(articles)
    return f"<PubmedArticleSet>\n{body}\n</PubmedArticleSet>".encode()


def _article(
    *,
    pmid: str = "1",
    title: str = "A cardiology study",
    abstract: str = "<Abstract><AbstractText>Findings here.</AbstractText></Abstract>",
    language: str | None = "eng",
    pubdate: str = "<PubDate><Year>2011</Year></PubDate>",
    journal_title: str = "<Title>Test Journal</Title>",
    mesh: str = "",
    include_article: bool = True,
) -> str:
    lang = f"<Language>{language}</Language>" if language is not None else ""
    article_block = (
        f"""
        <Article>
          <Journal>
            <JournalIssue>{pubdate}</JournalIssue>
            {journal_title}
          </Journal>
          <ArticleTitle>{title}</ArticleTitle>
          {abstract}
          {lang}
        </Article>
        """
        if include_article
        else ""
    )
    return dedent(f"""
        <PubmedArticle>
          <MedlineCitation>
            <PMID Version="1">{pmid}</PMID>
            {article_block}
            {mesh}
          </MedlineCitation>
        </PubmedArticle>
        """)


def _parse(xml: bytes) -> list[Document]:
    return list(parse_articles(io.BytesIO(xml)))


def test_valid_record_is_parsed() -> None:
    docs = _parse(_articleset(_article()))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.pmid == "1"
    assert doc.title == "A cardiology study"
    assert doc.abstract == "Findings here."
    assert doc.year == 2011
    assert doc.journal == "Test Journal"


def test_structured_abstract_sections_are_joined() -> None:
    abstract = (
        "<Abstract>"
        '<AbstractText Label="BACKGROUND">Metformin is common.</AbstractText>'
        '<AbstractText Label="RESULTS">It lowered mortality.</AbstractText>'
        "</Abstract>"
    )
    (doc,) = _parse(_articleset(_article(abstract=abstract)))
    assert doc.abstract == "Metformin is common. It lowered mortality."


def test_inline_markup_text_is_extracted() -> None:
    title = "Effect of <i>metformin</i> on HbA<sub>1c</sub>"
    (doc,) = _parse(_articleset(_article(title=title)))
    assert doc.title == "Effect of metformin on HbA1c"


def test_missing_abstract_is_skipped() -> None:
    assert _parse(_articleset(_article(abstract=""))) == []


def test_empty_whitespace_abstract_is_skipped() -> None:
    abstract = "<Abstract><AbstractText>   </AbstractText></Abstract>"
    assert _parse(_articleset(_article(abstract=abstract))) == []


def test_non_english_is_skipped() -> None:
    assert _parse(_articleset(_article(language="fre"))) == []


def test_missing_language_is_kept() -> None:
    (doc,) = _parse(_articleset(_article(language=None)))
    assert doc.pmid == "1"


def test_missing_pmid_is_skipped() -> None:
    assert _parse(_articleset(_article(pmid=""))) == []


def test_missing_article_element_is_skipped() -> None:
    assert _parse(_articleset(_article(include_article=False))) == []


def test_missing_title_is_skipped() -> None:
    assert _parse(_articleset(_article(title=""))) == []


def test_medline_date_year_fallback() -> None:
    (doc,) = _parse(
        _articleset(_article(pubdate="<PubDate><MedlineDate>2008 Jan-Feb</MedlineDate></PubDate>"))
    )
    assert doc.year == 2008


def test_missing_year_is_none() -> None:
    (doc,) = _parse(_articleset(_article(pubdate="<PubDate></PubDate>")))
    assert doc.year is None


def test_mesh_terms_extracted_in_order() -> None:
    def heading(name: str) -> str:
        return f"<MeshHeading><DescriptorName>{name}</DescriptorName></MeshHeading>"

    mesh = (
        "<MeshHeadingList>"
        + heading("Diabetes Mellitus, Type 2")
        + heading("Myocardial Infarction")
        + "</MeshHeadingList>"
    )
    (doc,) = _parse(_articleset(_article(mesh=mesh)))
    assert doc.mesh_terms == ("Diabetes Mellitus, Type 2", "Myocardial Infarction")


def test_missing_mesh_is_empty_tuple() -> None:
    (doc,) = _parse(_articleset(_article()))
    assert doc.mesh_terms == ()


def test_multiple_records_mixed_validity() -> None:
    xml = _articleset(
        _article(pmid="1"),
        _article(pmid="2", abstract=""),  # dropped
        _article(pmid="3", language="ger"),  # dropped
        _article(pmid="4"),
    )
    assert [d.pmid for d in _parse(xml)] == ["1", "4"]


def test_iter_documents_reads_plain_xml(tmp_path: Path) -> None:
    path = tmp_path / "sample.xml"
    path.write_bytes(_articleset(_article()))
    assert [d.pmid for d in iter_documents(path)] == ["1"]


def test_iter_documents_reads_gzip(tmp_path: Path) -> None:
    path = tmp_path / "sample.xml.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(_articleset(_article(pmid="42")))
    assert [d.pmid for d in iter_documents(path)] == ["42"]
