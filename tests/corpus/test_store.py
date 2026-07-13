"""Tests for the SQLite document store and its repository contract."""

from pathlib import Path

from meridian.corpus.records import Document
from meridian.corpus.store import DocumentStore, SqliteDocumentStore

_DOCS = [
    Document(
        pmid="18784090",
        title="10-year follow-up of intensive glucose control",
        abstract="Metformin reduced myocardial infarction and all-cause mortality.",
        year=2008,
        journal="NEJM",
        mesh_terms=("Diabetes Mellitus, Type 2", "Myocardial Infarction"),
    ),
    Document(
        pmid="30862451",
        title="Comparative effectiveness of glucose-lowering drugs",
        abstract="A meta-analysis of cardiovascular outcomes.",
        year=2019,
        journal="Lancet",
        mesh_terms=("Cardiovascular Diseases",),
    ),
]


def test_sqlite_store_satisfies_protocol() -> None:
    with SqliteDocumentStore(":memory:") as store:
        assert isinstance(store, DocumentStore)


def test_add_many_and_roundtrip() -> None:
    with SqliteDocumentStore(":memory:") as store:
        written = store.add_many(_DOCS)
        assert written == 2
        assert store.count() == 2
        fetched = store.get("18784090")
        assert fetched == _DOCS[0]


def test_get_missing_returns_none() -> None:
    with SqliteDocumentStore(":memory:") as store:
        assert store.get("does-not-exist") is None


def test_contains() -> None:
    with SqliteDocumentStore(":memory:") as store:
        store.add(_DOCS[0])
        assert store.contains("18784090") is True
        assert store.contains("nope") is False


def test_iter_documents_is_pmid_ordered() -> None:
    with SqliteDocumentStore(":memory:") as store:
        store.add_many(reversed(_DOCS))
        pmids = [doc.pmid for doc in store.iter_documents()]
        assert pmids == sorted(pmids)


def test_upsert_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite"
    with SqliteDocumentStore(db) as store:
        store.add_many(_DOCS)
        store.add_many(_DOCS)  # rebuild over the same records
        assert store.count() == 2


def test_upsert_replaces_existing() -> None:
    with SqliteDocumentStore(":memory:") as store:
        store.add(_DOCS[0])
        updated = Document(pmid="18784090", title="revised", abstract="new text", year=2009)
        store.add(updated)
        assert store.count() == 1
        assert store.get("18784090") == updated


def test_persists_across_connections(tmp_path: Path) -> None:
    db = tmp_path / "corpus.sqlite"
    with SqliteDocumentStore(db) as store:
        store.add_many(_DOCS)
    with SqliteDocumentStore(db) as reopened:
        assert reopened.count() == 2
        assert reopened.get("30862451") == _DOCS[1]
