"""Document store: a thin repository interface over SQLite.

The :class:`DocumentStore` protocol is the storage contract the rest of the
pipeline depends on; :class:`SqliteDocumentStore` is the Phase 1 implementation.
Keeping callers behind the protocol makes the Phase 10 Postgres swap a single-file
change (RAG.md §5).

Writes are upserts keyed by PMID, so re-ingesting the same raw files is idempotent
and a rebuild is deterministic.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType
from typing import Protocol, runtime_checkable

from meridian.corpus.records import Document

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    pmid       TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    abstract   TEXT NOT NULL,
    year       INTEGER,
    journal    TEXT,
    mesh_terms TEXT NOT NULL
);
"""


@runtime_checkable
class DocumentStore(Protocol):
    """Storage contract for normalized corpus records."""

    def add_many(self, documents: Iterable[Document]) -> int:
        """Insert or replace documents; return the number written."""
        ...

    def get(self, pmid: str) -> Document | None:
        """Return the document with this PMID, or ``None`` if absent."""
        ...

    def contains(self, pmid: str) -> bool:
        """Return whether a document with this PMID exists."""
        ...

    def count(self) -> int:
        """Return the number of stored documents."""
        ...

    def iter_documents(self) -> Iterator[Document]:
        """Iterate stored documents in PMID order (streaming)."""
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        pmid=row["pmid"],
        title=row["title"],
        abstract=row["abstract"],
        year=row["year"],
        journal=row["journal"],
        mesh_terms=tuple(json.loads(row["mesh_terms"])),
    )


class SqliteDocumentStore:
    """SQLite-backed :class:`DocumentStore`.

    Use as a context manager, or call :meth:`close` explicitly. Pass
    ``":memory:"`` for an ephemeral in-process database (used in tests).
    """

    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def add(self, document: Document) -> None:
        """Insert or replace a single document."""
        self.add_many((document,))

    def add_many(self, documents: Iterable[Document]) -> int:
        """Insert or replace documents; return the number written."""
        rows = (
            (
                doc.pmid,
                doc.title,
                doc.abstract,
                doc.year,
                doc.journal,
                json.dumps(list(doc.mesh_terms)),
            )
            for doc in documents
        )
        cursor = self._conn.executemany(
            "INSERT OR REPLACE INTO documents "
            "(pmid, title, abstract, year, journal, mesh_terms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return cursor.rowcount

    def get(self, pmid: str) -> Document | None:
        """Return the document with this PMID, or ``None`` if absent."""
        row = self._conn.execute("SELECT * FROM documents WHERE pmid = ?", (pmid,)).fetchone()
        return _row_to_document(row) if row is not None else None

    def contains(self, pmid: str) -> bool:
        """Return whether a document with this PMID exists."""
        row = self._conn.execute("SELECT 1 FROM documents WHERE pmid = ?", (pmid,)).fetchone()
        return row is not None

    def count(self) -> int:
        """Return the number of stored documents."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return int(row["n"])

    def iter_documents(self) -> Iterator[Document]:
        """Iterate stored documents in PMID order (streaming cursor)."""
        cursor = self._conn.execute("SELECT * FROM documents ORDER BY pmid")
        for row in cursor:
            yield _row_to_document(row)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> SqliteDocumentStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
