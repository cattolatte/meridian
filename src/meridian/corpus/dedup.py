"""De-duplicate parsed documents.

Two kinds of duplicates appear in PubMed baseline data:

1. **Exact PMID duplicates** — the same record occurring twice (e.g. across
   overlapping baseline files or update files). The first occurrence wins.
2. **Near-duplicate titles** — distinct PMIDs for what is effectively the same
   item (errata, reprints, versioned records). These are detected by a normalized
   title fingerprint (lowercased, alphanumeric-only, whitespace-collapsed); the
   first PMID seen for a fingerprint wins.

De-duplication is streaming and order-preserving, so a deterministic input order
yields a deterministic surviving set.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from meridian.corpus.records import Document

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def title_fingerprint(title: str) -> str:
    """Return a normalized fingerprint for near-duplicate title detection."""
    return _NON_ALNUM.sub(" ", title.lower()).strip()


def deduplicate(documents: Iterable[Document]) -> Iterator[Document]:
    """Yield documents with exact-PMID and near-duplicate-title duplicates removed.

    The first occurrence of each PMID and of each title fingerprint is kept.
    """
    seen_pmids: set[str] = set()
    seen_titles: set[str] = set()
    for document in documents:
        if document.pmid in seen_pmids:
            continue
        fingerprint = title_fingerprint(document.title)
        if fingerprint in seen_titles:
            continue
        seen_pmids.add(document.pmid)
        seen_titles.add(fingerprint)
        yield document
