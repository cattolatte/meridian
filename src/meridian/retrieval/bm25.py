"""From-scratch BM25 Okapi retrieval.

An inverted index over analyzed terms, scored with the standard BM25 Okapi formula.
IDF and postings depend only on the corpus, so ``k1``/``b`` are search-time
parameters and can be tuned on the dev split without rebuilding the index. Ranking
is deterministic: ties break by ascending document id.

This is the honest lexical baseline (RAG.md §7); no external search library is used.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass(frozen=True, slots=True)
class BM25Index:
    """A built BM25 index. Construct with :meth:`build`, query with :meth:`search`."""

    doc_ids: tuple[str, ...]
    doc_len: tuple[int, ...]
    avgdl: float
    postings: dict[str, list[tuple[int, int]]]
    idf: dict[str, float]
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    @classmethod
    def build(
        cls,
        corpus: Iterable[tuple[str, Sequence[str]]],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> BM25Index:
        """Build an index from ``(doc_id, analyzed_terms)`` pairs."""
        doc_ids: list[str] = []
        doc_len: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_freq: dict[str, int] = defaultdict(int)

        for doc_id, terms in corpus:
            index = len(doc_ids)
            doc_ids.append(doc_id)
            counts = Counter(terms)
            doc_len.append(sum(counts.values()))
            for term, term_freq in counts.items():
                postings[term].append((index, term_freq))
                doc_freq[term] += 1

        n_docs = len(doc_ids)
        avgdl = sum(doc_len) / n_docs if n_docs else 0.0
        idf = {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }
        return cls(
            doc_ids=tuple(doc_ids),
            doc_len=tuple(doc_len),
            avgdl=avgdl,
            postings=dict(postings),
            idf=idf,
            k1=k1,
            b=b,
        )

    def __len__(self) -> int:
        return len(self.doc_ids)

    def search(
        self,
        query_terms: Sequence[str],
        *,
        k: int = 10,
        k1: float | None = None,
        b: float | None = None,
    ) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(doc_id, score)`` pairs for the query terms.

        ``k1``/``b`` override the index defaults for this query (dev-split tuning).
        Ties break by ascending document id for deterministic output.
        """
        k1 = self.k1 if k1 is None else k1
        b = self.b if b is None else b

        scores: dict[int, float] = defaultdict(float)
        for term in set(query_terms):
            postings = self.postings.get(term)
            if postings is None:
                continue
            idf = self.idf[term]
            for index, term_freq in postings:
                length = self.doc_len[index]
                denominator = term_freq + k1 * (1.0 - b + b * length / self.avgdl)
                scores[index] += idf * term_freq * (k1 + 1.0) / denominator

        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.doc_ids[item[0]]))
        return [(self.doc_ids[index], score) for index, score in ranked[:k]]
