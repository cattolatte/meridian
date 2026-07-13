"""Extractive answerer v0 — verbatim cited sentences, or abstain.

Retrieve the top passages, split them into sentences, rank sentences by lexical
overlap with the question, and return the best ones **verbatim** with their PMID.
Because nothing is generated, the answer cannot hallucinate (RAG.md §2). When no
retrieved sentence overlaps the question, the answerer **abstains** and offers the
nearest passages instead — the abstain path is a feature, not an apology (RAG.md §3).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from meridian.retrieval.analyzer import simple_analyzer
from meridian.retrieval.pipeline import RetrievalHit, Retriever

Analyzer = Callable[[str], Sequence[str]]

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

_NOT_MEDICAL_ADVICE = "Research literature assistant. Not medical advice."

# Overlap on common function words is meaningless (every abstract contains "in",
# "the", "of"), so they are excluded when scoring sentence relevance. BM25 handles
# these via IDF; the extractive overlap needs an explicit filter.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
        "does",
        "do",
        "did",
        "not",
        "no",
        "but",
        "than",
        "these",
        "those",
        "we",
        "our",
        "their",
        "they",
        "there",
        "may",
        "can",
    }
)


def _content_terms(analyze: Analyzer, text: str) -> set[str]:
    return {term for term in analyze(text) if term not in _STOPWORDS}


@dataclass(frozen=True, slots=True)
class CitedSentence:
    """One answer sentence, quoted verbatim from a source, with its PMID."""

    text: str
    pmid: str
    overlap: int


@dataclass(frozen=True, slots=True)
class ExtractiveAnswer:
    """The extractive answer, or an abstention with nearest passages."""

    query: str
    abstained: bool
    sentences: tuple[CitedSentence, ...]
    sources: tuple[tuple[str, str], ...]  # (pmid, title), in citation order
    nearest: tuple[tuple[str, str], ...]  # (pmid, title), shown when abstaining


def _split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def answer_extractive(
    retriever: Retriever,
    query: str,
    *,
    k_passages: int = 5,
    max_sentences: int = 3,
    analyze: Analyzer = simple_analyzer,
) -> ExtractiveAnswer:
    """Answer ``query`` extractively from the top ``k_passages`` passages."""
    hits = retriever.retrieve(query, k=k_passages)
    query_terms = _content_terms(analyze, query)

    # (overlap, passage_rank, sentence_index) — the last two keep ordering
    # deterministic when overlaps tie.
    candidates: list[tuple[int, int, int, CitedSentence]] = []
    for passage_rank, hit in enumerate(hits):
        if hit.document is None:
            continue
        for sentence_index, sentence in enumerate(_split_sentences(hit.document.abstract)):
            overlap = len(query_terms & _content_terms(analyze, sentence))
            if overlap > 0:
                candidates.append(
                    (
                        overlap,
                        passage_rank,
                        sentence_index,
                        CitedSentence(sentence, hit.pmid, overlap),
                    )
                )

    if not candidates:
        nearest = tuple((hit.pmid, _title(hit)) for hit in hits[:3])
        return ExtractiveAnswer(query, abstained=True, sentences=(), sources=(), nearest=nearest)

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen: list[CitedSentence] = [item[3] for item in candidates[:max_sentences]]

    titles = {hit.pmid: _title(hit) for hit in hits}
    sources: list[tuple[str, str]] = []
    seen: set[str] = set()
    for cited in chosen:
        if cited.pmid not in seen:
            seen.add(cited.pmid)
            sources.append((cited.pmid, titles.get(cited.pmid, "")))

    return ExtractiveAnswer(
        query=query,
        abstained=False,
        sentences=tuple(chosen),
        sources=tuple(sources),
        nearest=(),
    )


def _title(hit: RetrievalHit) -> str:
    return hit.document.title if hit.document is not None else ""


def render_answer(answer: ExtractiveAnswer) -> str:
    """Render an :class:`ExtractiveAnswer` for the CLI, with the required banner."""
    lines: list[str] = []
    if answer.abstained:
        lines.append("ABSTAIN — retrieved literature does not directly answer this question.")
        if answer.nearest:
            lines.append("")
            lines.append("Nearest passages:")
            lines += [f"  PMID {pmid} — {title}" for pmid, title in answer.nearest]
    else:
        citation_number = {pmid: index for index, (pmid, _) in enumerate(answer.sources, start=1)}
        for sentence in answer.sentences:
            lines.append(f"{sentence.text} [{citation_number[sentence.pmid]}]")
        lines.append("")
        lines.append("Sources:")
        lines += [
            f"  [{index}] PMID {pmid} — {title}"
            for index, (pmid, title) in enumerate(answer.sources, start=1)
        ]
        lines.append("")
        lines.append("Confidence: GROUNDED (extractive — sentences quoted verbatim from sources)")

    lines.append("")
    lines.append(_NOT_MEDICAL_ADVICE)
    return "\n".join(lines)
