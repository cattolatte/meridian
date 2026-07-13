"""Answer generation.

Phase 2 ships the extractive answerer v0 (RAG.md §4.2): it returns corpus sentences
verbatim with their PMID citations, so it is definitionally incapable of
hallucination. It remains the system's shipped fallback for the life of the project;
the Zenith grounded generator (Phase 7) is layered on top, never replacing it.
"""

from meridian.answer.extractive import (
    CitedSentence,
    ExtractiveAnswer,
    answer_extractive,
    render_answer,
)

__all__ = ["CitedSentence", "ExtractiveAnswer", "answer_extractive", "render_answer"]
