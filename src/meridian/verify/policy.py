"""The fail-safe answer ladder (RAG.md §4.2, Phase 8 task 5).

Generate → verify. If every content sentence is entailed by its cited span, respond
GROUNDED. Otherwise fall back to the extractive answerer; if that too finds nothing,
abstain. No unverified generated claim reaches the user.
"""

from __future__ import annotations

from dataclasses import dataclass

from polaris.models import SentencePairClassifier
from polaris.tokenizers import BPETokenizer
from zenith.models.decoder import DecoderLM
from zenith.tokenizers.byte_tokenizer import ByteTokenizer

from meridian.answer.extractive import answer_extractive, render_answer
from meridian.corpus.store import DocumentStore
from meridian.generation.answerer import answer_grounded, render_grounded_answer
from meridian.retrieval.pipeline import Retriever
from meridian.verify.verifier import VerificationReport, verify_grounded_answer


@dataclass(frozen=True, slots=True)
class VerifiedAnswer:
    """The final answer after the fail-safe ladder, with its confidence label."""

    confidence: str  # GROUNDED | GROUNDED (extractive fallback) | ABSTAIN
    rendered: str
    report: VerificationReport | None


def answer_with_verification(
    retriever: Retriever,
    generator: DecoderLM,
    generator_tokenizer: ByteTokenizer,
    verifier: SentencePairClassifier,
    verifier_tokenizer: BPETokenizer,
    store: DocumentStore,
    query: str,
    *,
    k: int = 5,
) -> VerifiedAnswer:
    """Generate, verify, and apply the fail-safe ladder."""
    grounded = answer_grounded(retriever, generator, generator_tokenizer, query, k=k)
    if grounded.abstained:
        return VerifiedAnswer("ABSTAIN", render_grounded_answer(grounded), None)

    report = verify_grounded_answer(grounded, store, verifier, verifier_tokenizer)
    if report.grounded:
        rendered = render_grounded_answer(
            grounded,
            confidence=f"GROUNDED ({len(report.verdicts)} sentence(s) verified)",
        )
        return VerifiedAnswer("GROUNDED", rendered, report)

    # Fail-safe: verification did not pass -> extractive fallback -> abstain.
    extractive = answer_extractive(retriever, query, k_passages=k)
    if extractive.abstained:
        return VerifiedAnswer("ABSTAIN", render_answer(extractive), report)
    return VerifiedAnswer("GROUNDED (extractive fallback)", render_answer(extractive), report)
