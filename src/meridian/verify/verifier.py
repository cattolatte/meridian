"""Verify a grounded answer sentence-by-sentence against its cited spans.

For each content sentence, the NLI verifier checks whether the cited passage span
*entails* it. Un-entailed or uncited sentences are unsupported. Produces the
faithfulness metrics (RAG.md §7): citation precision, citation recall, hallucination
rate — and a ``grounded`` verdict (every content sentence cited and entailed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from polaris.collation import collate_pairs
from polaris.models import SentencePairClassifier
from polaris.tokenizers import BPETokenizer

from meridian.corpus.store import DocumentStore
from meridian.generation.answerer import GroundedAnswer
from meridian.verify.data import NLILabel, make_nli_samples

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True, slots=True)
class SentenceVerdict:
    """One answer sentence, its cited PMID (if any), and the NLI verdict."""

    sentence: str
    cited_pmid: str | None
    label: NLILabel | None  # None when the sentence carries no citation


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Per-sentence verdicts plus aggregate faithfulness metrics."""

    verdicts: tuple[SentenceVerdict, ...]
    citation_precision: float
    citation_recall: float
    hallucination_rate: float

    @property
    def grounded(self) -> bool:
        """True iff every content sentence is cited and entailed by its span."""
        return bool(self.verdicts) and all(v.label is NLILabel.ENTAILMENT for v in self.verdicts)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def _cited_pmid(sentence: str, passages: tuple[tuple[str, str], ...]) -> str | None:
    match = _CITATION.search(sentence)
    if match is None:
        return None
    n = int(match.group(1))
    return passages[n - 1][0] if 1 <= n <= len(passages) else None


def verify_grounded_answer(
    answer: GroundedAnswer,
    store: DocumentStore,
    model: SentencePairClassifier,
    tokenizer: BPETokenizer,
    *,
    max_length: int = 256,
) -> VerificationReport:
    """Run NLI over each content sentence vs its cited span; return the report."""
    sentences = _sentences(answer.text)
    pmids = [_cited_pmid(sentence, answer.passages) for sentence in sentences]

    # Batch the cited sentences through the NLI model.
    labels: dict[int, NLILabel] = {}
    premises: list[tuple[str, str]] = []  # (premise span, hypothesis sentence)
    index_map: list[int] = []
    for i, (sentence, pmid) in enumerate(zip(sentences, pmids, strict=True)):
        if pmid is None:
            continue
        doc = store.get(pmid)
        if doc is None:
            continue
        premises.append((doc.chunk_text(), sentence))
        index_map.append(i)

    if premises:
        vocab = tokenizer.vocabulary
        samples = make_nli_samples([(p, h, 0) for p, h in premises], tokenizer)
        batch = collate_pairs(
            samples,
            pad_id=vocab.pad_id or 0,
            cls_id=vocab.cls_id,
            sep_id=vocab.sep_id,
            max_length=max_length,
        )
        # Follow the model's device so a GPU/MPS-resident verifier works too.
        parameter = next(model.parameters(), None)
        if parameter is not None:
            batch = batch.to(parameter.device)
        model.eval()
        with torch.no_grad():
            predicted = model(batch).argmax(dim=-1).cpu().tolist()
        for i, label in zip(index_map, predicted, strict=True):
            labels[i] = NLILabel(int(label))

    verdicts = tuple(
        SentenceVerdict(sentence=s, cited_pmid=pmids[i], label=labels.get(i))
        for i, s in enumerate(sentences)
    )
    return VerificationReport(
        verdicts=verdicts,
        citation_precision=_precision(verdicts),
        citation_recall=_recall(verdicts),
        hallucination_rate=_hallucination_rate(verdicts),
    )


def _recall(verdicts: tuple[SentenceVerdict, ...]) -> float:
    if not verdicts:
        return 0.0
    cited = sum(1 for v in verdicts if v.cited_pmid is not None)
    return cited / len(verdicts)


def _precision(verdicts: tuple[SentenceVerdict, ...]) -> float:
    cited = [v for v in verdicts if v.label is not None]
    if not cited:
        return 0.0
    entailed = sum(1 for v in cited if v.label is NLILabel.ENTAILMENT)
    return entailed / len(cited)


def _hallucination_rate(verdicts: tuple[SentenceVerdict, ...]) -> float:
    if not verdicts:
        return 0.0
    unsupported = sum(1 for v in verdicts if v.label is not NLILabel.ENTAILMENT)
    return unsupported / len(verdicts)
