"""NLI training samples for the verifier.

The premise is a source span, the hypothesis an answer sentence; the label is one of
entail / neutral / contradict. Base training uses SNLI+MultiNLI, domain adaptation uses
SciNLI (RAG.md §6); this module tokenizes ``(premise, hypothesis, label)`` triples into
Polaris pair samples via ``collate_pairs``.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum

from polaris.tokenizers import BPETokenizer, Encoding

NLISample = tuple[Encoding, Encoding, int]


class NLILabel(IntEnum):
    """The three NLI classes (label ids match the classifier's output columns)."""

    ENTAILMENT = 0
    NEUTRAL = 1
    CONTRADICTION = 2


def make_nli_samples(
    examples: Iterable[tuple[str, str, int]],
    tokenizer: BPETokenizer,
) -> list[NLISample]:
    """Tokenize ``(premise, hypothesis, label)`` triples into NLI pair samples."""
    return [
        (tokenizer.encode(premise), tokenizer.encode(hypothesis), int(label))
        for premise, hypothesis, label in examples
    ]
