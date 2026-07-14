"""Build (query, passage, label) pair samples for reranker training.

Pointwise labels: a relevant pair is ``1``, an irrelevant pair is ``0``. MS MARCO
triples ``(query, positive, negative)`` expand to one positive and one negative
example each; mined PubMed pairs (Phase 5) domain-adapt the same way.
"""

from __future__ import annotations

from collections.abc import Iterable

from polaris.tokenizers import BPETokenizer, Encoding

PairSample = tuple[Encoding, Encoding, int]  # (query encoding, passage encoding, label)


def make_pair_samples(
    examples: Iterable[tuple[str, str, int]],
    tokenizer: BPETokenizer,
) -> list[PairSample]:
    """Tokenize ``(query, passage, label)`` examples into pair samples."""
    return [
        (tokenizer.encode(query), tokenizer.encode(passage), int(label))
        for query, passage, label in examples
    ]


def pairs_from_triples(
    triples: Iterable[tuple[str, str, str]],
) -> list[tuple[str, str, int]]:
    """Expand ``(query, positive, negative)`` triples into labeled pointwise pairs."""
    examples: list[tuple[str, str, int]] = []
    for query, positive, negative in triples:
        examples.append((query, positive, 1))
        examples.append((query, negative, 0))
    return examples
