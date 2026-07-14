"""Build contrastive training samples (ADR-0004 Stages A/B).

A contrastive sample is an ``(anchor, positive)`` pair of tokenized encodings — the
input to Polaris' ``collate_contrastive``. Stage A uses (query, relevant passage)
pairs from MS MARCO; Stage B uses self-mined ``(title, abstract)`` pairs and PQA-A
``(question, abstract)`` pairs. Hard negatives (Phase 5) are an optional third element.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from polaris.collation.contrastive import ContrastiveSample
from polaris.tokenizers import BPETokenizer

from meridian.corpus.records import Document


def make_contrastive_samples(
    pairs: Iterable[tuple[str, str]],
    tokenizer: BPETokenizer,
) -> list[ContrastiveSample]:
    """Tokenize ``(anchor, positive)`` text pairs into contrastive samples."""
    return [(tokenizer.encode(anchor), tokenizer.encode(positive)) for anchor, positive in pairs]


def make_contrastive_samples_with_negatives(
    triples: Iterable[tuple[str, str, Sequence[str]]],
    tokenizer: BPETokenizer,
) -> list[ContrastiveSample]:
    """Tokenize ``(anchor, positive, hard_negatives)`` triples (Phase 5 mining)."""
    return [
        (
            tokenizer.encode(anchor),
            tokenizer.encode(positive),
            [tokenizer.encode(negative) for negative in negatives],
        )
        for anchor, positive, negatives in triples
    ]


def mine_title_abstract_pairs(documents: Iterable[Document]) -> list[tuple[str, str]]:
    """Self-supervised Stage-B pairs: a document's title and abstract are positives."""
    return [(doc.title, doc.abstract) for doc in documents if doc.title and doc.abstract]
