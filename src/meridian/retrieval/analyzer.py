"""Text analysis for lexical retrieval.

An *analyzer* turns text into the sequence of terms an inverted index stores and
queries. The default :func:`simple_analyzer` is the conventional word analyzer BM25
uses — lowercase, split on runs of non-alphanumeric characters — which keeps the
Phase 2 baseline standard and independent of the trained tokenizer (ADR-0003). The
BM25 index takes the analyzer as an argument, so a BPE-backed analyzer can be
swapped in for comparison without touching the index.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")


def simple_analyzer(text: str) -> list[str]:
    """Lowercase and split ``text`` into alphanumeric terms."""
    return _TOKEN.findall(text.lower())
