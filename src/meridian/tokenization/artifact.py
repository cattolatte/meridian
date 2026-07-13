"""Versioned, self-describing tokenizer artifact (save / load).

Polaris' ``BPETokenizer`` has no serialization of its own, so Meridian defines a
small JSON artifact format. It captures everything needed to reproduce
tokenization — vocabulary, merges, end-of-word marker — plus the training metadata
(mix ratio, vocab size, corpus checksum) required by the claims-hygiene rule that a
pinned artifact fully determines its behavior.

(Tokenizer save/load is a natural upstream Polaris feature; until it lands there,
this reconstructs a ``BPETokenizer`` from its public ``vocabulary`` and its merge
list. The two private attributes read here are the merge sequence and end-of-word
marker, which the public constructor takes as arguments.)
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polaris.tokenizers import BPETokenizer, Vocabulary

#: Bump when the on-disk layout changes incompatibly.
FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class TokenizerArtifact:
    """A trained tokenizer plus its reproducibility metadata."""

    tokenizer: BPETokenizer
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format_version: int = FORMAT_VERSION


def _tokenizer_state(tokenizer: BPETokenizer) -> tuple[list[list[str]], str]:
    """Return the merge list and end-of-word marker used to reconstruct a tokenizer."""
    # BPETokenizer exposes no public accessor for its merges/end-of-word marker;
    # both are public constructor arguments, so reading them here is a stable
    # reconstruction contract (and a candidate upstream save/load feature).
    merges = [list(pair) for pair in tokenizer._merges]
    end_of_word = tokenizer._end_of_word
    return merges, end_of_word


def save_tokenizer(
    tokenizer: BPETokenizer,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write ``tokenizer`` and its training ``metadata`` to a JSON artifact."""
    merges, end_of_word = _tokenizer_state(tokenizer)
    payload = {
        "format_version": FORMAT_VERSION,
        "end_of_word": end_of_word,
        "vocabulary": tokenizer.vocabulary.to_dict(),
        "merges": merges,
        "metadata": dict(metadata or {}),
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_artifact(path: str | Path) -> TokenizerArtifact:
    """Load a tokenizer artifact, including its metadata and format version."""
    payload = json.loads(Path(path).read_text())
    version = payload["format_version"]
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported tokenizer artifact format_version {version}")
    vocabulary = Vocabulary.from_dict(payload["vocabulary"])
    merges = [tuple(pair) for pair in payload["merges"]]
    tokenizer = BPETokenizer(vocabulary, merges, end_of_word=payload["end_of_word"])
    return TokenizerArtifact(
        tokenizer=tokenizer,
        metadata=payload["metadata"],
        format_version=version,
    )


def load_tokenizer(path: str | Path) -> BPETokenizer:
    """Load just the ``BPETokenizer`` from an artifact."""
    return load_artifact(path).tokenizer
