"""Tokenizer training and serialization (Phase 1, Task 5).

Meridian's single BPE vocabulary is trained here on a mixed biomedical/general
corpus (:doc:`ADR-0003 </adr/0003-tokenizer-corpus-mix>`) using Polaris'
``train_bpe``, selected by a vocabulary-size sweep on the fertility metric, and
persisted as a versioned, self-describing artifact.
"""

from meridian.tokenization.artifact import (
    TokenizerArtifact,
    load_artifact,
    load_tokenizer,
    save_tokenizer,
)
from meridian.tokenization.fertility import fertility
from meridian.tokenization.training import SweepResult, sweep_vocabulary_sizes, train_tokenizer

__all__ = [
    "SweepResult",
    "TokenizerArtifact",
    "fertility",
    "load_artifact",
    "load_tokenizer",
    "save_tokenizer",
    "sweep_vocabulary_sizes",
    "train_tokenizer",
]
