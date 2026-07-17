"""Faithfulness verification — the system's conscience (Phase 8).

An NLI cross-encoder (Polaris ``SentencePairClassifier``, ``num_classes=3``) checks each
answer sentence against its cited passage span: entail / neutral / contradict. Sentences
that are not entailed by their cited span do not reach the user unchallenged — the
fail-safe ladder falls back to extractive or abstains (RAG.md §4.2).
"""

from meridian.verify.artifact import (
    NLIConfig,
    build_verifier,
    load_verifier,
    save_verifier,
)
from meridian.verify.data import NLILabel, make_nli_samples
from meridian.verify.policy import VerifiedAnswer, answer_with_verification
from meridian.verify.training import train_verifier
from meridian.verify.verifier import (
    SentenceVerdict,
    VerificationReport,
    verify_grounded_answer,
)

__all__ = [
    "NLIConfig",
    "NLILabel",
    "SentenceVerdict",
    "VerificationReport",
    "VerifiedAnswer",
    "answer_with_verification",
    "build_verifier",
    "load_verifier",
    "make_nli_samples",
    "save_verifier",
    "train_verifier",
    "verify_grounded_answer",
]
