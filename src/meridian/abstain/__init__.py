"""Abstention & calibration — knowing what it doesn't know (Phase 9).

Two gates decide whether to answer at all: **Gate 1** (retrieval confidence — score
threshold + top1/top-k margin) and **Gate 2** (an answerability classifier, a Polaris
pair head over the question and its top passages). Risk-coverage curves pick the
operating point (ADR-0007). Personal-advice / off-domain questions abstain (RAG.md §3).
"""

from meridian.abstain.answerability import (
    AnswerabilityConfig,
    answerable_probability,
    build_answerability,
    load_answerability,
    make_answerability_samples,
    save_answerability,
    train_answerability,
)
from meridian.abstain.calibration import operating_point, risk_coverage_curve
from meridian.abstain.gate import RetrievalConfidence, RetrievalGate

__all__ = [
    "AnswerabilityConfig",
    "RetrievalConfidence",
    "RetrievalGate",
    "answerable_probability",
    "build_answerability",
    "load_answerability",
    "make_answerability_samples",
    "operating_point",
    "risk_coverage_curve",
    "save_answerability",
    "train_answerability",
]
