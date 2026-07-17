"""Grounded answer generation (Phase 7).

A Zenith ``DecoderLM`` fine-tuned (LoRA SFT) to write cited answers over retrieved
passages, with citation-constrained decoding (malformed citations are structurally
impossible) and a reserved ``<abstain>`` token (ADR-0006). The extractive answerer
(Phase 2) remains the shipped fallback.
"""

from meridian.generation.answerer import (
    GroundedAnswer,
    answer_grounded,
    render_grounded_answer,
)
from meridian.generation.artifact import (
    GeneratorConfig,
    build_generator,
    load_generator,
    save_generator,
)
from meridian.generation.data import grounded_example, passages_from_hits
from meridian.generation.training import train_generator

__all__ = [
    "GeneratorConfig",
    "GroundedAnswer",
    "answer_grounded",
    "build_generator",
    "grounded_example",
    "load_generator",
    "passages_from_hits",
    "render_grounded_answer",
    "save_generator",
    "train_generator",
]
