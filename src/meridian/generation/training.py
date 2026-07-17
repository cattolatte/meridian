"""Grounded LoRA SFT for the generator (Phase 7).

Wraps Zenith's ``CausalLMTrainer`` over a ``GroundedInstructionDataset``: passages go in
the prompt, the cited answer (or abstain) is the masked target, and LoRA adapters are
trained. Meridian supplies the grounded examples; Zenith owns the training loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from zenith.instruct.grounded import GroundedInstructionDataset, GroundedTemplate
from zenith.models.decoder import DecoderLM
from zenith.peft.lora import LoraConfig
from zenith.tokenizers.byte_tokenizer import ByteTokenizer
from zenith.training.trainer import CausalLMTrainer, TrainingConfig

from meridian.generation.artifact import GeneratorConfig
from meridian.generation.data import GroundedExample


def train_generator(
    model: DecoderLM,
    examples: Sequence[GroundedExample],
    tokenizer: ByteTokenizer,
    config: GeneratorConfig,
    *,
    max_length: int = 512,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    save_path: str | Path,
    seed: int = 0,
) -> dict[str, Any]:
    """LoRA-SFT ``model`` on grounded ``examples``; return the trainer's metrics.

    Raises :class:`ValueError` if ``examples`` is empty.
    """
    if not examples:
        raise ValueError("no grounded examples to train on")

    dataset = GroundedInstructionDataset(
        list(examples), tokenizer, max_length=max_length, template=GroundedTemplate()
    )
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        use_lora=config.use_lora,
        lora=LoraConfig(rank=config.lora_rank, alpha=config.lora_alpha),
        save_path=str(save_path),
        log_samples=False,
    )
    trainer = CausalLMTrainer(model, tokenizer, training_config)
    metrics: dict[str, Any] = trainer.fit(dataset)
    return metrics
