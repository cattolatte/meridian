"""Stage-0 MLM pretraining and trunk transfer (ADR-0004).

Wraps Polaris' MLM pretraining pipeline (`pretrain`) to pretrain the encoder trunk on
the corpus, then transfers that trunk into a downstream head (`TextEmbedder`, and
later the reranker/NLI models) via `transfer_encoder_to`. This is what makes a
from-scratch bi-encoder viable; the Stage-0 checkpoint is kept and reused across
Phases 3/6/8 — one pretrained trunk, four heads.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from polaris.collation import collate
from polaris.models import HasEncoder
from polaris.pretraining import MaskedLanguageModel, pretrain
from polaris.pretraining.loop import PretrainEpoch
from polaris.tokenizers import BPETokenizer

from meridian.encoder.artifact import EmbedderConfig


def build_mlm(config: EmbedderConfig) -> MaskedLanguageModel:
    """Build a masked-language model whose trunk matches an ``EmbedderConfig``."""
    return MaskedLanguageModel(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        max_len=config.max_len,
        pad_id=config.pad_id,
    )


def mlm_pretrain(
    model: MaskedLanguageModel,
    texts: Sequence[str],
    tokenizer: BPETokenizer,
    *,
    mask_id: int,
    vocab_size: int,
    epochs: int = 1,
    batch_size: int = 32,
    max_length: int = 256,
    learning_rate: float = 1e-3,
    mask_probability: float = 0.15,
    seed: int = 0,
) -> tuple[PretrainEpoch, ...]:
    """MLM-pretrain ``model`` on ``texts``; return per-epoch metrics.

    Raises :class:`ValueError` if ``texts`` is empty.
    """
    if not texts:
        raise ValueError("no texts to pretrain on")

    pad_id = tokenizer.vocabulary.pad_id or 0
    encodings = [tokenizer.encode(text) for text in texts]
    batches = [
        collate(
            [(encoding, 0) for encoding in encodings[start : start + batch_size]],
            pad_id=pad_id,
            max_length=max_length,
        )
        for start in range(0, len(encodings), batch_size)
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    records: tuple[PretrainEpoch, ...] = pretrain(
        model,
        optimizer,
        batches,
        mask_id=mask_id,
        vocab_size=vocab_size,
        epochs=epochs,
        mask_probability=mask_probability,
        seed=seed,
    )
    return records


def initialize_from_mlm(model: MaskedLanguageModel, target: HasEncoder) -> None:
    """Copy the pretrained trunk from ``model`` into ``target`` (Stage-0 init)."""
    model.transfer_encoder_to(target)
