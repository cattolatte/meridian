"""Contrastive training for the dense retriever (ADR-0004 Stages A/B).

Wraps Polaris' ``collate_contrastive`` + ``train_contrastive`` (InfoNCE): batches the
samples, runs the optimizer, and returns per-epoch losses. Meridian does not
reimplement the objective — it supplies data and hyperparameters. The two-stage
curriculum (MS MARCO then domain pairs) is expressed by calling this twice on
different sample sets, on the same embedder.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from polaris.collation import collate_contrastive
from polaris.collation.contrastive import ContrastiveSample
from polaris.models import TextEmbedder
from polaris.training import train_contrastive


def train_retriever(
    embedder: TextEmbedder,
    samples: Sequence[ContrastiveSample],
    *,
    pad_id: int,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    temperature: float = 0.05,
    max_length: int = 256,
    symmetric: bool = True,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> list[float]:
    """Train ``embedder`` on contrastive ``samples``; return per-epoch losses.

    ``device`` moves the embedder onto an accelerator (CUDA/MPS) before training; Polaris'
    ``train_contrastive`` then follows the model's device for each batch. ``None`` keeps
    the model where it is (CPU by default).

    Raises :class:`ValueError` if ``samples`` is empty.
    """
    if not samples:
        raise ValueError("no contrastive samples to train on")

    if device is not None:
        embedder.to(device)
    batches = [
        collate_contrastive(
            samples[start : start + batch_size], pad_id=pad_id, max_length=max_length
        )
        for start in range(0, len(samples), batch_size)
    ]
    optimizer = torch.optim.Adam(embedder.parameters(), lr=learning_rate)
    losses: list[float] = train_contrastive(
        embedder,
        batches,
        optimizer=optimizer,
        epochs=epochs,
        temperature=temperature,
        symmetric=symmetric,
        seed=seed,
    )
    return losses
