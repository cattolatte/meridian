"""Train the cross-encoder reranker (pointwise BCE).

Batches ``(query, passage, label)`` samples with Polaris' ``collate_pairs`` (which
builds ``[CLS] query [SEP] passage [SEP]``), scores each with the ``num_classes=1``
``SentencePairClassifier``, and optimizes a binary cross-entropy over the single
relevance logit. Initialize the model from the Stage-0 trunk before calling this.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from polaris.collation import collate_pairs
from polaris.models import SentencePairClassifier
from torch import nn

from meridian.reranker.data import PairSample


def train_reranker(
    model: SentencePairClassifier,
    samples: Sequence[PairSample],
    *,
    pad_id: int,
    cls_id: int,
    sep_id: int,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    max_length: int = 256,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> list[float]:
    """Train ``model`` on pointwise pair ``samples``; return per-epoch mean losses.

    ``device`` moves the model and batches onto an accelerator (CUDA/MPS); ``None`` keeps
    them on CPU.

    Raises :class:`ValueError` if ``samples`` is empty.
    """
    if not samples:
        raise ValueError("no reranker samples to train on")

    torch.manual_seed(seed)
    batches = [
        collate_pairs(
            samples[start : start + batch_size],
            pad_id=pad_id,
            cls_id=cls_id,
            sep_id=sep_id,
            max_length=max_length,
        )
        for start in range(0, len(samples), batch_size)
    ]
    if device is not None:
        model.to(device)
        batches = [batch.to(device) for batch in batches]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    losses: list[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()
            logits = model(batch).squeeze(-1)  # (B,) relevance logit
            loss = loss_fn(logits, batch.labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / len(batches))
    return losses
