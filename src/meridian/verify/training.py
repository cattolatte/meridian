"""Train the NLI verifier (3-class cross-entropy).

Batches ``(premise, hypothesis, label)`` samples with ``collate_pairs`` and optimizes
cross-entropy over the three NLI classes. Initialize the model from the Stage-0 trunk
before calling this.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from polaris.collation import collate_pairs
from polaris.models import SentencePairClassifier
from torch import nn

from meridian.verify.data import NLISample


def train_verifier(
    model: SentencePairClassifier,
    samples: Sequence[NLISample],
    *,
    pad_id: int,
    cls_id: int,
    sep_id: int,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    max_length: int = 256,
    class_weights: Sequence[float] | None = None,
    device: torch.device | str | None = None,
    epoch_callback: Callable[[int, SentencePairClassifier], None] | None = None,
    seed: int = 0,
) -> list[float]:
    """Train ``model`` on NLI ``samples``; return per-epoch mean cross-entropy losses.

    ``class_weights`` (one per class) up-weights the loss on under-represented classes,
    so a minority label — e.g. PubMedQA ``maybe`` (11% of PQA-L) — is not drowned out by
    the majority. Pass inverse-frequency weights to counter class imbalance; ``None``
    keeps the unweighted cross-entropy.

    ``epoch_callback(epoch, model)`` is invoked after each epoch (0-indexed) with the model
    back in ``train`` mode on the next iteration, so a caller can evaluate on a held-out set
    and checkpoint the best epoch (early stopping) without disturbing training.

    Raises :class:`ValueError` if ``samples`` is empty.
    """
    if not samples:
        raise ValueError("no NLI samples to train on")

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
    weight = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
    if weight is not None and device is not None:
        weight = weight.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()
            logits = model(batch)  # (B, 3)
            loss = loss_fn(logits, batch.labels.long())
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / len(batches))
        if epoch_callback is not None:
            epoch_callback(epoch, model)
    return losses
