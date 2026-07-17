"""Gate 2 — answerability classifier.

A Polaris ``SentencePairClassifier(num_classes=2)`` over ``(question, top passages)``:
can the retrieved evidence answer this question at all? Trained with PQA unanswerable
negatives + synthetic off-domain questions (RAG.md §9). Shares the Stage-0 trunk.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from polaris.collation import collate_pairs
from polaris.models import SentencePairClassifier
from polaris.tokenizers import BPETokenizer, Encoding
from polaris.training import load_checkpoint, save_checkpoint
from torch import nn

FORMAT_VERSION = 1
_WEIGHTS_FILE = "weights.pt"
_CONFIG_FILE = "config.json"

AnswerabilitySample = tuple[Encoding, Encoding, int]  # (question, passages, label)


@dataclass(frozen=True, slots=True)
class AnswerabilityConfig:
    """Architecture of the 2-class answerability gate."""

    vocab_size: int
    num_classes: int = 2  # unanswerable / answerable
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    max_len: int = 512
    pad_id: int = 0
    pooling: str = "cls"


def build_answerability(config: AnswerabilityConfig) -> SentencePairClassifier:
    return SentencePairClassifier(**asdict(config))


def _join_passages(passages: Sequence[str]) -> str:
    return " ".join(passages)


def make_answerability_samples(
    examples: Iterable[tuple[str, Sequence[str], int]],
    tokenizer: BPETokenizer,
) -> list[AnswerabilitySample]:
    """Tokenize ``(question, passages, label)`` into pair samples (label 1 = answerable)."""
    return [
        (tokenizer.encode(question), tokenizer.encode(_join_passages(passages)), int(label))
        for question, passages, label in examples
    ]


def train_answerability(
    model: SentencePairClassifier,
    samples: Sequence[AnswerabilitySample],
    *,
    pad_id: int,
    cls_id: int,
    sep_id: int,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    max_length: int = 256,
    seed: int = 0,
) -> list[float]:
    """Train the answerability gate (2-class cross-entropy); return per-epoch losses."""
    if not samples:
        raise ValueError("no answerability samples to train on")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    losses: list[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch.labels.long())
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / len(batches))
    return losses


def answerable_probability(
    model: SentencePairClassifier,
    tokenizer: BPETokenizer,
    question: str,
    passages: Sequence[str],
    *,
    max_length: int = 256,
) -> float:
    """Return P(answerable) for a question given its top passages."""
    vocab = tokenizer.vocabulary
    samples = make_answerability_samples([(question, passages, 0)], tokenizer)
    batch = collate_pairs(
        samples,
        pad_id=vocab.pad_id or 0,
        cls_id=vocab.cls_id,
        sep_id=vocab.sep_id,
        max_length=max_length,
    )
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=-1)
    return float(probs[0, 1])


def save_answerability(
    model: SentencePairClassifier,
    config: AnswerabilityConfig,
    directory: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(directory / _WEIGHTS_FILE, model=model, metadata=dict(metadata or {}))
    payload = {
        "format_version": FORMAT_VERSION,
        "arch": asdict(config),
        "metadata": dict(metadata or {}),
    }
    (directory / _CONFIG_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_answerability(directory: str | Path) -> SentencePairClassifier:
    directory = Path(directory)
    payload = json.loads((directory / _CONFIG_FILE).read_text())
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"unsupported answerability artifact format_version {payload['format_version']}"
        )
    model = build_answerability(AnswerabilityConfig(**payload["arch"]))
    load_checkpoint(directory / _WEIGHTS_FILE, model=model)
    return model
