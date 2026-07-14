"""Versioned artifact for a trained cross-encoder reranker.

Stores the ``SentencePairClassifier`` architecture (so it can be rebuilt) and its
weights, so a pinned artifact reproduces the reranker's scores.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from polaris.models import SentencePairClassifier
from polaris.training import load_checkpoint, save_checkpoint

FORMAT_VERSION = 1
_WEIGHTS_FILE = "weights.pt"
_CONFIG_FILE = "config.json"


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    """Constructor arguments defining a ``SentencePairClassifier`` reranker."""

    vocab_size: int
    num_classes: int = 1  # a single relevance logit
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    max_len: int = 512
    pad_id: int = 0
    pooling: str = "cls"


def build_reranker(config: RerankerConfig) -> SentencePairClassifier:
    """Construct a ``SentencePairClassifier`` from its config."""
    return SentencePairClassifier(**asdict(config))


def save_reranker(
    model: SentencePairClassifier,
    config: RerankerConfig,
    directory: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write the reranker's weights and architecture config to ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(directory / _WEIGHTS_FILE, model=model, metadata=dict(metadata or {}))
    payload = {
        "format_version": FORMAT_VERSION,
        "arch": asdict(config),
        "metadata": dict(metadata or {}),
    }
    (directory / _CONFIG_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_reranker(directory: str | Path) -> SentencePairClassifier:
    """Rebuild a reranker from a saved artifact and load its weights."""
    directory = Path(directory)
    payload = json.loads((directory / _CONFIG_FILE).read_text())
    version = payload["format_version"]
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported reranker artifact format_version {version}")
    model = build_reranker(RerankerConfig(**payload["arch"]))
    load_checkpoint(directory / _WEIGHTS_FILE, model=model)
    return model
