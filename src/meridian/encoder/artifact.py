"""Versioned artifact for a trained ``TextEmbedder``.

Persists the embedder's architecture config (so it can be rebuilt) alongside its
weights (via Polaris' ``save_checkpoint``). A pinned artifact fully reproduces the
retriever's embeddings — the claims-hygiene requirement for the dense row in
``BENCHMARKS.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from polaris.models import TextEmbedder
from polaris.training import load_checkpoint, save_checkpoint

FORMAT_VERSION = 1
_WEIGHTS_FILE = "weights.pt"
_CONFIG_FILE = "config.json"


@dataclass(frozen=True, slots=True)
class EmbedderConfig:
    """The constructor arguments that define a ``TextEmbedder``'s architecture."""

    vocab_size: int
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    max_len: int = 512
    pad_id: int = 0
    projection_dim: int | None = None
    normalize: bool = True


def build_embedder(config: EmbedderConfig) -> TextEmbedder:
    """Construct a ``TextEmbedder`` from its architecture config."""
    return TextEmbedder(**asdict(config))


def save_embedder(
    embedder: TextEmbedder,
    config: EmbedderConfig,
    directory: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write the embedder's weights and architecture config to ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(directory / _WEIGHTS_FILE, model=embedder, metadata=dict(metadata or {}))
    payload = {
        "format_version": FORMAT_VERSION,
        "arch": asdict(config),
        "metadata": dict(metadata or {}),
    }
    (directory / _CONFIG_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_embedder(directory: str | Path) -> TextEmbedder:
    """Rebuild a ``TextEmbedder`` from a saved artifact and load its weights."""
    directory = Path(directory)
    payload = json.loads((directory / _CONFIG_FILE).read_text())
    version = payload["format_version"]
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported embedder artifact format_version {version}")
    embedder = build_embedder(EmbedderConfig(**payload["arch"]))
    load_checkpoint(directory / _WEIGHTS_FILE, model=embedder)
    return embedder
