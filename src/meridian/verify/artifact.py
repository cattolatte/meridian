"""Versioned artifact for the NLI faithfulness verifier.

The verifier is a Polaris ``SentencePairClassifier`` with ``num_classes=3`` (entail /
neutral / contradict), initialized from the shared Stage-0 MLM trunk (ADR-0004).
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
class NLIConfig:
    """Architecture of the 3-class NLI verifier."""

    vocab_size: int
    num_classes: int = 3  # entail / neutral / contradict
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    max_len: int = 512
    pad_id: int = 0
    pooling: str = "cls"


def build_verifier(config: NLIConfig) -> SentencePairClassifier:
    """Construct the NLI ``SentencePairClassifier`` from its config."""
    return SentencePairClassifier(**asdict(config))


def save_verifier(
    model: SentencePairClassifier,
    config: NLIConfig,
    directory: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write the verifier's weights and config to ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(directory / _WEIGHTS_FILE, model=model, metadata=dict(metadata or {}))
    payload = {
        "format_version": FORMAT_VERSION,
        "arch": asdict(config),
        "metadata": dict(metadata or {}),
    }
    (directory / _CONFIG_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_verifier(directory: str | Path) -> SentencePairClassifier:
    """Rebuild the verifier from a saved artifact and load its weights."""
    directory = Path(directory)
    payload = json.loads((directory / _CONFIG_FILE).read_text())
    version = payload["format_version"]
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported verifier artifact format_version {version}")
    model = build_verifier(NLIConfig(**payload["arch"]))
    load_checkpoint(directory / _WEIGHTS_FILE, model=model)
    return model
