"""Versioned artifact for the grounded generator (a Zenith ``DecoderLM``).

Stores the decoder architecture (to rebuild) and the trained weights. LoRA is tracked
so the adapters are re-injected before the weights load.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from zenith.models.decoder import DecoderConfig, DecoderLM
from zenith.peft.lora import LoraConfig, inject_lora

FORMAT_VERSION = 1
_WEIGHTS_FILE = "weights.pt"
_CONFIG_FILE = "config.json"


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Architecture + LoRA settings for the grounded generator."""

    vocab_size: int = 260  # Zenith ByteTokenizer vocabulary
    block_size: int = 512
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16


def build_generator(config: GeneratorConfig) -> DecoderLM:
    """Build a ``DecoderLM`` from a generator config (no LoRA injected)."""
    return DecoderLM(
        DecoderConfig(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
        )
    )


def _maybe_inject_lora(model: DecoderLM, config: GeneratorConfig) -> DecoderLM:
    if config.use_lora:
        inject_lora(model, LoraConfig(rank=config.lora_rank, alpha=config.lora_alpha))
    return model


def save_generator(
    model: DecoderLM,
    config: GeneratorConfig,
    directory: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write the generator's weights and config to ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / _WEIGHTS_FILE)
    payload = {
        "format_version": FORMAT_VERSION,
        "arch": asdict(config),
        "metadata": dict(metadata or {}),
    }
    (directory / _CONFIG_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_generator(directory: str | Path) -> DecoderLM:
    """Rebuild the generator (re-injecting LoRA if used) and load its weights."""
    directory = Path(directory)
    payload = json.loads((directory / _CONFIG_FILE).read_text())
    version = payload["format_version"]
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported generator artifact format_version {version}")
    config = GeneratorConfig(**payload["arch"])
    model = _maybe_inject_lora(build_generator(config), config)
    state = torch.load(directory / _WEIGHTS_FILE, weights_only=True)
    model.load_state_dict(state)
    return model
