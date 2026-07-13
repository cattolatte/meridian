"""Frozen-split checksum guard (house rule #4).

Verifies every committed split matches its registered checksum, and that the guard
detects tampering. If this test fails, a frozen split was edited or regenerated —
that must never happen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meridian.eval.qrels import EvalQuery, EvalSet, save_eval_set
from meridian.eval.splits import (
    ChecksumMismatchError,
    load_frozen_split,
    split_checksum,
)

_SPLITS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "splits"


def _registry() -> dict[str, str]:
    data: dict[str, str] = json.loads((_SPLITS_DIR / "checksums.json").read_text())
    return data


def test_committed_splits_match_registered_checksums() -> None:
    registry = _registry()
    assert registry, "no frozen splits registered"
    for filename, expected in registry.items():
        # Raises ChecksumMismatchError if the committed file was altered.
        eval_set = load_frozen_split(_SPLITS_DIR / filename, expected)
        assert len(eval_set) > 0


def test_guard_detects_tampering(tmp_path: Path) -> None:
    original = EvalSet("s", (EvalQuery("q1", "question", frozenset({"1"})),))
    path = tmp_path / "s.json"
    save_eval_set(original, path)
    good = split_checksum(original)

    tampered = EvalSet("s", (EvalQuery("q1", "question", frozenset({"999"})),))
    save_eval_set(tampered, path)  # someone edits the frozen file

    with pytest.raises(ChecksumMismatchError):
        load_frozen_split(path, good)
