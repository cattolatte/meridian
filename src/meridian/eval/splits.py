"""Frozen dev/test splits with a checksum guard (house rule #4).

The dev/test splits created in Phase 2 are **read-only forever**: never trained on,
never regenerated. This module computes a canonical content hash of an eval set and
verifies a loaded split against an expected hash, so accidental edits or
regeneration break loudly (a committed test asserts each split matches its
registered checksum).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from meridian.eval.qrels import EvalSet, load_eval_set, to_jsonable


class ChecksumMismatchError(RuntimeError):
    """Raised when a frozen split does not match its expected checksum."""


def split_checksum(eval_set: EvalSet) -> str:
    """Return the canonical SHA-256 hash of an eval set's content."""
    canonical = json.dumps(to_jsonable(eval_set), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_frozen_split(path: str | Path, expected_checksum: str) -> EvalSet:
    """Load a split and verify it matches ``expected_checksum``.

    Raises :class:`ChecksumMismatchError` if the content hash differs — the guard
    that keeps frozen splits immutable.
    """
    eval_set = load_eval_set(path)
    actual = split_checksum(eval_set)
    if actual != expected_checksum:
        raise ChecksumMismatchError(f"{path}: expected checksum {expected_checksum}, got {actual}")
    return eval_set
