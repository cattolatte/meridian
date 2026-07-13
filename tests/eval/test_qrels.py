"""Tests for eval-set types and canonical JSON I/O."""

from __future__ import annotations

from pathlib import Path

from meridian.eval.qrels import EvalQuery, EvalSet, load_eval_set, save_eval_set

_EVAL_SET = EvalSet(
    name="tiny",
    queries=(
        EvalQuery("q1", "question one", frozenset({"2", "1"})),
        EvalQuery("q2", "question two", frozenset({"3"})),
    ),
)


def test_len() -> None:
    assert len(_EVAL_SET) == 2


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "set.json"
    save_eval_set(_EVAL_SET, path)
    assert load_eval_set(path) == _EVAL_SET


def test_json_is_canonical(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    save_eval_set(_EVAL_SET, a)
    save_eval_set(_EVAL_SET, b)
    assert a.read_text() == b.read_text()
    # PMIDs are sorted regardless of frozenset iteration order.
    text = a.read_text()
    assert text.index('"1"') < text.index('"2"')
