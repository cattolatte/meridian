"""Tests for the lexical analyzer."""

from meridian.retrieval.analyzer import simple_analyzer


def test_lowercases_and_splits() -> None:
    assert simple_analyzer("Metformin, and the HEART!") == ["metformin", "and", "the", "heart"]


def test_keeps_alphanumeric_tokens() -> None:
    assert simple_analyzer("HbA1c dropped 2.5%") == ["hba1c", "dropped", "2", "5"]


def test_empty_text() -> None:
    assert simple_analyzer("   ") == []
