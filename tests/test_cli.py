"""End-to-end tests for the `meridian` CLI (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from meridian.cli import main

_SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_pubmed.xml"


def _ingest(db: Path) -> None:
    assert main(["ingest", str(_SAMPLE), "--db", str(db)]) == 0


def test_ingest_then_ask_grounded(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "corpus.sqlite"
    _ingest(db)
    capsys.readouterr()  # discard ingest output

    code = main(["ask", "beta-blockers mortality after myocardial infarction", "--db", str(db)])
    out = capsys.readouterr().out
    assert code == 0
    assert "PMID 10000001" in out
    assert "GROUNDED" in out
    assert "Not medical advice." in out


def test_ask_abstains_on_off_topic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "corpus.sqlite"
    _ingest(db)
    capsys.readouterr()

    code = main(["ask", "lattice gauge theory in particle physics", "--db", str(db)])
    out = capsys.readouterr().out
    assert code == 0
    assert "ABSTAIN" in out


def test_ask_without_store_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["ask", "anything", "--db", str(tmp_path / "missing.sqlite")])
    assert code == 1
    assert "not found" in capsys.readouterr().out


def test_ask_dense_missing_artifacts_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "corpus.sqlite"
    _ingest(db)
    capsys.readouterr()
    code = main(["ask", "heart failure", "--db", str(db), "--retriever", "dense"])
    out = capsys.readouterr().out
    assert code == 1
    assert "--embedder" in out  # dense requires artifacts


def test_ingest_reports_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "corpus.sqlite"
    code = main(["ingest", str(_SAMPLE), "--db", str(db)])
    out = capsys.readouterr().out
    assert code == 0
    assert "stored=6" in out
