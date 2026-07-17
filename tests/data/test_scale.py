"""Tests for scale-run dataset loaders (offline, tiny fixtures)."""

from __future__ import annotations

import json
from pathlib import Path

from meridian.data import (
    load_msmarco_triples,
    load_nli_jsonl,
    load_pqa_pairs,
    msmarco_pairs_from_triples,
    write_pairs_tsv,
)
from meridian.verify.data import NLILabel


def test_load_msmarco_triples_and_pairs(tmp_path: Path) -> None:
    tsv = tmp_path / "triples.tsv"
    tsv.write_text(
        "what is aspirin\taspirin is a drug\tmelanoma immunotherapy\n"
        "what is aspirin\taspirin is a drug\tanother negative\n"  # dup (query, pos)
        "malformed line without tabs\n"
        "diabetes\tmetformin lowers glucose\tbeta blockers\n"
    )
    triples = load_msmarco_triples(tsv)
    assert len(triples) == 3
    assert triples[0] == ("what is aspirin", "aspirin is a drug", "melanoma immunotherapy")

    pairs = msmarco_pairs_from_triples(triples)
    assert pairs == [
        ("what is aspirin", "aspirin is a drug"),
        ("diabetes", "metformin lowers glucose"),
    ]


def test_load_msmarco_triples_respects_cap(tmp_path: Path) -> None:
    tsv = tmp_path / "triples.tsv"
    tsv.write_text("".join(f"q{i}\tp{i}\tn{i}\n" for i in range(100)))
    assert len(load_msmarco_triples(tsv, max_examples=5)) == 5


def test_load_pqa_pairs(tmp_path: Path) -> None:
    path = tmp_path / "ori_pqaa.json"
    path.write_text(
        json.dumps(
            {
                "1": {"QUESTION": "Does X cause Y?", "CONTEXTS": ["Part one.", "Part two."]},
                "2": {"QUESTION": "  ", "CONTEXTS": ["no question"]},  # skipped
                "3": {"QUESTION": "Q3", "CONTEXTS": []},  # skipped (no context)
                "4": {"QUESTION": "Q4", "CONTEXTS": ["ctx4"]},
            }
        )
    )
    pairs = load_pqa_pairs(path)
    assert pairs == [("Does X cause Y?", "Part one. Part two."), ("Q4", "ctx4")]
    assert load_pqa_pairs(path, max_examples=1) == [("Does X cause Y?", "Part one. Part two.")]


def test_load_nli_jsonl_maps_labels_and_skips_ungraded(tmp_path: Path) -> None:
    path = tmp_path / "snli.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sentence1": "a cat sleeps",
                        "sentence2": "an animal rests",
                        "gold_label": "entailment",
                    }
                ),
                json.dumps(
                    {
                        "sentence1": "a cat sleeps",
                        "sentence2": "a dog runs",
                        "gold_label": "contradiction",
                    }
                ),
                json.dumps({"sentence1": "x", "sentence2": "y", "gold_label": "-"}),  # skipped
                "",  # blank line skipped
                json.dumps({"sentence1": "p", "sentence2": "q", "gold_label": "neutral"}),
            ]
        )
    )
    examples = load_nli_jsonl(path)
    assert [label for _, _, label in examples] == [
        int(NLILabel.ENTAILMENT),
        int(NLILabel.CONTRADICTION),
        int(NLILabel.NEUTRAL),
    ]


def test_load_nli_jsonl_scinli_fields(tmp_path: Path) -> None:
    path = tmp_path / "scinli.jsonl"
    path.write_text(
        json.dumps({"text1": "prior work", "text2": "in contrast we", "label": "contrasting"})
        + "\n"
    )
    examples = load_nli_jsonl(
        path, premise_field="text1", hypothesis_field="text2", label_field="label"
    )
    assert examples == [("prior work", "in contrast we", int(NLILabel.CONTRADICTION))]


def test_write_pairs_tsv_roundtrip_and_skips_malformed(tmp_path: Path) -> None:
    out = tmp_path / "pairs.tsv"
    written = write_pairs_tsv(
        [("query one", "passage one"), ("has\ttab", "bad"), ("query two", "passage two")], out
    )
    assert written == 2
    lines = out.read_text().splitlines()
    assert lines == ["query one\tpassage one", "query two\tpassage two"]
