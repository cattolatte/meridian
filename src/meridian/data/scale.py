"""Parse raw scale-run corpora into training-sample tuples.

Every loader streams its file and stops at ``max_examples`` so a smoke run reads only a
handful of lines from a multi-GB download. Formats follow each dataset's public release:

* **MS MARCO** ``triples.train.small.tsv`` — ``query<TAB>positive<TAB>negative`` per line.
  Feeds the reranker directly (triples) and the retriever after
  :func:`msmarco_pairs_from_triples` (deduplicated ``(query, positive)`` pairs).
* **PubMedQA PQA-A/PQA-L** ``ori_pqaa.json`` — ``{id: {"QUESTION", "CONTEXTS", ...}}``;
  :func:`load_pqa_pairs` yields ``(question, joined-contexts)`` retriever pairs.
* **SNLI / MultiNLI / SciNLI** JSONL — objects with sentence/label fields;
  :func:`load_nli_jsonl` yields ``(premise, hypothesis, label_id)`` with label ids matching
  :class:`meridian.verify.data.NLILabel` (entailment 0 / neutral 1 / contradiction 2).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from meridian.verify.data import NLILabel

_NLI_LABEL_IDS = {
    "entailment": int(NLILabel.ENTAILMENT),
    "neutral": int(NLILabel.NEUTRAL),
    "contradiction": int(NLILabel.CONTRADICTION),
    # SciNLI uses its own surface forms for the same three relations.
    "contrasting": int(NLILabel.CONTRADICTION),
    "reasoning": int(NLILabel.ENTAILMENT),
}


def load_msmarco_triples(
    path: str | Path, *, max_examples: int | None = None
) -> list[tuple[str, str, str]]:
    """Load ``(query, positive, negative)`` triples from an MS MARCO triples TSV."""
    triples: list[tuple[str, str, str]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            query, positive, negative = parts[0], parts[1], parts[2]
            if query and positive and negative:
                triples.append((query, positive, negative))
                if max_examples is not None and len(triples) >= max_examples:
                    break
    return triples


def msmarco_pairs_from_triples(
    triples: Iterable[tuple[str, str, str]],
) -> list[tuple[str, str]]:
    """Reduce ``(query, positive, negative)`` triples to unique ``(query, positive)`` pairs."""
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for query, positive, _negative in triples:
        key = (query, positive)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def load_pqa_pairs(path: str | Path, *, max_examples: int | None = None) -> list[tuple[str, str]]:
    """Load ``(question, joined-contexts)`` retriever pairs from a PubMedQA JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []
    for entry in data.values():
        question = entry.get("QUESTION", "").strip()
        context = " ".join(entry.get("CONTEXTS", [])).strip()
        if question and context:
            pairs.append((question, context))
            if max_examples is not None and len(pairs) >= max_examples:
                break
    return pairs


def load_nli_jsonl(
    path: str | Path,
    *,
    premise_field: str = "sentence1",
    hypothesis_field: str = "sentence2",
    label_field: str = "gold_label",
    max_examples: int | None = None,
) -> list[tuple[str, str, int]]:
    """Load ``(premise, hypothesis, label_id)`` NLI triples from a JSONL file.

    Rows with an unknown or absent gold label (e.g. SNLI's ``"-"``) are skipped so only
    labeled examples reach the verifier trainer. Field names default to SNLI/MultiNLI and
    can be overridden for other releases.
    """
    examples: list[tuple[str, str, int]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = _NLI_LABEL_IDS.get(str(row.get(label_field, "")).strip().lower())
            premise = str(row.get(premise_field, "")).strip()
            hypothesis = str(row.get(hypothesis_field, "")).strip()
            if label is None or not premise or not hypothesis:
                continue
            examples.append((premise, hypothesis, label))
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def write_pairs_tsv(pairs: Iterable[tuple[str, str]], path: str | Path) -> int:
    """Write ``(anchor, positive)`` pairs as the ``anchor<TAB>positive`` TSV the retriever
    driver reads; return the number of rows written. Pairs with embedded tabs/newlines are
    skipped so the TSV stays well-formed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for anchor, positive in pairs:
            if any(c in anchor or c in positive for c in ("\t", "\n")):
                continue
            handle.write(f"{anchor}\t{positive}\n")
            written += 1
    return written
