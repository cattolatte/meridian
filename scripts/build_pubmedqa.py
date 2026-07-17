#!/usr/bin/env python
"""Build a real retrieval corpus + frozen splits from PubMedQA PQA-L.

PQA-L is self-contained: each entry has a question, its source abstract (``CONTEXTS``),
and a yes/no/maybe label. This builds a document store keyed by PubMed id (one abstract
per entry) and a retrieval eval set that maps each question to its source PMID, then
splits it into frozen dev/test. This gives **real, reproducible** retrieval numbers
without the multi-gigabyte PubMed baseline (that full domain-filtered corpus is a
separate, heavier download — ADR-0001).

The raw PubMedQA text and the built store/splits are NOT committed (license: no bulk
raw redistribution); this script regenerates them, and the split checksums it prints
let anyone verify they built the same frozen splits.

    curl -sL https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json \\
        -o data/pubmedqa/ori_pqal.json
    uv run python scripts/build_pubmedqa.py --pqal data/pubmedqa/ori_pqal.json \\
        --db data/pubmedqa.sqlite --out-dir data/pubmedqa
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.eval.pubmedqa import build_eval_set_from_pubmedqa, split_dev_test
from meridian.eval.qrels import save_eval_set
from meridian.eval.splits import split_checksum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pqal", type=Path, required=True, help="PubMedQA ori_pqal.json")
    parser.add_argument("--db", type=Path, required=True, help="output SQLite document store")
    parser.add_argument("--out-dir", type=Path, required=True, help="dir for dev/test splits")
    parser.add_argument("--dev-fraction", type=float, default=0.5)
    args = parser.parse_args()

    data = json.loads(args.pqal.read_text())

    # One document per entry: PMID = the pubid key, abstract = joined contexts.
    documents = [
        Document(pmid=pubid, title="", abstract=" ".join(entry.get("CONTEXTS", [])))
        for pubid, entry in data.items()
        if entry.get("CONTEXTS")
    ]
    args.db.parent.mkdir(parents=True, exist_ok=True)
    with SqliteDocumentStore(args.db) as store:
        stored = store.add_many(documents)

    eval_set = build_eval_set_from_pubmedqa(data, name="pqal")
    dev, test = split_dev_test(eval_set, dev_fraction=args.dev_fraction)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_eval_set(dev, args.out_dir / "pqal_dev.json")
    save_eval_set(test, args.out_dir / "pqal_test.json")

    labels = Counter(entry["final_decision"] for entry in data.values())
    total = sum(labels.values())
    majority = max(labels.values()) / total

    print(f"documents stored: {stored}")
    print(f"dev queries: {len(dev)}  checksum: {split_checksum(dev)}")
    print(f"test queries: {len(test)}  checksum: {split_checksum(test)}")
    print(f"label distribution: {dict(labels)}")
    print(f"PubMedQA majority-class (yes) baseline: {majority:.3f}")


if __name__ == "__main__":
    main()
