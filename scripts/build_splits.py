#!/usr/bin/env python
"""Build frozen dev/test eval splits from PubMedQA (Phase 2, Task 2).

Reads a PubMedQA PQA-L JSON file, maps each question to its source PMID, splits
deterministically into dev/test, writes them, and prints their checksums for
registration in ``benchmarks/splits/checksums.json``.

**Frozen forever:** run this once. After the checksums are registered, the splits
are read-only (house rule #4) — never regenerate them.

Example
-------
    uv run python scripts/build_splits.py data/pubmedqa/ori_pqal.json \\
        --out-dir benchmarks/splits --name pqal
"""

from __future__ import annotations

import argparse
from pathlib import Path

from meridian.eval.pubmedqa import build_eval_set_from_pubmedqa, load_pubmedqa, split_dev_test
from meridian.eval.qrels import save_eval_set
from meridian.eval.splits import split_checksum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pubmedqa", type=Path, help="PubMedQA PQA-L JSON file")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory to write splits")
    parser.add_argument("--name", default="pqal", help="base name for the splits")
    parser.add_argument("--dev-fraction", type=float, default=0.5)
    args = parser.parse_args()

    data = load_pubmedqa(args.pubmedqa)
    eval_set = build_eval_set_from_pubmedqa(data, name=args.name)
    dev, test = split_dev_test(eval_set, dev_fraction=args.dev_fraction)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, filename in ((dev, f"{args.name}_dev.json"), (test, f"{args.name}_test.json")):
        path = args.out_dir / filename
        save_eval_set(split, path)
        print(f"{filename}: {len(split)} queries, checksum {split_checksum(split)}")
    print("Register these checksums in benchmarks/splits/checksums.json (then never regenerate).")


if __name__ == "__main__":
    main()
