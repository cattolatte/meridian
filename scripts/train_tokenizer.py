#!/usr/bin/env python
"""Train Meridian's BPE tokenizer from the document store (ADR-0003).

Runs the vocabulary-size sweep on a mix of biomedical chunks (from the store) and a
general-English corpus, reports fertility, trains the final tokenizer at the chosen
size, and writes a versioned artifact.

The general corpus is a plain-text file, one passage per line. ADR-0003's intended
source is a sample of MS MARCO passages (the general text the retriever also trains
on); any general-English line file works for a demonstration run.

Example
-------
    uv run python scripts/train_tokenizer.py --db data/corpus.sqlite \\
        --general data/msmarco_sample.txt --out artifacts/tokenizer.json \\
        --vocab-sizes 16000 32000 --report benchmarks/corpus.md
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from meridian.corpus.store import SqliteDocumentStore
from meridian.tokenization import (
    SweepResult,
    save_tokenizer,
    sweep_vocabulary_sizes,
    train_tokenizer,
)


def _read_general(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _checksum(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _render_sweep(results: list[SweepResult], mix_ratio: float) -> str:
    lines = [
        "## Tokenizer fertility sweep (ADR-0003)",
        "",
        f"Mix ratio: {mix_ratio:.2f} biomedical / {1 - mix_ratio:.2f} general.",
        "",
        "| Vocab size | Biomedical fertility | General fertility |",
        "|---|---|---|",
    ]
    lines += [
        f"| {r.vocab_size} | {r.biomedical_fertility:.3f} | {r.general_fertility:.3f} |"
        for r in results
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="SQLite document store path")
    parser.add_argument(
        "--general", type=Path, required=True, help="general-English text, one line per passage"
    )
    parser.add_argument("--out", type=Path, required=True, help="output tokenizer artifact path")
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[16000, 32000])
    parser.add_argument("--mix-ratio", type=float, default=0.7)
    parser.add_argument("--report", type=Path, help="append the fertility sweep table here")
    args = parser.parse_args()

    with SqliteDocumentStore(args.db) as store:
        biomedical = [doc.chunk_text() for doc in store.iter_documents()]
    general = _read_general(args.general)
    if not biomedical:
        parser.error("document store is empty; run scripts/ingest.py first")

    results = sweep_vocabulary_sizes(
        biomedical,
        general,
        args.vocab_sizes,
        biomedical_eval=biomedical,
        general_eval=general,
        mix_ratio=args.mix_ratio,
    )
    for result in results:
        print(
            f"vocab_size={result.vocab_size} "
            f"bio_fertility={result.biomedical_fertility:.3f} "
            f"general_fertility={result.general_fertility:.3f}"
        )

    # Default: the smallest swept size (fewer parameters) unless a larger size is
    # explicitly preferred; the sweep table lets a human confirm the trade-off.
    chosen = min(args.vocab_sizes)
    tokenizer = train_tokenizer(biomedical, general, vocab_size=chosen, mix_ratio=args.mix_ratio)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_tokenizer(
        tokenizer,
        args.out,
        metadata={
            "vocab_size": chosen,
            "mix_ratio": args.mix_ratio,
            "biomedical_documents": len(biomedical),
            "general_passages": len(general),
            "biomedical_checksum": _checksum(biomedical),
            "general_checksum": _checksum(general),
        },
    )
    print(f"wrote tokenizer artifact (vocab_size={chosen}): {args.out}")

    if args.report is not None:
        with args.report.open("a") as handle:
            handle.write("\n" + _render_sweep(results, args.mix_ratio))
        print(f"appended fertility sweep to {args.report}")


if __name__ == "__main__":
    main()
