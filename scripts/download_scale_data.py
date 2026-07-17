#!/usr/bin/env python
"""Fetch the scale-run training corpora into a local directory (run by a human).

This touches the network and is **not** part of the test suite. It downloads the datasets
that have stable direct URLs and prints exact manual steps for the ones behind a form or
Google Drive. After each download it prints the SHA-256 so you can record it alongside any
number you publish (claims hygiene, RAG.md 5). Licenses are summarised in
``docs/license-review.md``; MS MARCO and SciNLI are research-only.

    uv run python scripts/download_scale_data.py --out data/scale
    uv run python scripts/download_scale_data.py --out data/scale --only snli pqal
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    name: str
    filename: str
    url: str | None  # None => manual (form / Drive gated)
    note: str


SOURCES: tuple[Source, ...] = (
    Source(
        "snli",
        "snli_1.0.zip",
        "https://nlp.stanford.edu/projects/snli/snli_1.0.zip",
        "Verifier base. Unzip; use snli_1.0_train.jsonl with train_verifier.py.",
    ),
    Source(
        "multinli",
        "multinli_1.0.zip",
        "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip",
        "Verifier base (add to SNLI). Unzip; multinli_1.0_train.jsonl.",
    ),
    Source(
        "pqal",
        "ori_pqal.json",
        "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
        "PubMedQA PQA-L (1k labeled). Small; also used by the current benchmarks.",
    ),
    Source(
        "msmarco",
        "triples.train.small.tar.gz",
        None,
        "Retriever/reranker. Download triples.train.small.tar.gz from the MS MARCO passage "
        "ranking page (microsoft.github.io/msmarco), extract to triples.train.small.tsv. "
        "Research-only license.",
    ),
    Source(
        "pqaa",
        "ori_pqaa.json",
        None,
        "PubMedQA PQA-A (~211k). Follow the pubmedqa repo's data download (Google Drive) and "
        "place ori_pqaa.json here.",
    ),
    Source(
        "scinli",
        "scinli_train.jsonl",
        None,
        "Verifier domain adaptation. Request SciNLI from github.com/msadat3/SciNLI and place "
        "the train split here. Research-only license.",
    ),
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        total = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            total += len(chunk)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(f"  wrote {dest} ({total / 1e6:.1f} MB)  sha256={digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="download directory")
    parser.add_argument("--only", nargs="*", help="subset of source names")
    args = parser.parse_args()

    names = set(args.only) if args.only else {s.name for s in SOURCES}
    for source in SOURCES:
        if source.name not in names:
            continue
        print(f"[{source.name}] {source.note}")
        dest = args.out / source.filename
        if source.url is None:
            print(f"  MANUAL: place {source.filename} in {args.out}/\n")
            continue
        if dest.exists():
            print(f"  exists, skipping: {dest}\n")
            continue
        _download(source.url, dest)
        print()


if __name__ == "__main__":
    main()
