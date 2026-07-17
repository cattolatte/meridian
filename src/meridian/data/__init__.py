"""Loaders that turn raw scale-run corpora into Meridian's training-sample formats.

The heavy training sets (MS MARCO, SNLI/MultiNLI/SciNLI, PubMedQA PQA-A) are downloaded
out-of-band (see ``scripts/download_scale_data.py``); this package parses them into the
plain Python tuples the existing trainers already consume — no network, no framework
models, streaming with an explicit ``max_examples`` cap so a smoke run stays tiny.
"""

from meridian.data.scale import (
    load_msmarco_triples,
    load_nli_jsonl,
    load_pqa_pairs,
    msmarco_pairs_from_triples,
    write_pairs_tsv,
)

__all__ = [
    "load_msmarco_triples",
    "load_nli_jsonl",
    "load_pqa_pairs",
    "msmarco_pairs_from_triples",
    "write_pairs_tsv",
]
