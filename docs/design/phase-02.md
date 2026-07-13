# Phase 2 — BM25 baseline + eval harness + extractive E2E v0 (design doc)

- **Release:** v0.1.0 (first real release)
- **Goal:** the system answers questions end-to-end **before any neural retrieval
  exists**, and the measuring instruments are frozen.
- **Governing rules:** vertical slice (`meridian ask` works after this phase);
  baselines before models; frozen splits are read-only forever (house rule #4).

## Module layout

```
retrieval/
  analyzer.py   Text -> lexical terms (the conventional word analyzer BM25 uses).
  bm25.py       From-scratch BM25 Okapi: inverted index + scored top-k search.
  pipeline.py   Retriever protocol; BM25 retriever over the document store.
eval/
  qrels.py      EvalQuery / EvalSet (question -> relevant PMIDs) + JSON I/O.
  splits.py     Frozen dev/test splits with a checksum guard (house rule #4).
  metrics.py    Recall@k, MRR@10, nDCG@10.
  harness.py    Run a retriever over a split, compute metrics, write JSON results;
                optional MLflow logging (lazy import; never required by tests).
answer/
  extractive.py Extractive answerer v0: top passages -> best sentences by lexical
                overlap, returned verbatim with PMID citations (zero hallucination).
cli.py          `meridian ask` / `meridian ingest` (argparse; stdlib only).
```

## Key decisions

- **BM25 uses a conventional word analyzer, not the BPE tokenizer.** The honest
  lexical baseline every neural claim is measured against should be the standard
  one; subword BM25 is a non-standard variant. The analyzer is injected, so a
  BPE-backed analyzer remains a one-line swap for later comparison. This keeps the
  baseline independent of the Phase 1 tokenizer artifact (baselines before models).
- **k1/b are search-time parameters.** IDF and postings depend only on the corpus,
  so tuning k1/b needs no rebuild; defaults are 1.5 / 0.75, tuned on the dev split
  only.
- **Deterministic ranking.** Ties break by ascending PMID, so results are
  reproducible run to run (claims hygiene).
- **Frozen splits are checksum-guarded.** A canonical content hash of each committed
  split is asserted by a test; regeneration or accidental edits break CI. Real
  splits come from PubMedQA PQA-L (download); the mechanism is exercised now on a
  small committed fixture and applies unchanged to the real splits.
- **Extractive answerer is definitionally hallucination-free.** It returns corpus
  sentences verbatim with their PMID, ranked by lexical overlap with the query — the
  shipped fallback the whole system keeps forever (RAG.md §4.2).

## Testing strategy (offline-only)

- BM25: correctness against hand-computed scores on a tiny corpus; IDF monotonicity;
  determinism; k1/b effects; empty/unknown-term handling.
- Metrics: known-value checks for recall/MRR/nDCG, including edge cases (no
  relevant docs, perfect ranking).
- Splits: round-trip I/O; checksum guard detects tampering.
- Answerer + CLI: end-to-end on the sample corpus; every returned sentence carries a
  PMID that exists in the retrieved set.

## Environment constraint

Real BENCHMARKS numbers need the ingested PubMed corpus (Phase 1 download) and the
PubMedQA split (this phase's download) — deliberate networked steps the user runs.
All code and metrics are verified offline on synthetic fixtures; the BM25 row in
`benchmarks/BENCHMARKS.md` stays TBD until the real split is run. No number is
written from memory.

## Exit criteria tracking

| Criterion | Status |
|---|---|
| Frozen splits with checksum guard | done (`eval.splits` + guard test; `sample_dev.json` registered) |
| BM25 numbers in BENCHMARKS.md | mechanism done (`scripts/evaluate.py`); real row pending download |
| CLI demo works offline on the built index | done (`meridian ask` verified on the sample corpus) |
| ≥ 90% coverage held | maintained (97%) |

## Remaining user-triggered step

The retrieval baseline, eval harness, split-construction, and CLI are complete and
verified offline. Outstanding Phase 2 work needs downloads the user runs:

1. Ingest the real PubMed corpus (Phase 1 download).
2. `scripts/build_splits.py` on PubMedQA PQA-L → register `dev.json`/`test.json`
   checksums in `benchmarks/splits/checksums.json` (freeze once).
3. `scripts/evaluate.py --db … --split benchmarks/splits/dev.json` → fills the real
   BM25 row in `benchmarks/BENCHMARKS.md`. k1/b are tuned on **dev only** by
   re-running with `--k1/--b`; the test split is untouched until Phase 11.

Verified offline: `meridian ask` returns cited GROUNDED answers on the sample
corpus; `scripts/evaluate.py` scores the frozen `sample_dev.json` split (perfect on
the trivially-separable 6-doc sample) with the checksum guard enforced.
