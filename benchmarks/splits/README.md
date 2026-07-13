# Frozen evaluation splits

These files are **read-only forever** (house rule #4): never trained on, never
regenerated. Each split's canonical content hash is registered in `checksums.json`
and asserted by `tests/eval/test_splits.py`, so any edit or regeneration breaks CI.

- `sample_dev.json` — a tiny fixture over the committed `examples/sample_pubmed.xml`
  corpus, used to exercise the harness and the checksum guard offline.
- The real `dev.json` / `test.json` (built from PubMedQA PQA-L, mapping each question
  to its relevant PMIDs) are added here once, registered in `checksums.json`, and
  never touched again.

Format: an `EvalSet` (see `meridian.eval.qrels`) — a name plus a list of
`{query_id, question, relevant_pmids}` entries, serialized canonically.
