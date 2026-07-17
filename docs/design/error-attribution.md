# Error attribution study (Phase 11)

The definitive measured story of where the system fails. For every wrong end-to-end
answer on the frozen **test** split (its first and only use), we substitute an oracle for
each pipeline stage in turn and record which substitution *fixes* the answer; the failure
is attributed to the **earliest** stage whose oracle repairs it (`meridian.eval.attribution`).

## Method

Pipeline order and their oracles:

| Stage | Oracle substitution | "Fixed" means |
|---|---|---|
| **retrieval** | inject the gold passage(s) into the candidate set | correct with perfect recall |
| **rerank** | rank the gold passage first | correct given retrieval found it |
| **generation** | use the gold answer sentences | correct given the right passages |
| **verification** | a perfect verifier verdict | correct answer wrongly suppressed / passed |

`attribute_failure({stage: fixed})` returns the earliest stage that fixes the case;
`attribution_study(cases)` aggregates the breakdown (the attribution pie).

## Outputs (from the real test run)

- **Attribution pie** — fraction of failures per stage. **TBD** (real test run).
- **PubMedQA headline** — yes/no/maybe accuracy of the full pipeline
  (`pubmedqa_accuracy`). **TBD**.
- **Ablation narrative** (in `benchmarks/BENCHMARKS.md`) — which components paid rent,
  which didn't, including honest negative results.

## Honesty clause

Every number is produced by a committed script over the frozen test split, run once. If a
component did not help (e.g. dense < BM25, or the reranker's nDCG lift was marginal), it is
reported as such — the honest-benchmark culture is the brand (RAG.md §7).

*Pending: the real test-split campaign requires all trained artifacts (retriever, reranker,
generator, verifier) and is run once, at the end.*
