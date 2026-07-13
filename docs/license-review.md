# Dataset & Corpus License Review (Phase 0 memo)

- **Date:** 2026-07-13
- **Purpose:** record the terms under which each planned data source may be used,
  before any download or training occurs. Rule: **no dataset with credentialing
  or DUA requirements in the core path** (MIMIC, MedNLI, n2c2 excluded by design).

> This memo records the project's understanding as of the date above; each terms
> page should be re-checked at the phase where the dataset is first downloaded.
> This is not legal advice.

## Summary table

| Source | Planned use | Terms (as understood) | Allowed for this project? |
|---|---|---|---|
| PubMed baseline (NLM) | Corpus (~200K filtered abstracts) — Phase 1 | NLM Data Terms & Conditions: bulk data freely available; NLM does not claim copyright on abstracts, but some abstracts may carry third-party copyright; no NLM endorsement may be implied; the "PubMed"/NLM name and logo may not be used to suggest endorsement | **Yes** — ingest, index, and quote abstracts with PMID attribution; carry a "not endorsed by NLM" note; redistribute derived indexes/embeddings, **not** bulk raw abstract dumps |
| PubMedQA — PQA-L (1K expert), PQA-A (211K auto) | E2E benchmark + frozen splits (Phase 2); grounded-SFT data (Phase 7) | Released publicly on GitHub under MIT; derived from PubMed abstracts (see NLM row) | **Yes** — use for evaluation and training; attribute the PubMedQA paper; PubMed attribution rules apply to the underlying abstracts |
| MS MARCO passage triples | Stage-A retriever/reranker training (Phase 3, 6) | Microsoft "MS MARCO" terms: free for **non-commercial research**; no redistribution of the dataset itself | **Yes for this portfolio/research project** — train on triples; do not redistribute the raw dataset; note the non-commercial restriction anywhere results are published |
| SNLI | NLI verifier base training (Phase 8) | CC BY-SA 4.0 | **Yes** — attribute; derived models are fine |
| MultiNLI | NLI verifier base training (Phase 8) | Mixed per-genre licenses; released for research; majority permissive (OANC public-domain + others) | **Yes for research use** — attribute; treat as research-only |
| SciNLI | NLI verifier domain adaptation (Phase 8) | Released for research use with attribution (ACL 2022) | **Yes** — attribute the SciNLI paper; research use |
| Title↔abstract pairs (self-mined from corpus) | Stage-B retriever domain adaptation (Phase 3, 5) | Derived from the PubMed corpus above | **Yes** — inherits the PubMed/NLM terms; distribute as derived pairs/embeddings, not raw dumps |
| BioASQ | Optional extra benchmark (not a dependency) | Free registration required; research use | **Optional only** — never on the core path; not a build dependency |

## Excluded by design (credentialing / DUA)

MIMIC, MedNLI, n2c2, and any other DUA- or credentialing-gated corpus are **out of
scope for the core path** and must not be added without a superseding ADR. They may
appear only as clearly-labeled optional stretch work with their access requirements
documented.

## Practical rules adopted

1. **Attribution everywhere.** Every published result names its data sources; PMID
   citations accompany any quoted abstract text.
2. **No raw redistribution.** The repo ships code and *derived* artifacts (indexes,
   embeddings, trained weights, eval splits as PMID lists) — never bulk raw
   third-party corpora.
3. **Non-commercial honesty.** Because MS MARCO is non-commercial-research-only,
   Meridian as a whole is presented as a non-commercial research/portfolio project.
4. **Re-verify at download time.** This memo is revisited in the phase where each
   dataset is first fetched; any change in terms is recorded by updating this file
   with a dated note.
