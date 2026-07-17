# Phase 12 - v1.0: docs, demo, release (design doc)

- **Release:** v1.0.0 (**gated** - see below)
- **Goal:** ship it like Zenith shipped 1.0.

## What this phase does (in the repo)

- **README**: architecture diagram (online path), quickstart, serving quickstart, a
  guardrails section ("what this is *not*"), model-card link, and the honest status note.
- **MODEL_CARD.md**: component table updated to "implemented" (pipelines run offline;
  quality numbers pending the real runs).
- **API-stability pass**: the public surface (`meridian ask`/`ingest`, the `serving` app,
  the artifact formats) is stable and semver-versioned.

## The v1.0 / PyPI gate (honest scope)

The plan's v1.0 definition requires **documented component quality, a complete
BENCHMARKS.md, and a PyPI release**. Those depend on the real training campaign (PubMed +
PubMedQA + MS MARCO downloads, MPS/CUDA training, the Phase-11 test-split run) - a
deliberate, networked/compute step the user runs. Therefore:

- **The version is not bumped to 1.0.0, and nothing is published to PyPI, until those
  numbers exist and the user chooses to publish.** Doing so earlier would violate claims
  hygiene (numbers) and the project's release rules (no PyPI/GitHub release without
  explicit instruction).
- Everything *reproducible offline* is done and shipped; the release machinery
  (semver, trusted-publishing workflow) is in place and one command away.

## Remaining user-triggered steps to reach v1.0

1. Run the real training campaign (all phases' `scripts/*` on the downloaded data),
   filling every `TBD` in `benchmarks/BENCHMARKS.md` from committed scripts + MLflow ids.
2. Record the demo GIF (SSE streaming) for the README.
3. Tag `v1.0.0` and publish to PyPI (trusted publishing) - on explicit instruction.
4. Write the `Meridian.md` portfolio file and update the resume from harness numbers only
   (personal files, outside this repo).

## Exit criteria tracking

| Criterion | Status |
|---|---|
| README: architecture, quickstart, guardrails, model card | **done** |
| `pip install meridian-rag` works | pending PyPI publish (gated on real numbers + instruction) |
| A stranger can reproduce the headline benchmark from the README | pending the real campaign |
| >= 90% coverage held | maintained |
