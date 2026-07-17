# Phase 11 - Evaluation campaign + error attribution (design doc)

- **Release:** v0.10.0
- **Goal:** the definitive measured story of the whole system (the research artifact).

## Module layout

```
eval/attribution.py   attribute_failure / attribution_study - oracle-substitution
                      error attribution (retrieval / rerank / generation / verification).
eval/pubmedqa.py      pubmedqa_accuracy - the yes/no/maybe headline number.
docs/design/error-attribution.md   the method + committed results.
```

## Key decisions

- **Attribute to the earliest fixing stage.** A wrong answer is blamed on the first stage
  (in pipeline order) whose oracle substitution repairs it - the stage that first broke
  the chain. Compound failures with no single fix are `unattributed`.
- **Test split used once.** The full campaign (all retrieval configs x +/-reranker x
  extractive/generated x gates) touches the frozen **test** split for the first and only
  time here; it is never inspected earlier (house rule #4).
- **Every claim reproducible.** The attribution pie, the PubMedQA headline, and the
  ablation narrative all come from committed scripts over the frozen split + MLflow ids.

## Testing strategy (offline-only)

- Attribution: earliest-stage logic, unattributed case, aggregate counts/fractions.
- PubMedQA scoring: accuracy (case-insensitive), missing prediction = wrong, empty gold
  rejected.

## Environment constraint

The campaign is data- and artifact-heavy: it needs the trained retriever, reranker,
generator, and verifier and the frozen test split, and is run **once** at the end. The
attribution and scoring building blocks are verified offline; the numbers (attribution
pie, PubMedQA headline, ablation table) are filled from that single real run - none from
memory.

## Exit criteria tracking

| Criterion | Status |
|---|---|
| BENCHMARKS.md complete + reproducible | building blocks done; real numbers pending the campaign |
| Attribution doc merged | **done** - [error-attribution.md](error-attribution.md) (method + pending results) |
| >= 90% coverage held | maintained |
