# Phase 9 - Abstention & calibration (design doc)

- **Release:** v0.8.0
- **Goal:** the system knows what it doesn't know, measurably.
- **Upstream:** Gate 2 reuses Polaris `SentencePairClassifier(num_classes=2)` (available).
- **Governing ADR:** [0007](../adr/0007-abstention-operating-point.md) (operating point).

## Module layout (`src/meridian/abstain/`)

```
gate.py           Gate 1 - RetrievalGate: abstain on low retrieval confidence
                  (top score + top-1/top-k margin).
answerability.py  Gate 2 - answerability classifier (pair head over question + top
                  passages); config/artifact/data/training + answerable_probability.
calibration.py    risk_coverage_curve / operating_point: error rate vs coverage,
                  pick the operating point (~80% coverage, ADR-0007).
```

## Key decisions

- **Two gates, cheap then decisive.** Gate 1 is a pure function of the ranking (no model);
  Gate 2 is a small pair-head classifier over `(question, concatenated top passages)`.
  Either firing ⇒ abstain, before the generator runs.
- **Margin matters.** A high top score with a flat tail is *less* confident than a high
  top score with a big margin; Gate 1 uses both.
- **Operating point from measured curves.** Thresholds come from the dev risk-coverage
  curve, not guesses (claims hygiene); ~80% coverage is the recommended target.
- **Off-domain / personal advice abstains ~always** (ADR-0007) - a hard requirement.

## Testing strategy (offline-only)

- Gate 1: confidence (top + margin) and the two-threshold decision; empty hits.
- Risk-coverage: ordering by confidence, coverage/error at each prefix, operating point
  at a target coverage; empty rejected.
- Gate 2: training loss decreases on answerable-vs-unanswerable examples; probability in
  [0, 1]; empty rejected.

## Environment constraint

The chosen thresholds, the committed risk-coverage plot, and the off-domain abstain rate
need the trained retriever + answerability gate on the frozen dev split and the 100-question
off-domain battery. The gates, calibration, and metrics are verified offline with tiny
models; the numbers and operating point come from the real run.

## Exit criteria tracking

| Criterion | Status |
|---|---|
| Risk-coverage plot committed | curve/operating-point implemented; plot pending real run |
| Off-domain abstain rate measured | battery + gates ready; number pending real run |
| Operating point in config + ADR | ADR-0007 (procedure + target); concrete thresholds pending |
| >= 90% coverage held | maintained |
