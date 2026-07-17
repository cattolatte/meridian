# ADR-0007: Abstention operating point

- **Status:** Accepted (thresholds provisional pending the real risk-coverage curve)
- **Date:** 2026-07-14

## Context

Phase 9 makes the system selectively answer: two gates (retrieval confidence, then an
answerability classifier) decide whether to answer at all. We must fix **how the
operating point is chosen** from the risk-coverage trade-off, and what target to aim for.
Selective answering with calibrated confidence is the most senior design signal in the
system (RAG.md §3) — but only if the operating point is picked from measured curves, not
guessed.

## Decision

1. **Two gates, in order.** Gate 1 (`RetrievalGate`) abstains before generation on low
   retrieval confidence (top score + top-1/top-k margin). Gate 2 (`answerable_probability`)
   abstains when the answerability classifier's P(answerable) is below threshold. Either
   gate firing ⇒ abstain.
2. **Operating point from the risk-coverage curve.** Thresholds are selected on the
   **frozen dev split** via `risk_coverage_curve` / `operating_point`, targeting the best
   accuracy at **~80% coverage** (answer 80% of answerable questions, abstain on the rest).
   The chosen thresholds are recorded in config, and the risk-coverage plot is committed.
3. **Personal-advice / off-domain questions abstain ≈ always.** The off-domain battery
   (100 questions the corpus cannot answer, incl. personal medical advice) measures the
   abstain rate; personal-advice questions must abstain nearly always — a hard product
   requirement, not a tuned trade-off.
4. **Reaffirm from real data.** The concrete threshold values are set from the real dev
   risk-coverage curve (trained retriever + answerability gate); until then the gates
   default to permissive thresholds and the mechanism is exercised offline.

## Alternatives considered

- **Single retrieval-score threshold only:** simpler but blind to flat rankings and to
  well-retrieved-but-unanswerable questions; Gate 2 catches the latter.
- **Maximize coverage:** rejected — the whole point is to trade coverage for accuracy;
  80% is the recommended balance, revisited from the curve.

## Consequences

- The serving path (Phase 10) consults both gates before generating; an abstention shows
  the nearest passages and the "not medical advice" banner.
- The risk-coverage curve, chosen operating point, and off-domain abstain rate populate
  `benchmarks/BENCHMARKS.md` from committed scripts (claims hygiene).
