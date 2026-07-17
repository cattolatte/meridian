# Phase 8 — Faithfulness verifier + citation checking (design doc)

- **Release:** v0.7.0
- **Goal:** the system's conscience — no unverified sentence reaches the user.
- **Upstream:** consumes Polaris `SentencePairClassifier(num_classes=3)` (already
  released, 1.4.0); initialized from the Stage-0 MLM trunk (ADR-0004).

## Module layout (`src/meridian/verify/`)

```
artifact.py   NLIConfig (num_classes=3) + build_verifier + versioned save/load.
data.py       NLILabel (entail/neutral/contradict) + make_nli_samples.
training.py   train_verifier — 3-class cross-entropy over collate_pairs batches
              (SNLI+MultiNLI base -> SciNLI adaptation, RAG.md §6).
verifier.py   verify_grounded_answer — sentence-split the answer, NLI(cited span ->
              sentence) per content sentence, and the faithfulness metrics.
policy.py     answer_with_verification — the fail-safe ladder (generate -> verify ->
              extractive fallback -> abstain).
```

## Key decisions

- **NLI = the same pair head, three classes.** The verifier is a
  `SentencePairClassifier(num_classes=3)` — one architecture (Polaris) does rerank
  (1 class), answerability (2), and NLI (3), all from the shared Stage-0 trunk.
- **Entailment is the bar.** A sentence is *supported* only if its cited span is
  predicted `ENTAILMENT`. Neutral or contradict = unsupported; uncited = unsupported.
- **Metrics (RAG.md §7).** citation recall = fraction of content sentences carrying a
  citation; citation precision = fraction of cited sentences whose span entails;
  hallucination rate = fraction of content sentences not entailed by a citation.
- **Fail-safe ladder.** `answer_with_verification`: if every content sentence is
  entailed → GROUNDED; else fall back to the extractive answerer; if that abstains →
  ABSTAIN. A generated answer is never shown unverified.

## Testing strategy (offline-only)

- Training loss decreases on separable NLI examples; artifact round-trip; format guard.
- `verify_grounded_answer`: correct per-sentence verdicts and metric ranges; an uncited
  answer is never `grounded`.
- Fail-safe ladder: abstains when the generator abstains; otherwise resolves to a
  labelled outcome and always carries the "Not medical advice" banner.

## Environment constraint

Real citation-precision / hallucination numbers and the verifier–human agreement
(hand-audit of 100 triples) need the trained verifier (SNLI/MultiNLI/SciNLI) and the
real generated answers. The verification pipeline, metrics, and fail-safe ladder are
verified offline with tiny models; the numbers come from the real run.

## Exit criteria tracking

| Criterion | Status |
|---|---|
| Faithfulness metrics in BENCHMARKS.md | metrics implemented; real values pending trained verifier |
| Hand-audit agreement documented | pending (100-triple audit on real verifier) |
| Fallback ladder covered by tests | **done** — `answer_with_verification` + tests |
| ≥ 90% coverage held | maintained |
