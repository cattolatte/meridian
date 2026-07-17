# ADR-0006: Answer format and citation/abstain semantics

- **Status:** Accepted
- **Date:** 2026-07-14

## Context

Phase 7 introduces the Zenith grounded generator. Before training it, we must fix the
**answer format**: how citations are written, how the answer maps back to passages, and
what abstention looks like. The format is the contract between the generator, the
citation-constrained decoder (Zenith `AllowedTokens`), and the Phase 8 verifier.

## Decision

1. **Inline bracketed citations `[n]`.** The answer text carries citations as `[n]`,
   where `n` is the 1-based index of a passage in the prompt context. `n` maps to the
   passage's PMID (the prompt lists passages as `[pmid] text`, but citations index by
   position, so decoding constraints stay a small fixed digit set).
2. **Citation-constrained decoding.** Immediately after `[`, the generator may only emit
   a digit `1..k` (k = number of retrieved passages, k ≤ 9 for single-digit indices) via
   Zenith's `AllowedTokens(trigger_ids={'['}, allowed_ids={'1'..'k'})`. A citation to a
   passage not in context is therefore **structurally impossible**; malformed-citation
   rate is ~0 by construction.
3. **Structured answer object.** The generator's output is parsed into a `GroundedAnswer`:
   the generated text, the ordered `(n, pmid, title)` citations it used, and the retrieved
   passages. Downstream (Phase 8) verifies each content sentence against its cited span.
4. **Abstention via the `<abstain>` token.** Unanswerable questions train the generator to
   emit Zenith's reserved abstain token; `Generator.abstained(ids)` detects it. An
   abstaining answer carries no claims and shows the nearest passages (RAG.md §3).
5. **Confidence labels.** A freshly generated, unverified answer is labelled
   **GENERATED**; it becomes **GROUNDED** only after the Phase 8 faithfulness verifier
   passes every content sentence. Abstentions are **ABSTAIN**. Every surface carries
   "Research literature assistant. Not medical advice."

## Alternatives considered

- **Citations by PMID (`[PMID 18784090]`):** unconstrainable at the digit level and
  verbose; rejected in favor of positional `[n]` with a PMID map.
- **Free-form citations (post-hoc extraction):** allows hallucinated references; rejected
  — constrained decoding is the whole point.
- **No abstain token (string sentinel):** brittle; the reserved token is first-class and
  detectable.

## Consequences

- The generator is trained with the grounded-SFT format (Zenith `GroundedInstructionDataset`)
  where passages are numbered and answers cite `[n]`.
- Single-digit indices cap the reranked context the generator reads at ~9 passages
  (k ≈ 5 in practice, RAG.md §4.2) — comfortably within budget; multi-digit citation
  constraints are a later refinement if needed.
- The extractive answerer (Phase 2) remains the shipped fallback forever; generation is
  additive and gated behind verification.
