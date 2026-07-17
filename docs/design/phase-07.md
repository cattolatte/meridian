# Phase 7 — Grounded generator (Zenith) (design doc)

- **Release:** v0.6.0
- **Goal:** fluent cited answers — with the extractive path retained as fallback forever.
- **Upstream:** consumes `zenith-nlp==1.1.0` — `Generator.generate_ids(logits_constraint=…)`,
  `AllowedTokens`, `Generator.abstained`, `instruct.grounded` (`GroundedTemplate`,
  `GroundedInstructionDataset`), plus existing LoRA (`CausalLMTrainer(use_lora=…)`).
- **Governing ADR:** [0006](../adr/0006-answer-format.md) (answer format, citations, abstain).

## Module layout (`src/meridian/generation/`)

```
artifact.py   GeneratorConfig + build_generator (Zenith DecoderLM) + versioned save/load
              (weights + arch; re-injects LoRA on load).
data.py       grounded_example / passages_from_hits — (question, [(pmid,text)], answer|None).
training.py   train_generator: LoRA SFT via CausalLMTrainer over a
              GroundedInstructionDataset (Zenith owns the loop).
answerer.py   answer_grounded: retrieve → constrained-decode a cited answer (or abstain)
              → parse [n] citations back to PMIDs; GroundedAnswer + render.
```

## Key decisions

- **Positional `[n]` citations, constrained at the digit.** After `[`, decoding is
  restricted to `1..k` via `AllowedTokens` — a citation to a passage not in context is
  structurally impossible (ADR-0006). `[n]` maps to the n-th retrieved passage's PMID.
- **Abstain is a first-class token.** Zenith's reserved `<abstain>` (`ByteTokenizer.abstain_id`)
  is a stop token; `Generator.abstained` detects it. No claims, nearest passages shown.
- **GENERATED, not yet GROUNDED.** A fresh answer is labelled GENERATED; the Phase 8
  verifier promotes it to GROUNDED. Extractive stays the shipped fallback (default
  `--answerer extractive`); generation is `--answerer generated --generator DIR`.
- **Byte tokenizer for the generator.** The generator uses Zenith's `ByteTokenizer`
  (vocab 260, includes the abstain token) — decoupled from the Polaris encoder vocabulary.

## Testing strategy (offline-only)

- Artifact round-trip reproduces logits; format-version guard.
- `answer_grounded` on a tiny random model: prose is gibberish (untrained) **but** every
  parsed citation maps to a retrieved PMID (the constraint + parsing under test); abstains
  when no passages are retrieved.
- Render shows citations, GENERATED/ABSTAIN, and the "Not medical advice" banner.
- Training wrapper: empty examples rejected; the real SFT run is deferred.

## Environment constraint

Fluent, correct cited prose needs the real grounded-SFT dataset (PQA + citation alignment
+ hand-audit) and a real LoRA SFT run on the ~30–125M generator. The *mechanism*
(constrained decoding, abstain, citation parsing, LoRA SFT wiring) is verified offline with
a tiny model; answer quality and the ≥95% schema-validity number come from the real run.

## Exit criteria tracking

| Criterion | Status |
|---|---|
| Schema-valid cited answers on ≥95% dev | mechanism done (citations structurally valid by construction); real % pending SFT run |
| Extractive fallback one flag away | **done** — `--answerer extractive` is the default |
| Malformed-citation rate ~0 via constrained decoding | **done** (by construction; a cited `[n]` always references a retrieved passage) |
| ≥ 90% coverage held | maintained (96%) |

## Remaining user-triggered step

Build the grounded-SFT dataset from PQA-A/PQA-L (align answer sentences to source spans,
hand-audit 50), `train_generator` (LoRA SFT) → `save_generator`, then
`meridian ask --answerer generated --generator DIR`. Phase 8 adds the verifier that turns
GENERATED into GROUNDED.
