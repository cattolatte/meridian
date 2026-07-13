# Phase 0 — Scaffolding, licenses, scope lock (design doc)

- **Release:** v0.0.1
- **Goal:** empty-but-industrial repo; all legal/naming questions answered; scope
  frozen. No ML functionality.

## What was built

- **Package skeleton.** `src/meridian/` (src layout, `py.typed`), `tests/`,
  `docs/adr/`, `docs/design/`, `benchmarks/`, `scripts/`. uv-managed project with a
  single pinned dependency set.
- **Quality gates.** Black + Ruff + `mypy --strict` + pytest with a ≥ 90% coverage
  gate, wired identically into `.pre-commit-config.yaml` and CI.
- **CI.** GitHub Actions matrix: Python 3.12/3.13 × Ubuntu/macOS, running format
  check, lint, typecheck, and offline tests.
- **Legal.** [License review memo](../license-review.md) covering PubMed/NLM,
  PubMedQA, MS MARCO (non-commercial research), SNLI/MultiNLI, SciNLI, and the
  self-mined pairs. Credentialed corpora (MIMIC/MedNLI/n2c2) excluded by design.
- **Scope lock.** [ADR-0001](../adr/0001-scope.md): domains (cardiology +
  endocrinology + oncology), ~200K-abstract initial corpus, model-size budget,
  and a $150 rented-GPU cap.
- **Community files.** README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CHANGELOG,
  MODEL_CARD placeholder, BENCHMARKS scaffold, issue/PR templates.

## Decisions & rationale

- **Pinned upstreams:** `polaris-nlp==1.1.0`, `zenith-nlp==1.0.0` (exact pins per
  the house rule that framework extensions land upstream and are consumed as
  releases). The `[ext]` capabilities noted in RAG.md §4.3 are tracked as upstream
  issues and consumed as future pinned minor releases, never vendored here.
- **Python floor 3.12** to match Polaris's `requires-python` and the CI matrix.
- **No speculative infrastructure.** Serving deps (FastAPI), MLflow, Hydra, and
  SQLite tooling are deliberately *not* added yet — they arrive with the phases
  that first need them (Phase 1+), keeping the Phase 0 dependency surface minimal.

## Name & availability check (Task 1)

- `meridian-rag` — available on PyPI as of 2026-07-13 (HTTP 404 on the JSON API);
  fallbacks `meridian-nlp` / `meridian-qa` recorded in ADR-0001, unused.
- GitHub repo target: `github.com/cattolatte/meridian`.

## Exit criteria

| Criterion | Status |
|---|---|
| CI green on empty package | ready (CI defined; runs on first push) |
| ADR-0001 merged | done |
| License memo committed | done |
| Names registered | `meridian-rag` verified available; registration is a manual publish step (no PyPI push this phase per project git rules) |

## Deviations from plan

- **Upstream-extension issue lists (Task 6)** are to be filed in the Polaris and
  Zenith repos, not this one; recorded here as a pointer since this session does not
  push to or modify remote repositories.
- **PyPI/GitHub registration** is verified-available but not executed — publishing
  and remote changes are explicitly deferred until instructed.
