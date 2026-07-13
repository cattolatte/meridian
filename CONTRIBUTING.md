# Contributing to Meridian

Thanks for your interest. Meridian is primarily a solo research/portfolio project
built in strictly ordered phases, but issues and PRs are welcome.

## Ground rules

1. **Zero external NLP-model dependencies.** No Hugging Face models,
   sentence-transformers, embedding APIs, FAISS, or vector DBs. Encoders come from
   `polaris-nlp`, generation from `zenith-nlp` (pinned); everything else is built
   here on PyTorch/NumPy primitives. PRs that violate this are declined regardless
   of quality.
2. **Framework extensions go upstream.** New Polaris/Zenith capabilities belong in
   those repos, released and consumed as pinned versions — never vendored or
   monkey-patched here.
3. **Vertical slices.** `main` must run end-to-end after every merge.
4. **ADR discipline.** Architectural decisions need a numbered ADR in `docs/adr/`
   before implementation. Open an issue first for anything architectural.
5. **Claims hygiene.** Benchmark numbers appear only if reproduced by a committed,
   seeded script with the MLflow run ID referenced.

## Development setup

```bash
uv sync --extra dev
uv run pre-commit install
```

## Quality gates (must pass before every commit)

```bash
uv run black --check src tests
uv run ruff check src tests
uv run mypy --strict
uv run pytest   # offline-only; coverage must stay ≥ 90%
```

Tests must not touch the network. If a test needs data, generate it in the test
or commit a small fixture.

## Commit / PR conventions

- Semantic versioning; user-facing changes get a `CHANGELOG.md` entry under
  *Unreleased*.
- Keep PRs scoped to the current roadmap phase (see the phase design docs in
  `docs/design/`); out-of-phase ideas go in an issue, not the code.
