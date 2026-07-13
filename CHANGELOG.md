# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Phase 1 — corpus ingestion + tokenizer.**
  - `meridian.corpus`: normalized `Document` record (ADR-0002 one-chunk-per-doc),
    MeSH domain filter for the ADR-0001 domains, streaming PubMed XML parser
    (structured abstracts, language/empty filtering), exact + near-duplicate-title
    de-duplication, resumable checksummed baseline downloader, idempotent SQLite
    document store behind a repository protocol, and corpus statistics.
  - `meridian.tokenization`: mixed-corpus BPE training via Polaris (ADR-0003),
    fertility metric, vocabulary-size sweep, and a versioned tokenizer artifact.
  - `scripts/ingest.py` (one-command store rebuild + stats report) and
    `scripts/train_tokenizer.py` (sweep + versioned artifact).
  - ADR-0002 (chunking), ADR-0003 (tokenizer corpus mix), Phase 1 design doc,
    `benchmarks/corpus.md` report, and a committed offline sample fixture.

## [0.0.1] - 2026-07-13

### Added

- Repository scaffolding: `src/` layout, offline-only test suite, uv project,
  Black/Ruff/strict-mypy/pytest configuration with a ≥ 90% coverage gate.
- CI: GitHub Actions matrix (Python 3.12/3.13 × Ubuntu/macOS) running lint,
  format check, typecheck, and tests.
- ADR-0001 (project scope), dataset license review memo, ADR process docs.
- Community files: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue/PR templates.
- Pinned upstream frameworks: `polaris-nlp==1.1.0`, `zenith-nlp==1.0.0`.
