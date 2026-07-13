# Meridian

[![CI](https://github.com/cattolatte/meridian/actions/workflows/ci.yml/badge.svg)](https://github.com/cattolatte/meridian/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/cattolatte/meridian?sort=semver)](https://github.com/cattolatte/meridian/releases)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://github.com/cattolatte/meridian)
[![License: MIT](https://img.shields.io/github/license/cattolatte/meridian)](LICENSE)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> A from-scratch grounded RAG engine over biomedical literature (PubMed).
> Every ML component is trained in-house — tokenizer, dense retriever, reranker,
> faithfulness verifier ([Polaris](https://github.com/cattolatte/Polaris)), and the
> cited-answer generator ([Zenith](https://github.com/cattolatte/zenith-nlp-framework)).
> **Every answer is cited, verified, or refused.**

**Research literature assistant. Not medical advice.**

## Status

Pre-alpha (`v0.0.1`) — Phase 0: scaffolding. The system is built in strictly
ordered vertical slices; end-to-end question answering ships from `v0.1.0`.

## Design principles

- **Zero external NLP-model dependencies.** No Hugging Face models, no embedding
  APIs, no FAISS or vector DBs. Encoders come from `polaris-nlp`, generation from
  `zenith-nlp` (both pinned); everything else — BM25, IVF/HNSW ANN indexes,
  serving — is built in this repo on PyTorch/NumPy primitives.
- **Baselines before models.** BM25 and brute-force search exist before any neural
  component, so every neural claim has an honest denominator.
- **Grounded or silent.** Generation is citation-constrained, every claim sentence
  is verified by an NLI entailment check, and the system abstains when retrieval
  confidence or answerability is low.
- **The eval harness is the product.** Every published number is reproducible from
  a committed, seeded script. See [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md).

## Quickstart (offline demo)

The repository ships a tiny synthetic corpus so the vertical slice runs with no
downloads:

```bash
uv sync
uv run meridian ingest examples/sample_pubmed.xml --db build/corpus.sqlite
uv run meridian ask "Does metformin reduce cardiovascular mortality in type 2 diabetes?" \
    --db build/corpus.sqlite
```

You get cited sentences quoted verbatim from the corpus, a `GROUNDED` badge, and the
"Not medical advice" banner — or an `ABSTAIN` when nothing relevant is retrieved. On
the real corpus, replace the sample file with downloaded PubMed baseline files. (The
sample is fabricated data for demonstration only.)

## Repository layout

```
src/meridian/   library code
tests/          offline-only test suite (coverage gate ≥ 90%)
docs/adr/       Architecture Decision Records
docs/design/    per-phase design docs
benchmarks/     benchmark results + reproduction scripts
scripts/        operational scripts (ingest, index builds, releases)
```

## Development

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pre-commit install
uv run pytest
```

Quality gates (enforced in CI and pre-commit): Black, Ruff, `mypy --strict`,
pytest with ≥ 90% coverage. Tests never touch the network.

## License

[MIT](LICENSE). Corpus and benchmark data carry their own terms — see
[docs/license-review.md](docs/license-review.md).
