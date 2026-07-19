# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026-07-19

### Fixed

- **Package metadata for the first PyPI publication.** `Development Status` was still
  `2 - Pre-Alpha`, which contradicted a 1.0 release on the rendered PyPI page; it is now
  `4 - Beta` (honest for a laptop-scale, production-*inspired* project). Added a
  `Documentation` URL pointing at BENCHMARKS.md.

## [1.0.0] - 2026-07-19

First stable release. All 12 build phases are implemented and **every benchmark section
carries real measured numbers** — the release gate set in Phase 12 (no `TBD` cells, no
estimated metrics). Scope is honest: laptop-scale, production-*inspired*, with three
items explicitly not run (the ~200K PubMed corpus, the Phase-7 generator pipeline, and
verifier–human agreement) rather than estimated.

### Added

- **Full benchmark sweep — every section measured, no `TBD` cells.** ANN on the real index
  (HNSW recall@10 0.996 @ 0.262 ms, with `tracemalloc` memory reporting); NLI verifier
  quality (SNLI dev **0.783**, chance 0.333) with the tokenizer/data/learning-rate ablation;
  faithfulness via the new `scripts/benchmark_faithfulness.py` (citation precision 0.751,
  hallucination 0.437 — the verifier's *out-of-domain* error, since extractive answers quote
  verbatim); real Gate-1 calibration (0.801 coverage @ 0.000 error); and per-stage latency
  with all trained artifacts (rerank 421 ms vs BM25 1.33 ms). Measured JSON is committed in
  `benchmarks/results/` and the charts regenerate from it via `scripts/plot_summary.py`
  (`retrieval_ablation`, `verifier_progression`, `latency_breakdown`). Items that were not
  run — ~200K PubMed corpus, generator pipeline, verifier–human agreement — say so instead
  of carrying an estimate.
- **Verifier early stopping + LR control.** `train_verifier.py --eval-every N` evaluates a
  held-out set each N epochs and keeps the **best** checkpoint (it caught the peak epoch
  where a long run would otherwise have shipped an overfit final epoch); `--learning-rate`
  exposes the knob that took the 384×6 model from 0.607 to 0.783.
- **Reranker trainable without MS MARCO.** `train_reranker.py --pqal` mines BM25 hard
  negatives from PubMedQA, unblocking the reranker (and its latency benchmark) without the
  gated multi-GB download.
- **Real benchmark campaign on PubMedQA PQA-L.** A self-contained 1000-abstract retrieval
  task (clean train/dev/test, no leakage): BM25 vs from-scratch dense vs reranker, a
  seed-averaged Stage-0 ablation (`scripts/ablate_stage0.py`, `scripts/variance_dense.py`),
  the PubMedQA yes/no/maybe classifier (`scripts/train_pubmedqa_classifier.py`, with
  MLM-pretrain and class-weight options), and per-stage latency
  (`scripts/benchmark_latency.py`). Real numbers now populate BENCHMARKS/README.
- **GPU / accelerator support + scale-run wiring.** `meridian.device.resolve_device`
  (CUDA/MPS/CPU) threaded through the trainers and the encoder; `meridian.data.scale`
  loaders (MS MARCO / PubMedQA PQA-A / SNLI-MultiNLI-SciNLI); `scripts/train_reranker.py`,
  `scripts/train_verifier.py` (with `--eval-nli` self-scoring), `--pqa`/`--msmarco-triples`
  on the retriever driver, `scripts/download_scale_data.py`, and the `docs/scale-runs.md`
  runbook. Verified training on Apple MPS.
- **Phase 12 — v1.0 docs & demo (release gated on the real training campaign).**
  - README: architecture diagram (online path), serving quickstart, a guardrails
    section, and an honest status note; MODEL_CARD component statuses. The v1.0.0 release
    was gated on real benchmark numbers existing (claims hygiene + release rules) — that
    gate is satisfied by the measured campaign above, which is why 1.0.0 ships now (PyPI
    publication follows in 1.0.1).
- **Phase 11 — evaluation campaign + error attribution.**
  - `meridian.eval.attribution`: oracle-substitution error attribution
    (`attribute_failure` / `attribution_study`) blaming each wrong answer on the
    earliest pipeline stage an oracle repairs.
  - `meridian.eval.pubmedqa.pubmedqa_accuracy`: the yes/no/maybe headline scorer.
  - `docs/design/error-attribution.md` (method + pending real results).
- **Phase 10 — serving & performance.**
  - `meridian.serving`: a FastAPI app (`/health`, `/passages`, `/ask`, SSE
    `/ask/stream`, `/metrics`) over the retrieval + extractive-answer pipeline, with
    per-stage latency instrumentation (`StageTimer`).
  - `scripts/serve.py`, `Dockerfile`, `docker-compose.yml` (api + demo UI + Postgres
    profile), and a zero-framework `demo/index.html` (clickable citations,
    GROUNDED/ABSTAIN badge, disclaimer). `serving` optional extra;
    `SqliteDocumentStore` now serves a threaded server.
- **Phase 9 — abstention & calibration.**
  - `meridian.abstain`: Gate 1 `RetrievalGate` (top score + top-1/top-k margin);
    Gate 2 answerability classifier (Polaris `SentencePairClassifier`,
    `num_classes=2`) config/artifact/data/training + `answerable_probability`;
    `risk_coverage_curve`/`operating_point` for selective answering; ADR-0007.
- **Phase 8 — faithfulness verifier (NLI) + citation checking.**
  - `meridian.verify`: NLI verifier (Polaris `SentencePairClassifier`,
    `num_classes=3`) config/artifact, `make_nli_samples`, 3-class `train_verifier`;
    `verify_grounded_answer` (per-sentence entailment of cited spans) with citation
    precision / recall / hallucination-rate metrics; and `answer_with_verification`
    — the fail-safe ladder (generate → verify → extractive fallback → abstain).
- **Phase 7 — grounded generator (Zenith).**
  - Bumps the pin to `zenith-nlp==1.1.0` (constrained-decoding hook, `AllowedTokens`,
    `Generator.abstained`, `instruct.grounded`).
  - `meridian.generation`: `GeneratorConfig`/artifact (Zenith `DecoderLM`, LoRA-aware
    save/load), grounded-SFT example builders, `train_generator` (LoRA SFT via
    `CausalLMTrainer`), and `answer_grounded` — retrieve → citation-constrained decode
    (a `[n]` can only reference a retrieved passage) → parse citations to PMIDs, or
    abstain.
  - CLI `meridian ask --answerer generated --generator DIR`; ADR-0006 (answer format).
  - Extractive answering remains the default fallback.
- **Phase 6 — cross-encoder reranker.**
  - Bumps the pin to `polaris-nlp==1.4.0` (SentencePairClassifier, collate_pairs,
    special-token reservation, native BPETokenizer save/load).
  - `meridian.tokenization`: the tokenizer now reserves `<mask>`/`<cls>`/`<sep>` so
    one vocabulary serves the MLM, embedder, and pair models.
  - `meridian.reranker`: `SentencePairClassifier` (num_classes=1) config/artifact,
    pointwise pair-sample builders, and BCE `train_reranker`.
  - `meridian.retrieval.rerank.RerankingRetriever`: rerank a base retriever's
    top-N candidates with the cross-encoder; `meridian ask --rerank --reranker DIR`
    and `scripts/evaluate.py --rerank`.
- **Phase 5 — hard negatives, retriever v2, hybrid retrieval.**
  - `meridian.encoder.mining.mine_hard_negatives`: pool top non-relevant passages
    from BM25 + dense into `(anchor, positive, negatives)` triples for retraining.
  - `meridian.retrieval.hybrid.HybridRetriever`: reciprocal rank fusion of BM25 +
    dense; wired as `meridian ask --retriever hybrid` and in `scripts/evaluate.py`.
  - `meridian.eval.sample_misses` + `docs/design/failure-taxonomy.md` scaffold.
- **Phase 4 — ANN index from scratch (IVF + HNSW).**
  - `meridian.retrieval.ann`: `VectorIndex` protocol; seeded Lloyd k-means
    (k-means++ init); `IVFIndex` (nlist cells, nprobe search, exact rerank);
    `HNSWIndex` (layered navigable small-world graph, greedy build/search); a
    `build_ann_index` dispatcher. No external ANN library.
  - `DenseRetriever` now searches any backend; CLI `meridian ask --ann
    {none,ivf,hnsw}` and `scripts/evaluate.py --ann`.
  - `scripts/benchmark_ann.py` + committed recall/latency figure; ADR-0005
    (default backend). `matplotlib` added as the optional `benchmark` extra.
- **Phase 3 — dense retriever: Polaris bi-encoder + contrastive training.**
  - Bumps the pin to `polaris-nlp==1.2.0`; adds `torch` as a direct dependency.
  - `meridian.retrieval`: brute-force `EmbeddingIndex` (memory-mapped float32
    shards, exact cosine top-k — the Phase-4 ANN ground truth) and
    `DenseRetriever` behind the `Retriever` protocol; a `build_retriever` factory.
  - `meridian.encoder`: corpus embedding, the ADR-0004 curriculum — Stage-0 MLM
    pretraining + trunk transfer, Stage A/B contrastive training (InfoNCE),
    contrastive-sample builders — and a versioned embedder artifact.
  - `meridian.tokenization.special_tokens`: append a `<mask>` token without
    renumbering existing ids (ADR-0003 superseding note).
  - CLI `meridian ask --retriever dense`; `scripts/train_retriever.py`,
    `scripts/embed_corpus.py`, and dense support in `scripts/evaluate.py`.
  - ADR-0004 (training curriculum) and the Phase 3 design doc.
- **Phase 2 — BM25 baseline + eval harness + extractive E2E v0.**
  - `meridian.retrieval`: from-scratch BM25 Okapi (inverted index, search-time
    k1/b, deterministic ranking) behind a `Retriever` protocol, with an injectable
    text analyzer.
  - `meridian.eval`: binary-relevance metrics (Recall@k, MRR@10, nDCG@10),
    EvalSet/qrels types, frozen dev/test splits with a SHA-256 checksum guard
    (house rule #4), a JSON-writing runner with optional MLflow logging, and
    PubMedQA split construction.
  - `meridian.answer`: extractive answerer v0 — verbatim cited sentences or
    abstain — and its CLI rendering.
  - `meridian` CLI (`ask`, `ingest`) as the console entry point;
    `scripts/evaluate.py` and `scripts/build_splits.py`; `benchmarks/splits/`
    sample split; `mlflow` optional `tracking` extra.
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

### Changed

- **Corrected the retrieval headline (claims hygiene).** A clean, seed-averaged ablation
  showed MLM Stage-0 pretraining is *within run-to-run noise* on this task (dense R@5
  0.371 ± 0.022 vs random-init 0.382 ± 0.023); the robust ~38× lever is the switch to
  supervised question→abstract pairs, not pretraining. An earlier draft over-credited the
  MLM "curriculum"; BENCHMARKS/README/MODEL_CARD rewritten to match the measurement.

### Fixed

- **`verify_grounded_answer` crashed on GPU/MPS** — it never moved its collated batch to
  the model's device. It now follows the model's device (and returns predictions on CPU).
- **Reranker catastrophe → graceful degradation.** A from-scratch cross-encoder overfits
  limited data and, applied purely, dragged BM25's R@5 below random (0.029). `RerankingRetriever`
  now supports reciprocal-rank fusion with the base (`base_weight`) and breaks ties by base
  rank, recovering R@5 to 0.983; exposed via the retriever factory.
- **`.gitignore`** `data/` matched at any depth and hid the new `src/meridian/data` package;
  anchored to `/data/` so datasets stay ignored but the package is tracked.

## [0.0.1] - 2026-07-13

### Added

- Repository scaffolding: `src/` layout, offline-only test suite, uv project,
  Black/Ruff/strict-mypy/pytest configuration with a ≥ 90% coverage gate.
- CI: GitHub Actions matrix (Python 3.12/3.13 × Ubuntu/macOS) running lint,
  format check, typecheck, and tests.
- ADR-0001 (project scope), dataset license review memo, ADR process docs.
- Community files: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue/PR templates.
- Pinned upstream frameworks: `polaris-nlp==1.1.0`, `zenith-nlp==1.0.0`.
