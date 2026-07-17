# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Phase 12 — v1.0 docs & demo (release gated on the real training campaign).**
  - README: architecture diagram (online path), serving quickstart, a guardrails
    section, and an honest status note; MODEL_CARD component statuses updated to
    "implemented (untrained on real data)". The v1.0.0 / PyPI release is deliberately
    deferred until the real benchmark numbers exist (claims hygiene + release rules).
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

## [0.0.1] - 2026-07-13

### Added

- Repository scaffolding: `src/` layout, offline-only test suite, uv project,
  Black/Ruff/strict-mypy/pytest configuration with a ≥ 90% coverage gate.
- CI: GitHub Actions matrix (Python 3.12/3.13 × Ubuntu/macOS) running lint,
  format check, typecheck, and tests.
- ADR-0001 (project scope), dataset license review memo, ADR process docs.
- Community files: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue/PR templates.
- Pinned upstream frameworks: `polaris-nlp==1.1.0`, `zenith-nlp==1.0.0`.
