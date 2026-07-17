# scripts/

Operational scripts: corpus builds, model training, benchmark reproduction, serving.
Every published number comes from a committed, seeded script here (claims hygiene).

## Corpus & data

- `ingest.py` — rebuild the SQLite document store from raw PubMed XML (+ stats).
- `build_pubmedqa.py` — build a store + frozen dev/test splits from PubMedQA PQA-L.
- `build_splits.py` — frozen dev/test splits with the checksum guard.
- `download_scale_data.py` — fetch the scale-run corpora (MS MARCO / SNLI / MultiNLI /
  PQA-A / SciNLI); see [`docs/scale-runs.md`](../docs/scale-runs.md).

## Training

- `train_tokenizer.py` — mixed-corpus BPE tokenizer (vocab sweep + fertility).
- `train_retriever.py` — dense bi-encoder (Stage-0 MLM → contrastive); `--pqa`,
  `--msmarco-triples`, `--device`.
- `train_reranker.py` — cross-encoder reranker on MS MARCO triples; `--device`.
- `train_verifier.py` — 3-class NLI verifier on SNLI/MultiNLI/SciNLI; `--eval-nli`,
  `--device`.
- `train_pubmedqa_dense.py` — leakage-free dense baseline on PubMedQA.
- `train_pubmedqa_classifier.py` — PubMedQA yes/no/maybe classifier (MLM-pretrain /
  class-weight options).
- `embed_corpus.py` — embed the corpus into an index for dense retrieval.

## Evaluation & benchmarks

- `evaluate.py` — retrieval metrics on a frozen split (bm25/dense/hybrid, ±rerank, ±ann).
- `ablate_stage0.py` — retrieval ablation (BM25/dense/reranker) + reranker-fusion check.
- `variance_dense.py` — seed-averaged Stage-0 (MLM vs random-init) study.
- `campaign_pubmedqa.py` — full retrieval campaign on PubMedQA.
- `benchmark_ann.py` — ANN recall/latency trade-off.
- `benchmark_latency.py` — per-stage serving latency (P50/P95).
- `benchmark_calibration.py` — risk-coverage / abstention curve.

## Serving

- `serve.py` — FastAPI app over the retrieval + extractive-answer pipeline.
