# Scale runs — training Meridian's models on the full corpora

The committed benchmarks are laptop-scale (PubMedQA PQA-L, ~590–800 examples). This doc is
the runbook for the **scale runs** the plan budgets a rented GPU for (ADR-0001): training
the retriever, reranker, and NLI verifier on the heavy public corpora. Everything here is
wired and smoke-tested on CPU/MPS; the actual multi-hour training runs on a GPU
(`--device cuda`).

> **Metrics are `TBD` until run.** No number from a scale run appears in
> [BENCHMARKS.md](../benchmarks/BENCHMARKS.md) until a committed, seeded run produces it
> (RAG.md §5). This doc describes *how*, not results.

## What you need

| Model | Corpus | Direct download? | License |
|---|---|---|---|
| Retriever (Stage A) | MS MARCO `triples.train.small.tsv` | manual (MS MARCO page) | research-only |
| Retriever (Stage B) | PubMedQA **PQA-A** `ori_pqaa.json` | manual (Google Drive) | see license-review |
| Reranker | MS MARCO triples (same as above) | — | research-only |
| Verifier (base) | SNLI + MultiNLI `*_train.jsonl` | yes | permissive |
| Verifier (adapt) | SciNLI train split | manual (request) | research-only |
| Corpus/index | 200K–1M PubMed abstracts | via `ingest.py` | NLM terms |

Licenses are summarised in [license-review.md](license-review.md). MS MARCO and SciNLI are
**research-only** — do not use for commercial purposes.

## 1. Download

```bash
uv run python scripts/download_scale_data.py --out data/scale
```

Fetches SNLI, MultiNLI, and PQA-L directly; prints exact manual steps for MS MARCO, PQA-A,
and SciNLI (form/Drive-gated). It records the SHA-256 of each file — keep those next to any
number you later publish.

## 2. Train the tokenizer (once)

```bash
uv run python scripts/train_tokenizer.py --corpus data/scale/... --out artifacts/tokenizer.json
```

## 3. Train each model (GPU)

All three drivers take `--device cuda` (or `auto` → CUDA/MPS/CPU) and a `--max-*` cap for a
smoke run. Drop the cap for the real run and raise `--embed-dim`/`--num-layers`/`--epochs`.

```bash
# Retriever — Stage-0 MLM on the corpus, then contrastive on MS MARCO + PQA-A pairs
uv run python scripts/train_retriever.py --db data/corpus.sqlite \
    --tokenizer artifacts/tokenizer.json \
    --msmarco-triples data/scale/triples.train.small.tsv \
    --pqa data/scale/ori_pqaa.json \
    --out artifacts/embedder --embed-dim 256 --num-layers 4 --epochs 3 --device cuda

# Reranker — cross-encoder on MS MARCO triples
uv run python scripts/train_reranker.py --tokenizer artifacts/tokenizer.json \
    --msmarco-triples data/scale/triples.train.small.tsv \
    --out artifacts/reranker --embed-dim 256 --num-layers 4 --epochs 2 --device cuda

# Verifier — SNLI + MultiNLI base, then SciNLI domain adaptation
uv run python scripts/train_verifier.py --tokenizer artifacts/tokenizer.json \
    --nli data/scale/snli_1.0/snli_1.0_train.jsonl \
    --nli data/scale/multinli_1.0/multinli_1.0_train.jsonl \
    --out artifacts/verifier --embed-dim 256 --num-layers 4 --epochs 2 --device cuda
```

Smoke-check any driver first with tiny caps on your laptop, e.g. add
`--max-examples 2000 --device auto --epochs 1`.

## 4. Build the index and evaluate

```bash
uv run python scripts/embed_corpus.py --embedder artifacts/embedder ...   # build embeddings
uv run python scripts/evaluate.py ...                                     # metrics on frozen dev
```

Write the resulting numbers into BENCHMARKS.md with the run/seed that produced them.

## Notes & honest caveats

- **Shared trunk (optimization, not yet wired).** Each driver currently MLM-pretrains its
  own trunk. The plan's "one Stage-0 trunk, four heads" would pretrain once and transfer;
  only `MaskedLanguageModel.transfer_encoder_to` is exposed today, so a shared-trunk
  checkpoint step is the natural next improvement.
- **The ceiling is still a small model.** These runs bring the models to *useful* small-model
  quality (a dense retriever that beats BM25 on vocabulary-mismatched queries, a reranker
  that helps, a working NLI verifier) — **not** GPT-4-class accuracy. Encoders are ~10–30M,
  the generator ~30–125M by design (RAG.md §2).
- **Generator (Zenith) SFT** is not covered here yet — the grounded-generation SFT on PQA-A
  is a separate driver (Phase 7) still to be wired.
