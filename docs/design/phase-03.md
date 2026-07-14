# Phase 3 — Dense retriever: Polaris bi-encoder + contrastive training (design doc)

- **Release:** v0.2.0
- **Goal:** a trained embedding model and dense retrieval wired in behind the same
  CLI (brute-force search for now), with an honest ablation table vs BM25.
- **Upstream:** consumes `polaris-nlp==1.2.0` (`TextEmbedder`, `mean_pool`,
  `transfer_encoder_to`, `collate_contrastive`, `info_nce_loss`, `train_contrastive`).
- **Governing ADR:** [0004](../adr/0004-training-curriculum.md) (training curriculum).

## Module layout (`src/meridian/`)

```
encoder/
  embed.py        embed_documents(): batch-encode chunk text -> float32 matrix
                  using a Polaris TextEmbedder + the trained tokenizer.
  training.py     Stage 0 (MLM) / Stage A/B (contrastive) pipelines wrapping
                  polaris.pretraining.pretrain and polaris.training.train_contrastive;
                  seeded, checkpointed. (Milestone 2.)
  data.py         Build contrastive samples: MS MARCO triples + mined
                  title↔abstract / PQA-A pairs. (Milestone 2.)
retrieval/
  embedding_index.py  EmbeddingIndex: memory-mapped float32 embedding shards + a
                      PMID map + brute-force top-k search (cosine via dot product on
                      normalized vectors). The Phase-4 ANN indexes slot in behind the
                      same search interface.
  dense.py            DenseRetriever(Retriever): encode query -> brute-force search
                      -> RetrievalHits, resolving PMIDs to Documents.
```

Dense retrieval is **additive**: the CLI keeps BM25 as the default; dense is selected
explicitly (a trained embedder + a built embedding index must exist). Main stays
runnable end-to-end after every merge (vertical slice).

## Milestones

1. **Dense retrieval infrastructure** *(this milestone)* — `EmbeddingIndex` (brute-force
   search over memory-mapped shards) + `embed_documents` + `DenseRetriever`, verified
   offline with a small randomly-initialized `TextEmbedder`. No training yet; proves the
   query→embed→search→cite path against the `Retriever` protocol.
2. **Training pipelines** — Stage 0 MLM pretrain (wrap `pretrain`), Stage A/B contrastive
   (wrap `train_contrastive`), data builders, seeded configs, checkpoint artifacts.
3. **Scripts + CLI + ablation** — `scripts/train_retriever.py`, `scripts/embed_corpus.py`,
   `meridian ask --retriever dense`, and the `{random vs Stage-0}` / `{A vs A+B}`
   ablation harness → dense row in `BENCHMARKS.md`.

## Key decisions

- **Normalized embeddings + dot product.** `TextEmbedder(normalize=True)` emits unit
  vectors, so cosine similarity is a plain matmul — the brute-force ground truth the
  Phase-4 ANN indexes are measured against.
- **Memory-mapped float32 shards.** The corpus embedding matrix is stored as `.npy`
  memmaps so a 200K×D matrix never needs to fit in RAM at query time (RAG.md §4.1).
- **Search interface is index-agnostic.** `EmbeddingIndex.search(query_vec, k)` is the
  seam; brute force now, IVF/HNSW in Phase 4, no caller changes.
- **One trunk, reused.** The Stage-0 MLM checkpoint is kept for Phases 6/8 (ADR-0004).

## Testing strategy (offline-only)

- Brute-force search correctness against a hand-computed nearest-neighbour on tiny
  vectors; determinism; top-k and tie-breaking; memmap round-trip.
- `embed_documents` shape/normalization with a small random `TextEmbedder`; empty-corpus
  guard.
- `DenseRetriever` satisfies the `Retriever` protocol and returns the nearest document
  first on a separable toy corpus (seeded random embedder is enough to exercise the path).
- Training pipelines (Milestone 2) tested with tiny models/epochs: loss decreases,
  checkpoints round-trip, Stage-0 transfer copies the trunk.

## Environment constraint (honest note)

Real embeddings and the ablation table require the ingested PubMed corpus, an MS MARCO
sample, and MPS/CUDA training — deliberate networked/heavy steps the user runs. All code
and the search path are verified offline with small synthetic models; the dense row in
`BENCHMARKS.md` stays TBD until a real training run, and no metric is written from memory.

## Exit criteria tracking

| Criterion | Status |
|---|---|
| Reproducible training config + seeds | done (seeded `EmbedderConfig` + scripts; Hydra deferred — configs are plain dataclasses the scripts drive) |
| Dense retrieval live behind the same CLI | **done** — `meridian ask --retriever dense` verified end-to-end on the sample corpus |
| Ablation table committed | mechanism done (`--random-init` + `scripts/evaluate.py --retriever dense`); real numbers pending training run |
| ≥ 90% coverage held | maintained (97%) |

## Remaining user-triggered step

The full curriculum, dense search, artifacts, scripts, and CLI wiring are complete and
verified offline (train_retriever → embed_corpus → `meridian ask --retriever dense`
runs end-to-end on the sample corpus with a tiny model). Outstanding work needs the
real corpus + MS MARCO + MPS/CUDA compute:

1. `scripts/train_retriever.py` with real MS MARCO pairs (Stage A) + `--mine-title-abstract`
   (Stage B), full-size (`--embed-dim 256 --num-layers 4`).
2. `scripts/embed_corpus.py` to build the corpus index.
3. `scripts/evaluate.py --retriever dense` on the frozen dev split → dense row in
   `BENCHMARKS.md`; run with/without `--random-init` and Stage B for the ADR-0004
   ablations. If dense < BM25, publish and diagnose (honesty clause).

**Note on Hydra:** the plan named Hydra for training config; Phase 3 uses seeded
dataclass configs driven by argparse scripts, which meets "reproducible config + seeds"
without the Hydra dependency. If sweep orchestration is needed later, Hydra can wrap the
same config objects — recorded here as a deliberate, minor deviation.
