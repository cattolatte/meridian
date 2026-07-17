#!/usr/bin/env python
"""Clean Stage-0 ablation for dense retrieval + reranker-fusion check (real PubMedQA).

The headline campaign number conflated two changes (MLM pretraining *and* the switch to
supervised pairs). This isolates **MLM Stage-0** alone: identical corpus, split, pairs,
seed, and contrastive schedule -- only the trunk initialization changes:

    * random-init            -- no pretraining
    * mlm-2ep                -- the campaign's light pretrain
    * mlm-strong             -- more MLM epochs (does a better trunk lift retrieval?)

It also checks the reranker robustness fix: pure rerank (``base_weight=0``) vs a
base-dominant fusion, on BM25 candidates, to show graceful degradation.

    uv run python scripts/ablate_stage0.py --pqal data/pubmedqa/ori_pqal.json \\
        --out data/pubmedqa/ablation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.encoder.artifact import EmbedderConfig, build_embedder
from meridian.encoder.data import make_contrastive_samples
from meridian.encoder.embed import embed_documents
from meridian.encoder.mining import mine_hard_negatives
from meridian.encoder.pretrain import build_mlm, initialize_from_mlm, mlm_pretrain
from meridian.encoder.training import train_retriever
from meridian.eval.harness import run_evaluation
from meridian.eval.qrels import EvalQuery, EvalSet
from meridian.reranker.artifact import RerankerConfig, build_reranker
from meridian.reranker.data import make_pair_samples
from meridian.reranker.training import train_reranker
from meridian.retrieval.dense import DenseRetriever
from meridian.retrieval.embedding_index import EmbeddingIndex
from meridian.retrieval.pipeline import BM25Retriever
from meridian.retrieval.rerank import RerankingRetriever
from meridian.tokenization.training import train_tokenizer


def _bucket(pubid: str) -> int:
    return int(hashlib.sha256(f"split:{pubid}".encode()).hexdigest(), 16) % 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pqal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--strong-mlm-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = json.loads(args.pqal.read_text())
    contexts = {k: " ".join(v.get("CONTEXTS", [])) for k, v in data.items() if v.get("CONTEXTS")}
    docs = [Document(pmid=k, title="", abstract=text) for k, text in contexts.items()]
    train_ids = [k for k in contexts if _bucket(k) < 60]
    dev_ids = [k for k in contexts if 60 <= _bucket(k) < 80]
    results: dict[str, object] = {"n_train": len(train_ids), "n_dev": len(dev_ids)}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump() -> None:
        args.out.write_text(json.dumps(results, indent=2) + "\n")

    print(f"corpus {len(docs)} | train {len(train_ids)} | dev {len(dev_ids)}", flush=True)
    dump()

    with SqliteDocumentStore(":memory:") as store:
        store.add_many(docs)
        dev = EvalSet(
            "pqal-dev", tuple(EvalQuery(k, data[k]["QUESTION"], frozenset({k})) for k in dev_ids)
        )
        tokenizer = train_tokenizer(
            list(contexts.values()),
            ["a general english passage about various topics"] * 200,
            vocab_size=args.vocab_size,
        )
        vocab = tokenizer.vocabulary
        config = EmbedderConfig(
            vocab_size=vocab.size, embed_dim=args.embed_dim, num_layers=2, pad_id=vocab.pad_id or 0
        )
        pairs = [(data[k]["QUESTION"], contexts[k]) for k in train_ids]
        print("tokenizer trained", flush=True)

        bm25 = BM25Retriever.from_store(store)
        results["bm25"] = run_evaluation(bm25, dev).metrics
        print(f"bm25: {results['bm25']}", flush=True)
        dump()

        def dense_run(mlm_epochs: int | None) -> dict[str, float]:
            torch.manual_seed(args.seed)
            embedder = build_embedder(config)
            if mlm_epochs is not None:
                mlm = build_mlm(config)
                mlm_pretrain(
                    mlm,
                    list(contexts.values()),
                    tokenizer,
                    mask_id=vocab.mask_id,
                    vocab_size=vocab.size,
                    epochs=mlm_epochs,
                    seed=args.seed,
                )
                initialize_from_mlm(mlm, embedder)
            train_retriever(
                embedder,
                make_contrastive_samples(pairs, tokenizer),
                pad_id=config.pad_id,
                epochs=args.epochs,
                batch_size=32,
                max_length=256,
                seed=args.seed,
            )
            pmids, vectors = embed_documents(
                embedder, tokenizer, store.iter_documents(), max_length=256
            )
            dense = DenseRetriever(embedder, tokenizer, EmbeddingIndex.build(pmids, vectors), store)
            return run_evaluation(dense, dev).metrics

        for name, mlm_epochs in (
            ("dense_random_init", None),
            ("dense_mlm2", 2),
            (f"dense_mlm{args.strong_mlm_epochs}", args.strong_mlm_epochs),
        ):
            results[name] = dense_run(mlm_epochs)
            print(f"{name}: {results[name]}", flush=True)
            dump()

        # --- reranker fusion check: pure vs base-dominant on BM25 candidates ---
        rr_config = RerankerConfig(
            vocab_size=vocab.size, embed_dim=args.embed_dim, num_layers=2, pad_id=vocab.pad_id or 0
        )
        torch.manual_seed(args.seed)
        reranker = build_reranker(rr_config)
        triples = mine_hard_negatives(
            [bm25], store, [(data[k]["QUESTION"], k) for k in train_ids], num_negatives=2, pool=20
        )
        examples: list[tuple[str, str, int]] = []
        for q, positive, negatives in triples:
            examples.append((q, positive, 1))
            examples.extend((q, neg, 0) for neg in negatives)
        train_reranker(
            reranker,
            make_pair_samples(examples, tokenizer),
            pad_id=vocab.pad_id or 0,
            cls_id=vocab.cls_id,
            sep_id=vocab.sep_id,
            epochs=args.epochs,
            batch_size=16,
            max_length=256,
            seed=args.seed,
        )
        for name, base_weight in (("bm25_rerank_pure", 0.0), ("bm25_rerank_fused", 5.0)):
            reranked = RerankingRetriever(
                bm25, reranker, tokenizer, store, candidates=100, base_weight=base_weight
            )
            results[name] = run_evaluation(reranked, dev).metrics
            print(f"{name}: {results[name]}", flush=True)
            dump()

    print("ablation complete", flush=True)
    dump()


if __name__ == "__main__":
    main()
