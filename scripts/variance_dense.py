#!/usr/bin/env python
"""Multi-seed variance study: does MLM Stage-0 robustly help dense retrieval?

A single-seed ablation is not enough to claim a pretraining effect when the model is
tiny and the data limited: run-to-run init/seed variance can dwarf it. This runs the
dense retriever across several seeds for two trunks -- random-init and MLM-pretrained --
on the real PubMedQA dev split, and reports per-seed Recall@5/@20 plus mean +/- std, so
any MLM claim is made against the noise floor (claims hygiene; RAG.md 9).

    uv run python scripts/variance_dense.py --pqal data/pubmedqa/ori_pqal.json \\
        --out data/pubmedqa/variance.json --seeds 0,1,2,3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.encoder.artifact import EmbedderConfig, build_embedder
from meridian.encoder.data import make_contrastive_samples
from meridian.encoder.embed import embed_documents
from meridian.encoder.pretrain import build_mlm, initialize_from_mlm, mlm_pretrain
from meridian.encoder.training import train_retriever
from meridian.eval.harness import run_evaluation
from meridian.eval.qrels import EvalQuery, EvalSet
from meridian.retrieval.dense import DenseRetriever
from meridian.retrieval.embedding_index import EmbeddingIndex
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
    parser.add_argument("--mlm-epochs", type=int, default=2)
    parser.add_argument("--seeds", type=str, default="0,1,2,3")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    data = json.loads(args.pqal.read_text())
    contexts = {k: " ".join(v.get("CONTEXTS", [])) for k, v in data.items() if v.get("CONTEXTS")}
    docs = [Document(pmid=k, title="", abstract=text) for k, text in contexts.items()]
    train_ids = [k for k in contexts if _bucket(k) < 60]
    dev_ids = [k for k in contexts if 60 <= _bucket(k) < 80]
    results: dict[str, object] = {"n_train": len(train_ids), "n_dev": len(dev_ids), "seeds": seeds}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump() -> None:
        args.out.write_text(json.dumps(results, indent=2) + "\n")

    print(f"corpus {len(docs)} | train {len(train_ids)} | dev {len(dev_ids)} | seeds {seeds}")
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

        def dense_r5(seed: int, pretrain: bool) -> tuple[float, float]:
            torch.manual_seed(seed)
            embedder = build_embedder(config)
            if pretrain:
                mlm = build_mlm(config)
                mlm_pretrain(
                    mlm,
                    list(contexts.values()),
                    tokenizer,
                    mask_id=vocab.mask_id,
                    vocab_size=vocab.size,
                    epochs=args.mlm_epochs,
                    seed=seed,
                )
                initialize_from_mlm(mlm, embedder)
            train_retriever(
                embedder,
                make_contrastive_samples(pairs, tokenizer),
                pad_id=config.pad_id,
                epochs=args.epochs,
                batch_size=32,
                max_length=256,
                seed=seed,
            )
            pmids, vectors = embed_documents(
                embedder, tokenizer, store.iter_documents(), max_length=256
            )
            dense = DenseRetriever(embedder, tokenizer, EmbeddingIndex.build(pmids, vectors), store)
            metrics = run_evaluation(dense, dev).metrics
            return metrics["recall@5"], metrics["recall@20"]

        for arm in ("random_init", "mlm"):
            r5s: list[float] = []
            r20s: list[float] = []
            for seed in seeds:
                r5, r20 = dense_r5(seed, pretrain=(arm == "mlm"))
                r5s.append(r5)
                r20s.append(r20)
                print(f"{arm} seed={seed} R@5={r5:.3f} R@20={r20:.3f}", flush=True)
                results[arm] = {
                    "recall@5": r5s,
                    "recall@20": r20s,
                    "r5_mean": statistics.mean(r5s),
                    "r5_std": statistics.pstdev(r5s) if len(r5s) > 1 else 0.0,
                    "r20_mean": statistics.mean(r20s),
                    "r20_std": statistics.pstdev(r20s) if len(r20s) > 1 else 0.0,
                }
                dump()

    print("variance study complete", flush=True)
    dump()


if __name__ == "__main__":
    main()
