#!/usr/bin/env python
"""Full real retrieval campaign on PubMedQA with a proper train/dev/test split.

Trains the actual ADR-0004 curriculum on real data — **no leakage** (dense/reranker see
only TRAIN questions; dev/test questions are held out) — and reports BM25 vs dense vs
hybrid vs +rerank on the frozen dev split. Writes results incrementally so progress is
visible while it runs.

    uv run python scripts/campaign_pubmedqa.py --pqal data/pubmedqa/ori_pqal.json \\
        --out data/pubmedqa/campaign.json
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
from meridian.retrieval.hybrid import HybridRetriever
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
    parser.add_argument("--mlm-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
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
            "pqal-dev",
            tuple(EvalQuery(k, data[k]["QUESTION"], frozenset({k})) for k in dev_ids),
        )

        tokenizer = train_tokenizer(
            list(contexts.values()),
            ["a general english passage about various topics"] * 200,
            vocab_size=args.vocab_size,
        )
        vocab = tokenizer.vocabulary
        print("tokenizer trained", flush=True)

        bm25 = BM25Retriever.from_store(store)
        results["bm25"] = run_evaluation(bm25, dev).metrics
        print(f"BM25: {results['bm25']}", flush=True)
        dump()

        # --- dense: MLM Stage-0 pretrain -> supervised contrastive on TRAIN pairs ---
        config = EmbedderConfig(
            vocab_size=vocab.size, embed_dim=args.embed_dim, num_layers=2, pad_id=vocab.pad_id or 0
        )
        torch.manual_seed(args.seed)
        mlm = build_mlm(config)
        mlm_pretrain(
            mlm,
            list(contexts.values()),
            tokenizer,
            mask_id=vocab.mask_id,
            vocab_size=vocab.size,
            epochs=args.mlm_epochs,
            seed=args.seed,
        )
        embedder = build_embedder(config)
        initialize_from_mlm(mlm, embedder)
        print("MLM pretrained + transferred to embedder", flush=True)

        pairs = [(data[k]["QUESTION"], contexts[k]) for k in train_ids]  # supervised q->abstract
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
        index = EmbeddingIndex.build(pmids, vectors)
        dense = DenseRetriever(embedder, tokenizer, index, store)
        results["dense"] = run_evaluation(dense, dev).metrics
        print(f"dense: {results['dense']}", flush=True)
        dump()

        hybrid = HybridRetriever([bm25, dense], store)
        results["hybrid"] = run_evaluation(hybrid, dev).metrics
        print(f"hybrid: {results['hybrid']}", flush=True)
        dump()

        # --- reranker: init from Stage-0 trunk, train on TRAIN pairs + BM25 hard negs ---
        rr_config = RerankerConfig(
            vocab_size=vocab.size, embed_dim=args.embed_dim, num_layers=2, pad_id=vocab.pad_id or 0
        )
        reranker = build_reranker(rr_config)
        mlm.transfer_encoder_to(reranker)
        train_queries = [(data[k]["QUESTION"], k) for k in train_ids]
        triples = mine_hard_negatives([bm25], store, train_queries, num_negatives=2, pool=20)
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
        for base_name, base in (("bm25", bm25), ("hybrid", hybrid)):
            reranked = RerankingRetriever(base, reranker, tokenizer, store, candidates=100)
            results[f"{base_name}+rerank"] = run_evaluation(reranked, dev).metrics
            print(f"{base_name}+rerank: {results[f'{base_name}+rerank']}", flush=True)
            dump()

    print("campaign complete", flush=True)
    dump()


if __name__ == "__main__":
    main()
