#!/usr/bin/env python
"""Real, leakage-free dense-retriever run on the PubMedQA corpus, end to end.

Trains a small bi-encoder with **self-supervised** contrastive pairs derived from the
abstracts themselves (first half <-> second half of each abstract) — never the eval
questions, so there is no train-on-test leakage. Then embeds the corpus and evaluates
dense retrieval on the frozen dev split. This produces a *real* dense row; because the
only training signal is self-supervised (no MS MARCO / PQA-A), dense is expected to
trail BM25 here — the honest, plan-anticipated outcome (RAG.md §9).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from meridian.corpus.store import SqliteDocumentStore
from meridian.encoder.artifact import EmbedderConfig, build_embedder
from meridian.encoder.data import make_contrastive_samples
from meridian.encoder.embed import embed_documents
from meridian.encoder.training import train_retriever
from meridian.eval.harness import run_evaluation, write_results
from meridian.eval.splits import load_frozen_split
from meridian.retrieval.dense import DenseRetriever
from meridian.retrieval.embedding_index import EmbeddingIndex
from meridian.tokenization.training import train_tokenizer


def _self_supervised_pairs(abstracts: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for text in abstracts:
        words = text.split()
        if len(words) < 8:
            continue
        mid = len(words) // 2
        pairs.append((" ".join(words[:mid]), " ".join(words[mid:])))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checksum", required=True)
    parser.add_argument("--out", type=Path, required=True, help="results JSON")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with SqliteDocumentStore(args.db) as store:
        abstracts = [doc.chunk_text() for doc in store.iter_documents()]
        print(f"corpus: {len(abstracts)} abstracts")

        tokenizer = train_tokenizer(
            abstracts,
            ["a general english passage about various topics"] * 200,
            vocab_size=args.vocab_size,
        )
        print("tokenizer trained")

        pairs = _self_supervised_pairs(abstracts)
        print(f"self-supervised pairs: {len(pairs)}")
        config = EmbedderConfig(
            vocab_size=tokenizer.vocabulary.size,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            pad_id=tokenizer.vocabulary.pad_id or 0,
        )
        torch.manual_seed(args.seed)
        embedder = build_embedder(config)
        losses = train_retriever(
            embedder,
            make_contrastive_samples(pairs, tokenizer),
            pad_id=config.pad_id,
            epochs=args.epochs,
            batch_size=32,
            max_length=256,
            seed=args.seed,
        )
        print(f"contrastive training done; final loss {losses[-1]:.4f}")

        pmids, vectors = embed_documents(
            embedder, tokenizer, store.iter_documents(), max_length=256
        )
        index = EmbeddingIndex.build(pmids, vectors)
        retriever = DenseRetriever(embedder, tokenizer, index, store)
        print("corpus embedded; evaluating dense retrieval on dev...")

        eval_set = load_frozen_split(args.split, args.checksum)
        result = run_evaluation(retriever, eval_set)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_results(result, args.out)
    for name, value in sorted(result.metrics.items()):
        print(f"dense {name}: {value:.4f}")
    print(f"(n_queries={result.n_queries}) -> {args.out}")


if __name__ == "__main__":
    main()
