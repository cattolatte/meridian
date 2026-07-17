#!/usr/bin/env python
"""Train the dense bi-encoder retriever (ADR-0004 curriculum).

Stage 0 (optional): MLM-pretrain the trunk on the corpus. Stage A/B: contrastive
training on the supplied pairs. The `--random-init` flag skips Stage 0 for the
ablation baseline. Writes a versioned embedder artifact.

Pairs are a TSV file of ``anchor<TAB>positive`` lines. ADR-0004: Stage A uses MS MARCO
(query, passage) pairs; Stage B uses self-mined (title, abstract) and PQA-A pairs.
With ``--mine-title-abstract`` the store's own (title, abstract) pairs are added.

Example
-------
    uv run python scripts/train_retriever.py --db data/corpus.sqlite \\
        --tokenizer artifacts/tokenizer.json --pairs data/msmarco_pairs.tsv \\
        --mine-title-abstract --out artifacts/embedder --embed-dim 256 --epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from meridian.corpus.store import SqliteDocumentStore
from meridian.data import load_msmarco_triples, load_pqa_pairs, msmarco_pairs_from_triples
from meridian.device import resolve_device
from meridian.encoder.artifact import EmbedderConfig, build_embedder, save_embedder
from meridian.encoder.data import make_contrastive_samples, mine_title_abstract_pairs
from meridian.encoder.pretrain import build_mlm, initialize_from_mlm, mlm_pretrain
from meridian.encoder.training import train_retriever
from meridian.tokenization.artifact import load_tokenizer
from meridian.tokenization.special_tokens import ensure_mask_token


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        if "\t" in line:
            anchor, positive = line.split("\t", 1)
            if anchor.strip() and positive.strip():
                pairs.append((anchor.strip(), positive.strip()))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output embedder artifact dir")
    parser.add_argument("--pairs", type=Path, help="TSV of anchor<TAB>positive contrastive pairs")
    parser.add_argument(
        "--pqa", type=Path, help="PubMedQA JSON (PQA-A/L) -> question/context pairs"
    )
    parser.add_argument(
        "--msmarco-triples", type=Path, help="MS MARCO triples TSV -> (query, positive) pairs"
    )
    parser.add_argument("--max-pairs", type=int, help="cap examples per source (smoke runs)")
    parser.add_argument(
        "--mine-title-abstract", action="store_true", help="add (title, abstract) pairs"
    )
    parser.add_argument("--random-init", action="store_true", help="skip Stage-0 MLM (ablation)")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--mlm-epochs", type=int, default=1)
    parser.add_argument(
        "--device", default="auto", help="auto | cpu | cuda | mps (scale runs: cuda)"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device: {device}")
    tokenizer, mask_id = ensure_mask_token(load_tokenizer(args.tokenizer))
    config = EmbedderConfig(
        vocab_size=tokenizer.vocabulary.size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        pad_id=tokenizer.vocabulary.pad_id or 0,
    )

    pairs: list[tuple[str, str]] = []
    if args.pairs is not None:
        pairs += _read_pairs(args.pairs)
    if args.pqa is not None:
        pairs += load_pqa_pairs(args.pqa, max_examples=args.max_pairs)
    if args.msmarco_triples is not None:
        triples = load_msmarco_triples(args.msmarco_triples, max_examples=args.max_pairs)
        pairs += msmarco_pairs_from_triples(triples)
    with SqliteDocumentStore(args.db) as store:
        if args.mine_title_abstract:
            pairs += mine_title_abstract_pairs(store.iter_documents())
        corpus_texts = [doc.chunk_text() for doc in store.iter_documents()]
    if not pairs:
        parser.error("no training pairs (pass --pairs / --pqa / --msmarco-triples / --mine-*)")

    torch.manual_seed(args.seed)
    embedder = build_embedder(config)
    if not args.random_init:
        mlm = build_mlm(config)
        mlm_pretrain(
            mlm,
            corpus_texts,
            tokenizer,
            mask_id=mask_id,
            vocab_size=config.vocab_size,
            epochs=args.mlm_epochs,
            device=device,
            seed=args.seed,
        )
        initialize_from_mlm(mlm, embedder)
        print(f"stage 0: MLM-pretrained trunk on {len(corpus_texts)} documents")

    samples = make_contrastive_samples(pairs, tokenizer)
    losses = train_retriever(
        embedder, samples, pad_id=config.pad_id, epochs=args.epochs, device=device, seed=args.seed
    )
    print(f"contrastive training on {len(samples)} pairs; final loss {losses[-1]:.4f}")

    save_embedder(
        embedder,
        config,
        args.out,
        metadata={
            "random_init": args.random_init,
            "num_pairs": len(samples),
            "epochs": args.epochs,
            "final_loss": losses[-1],
        },
    )
    print(f"wrote embedder artifact -> {args.out}")


if __name__ == "__main__":
    main()
