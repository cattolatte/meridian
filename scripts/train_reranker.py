#!/usr/bin/env python
"""Train the cross-encoder reranker at scale (ADR-0004 Phase 5/6).

Consumes MS MARCO ``triples.train.small.tsv`` (query, positive, negative) directly:
optional Stage-0 MLM pretrain on the passages, then pointwise cross-encoder training on
the expanded pairs. Writes a versioned reranker artifact. Runs on a GPU with ``--device
cuda`` (scale) or CPU/MPS for a smoke run.

Example (smoke)
---------------
    uv run python scripts/train_reranker.py --tokenizer artifacts/tokenizer.json \\
        --msmarco-triples data/scale/triples.train.small.tsv --max-examples 2000 \\
        --out artifacts/reranker --device auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from meridian.data import load_msmarco_triples
from meridian.device import resolve_device
from meridian.encoder.artifact import EmbedderConfig
from meridian.encoder.pretrain import build_mlm, initialize_from_mlm, mlm_pretrain
from meridian.reranker.artifact import RerankerConfig, build_reranker, save_reranker
from meridian.reranker.data import make_pair_samples, pairs_from_triples
from meridian.reranker.training import train_reranker
from meridian.tokenization.artifact import load_tokenizer
from meridian.tokenization.special_tokens import ensure_mask_token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output reranker artifact dir")
    parser.add_argument("--msmarco-triples", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, help="cap triples (smoke runs)")
    parser.add_argument("--random-init", action="store_true", help="skip Stage-0 MLM (ablation)")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--mlm-epochs", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device: {device}")
    tokenizer, mask_id = ensure_mask_token(load_tokenizer(args.tokenizer))
    pad_id = tokenizer.vocabulary.pad_id or 0

    triples = load_msmarco_triples(args.msmarco_triples, max_examples=args.max_examples)
    if not triples:
        parser.error(f"no triples read from {args.msmarco_triples}")
    examples = pairs_from_triples(triples)
    print(f"{len(triples)} triples -> {len(examples)} pointwise pairs")

    rr_config = RerankerConfig(
        vocab_size=tokenizer.vocabulary.size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        pad_id=pad_id,
    )
    torch.manual_seed(args.seed)
    reranker = build_reranker(rr_config)
    if not args.random_init:
        passages = list({p for _, p, _ in triples} | {n for _, _, n in triples})
        mlm = build_mlm(
            EmbedderConfig(
                vocab_size=tokenizer.vocabulary.size,
                embed_dim=args.embed_dim,
                num_layers=args.num_layers,
                pad_id=pad_id,
            )
        )
        mlm_pretrain(
            mlm,
            passages,
            tokenizer,
            mask_id=mask_id,
            vocab_size=tokenizer.vocabulary.size,
            epochs=args.mlm_epochs,
            device=device,
            seed=args.seed,
        )
        initialize_from_mlm(mlm, reranker)
        print(f"stage 0: MLM-pretrained trunk on {len(passages)} passages")

    losses = train_reranker(
        reranker,
        make_pair_samples(examples, tokenizer),
        pad_id=pad_id,
        cls_id=tokenizer.vocabulary.cls_id,
        sep_id=tokenizer.vocabulary.sep_id,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
    )
    print(f"trained on {len(examples)} pairs; final loss {losses[-1]:.4f}")

    save_reranker(
        reranker,
        rr_config,
        args.out,
        metadata={"num_pairs": len(examples), "epochs": args.epochs, "final_loss": losses[-1]},
    )
    print(f"wrote reranker artifact -> {args.out}")


if __name__ == "__main__":
    main()
