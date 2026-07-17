#!/usr/bin/env python
"""Train the NLI faithfulness verifier at scale (ADR-0004 Phase 8).

Consumes SNLI/MultiNLI (and SciNLI) JSONL directly: optional Stage-0 MLM pretrain on the
sentences, then 3-class NLI training (entail / neutral / contradict). Pass ``--nli`` once
per file (e.g. SNLI then MultiNLI for the base, then SciNLI to domain-adapt). Writes a
versioned verifier artifact. Use ``--device cuda`` for scale runs.

Example (smoke)
---------------
    uv run python scripts/train_verifier.py --tokenizer artifacts/tokenizer.json \\
        --nli data/scale/snli_1.0_train.jsonl --max-examples 5000 \\
        --out artifacts/verifier --device auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from meridian.data import load_nli_jsonl
from meridian.device import resolve_device
from meridian.encoder.artifact import EmbedderConfig
from meridian.encoder.pretrain import build_mlm, initialize_from_mlm, mlm_pretrain
from meridian.tokenization.artifact import load_tokenizer
from meridian.tokenization.special_tokens import ensure_mask_token
from meridian.verify.artifact import NLIConfig, build_verifier, save_verifier
from meridian.verify.data import make_nli_samples
from meridian.verify.training import train_verifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output verifier artifact dir")
    parser.add_argument(
        "--nli", type=Path, action="append", required=True, help="NLI JSONL (repeatable)"
    )
    parser.add_argument("--premise-field", default="sentence1")
    parser.add_argument("--hypothesis-field", default="sentence2")
    parser.add_argument("--label-field", default="gold_label")
    parser.add_argument("--max-examples", type=int, help="cap per file (smoke runs)")
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

    examples: list[tuple[str, str, int]] = []
    for nli_path in args.nli:
        loaded = load_nli_jsonl(
            nli_path,
            premise_field=args.premise_field,
            hypothesis_field=args.hypothesis_field,
            label_field=args.label_field,
            max_examples=args.max_examples,
        )
        print(f"{nli_path.name}: {len(loaded)} labeled examples")
        examples += loaded
    if not examples:
        parser.error("no labeled NLI examples read")

    config = NLIConfig(
        vocab_size=tokenizer.vocabulary.size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        pad_id=pad_id,
    )
    torch.manual_seed(args.seed)
    verifier = build_verifier(config)
    if not args.random_init:
        sentences = list({p for p, _, _ in examples} | {h for _, h, _ in examples})
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
            sentences,
            tokenizer,
            mask_id=mask_id,
            vocab_size=tokenizer.vocabulary.size,
            epochs=args.mlm_epochs,
            device=device,
            seed=args.seed,
        )
        initialize_from_mlm(mlm, verifier)
        print(f"stage 0: MLM-pretrained trunk on {len(sentences)} sentences")

    losses = train_verifier(
        verifier,
        make_nli_samples(examples, tokenizer),
        pad_id=pad_id,
        cls_id=tokenizer.vocabulary.cls_id,
        sep_id=tokenizer.vocabulary.sep_id,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
    )
    print(f"trained on {len(examples)} examples; final loss {losses[-1]:.4f}")

    save_verifier(
        verifier,
        config,
        args.out,
        metadata={"num_examples": len(examples), "epochs": args.epochs, "final_loss": losses[-1]},
    )
    print(f"wrote verifier artifact -> {args.out}")


if __name__ == "__main__":
    main()
