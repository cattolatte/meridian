#!/usr/bin/env python
"""Train and evaluate a PubMedQA yes/no/maybe classifier (the headline number).

A 3-class Polaris ``SentencePairClassifier`` over ``(question, context)`` predicts the
final decision. Trained on an 80% partition of PQA-L, evaluated on the held-out 20%
(disjoint questions — no leakage). Reports accuracy vs the 0.552 majority-class baseline.
This is a *real* number; a small, from-scratch model on ~800 examples will not match
large fine-tuned models — the honest, plan-anticipated scope (RAG.md §2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from meridian.eval.pubmedqa import pubmedqa_accuracy
from meridian.tokenization.training import train_tokenizer
from meridian.verify.artifact import NLIConfig, build_verifier
from meridian.verify.data import make_nli_samples
from meridian.verify.training import train_verifier

_LABELS = ("yes", "no", "maybe")
_LABEL_ID = {label: i for i, label in enumerate(_LABELS)}


def _is_train(pubid: str, salt: str = "cls") -> bool:
    digest = hashlib.sha256(f"{salt}:{pubid}".encode()).hexdigest()
    return int(digest, 16) % 100 < 80


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pqal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="results JSON")
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = json.loads(args.pqal.read_text())
    contexts = {k: " ".join(v.get("CONTEXTS", [])) for k, v in data.items()}

    train_examples = [
        (v["QUESTION"], contexts[k], _LABEL_ID[v["final_decision"]])
        for k, v in data.items()
        if _is_train(k) and contexts[k]
    ]
    eval_items = [
        (k, v["QUESTION"], contexts[k], v["final_decision"])
        for k, v in data.items()
        if not _is_train(k) and contexts[k]
    ]
    print(f"train: {len(train_examples)}  eval: {len(eval_items)}")

    tokenizer = train_tokenizer(
        list(contexts.values()),
        ["a general english passage about various topics"] * 200,
        vocab_size=args.vocab_size,
    )
    v = tokenizer.vocabulary
    print("tokenizer trained")

    torch.manual_seed(args.seed)
    model = build_verifier(
        NLIConfig(vocab_size=v.size, embed_dim=args.embed_dim, num_layers=2, pad_id=v.pad_id or 0)
    )
    losses = train_verifier(
        model,
        make_nli_samples(train_examples, tokenizer),
        pad_id=v.pad_id or 0,
        cls_id=v.cls_id,
        sep_id=v.sep_id,
        epochs=args.epochs,
        batch_size=16,
        max_length=256,
        seed=args.seed,
    )
    print(f"training done; final loss {losses[-1]:.4f}")

    from polaris.collation import collate_pairs

    model.eval()
    predictions: dict[str, str] = {}
    gold: dict[str, str] = {}
    for pubid, question, context, label in eval_items:
        gold[pubid] = label
        batch = collate_pairs(
            make_nli_samples([(question, context, 0)], tokenizer),
            pad_id=v.pad_id or 0,
            cls_id=v.cls_id,
            sep_id=v.sep_id,
            max_length=256,
        )
        with torch.no_grad():
            predictions[pubid] = _LABELS[int(model(batch).argmax(dim=-1)[0])]

    accuracy = pubmedqa_accuracy(predictions, gold)
    from collections import Counter

    majority = max(Counter(gold.values()).values()) / len(gold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"accuracy": accuracy, "majority": majority, "n": len(gold)}) + "\n"
    )
    print(
        f"PubMedQA accuracy: {accuracy:.4f}  (majority {majority:.4f}, n={len(gold)}) -> {args.out}"
    )


if __name__ == "__main__":
    main()
