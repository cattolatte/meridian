"""Tests for device resolution."""

from __future__ import annotations

import torch

from meridian.device import resolve_device


def test_resolve_explicit_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto_returns_available_device() -> None:
    device = resolve_device("auto")
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


def test_resolve_honors_explicit_spec() -> None:
    assert resolve_device("cuda:1") == torch.device("cuda:1")


def test_trainers_accept_device_cpu() -> None:
    """The device kwarg threads through every trainer + encoder (exercised on cpu)."""
    from meridian.encoder.artifact import EmbedderConfig, build_embedder
    from meridian.encoder.data import make_contrastive_samples
    from meridian.encoder.embed import encode_texts
    from meridian.encoder.pretrain import build_mlm, initialize_from_mlm, mlm_pretrain
    from meridian.encoder.training import train_retriever
    from meridian.reranker.artifact import RerankerConfig, build_reranker
    from meridian.reranker.data import make_pair_samples
    from meridian.reranker.training import train_reranker
    from meridian.tokenization.training import train_tokenizer
    from meridian.verify.artifact import NLIConfig, build_verifier
    from meridian.verify.data import make_nli_samples
    from meridian.verify.training import train_verifier

    dev = torch.device("cpu")
    texts = ["heart failure mortality beta blockers", "diabetes glucose metformin"] * 4
    tok = train_tokenizer(texts, ["a general english passage"] * 4, vocab_size=120)
    v = tok.vocabulary
    ecfg = EmbedderConfig(vocab_size=v.size, embed_dim=16, num_layers=1, pad_id=v.pad_id or 0)

    mlm = build_mlm(ecfg)
    mlm_pretrain(mlm, texts, tok, mask_id=v.mask_id, vocab_size=v.size, epochs=1, device=dev)
    emb = build_embedder(ecfg)
    initialize_from_mlm(mlm, emb)
    pairs = [("heart failure", "beta blockers reduce mortality"), ("diabetes", "metformin")] * 4
    train_retriever(emb, make_contrastive_samples(pairs, tok), pad_id=ecfg.pad_id, device=dev)
    assert encode_texts(emb, tok, ["heart failure"], device=dev).shape == (1, 16)

    rr = build_reranker(
        RerankerConfig(vocab_size=v.size, embed_dim=16, num_layers=1, pad_id=v.pad_id or 0)
    )
    train_reranker(
        rr,
        make_pair_samples([("q", "p", 1), ("q", "n", 0)] * 3, tok),
        pad_id=v.pad_id or 0,
        cls_id=v.cls_id,
        sep_id=v.sep_id,
        device=dev,
    )
    nli = build_verifier(
        NLIConfig(vocab_size=v.size, embed_dim=16, num_layers=1, pad_id=v.pad_id or 0)
    )
    losses = train_verifier(
        nli,
        make_nli_samples([("a b", "c d", 0), ("e f", "g h", 2)] * 3, tok),
        pad_id=v.pad_id or 0,
        cls_id=v.cls_id,
        sep_id=v.sep_id,
        class_weights=[1.0, 1.0, 2.0],
        device=dev,
    )
    assert all(isinstance(x, float) for x in losses)
