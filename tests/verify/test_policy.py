"""Tests for the fail-safe answer ladder."""

from __future__ import annotations

import torch
from zenith.tokenizers.byte_tokenizer import ByteTokenizer

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.generation.artifact import GeneratorConfig, build_generator
from meridian.retrieval.pipeline import BM25Retriever
from meridian.tokenization.training import train_tokenizer
from meridian.verify.artifact import NLIConfig, build_verifier
from meridian.verify.policy import answer_with_verification

_DOCS = [
    Document(pmid="10", title="Metformin", abstract="Metformin lowers cardiovascular mortality."),
    Document(pmid="20", title="Melanoma", abstract="Checkpoint immunotherapy responses."),
]
_BIO = ["metformin mortality", "melanoma immunotherapy"] * 4
_GENERAL = ["a general english passage here"] * 4


def test_abstains_when_generator_abstains() -> None:
    torch.manual_seed(0)
    tok = train_tokenizer(_BIO, _GENERAL, vocab_size=200)
    gen = build_generator(
        GeneratorConfig(embed_dim=32, num_layers=1, num_heads=2, ff_dim=64, use_lora=False)
    )
    ver = build_verifier(
        NLIConfig(
            vocab_size=tok.vocabulary.size,
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            ff_dim=32,
            max_len=64,
            pad_id=tok.vocabulary.pad_id or 0,
        )
    )
    with SqliteDocumentStore(":memory:") as store:
        store.add_many(_DOCS)
        # No lexical overlap -> retriever returns nothing -> generator abstains.
        result = answer_with_verification(
            BM25Retriever.from_store(store), gen, ByteTokenizer(), ver, tok, store, "xyzzy quux"
        )
        assert result.confidence == "ABSTAIN"
        assert "Not medical advice." in result.rendered


def test_falls_back_to_extractive_when_unverified() -> None:
    # Untrained verifier almost never returns all-entailment, so a generated answer
    # fails verification and the ladder falls back to extractive (or abstains).
    torch.manual_seed(0)
    tok = train_tokenizer(_BIO, _GENERAL, vocab_size=200)
    gen = build_generator(
        GeneratorConfig(embed_dim=32, num_layers=1, num_heads=2, ff_dim=64, use_lora=False)
    )
    ver = build_verifier(
        NLIConfig(
            vocab_size=tok.vocabulary.size,
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            ff_dim=32,
            max_len=64,
            pad_id=tok.vocabulary.pad_id or 0,
        )
    )
    with SqliteDocumentStore(":memory:") as store:
        store.add_many(_DOCS)
        result = answer_with_verification(
            BM25Retriever.from_store(store),
            gen,
            ByteTokenizer(),
            ver,
            tok,
            store,
            "metformin mortality",
            k=2,
        )
        assert result.confidence in {"GROUNDED (extractive fallback)", "ABSTAIN", "GROUNDED"}
        assert "Not medical advice." in result.rendered
