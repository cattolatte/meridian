"""Tests for the reranking retrieval stage."""

from __future__ import annotations

import pytest
import torch
from polaris.tokenizers import BPETokenizer, train_bpe

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.reranker.artifact import RerankerConfig, build_reranker
from meridian.reranker.data import make_pair_samples
from meridian.reranker.training import train_reranker
from meridian.retrieval.pipeline import BM25Retriever, RetrievalHit, Retriever
from meridian.retrieval.rerank import RerankingRetriever
from meridian.tokenization.training import train_tokenizer

_DOCS = [
    Document(pmid="1", title="Heart failure", abstract="beta blockers reduce mortality"),
    Document(pmid="2", title="Diabetes", abstract="metformin lowers glucose"),
    Document(pmid="3", title="Melanoma", abstract="checkpoint immunotherapy responses"),
]
_BIO = ["heart failure mortality", "diabetes glucose metformin", "melanoma immunotherapy"] * 4
_GENERAL = ["a general english passage here"] * 4

_QUERY = "heart failure mortality beta blockers"


class _FixedRetriever:
    """Return a fixed ranking of PMIDs (base ordering the reranker will rearrange)."""

    def __init__(self, order: list[str], store: SqliteDocumentStore) -> None:
        self._order = order
        self._store = store

    def retrieve(self, query: str, *, k: int = 10) -> list[RetrievalHit]:
        return [
            RetrievalHit(pmid=p, score=-i, document=self._store.get(p))
            for i, p in enumerate(self._order[:k])
        ]


def _store() -> SqliteDocumentStore:
    store = SqliteDocumentStore(":memory:")
    store.add_many(_DOCS)
    return store


def _tok() -> BPETokenizer:
    return train_tokenizer(_BIO, _GENERAL, vocab_size=200)


def _trained_reranker(tok: BPETokenizer) -> object:
    """A tiny reranker trained (memorized) to prefer doc 1 for the heart-failure query."""
    torch.manual_seed(0)
    config = RerankerConfig(
        vocab_size=tok.vocabulary.size,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
        max_len=64,
        pad_id=tok.vocabulary.pad_id or 0,
    )
    model = build_reranker(config)
    # Train on the actual document chunk texts so scoring is deterministic.
    examples = [
        (_QUERY, _DOCS[0].chunk_text(), 1),
        (_QUERY, _DOCS[2].chunk_text(), 0),
        (_QUERY, _DOCS[1].chunk_text(), 0),
    ] * 5
    train_reranker(
        model,
        make_pair_samples(examples, tok),
        pad_id=tok.vocabulary.pad_id or 0,
        cls_id=tok.vocabulary.cls_id,
        sep_id=tok.vocabulary.sep_id,
        epochs=30,
        batch_size=4,
        learning_rate=5e-3,
    )
    return model


def test_satisfies_protocol() -> None:
    tok = _tok()
    with _store() as store:
        rr = RerankingRetriever(BM25Retriever.from_store(store), _trained_reranker(tok), tok, store)
        assert isinstance(rr, Retriever)


def test_returns_at_most_k() -> None:
    tok = _tok()
    with _store() as store:
        base = _FixedRetriever(["1", "2", "3"], store)
        rr = RerankingRetriever(base, _trained_reranker(tok), tok, store, candidates=10)
        assert len(rr.retrieve(_QUERY, k=2)) == 2


def test_reorders_base_ranking_toward_relevant() -> None:
    tok = _tok()
    model = _trained_reranker(tok)
    with _store() as store:
        # Base puts the irrelevant melanoma doc first; the reranker should demote it.
        base = _FixedRetriever(["3", "1"], store)
        rr = RerankingRetriever(base, model, tok, store, candidates=10)
        top = rr.retrieve(_QUERY, k=1)[0]
        assert top.pmid == "1"


def test_requires_cls_sep_tokens() -> None:
    maskless = train_bpe(
        [["heart", "failure"]] * 3, vocab_size=40, unk_token="<unk>", pad_token="<pad>"
    )
    with _store() as store, pytest.raises(ValueError):
        RerankingRetriever(
            BM25Retriever.from_store(store), _trained_reranker(_tok()), maskless, store
        )
