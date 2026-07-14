"""Encoder-side utilities: embedding the corpus and training the bi-encoder.

Phase 3 consumes the Polaris ``TextEmbedder`` (``polaris-nlp==1.2.0``). This package
holds the Meridian-side glue — batch-encoding documents to vectors, and (Milestone 2)
the MLM → contrastive training pipelines (ADR-0004) that wrap Polaris's trainers.
"""

from meridian.encoder.artifact import (
    EmbedderConfig,
    build_embedder,
    load_embedder,
    save_embedder,
)
from meridian.encoder.data import (
    make_contrastive_samples,
    make_contrastive_samples_with_negatives,
    mine_title_abstract_pairs,
)
from meridian.encoder.embed import embed_documents, encode_texts
from meridian.encoder.training import train_retriever

__all__ = [
    "EmbedderConfig",
    "build_embedder",
    "embed_documents",
    "encode_texts",
    "load_embedder",
    "make_contrastive_samples",
    "make_contrastive_samples_with_negatives",
    "mine_title_abstract_pairs",
    "save_embedder",
    "train_retriever",
]
