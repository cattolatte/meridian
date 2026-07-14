"""Encoder-side utilities: embedding the corpus and training the bi-encoder.

Phase 3 consumes the Polaris ``TextEmbedder`` (``polaris-nlp==1.2.0``). This package
holds the Meridian-side glue — batch-encoding documents to vectors, and (Milestone 2)
the MLM → contrastive training pipelines (ADR-0004) that wrap Polaris's trainers.
"""

from meridian.encoder.embed import embed_documents, encode_texts

__all__ = ["embed_documents", "encode_texts"]
