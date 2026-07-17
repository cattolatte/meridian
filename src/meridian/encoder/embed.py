"""Batch-encode text to embeddings with a Polaris ``TextEmbedder``.

Turns documents (or a query) into a float32 embedding matrix, ready for
:class:`~meridian.retrieval.embedding_index.EmbeddingIndex`. Padding and the
attention mask are built here from the trained BPE tokenizer; the embedder emits
L2-normalized vectors, so downstream dot products are cosine similarities.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
import torch
from polaris.models import TextEmbedder
from polaris.tokenizers import BPETokenizer

from meridian.corpus.records import Document


def _pad_batch(
    id_lists: Sequence[Sequence[int]], *, pad_id: int, max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad/truncate token-id lists into aligned ``input_ids`` and ``attention_mask``."""
    trimmed = [list(ids[:max_length]) for ids in id_lists]
    width = max((len(ids) for ids in trimmed), default=1) or 1
    input_ids = torch.full((len(trimmed), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(trimmed), width), dtype=torch.long)
    for row, ids in enumerate(trimmed):
        if ids:
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
    return input_ids, attention_mask


def encode_texts(
    embedder: TextEmbedder,
    tokenizer: BPETokenizer,
    texts: Sequence[str],
    *,
    max_length: int = 256,
    batch_size: int = 32,
    device: torch.device | str | None = None,
) -> npt.NDArray[np.float32]:
    """Encode ``texts`` into an ``(N, D)`` float32 embedding matrix.

    ``device`` moves the embedder and inputs onto an accelerator (CUDA/MPS); vectors are
    always returned as a CPU float32 array. ``None`` keeps the model where it is.
    """
    if not texts:
        return np.zeros((0, embedder.embedding_dim), dtype=np.float32)

    pad_id = tokenizer.vocabulary.pad_id
    if pad_id is None:
        pad_id = 0

    if device is not None:
        embedder.to(device)
    embedder.eval()
    chunks: list[npt.NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            id_lists = [list(tokenizer.encode(text).ids) for text in batch]
            input_ids, attention_mask = _pad_batch(id_lists, pad_id=pad_id, max_length=max_length)
            if device is not None:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
            vectors = embedder.encode(input_ids, attention_mask)
            chunks.append(vectors.detach().cpu().to(torch.float32).numpy())
    return np.concatenate(chunks, axis=0)


def embed_documents(
    embedder: TextEmbedder,
    tokenizer: BPETokenizer,
    documents: Iterable[Document],
    *,
    max_length: int = 256,
    batch_size: int = 32,
    device: torch.device | str | None = None,
) -> tuple[list[str], npt.NDArray[np.float32]]:
    """Embed each document's chunk text; return parallel PMIDs and the matrix."""
    docs = list(documents)
    pmids = [doc.pmid for doc in docs]
    texts = [doc.chunk_text() for doc in docs]
    vectors = encode_texts(
        embedder, tokenizer, texts, max_length=max_length, batch_size=batch_size, device=device
    )
    return pmids, vectors
