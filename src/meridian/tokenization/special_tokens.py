"""Add a ``<mask>`` token to a trained tokenizer (ADR-0003 superseding note).

ADR-0003 deferred the MLM mask token to Phase 3 with one constraint: introducing it
must **not** shift the ids of an already-trained vocabulary. This helper appends a
mask token at the end of the vocabulary (id = current size), so every existing token
keeps its id and normal-text encoding is unchanged — only the new id is added for the
MLM masking process to reference.
"""

from __future__ import annotations

from polaris.tokenizers import BPETokenizer, Vocabulary

DEFAULT_MASK_TOKEN = "<mask>"


def ensure_mask_token(
    tokenizer: BPETokenizer, *, mask_token: str = DEFAULT_MASK_TOKEN
) -> tuple[BPETokenizer, int]:
    """Return a tokenizer that has a mask token, plus its id.

    If the tokenizer already defines a mask token, it is returned unchanged. Otherwise
    the mask token is appended at the next free id (no existing id is renumbered) and a
    new :class:`BPETokenizer` with the same merges is returned.
    """
    vocabulary = tokenizer.vocabulary
    if vocabulary.mask_token is not None:
        return tokenizer, vocabulary.mask_id

    payload = vocabulary.to_dict()
    token_to_id = dict(payload["token_to_id"])
    if mask_token not in token_to_id:
        token_to_id[mask_token] = len(token_to_id)  # append; existing ids untouched
    extended = Vocabulary(
        token_to_id=token_to_id,
        unk_token=payload["unk_token"],
        pad_token=payload["pad_token"],
        mask_token=mask_token,
    )
    # BPETokenizer exposes no public accessor for its merges / end-of-word marker;
    # both are public constructor arguments (see tokenization.artifact).
    rebuilt = BPETokenizer(extended, tokenizer._merges, end_of_word=tokenizer._end_of_word)
    return rebuilt, extended.mask_id
