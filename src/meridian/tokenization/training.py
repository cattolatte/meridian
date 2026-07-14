"""Train Meridian's BPE tokenizer on a mixed corpus (ADR-0003).

The tokenizer learns from a biomedical corpus (PubMed title+abstract chunks) mixed
with a general-English corpus (MS MARCO passages), combined by *word count* so the
ratio reflects what the merge-learner actually sees. Training itself is delegated to
Polaris' ``train_bpe``; this module handles the corpus mix and the vocabulary-size
sweep. Everything is deterministic: the general portion is taken as a stable prefix
sized to the target ratio, so a rebuild from the same inputs yields the same
tokenizer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from polaris.tokenizers import BPETokenizer, train_bpe

from meridian.tokenization.fertility import fertility

DEFAULT_MIX_RATIO = 0.7
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_PAD_TOKEN = "<pad>"
# Reserved so one vocabulary serves every model: MLM masking needs <mask>; the
# cross-encoder pair head (Phase 6+) needs <cls>/<sep> (ADR-0003 deferred these to
# later phases — Polaris 1.3.0 reserves them at train time without shifting ids).
DEFAULT_MASK_TOKEN = "<mask>"
DEFAULT_CLS_TOKEN = "<cls>"
DEFAULT_SEP_TOKEN = "<sep>"


def _mixed_word_sequences(
    biomedical_texts: Iterable[str],
    general_texts: Iterable[str],
    mix_ratio: float,
) -> list[list[str]]:
    """Combine the two corpora into word sequences at the target word-count ratio.

    ``mix_ratio`` is the biomedical fraction of total words. The biomedical corpus
    is used in full; general sequences are added in order until they make up the
    remaining fraction (or are exhausted).
    """
    if not 0.0 < mix_ratio <= 1.0:
        raise ValueError("mix_ratio must be in (0, 1]")

    bio_seqs = [text.split() for text in biomedical_texts]
    bio_seqs = [seq for seq in bio_seqs if seq]
    bio_words = sum(len(seq) for seq in bio_seqs)
    if bio_words == 0:
        raise ValueError("biomedical corpus is empty")

    target_general_words = round(bio_words * (1.0 - mix_ratio) / mix_ratio)
    selected_general: list[list[str]] = []
    general_words = 0
    for text in general_texts:
        if general_words >= target_general_words:
            break
        seq = text.split()
        if not seq:
            continue
        selected_general.append(seq)
        general_words += len(seq)

    return bio_seqs + selected_general


def train_tokenizer(
    biomedical_texts: Iterable[str],
    general_texts: Iterable[str],
    *,
    vocab_size: int,
    mix_ratio: float = DEFAULT_MIX_RATIO,
    min_frequency: int = 1,
    unk_token: str = DEFAULT_UNK_TOKEN,
    pad_token: str = DEFAULT_PAD_TOKEN,
    mask_token: str = DEFAULT_MASK_TOKEN,
    cls_token: str = DEFAULT_CLS_TOKEN,
    sep_token: str = DEFAULT_SEP_TOKEN,
) -> BPETokenizer:
    """Train a BPE tokenizer on the mixed corpus and return it.

    Reserves ``<unk>``/``<pad>``/``<mask>``/``<cls>``/``<sep>`` so the same vocabulary
    serves the MLM, embedder, and cross-encoder pair models.
    """
    sequences = _mixed_word_sequences(biomedical_texts, general_texts, mix_ratio)
    return train_bpe(
        sequences,
        vocab_size=vocab_size,
        unk_token=unk_token,
        pad_token=pad_token,
        mask_token=mask_token,
        cls_token=cls_token,
        sep_token=sep_token,
        min_frequency=min_frequency,
    )


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Fertility of one trained vocabulary size on the two held-out samples."""

    vocab_size: int
    biomedical_fertility: float
    general_fertility: float


def sweep_vocabulary_sizes(
    biomedical_texts: Sequence[str],
    general_texts: Sequence[str],
    vocab_sizes: Iterable[int],
    *,
    biomedical_eval: Sequence[str],
    general_eval: Sequence[str],
    mix_ratio: float = DEFAULT_MIX_RATIO,
    min_frequency: int = 1,
) -> list[SweepResult]:
    """Train each vocabulary size and measure fertility on the held-out samples.

    Returns one :class:`SweepResult` per size, in the given order — the evidence
    table recorded in ``benchmarks/corpus.md`` for the ADR-0003 default choice.
    """
    results: list[SweepResult] = []
    for size in vocab_sizes:
        tokenizer = train_tokenizer(
            biomedical_texts,
            general_texts,
            vocab_size=size,
            mix_ratio=mix_ratio,
            min_frequency=min_frequency,
        )
        results.append(
            SweepResult(
                vocab_size=size,
                biomedical_fertility=fertility(tokenizer, biomedical_eval),
                general_fertility=fertility(tokenizer, general_eval),
            )
        )
    return results
