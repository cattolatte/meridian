"""Fertility: mean subword tokens per whitespace word.

Fertility is the selection metric for the ADR-0003 vocabulary-size sweep. A lower
value means the tokenizer fragments words less; it is reported separately on
biomedical and general held-out text because the trade-off between them is exactly
what the mixed-corpus decision balances.
"""

from __future__ import annotations

from collections.abc import Iterable

from polaris.tokenizers import Tokenizer


def fertility(tokenizer: Tokenizer, texts: Iterable[str]) -> float:
    """Return mean subword tokens per whitespace word over ``texts``.

    Words are counted by whitespace splitting; subword tokens by the tokenizer's
    own ``tokenize``. Texts with no words contribute nothing. Raises
    :class:`ValueError` if the sample contains no words at all.
    """
    total_tokens = 0
    total_words = 0
    for text in texts:
        words = text.split()
        if not words:
            continue
        total_words += len(words)
        total_tokens += len(tokenizer.tokenize(text))
    if total_words == 0:
        raise ValueError("cannot compute fertility on an empty word sample")
    return total_tokens / total_words
