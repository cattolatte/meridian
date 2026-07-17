"""Grounded answering: retrieve → generate a cited answer, or abstain (ADR-0006).

Formats the retrieved passages into the prompt, generates with citation-constrained
decoding (a `[n]` may only reference a retrieved passage) and an abstain stop token,
then parses the `[n]` citations back to PMIDs. A freshly generated answer is labelled
GENERATED; it becomes GROUNDED only after the Phase 8 verifier passes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from zenith.generation.constraints import AllowedTokens
from zenith.generation.generator import Generator
from zenith.instruct.grounded import GroundedTemplate
from zenith.models.decoder import DecoderLM
from zenith.tokenizers.byte_tokenizer import ByteTokenizer

from meridian.retrieval.pipeline import Retriever

_CITATION = re.compile(r"\[(\d+)\]")
_NOT_MEDICAL_ADVICE = "Research literature assistant. Not medical advice."
_MAX_CITED_PASSAGES = 9  # single-digit citation indices (ADR-0006)


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """A generated cited answer, or an abstention."""

    query: str
    abstained: bool
    text: str
    citations: tuple[tuple[int, str, str], ...]  # (n, pmid, title) actually cited
    passages: tuple[tuple[str, str], ...]  # (pmid, title) retrieved, in order


def answer_grounded(
    retriever: Retriever,
    model: DecoderLM,
    tokenizer: ByteTokenizer,
    query: str,
    *,
    k: int = 5,
    max_new_tokens: int = 128,
) -> GroundedAnswer:
    """Retrieve, generate a cited answer with constrained decoding, and parse it."""
    hits = [hit for hit in retriever.retrieve(query, k=k) if hit.document is not None]
    passages = [(hit.pmid, hit.document.chunk_text()) for hit in hits if hit.document]
    titles = {hit.pmid: hit.document.title for hit in hits if hit.document}
    ordered = tuple((pmid, titles.get(pmid, "")) for pmid, _ in passages)

    if not passages:
        return GroundedAnswer(query, abstained=True, text="", citations=(), passages=())

    prompt = GroundedTemplate().format_prompt(query, passages)
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    n_valid = min(len(passages), _MAX_CITED_PASSAGES)
    constraint = AllowedTokens(
        trigger_ids={ord("[")},
        allowed_ids={ord(str(i)) for i in range(1, n_valid + 1)},
    )
    generator = Generator(model, tokenizer)
    output = generator.generate_ids(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        logits_constraint=constraint,
        stop_ids={tokenizer.abstain_id, tokenizer.eos_id},
    )
    if generator.abstained(output):
        return GroundedAnswer(query, abstained=True, text="", citations=(), passages=ordered)

    generated_ids = output[0, prompt_ids.shape[1] :].tolist()
    text = tokenizer.decode(generated_ids).strip()
    citations = _parse_citations(text, passages, titles)
    return GroundedAnswer(query, abstained=False, text=text, citations=citations, passages=ordered)


def _parse_citations(
    text: str,
    passages: list[tuple[str, str]],
    titles: dict[str, str],
) -> tuple[tuple[int, str, str], ...]:
    """Map the ``[n]`` markers in ``text`` to their passage PMIDs, in first-use order."""
    seen: set[int] = set()
    citations: list[tuple[int, str, str]] = []
    for match in _CITATION.finditer(text):
        n = int(match.group(1))
        if 1 <= n <= len(passages) and n not in seen:
            seen.add(n)
            pmid = passages[n - 1][0]
            citations.append((n, pmid, titles.get(pmid, "")))
    return tuple(citations)


def render_grounded_answer(answer: GroundedAnswer) -> str:
    """Render a :class:`GroundedAnswer` for the CLI, with the required banner."""
    lines: list[str] = []
    if answer.abstained:
        lines.append("ABSTAIN — the generator did not produce a grounded answer.")
        if answer.passages:
            lines.append("")
            lines.append("Nearest passages:")
            lines += [f"  PMID {pmid} — {title}" for pmid, title in answer.passages[:3]]
    else:
        lines.append(answer.text)
        lines.append("")
        lines.append("Sources:")
        lines += [f"  [{n}] PMID {pmid} — {title}" for n, pmid, title in answer.citations]
        lines.append("")
        lines.append(
            "Confidence: GENERATED (citations constrained to retrieved passages; unverified)"
        )
    lines.append("")
    lines.append(_NOT_MEDICAL_ADVICE)
    return "\n".join(lines)
