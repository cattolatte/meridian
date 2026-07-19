"""Meridian — a from-scratch grounded RAG engine over biomedical literature.

Every ML component is trained in-house: the tokenizer, dense retriever,
reranker, and faithfulness verifier (Polaris), and the cited-answer
generator (Zenith). Every answer is cited, verified, or refused.
"""

__version__ = "1.0.0"
