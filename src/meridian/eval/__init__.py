"""Evaluation harness: the measuring instruments (RAG.md §7).

Retrieval metrics, eval-set/qrels types, frozen dev/test splits guarded by a
checksum (house rule #4), and a runner that scores a retriever and writes
reproducible JSON results.
"""

from meridian.eval.harness import EvalResult, run_evaluation
from meridian.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from meridian.eval.qrels import EvalQuery, EvalSet, load_eval_set, save_eval_set
from meridian.eval.splits import ChecksumMismatchError, load_frozen_split, split_checksum

__all__ = [
    "ChecksumMismatchError",
    "EvalQuery",
    "EvalResult",
    "EvalSet",
    "load_eval_set",
    "load_frozen_split",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "run_evaluation",
    "save_eval_set",
    "split_checksum",
]
