"""Evaluation harness: the measuring instruments (RAG.md §7).

Retrieval metrics, eval-set/qrels types, frozen dev/test splits guarded by a
checksum (house rule #4), and a runner that scores a retriever and writes
reproducible JSON results.
"""

from meridian.eval.attribution import AttributionStudy, attribute_failure, attribution_study
from meridian.eval.harness import EvalResult, run_evaluation
from meridian.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from meridian.eval.misses import MissRecord, sample_misses
from meridian.eval.pubmedqa import pubmedqa_accuracy
from meridian.eval.qrels import EvalQuery, EvalSet, load_eval_set, save_eval_set
from meridian.eval.splits import ChecksumMismatchError, load_frozen_split, split_checksum

__all__ = [
    "AttributionStudy",
    "ChecksumMismatchError",
    "EvalQuery",
    "EvalResult",
    "EvalSet",
    "MissRecord",
    "attribute_failure",
    "attribution_study",
    "load_eval_set",
    "load_frozen_split",
    "mrr_at_k",
    "ndcg_at_k",
    "pubmedqa_accuracy",
    "recall_at_k",
    "run_evaluation",
    "sample_misses",
    "save_eval_set",
    "split_checksum",
]
