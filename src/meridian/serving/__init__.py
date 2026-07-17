"""Serving layer — the whole pipeline behind one API (Phase 10).

A FastAPI app exposing ``/ask`` (with an SSE streaming variant), ``/passages``,
``/health``, and ``/metrics`` over the retrieval + answering pipeline. int8
quantization and speculative decoding are provided by Zenith for the generated path;
this layer wires the pipeline and instruments per-stage latency (RAG.md §5).
"""

from meridian.serving.app import create_app
from meridian.serving.instrumentation import StageTimer

__all__ = ["StageTimer", "create_app"]
