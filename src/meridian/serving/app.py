"""The Meridian FastAPI application.

Endpoints (RAG.md §5):
- ``GET /health``   — liveness + document count.
- ``GET /passages`` — top retrieved passages for a query.
- ``POST /ask``     — a cited extractive answer (or abstain), with the disclaimer.
- ``GET /ask/stream`` — the same, as Server-Sent Events (retrieval → answer → done).
- ``GET /metrics``  — request counts and per-stage latency percentiles.

The extractive answerer is the default serving path (hallucination-free); the generated
path is layered on with the trained generator/verifier. Every answer carries
"Not medical advice".
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from meridian.answer.extractive import ExtractiveAnswer, answer_extractive
from meridian.corpus.store import DocumentStore
from meridian.retrieval.pipeline import Retriever
from meridian.serving.instrumentation import StageTimer

_DISCLAIMER = "Research literature assistant. Not medical advice."


class AskRequest(BaseModel):
    """A question plus retrieval/answer knobs."""

    question: str
    k: int = 5
    sentences: int = 3


def _answer_payload(answer: ExtractiveAnswer) -> dict[str, object]:
    if answer.abstained:
        return {
            "query": answer.query,
            "abstained": True,
            "confidence": "ABSTAIN",
            "nearest": [{"pmid": pmid, "title": title} for pmid, title in answer.nearest],
            "disclaimer": _DISCLAIMER,
        }
    numbering = {pmid: i for i, (pmid, _) in enumerate(answer.sources, start=1)}
    return {
        "query": answer.query,
        "abstained": False,
        "confidence": "GROUNDED",
        "sentences": [
            {"text": s.text, "pmid": s.pmid, "citation": numbering[s.pmid]}
            for s in answer.sentences
        ],
        "sources": [
            {"citation": i, "pmid": pmid, "title": title}
            for i, (pmid, title) in enumerate(answer.sources, start=1)
        ],
        "disclaimer": _DISCLAIMER,
    }


def create_app(retriever: Retriever, store: DocumentStore) -> FastAPI:
    """Build the FastAPI app over a retriever and document store."""
    app = FastAPI(title="Meridian", description="Grounded biomedical RAG. Not medical advice.")
    app.state.retriever = retriever
    app.state.store = store
    app.state.timer = StageTimer()
    app.state.counters = {"health": 0, "passages": 0, "ask": 0}

    @app.get("/health")
    def health() -> dict[str, object]:
        app.state.counters["health"] += 1
        return {"status": "ok", "documents": store.count()}

    @app.get("/passages")
    def passages(q: str, k: int = 5) -> dict[str, object]:
        app.state.counters["passages"] += 1
        with app.state.timer.stage("search"):
            hits = retriever.retrieve(q, k=k)
        return {
            "query": q,
            "passages": [
                {
                    "pmid": hit.pmid,
                    "score": hit.score,
                    "title": hit.document.title if hit.document else None,
                }
                for hit in hits
            ],
        }

    @app.post("/ask")
    def ask(request: AskRequest) -> dict[str, object]:
        app.state.counters["ask"] += 1
        with app.state.timer.stage("answer"):
            answer = answer_extractive(
                retriever, request.question, k_passages=request.k, max_sentences=request.sentences
            )
        return _answer_payload(answer)

    @app.get("/ask/stream")
    def ask_stream(q: str, k: int = 5, sentences: int = 3) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            hits = retriever.retrieve(q, k=k)
            retrieval = [{"pmid": h.pmid, "score": h.score} for h in hits]
            yield f"event: retrieval\ndata: {json.dumps(retrieval)}\n\n"
            answer = answer_extractive(retriever, q, k_passages=k, max_sentences=sentences)
            yield f"event: answer\ndata: {json.dumps(_answer_payload(answer))}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        return {"counters": app.state.counters, "latency_ms": app.state.timer.percentiles()}

    return app
