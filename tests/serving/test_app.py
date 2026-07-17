"""Offline tests for the FastAPI serving layer (TestClient, no network)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from meridian.corpus.records import Document
from meridian.corpus.store import SqliteDocumentStore
from meridian.retrieval.pipeline import BM25Retriever
from meridian.serving.app import create_app

_DOCS = [
    Document(pmid="1", title="Heart failure", abstract="beta blockers reduce mortality"),
    Document(pmid="2", title="Melanoma", abstract="checkpoint immunotherapy responses"),
]


@pytest.fixture()
def client() -> Iterator[TestClient]:
    store = SqliteDocumentStore(":memory:")
    store.add_many(_DOCS)
    app = create_app(BM25Retriever.from_store(store), store)
    with TestClient(app) as test_client:
        yield test_client
    store.close()


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "documents": 2}


def test_passages(client: TestClient) -> None:
    response = client.get("/passages", params={"q": "beta blockers mortality", "k": 2})
    body = response.json()
    assert body["query"] == "beta blockers mortality"
    assert body["passages"][0]["pmid"] == "1"
    assert body["passages"][0]["title"] == "Heart failure"


def test_ask_grounded(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "beta blockers mortality", "k": 3})
    body = response.json()
    assert body["abstained"] is False
    assert body["confidence"] == "GROUNDED"
    assert body["sources"][0]["pmid"] == "1"
    assert body["disclaimer"].endswith("Not medical advice.")


def test_ask_abstains_off_topic(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "lattice gauge theory physics"})
    body = response.json()
    assert body["abstained"] is True
    assert body["confidence"] == "ABSTAIN"
    assert body["disclaimer"].endswith("Not medical advice.")


def test_ask_stream_emits_events(client: TestClient) -> None:
    response = client.get("/ask/stream", params={"q": "beta blockers mortality"})
    assert response.status_code == 200
    text = response.text
    assert "event: retrieval" in text
    assert "event: answer" in text
    assert "event: done" in text


def test_metrics_counts_requests(client: TestClient) -> None:
    client.get("/health")
    client.post("/ask", json={"question": "beta blockers"})
    metrics = client.get("/metrics").json()
    assert metrics["counters"]["ask"] >= 1
    assert "answer" in metrics["latency_ms"]
