# Phase 10 - Serving & performance (design doc)

- **Release:** v0.9.0
- **Goal:** the whole pipeline behind one API, with earned latency numbers.
- **Upstream:** int8 quantization and speculative decoding already exist in Zenith
  (`zenith.quantize`, `Generator.speculative_generate`); SSE patterns too. No new upstream.

## Module layout

```
serving/app.py            FastAPI app: /health, /passages, /ask, /ask/stream (SSE), /metrics.
serving/instrumentation.py StageTimer - per-stage wall-clock (P50/P95).
scripts/serve.py          uvicorn entry over a document store.
Dockerfile, docker-compose.yml, demo/index.html  the demo stack.
```

## Key decisions

- **Extractive is the default serving path.** `/ask` returns cited, verbatim sentences
  (hallucination-free) so the API is useful before the generator is trained; the
  generated+verified path layers on with the trained artifacts.
- **SSE for streaming.** `/ask/stream` emits `retrieval` → `answer` → `done` events; with
  the generated path this becomes token streaming + a final verification verdict.
- **Instrumented, not asserted.** `StageTimer` records per-stage latency; `/metrics`
  reports P50/P95 - the earned numbers (RAG.md §7), filled under real load.
- **Store stays behind the protocol.** SQLite backs serving today (`check_same_thread=False`
  for the threadpool); Postgres is the swap target (compose `postgres` profile), one
  `DocumentStore` implementation away.
- **Demo UI is zero-framework.** A single static page: query box, streamed answer,
  clickable PubMed citations, GROUNDED/ABSTAIN badge, and the "not medical advice" banner.

## Testing strategy (offline-only)

- FastAPI `TestClient` (no network): `/health`, `/passages`, `/ask` (grounded + abstain),
  `/ask/stream` event framing, `/metrics` counters. `StageTimer` percentiles.

## Environment constraint

The P50/P95 latency table, tokens/sec, and the int8 / speculative-decoding deltas are
**earned** - measured by a committed benchmark under fixed load on the real corpus +
trained models. The API, instrumentation, and demo stack are verified offline; the
numbers come from the real run. `docker compose up` builds a working demo from the sample
fixture.

## Exit criteria tracking

| Criterion | Status |
|---|---|
| `docker compose up` -> working demo from a clean machine | compose + Dockerfile + demo UI committed (builds on the sample fixture) |
| Latency table in BENCHMARKS.md | instrumentation ready; numbers pending real load |
| SSE demo | `/ask/stream` implemented; GIF pending real generator |
| >= 90% coverage held | maintained |
