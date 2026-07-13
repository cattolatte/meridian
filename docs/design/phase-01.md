# Phase 1 — Corpus ingestion + tokenizer (design doc)

- **Release:** v0.1.0-alpha
- **Goal:** a clean, deduplicated document store built from raw PubMed files by one
  command, plus a trained, versioned BPE tokenizer and a corpus statistics report.
- **Governing ADRs:** [0001](../adr/0001-scope.md) (domains, size), [0002](../adr/0002-chunking.md)
  (chunk = title+abstract), [0003](../adr/0003-tokenizer-corpus-mix.md) (tokenizer mix).

## Module layout (`src/meridian/`)

```
corpus/
  records.py    Document — the normalized record (PMID, title, abstract, year,
                journal, MeSH terms); chunk_text() composes the ADR-0002 unit.
  mesh.py       Domain MeSH filter (ADR-0001: cardiology, endocrinology, oncology).
  parser.py     PubMed baseline XML -> Document iterator; language/empty filtering.
  dedup.py      PMID de-duplication + near-duplicate title hashing.
  download.py   Resumable, checksummed downloader for baseline .xml.gz files.
  store.py      SQLite document store behind a DocumentStore repository interface
                (Postgres swap deferred to Phase 10).
  stats.py      Corpus statistics (counts, token estimate, year histogram, MeSH).
tokenization/
  artifact.py   Versioned tokenizer artifact: save/load Polaris BPE (vocab+merges)
                with training metadata. (Polaris ships no save/load — upstream TODO.)
  training.py   train_tokenizer(): mixed-corpus BPE via polaris.train_bpe; sweep.
  fertility.py  Fertility (tokens/word) metric for ADR-0003 selection.
```

The public online pipeline (retrieve/generate) does not exist yet; Phase 1 is
offline ingestion only. `scripts/ingest.py` is the one-command entry point.

## Design decisions & rationale

- **Repository interface over SQLite.** `store.py` exposes a `DocumentStore`
  protocol; the SQLite implementation is one class behind it, so the Phase 10
  Postgres swap changes one file and no callers.
- **Streaming everywhere.** Parser and store APIs take/return iterators so a
  200K-abstract (~GB of XML) rebuild never holds the corpus in memory.
- **Determinism.** Dedup title-hashing and tokenizer training are seeded/order-
  stable so a rebuild from the same raw files yields byte-identical artifacts.
- **Tokenizer artifact is self-describing.** It stores vocab, merges, end-of-word
  marker, mix ratio, vocab size, source sample sizes, and a corpus checksum, so a
  version pin fully reproduces tokenization (claims-hygiene requirement).

## Testing strategy (offline-only, ≥ 90% coverage)

- Small synthetic **PubMed XML fixtures** exercise the parser: valid records,
  missing/empty abstract, non-English, structured (multi-`<AbstractText>`)
  abstracts, missing MeSH, malformed entries.
- Dedup tests cover exact PMID collisions and near-duplicate titles.
- Store tests use a temp-file / in-memory SQLite database; round-trip and
  idempotent-rebuild assertions.
- Downloader tests use a local `file://`/fixture source and a fake interruption to
  assert resume + checksum verification — **no network**.
- Tokenizer tests train on tiny corpora and assert round-trip encode/decode,
  artifact save/load equivalence, and fertility monotonicity.

## Environment constraint (honest note)

Real corpus numbers (~200K abstracts) require downloading the PubMed baseline, which
needs network access the user runs deliberately — CI/tests never do. `scripts/ingest.py`
produces the real `benchmarks/corpus.md` when pointed at downloaded baseline files;
until then the report is validated end-to-end on a committed **sample** fixture and
its real-corpus rows are marked pending. No corpus number is stated from memory
(claims hygiene).

## Exit criteria tracking

| Criterion | Status |
|---|---|
| One-command `scripts/ingest.py` rebuilds the store from raw files | in progress |
| Tokenizer artifact versioned | in progress |
| Stats report in `benchmarks/corpus.md` | in progress |
| Parser edge-case tests | in progress |
| Offline tests, ≥ 90% coverage held | maintained |
