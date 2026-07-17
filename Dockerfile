# Meridian API image. Not medical advice.
FROM python:3.12-slim

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY examples ./examples

# Install the package with the serving extra (no dev tooling in the image).
RUN uv sync --extra serving --no-dev

EXPOSE 8000

# Build a demo store from the sample fixture, then serve it. Mount a real store at
# /data/corpus.sqlite and override this command to serve the real corpus.
CMD ["sh", "-c", "uv run meridian ingest examples/sample_pubmed.xml --db /data/corpus.sqlite && uv run python scripts/serve.py --db /data/corpus.sqlite --host 0.0.0.0 --port 8000"]
