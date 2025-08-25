FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY mcp-sgr/pyproject.toml /app/
COPY mcp-sgr /app/mcp-sgr

RUN pip install --upgrade pip && pip wheel --no-cache-dir --no-deps --wheel-dir /wheels \
    .[ollama,langchain,autogen,crewai]

FROM python:3.11-slim
LABEL maintainer="MCP-SGR Team <team@mcp-sgr.dev>"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create non-root user
RUN useradd -ms /bin/bash appuser

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache --no-index --find-links /wheels mcp-sgr \
 && rm -rf /wheels

# Ensure data/log directories exist and are writable
RUN mkdir -p /app/data /app/logs \
 && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["mcp-sgr"]