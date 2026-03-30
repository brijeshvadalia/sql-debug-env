# =============================================================================
# SQL Debug Environment — Dockerfile
# =============================================================================
# Multi-stage build:
#   builder  → installs dependencies with uv (fast, cached)
#   runtime  → slim production image
#
# HF Spaces:
#   - Listens on port 7860 (required)
#   - Set ENABLE_WEB_INTERFACE=true to mount Gradio UI at /web
#
# Local dev:
#   docker build -t sql-debug-env:latest .
#   docker run -p 7860:7860 sql-debug-env:latest
#
# With web UI:
#   docker run -p 7860:7860 -e ENABLE_WEB_INTERFACE=true sql-debug-env:latest
# =============================================================================

# ── Builder stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv (fast Python package manager)
RUN pip install --no-cache-dir uv==0.4.29

# Copy dependency specs first (layer cache friendly)
COPY pyproject.toml ./

# Install all dependencies into a virtual environment
RUN uv venv .venv && \
    uv pip install --no-cache-dir -e ".[dev]" --python .venv/bin/python || \
    uv pip install --no-cache-dir \
        "fastapi>=0.111.0" \
        "uvicorn[standard]>=0.30.0" \
        "pydantic>=2.7.0" \
        "httpx>=0.27.0" \
        "openai>=1.30.0" \
        "gradio>=4.36.0" \
        --python .venv/bin/python

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root user (good practice for HF Spaces)
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser . /app/

# HF Spaces requires files to be owned by user 1000
RUN chown -R appuser:appuser /app

USER appuser

# Activate venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# HF Spaces uses port 7860
EXPOSE 7860

# Health check — validates the /health endpoint on startup
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()" \
    || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info", \
     "--timeout-keep-alive", "30"]
