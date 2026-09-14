# ═══════════════════════════════════════════════════════════════
# MedAI — Dockerfile
# Production container for Render deployment
# ═══════════════════════════════════════════════════════════════

# ─── Stage 1: Builder ───
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /install

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install \
    --prefix=/runtime \
    --no-cache-dir \
    -r requirements.txt


# ─── Stage 2: Runtime ───
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    LOG_LEVEL=INFO

# Runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r medai && \
    useradd -r -g medai medai

WORKDIR /app

# Copy installed Python packages
COPY --from=builder /runtime /usr/local

# Copy application
COPY --chown=medai:medai . .

# Required directories
RUN mkdir -p logs models data && \
    chown -R medai:medai /app

# Run as non-root user
USER medai

# Render supplies the actual PORT at runtime.
# This is only documentation for the container.
EXPOSE 10000

# Container health check
# Uses the PORT supplied by Render, falling back to 10000.
HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=120s \
    --retries=3 \
    CMD sh -c 'curl -f http://localhost:${PORT:-10000}/health || exit 1'

# ─── Startup ───
# IMPORTANT:
# - Do not generate the medical PDF here.
# - Do not initialize ML models here.
# - Do not use 2 Gunicorn workers on the 512 MB Render instance.
# - Start Gunicorn immediately so Render can detect the HTTP port.
#
# Render provides $PORT automatically.
# If PORT is unavailable, 10000 is used as fallback.

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --worker-class sync --timeout 120 --keep-alive 5 --access-logfile - --error-logfile - --log-level info app:app"]