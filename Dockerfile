# ═══════════════════════════════════════════════════════════════
# MedAI — Dockerfile
# Multi-stage build for production-ready container
# ═══════════════════════════════════════════════════════════════

# ─── Stage 1: Builder (installs dependencies) ───
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /install

# Install system deps for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --prefix=/runtime --no-cache-dir -r requirements.txt

# ─── Stage 2: Final Runtime Image ───
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    FLASK_ENV=production \
    LOG_LEVEL=INFO

# Install only runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r medai && useradd -r -g medai medai

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /runtime /usr/local

# Copy application source
COPY --chown=medai:medai . .

# Create required directories
RUN mkdir -p logs models data && chown -R medai:medai /app

# Switch to non-root user
USER medai

# Expose application port
EXPOSE 5000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# ─── Startup ───
# Generate PDF + train ML models on first boot, then start Gunicorn
CMD ["sh", "-c", "\
  python src/pipelines/generate_medical_pdf.py && \
  python -c 'from src.ml.predictor import DiseasePredictor; DiseasePredictor()' && \
  gunicorn --bind 0.0.0.0:$PORT \
           --workers 2 \
           --worker-class sync \
           --timeout 120 \
           --keep-alive 5 \
           --access-logfile logs/access.log \
           --error-logfile logs/error.log \
           --log-level info \
           app:app"]
