# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for rasterio (GDAL) and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        gcc \
        g++ \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into /build/venv
COPY requirements.txt .
RUN python -m venv /build/venv && \
    /build/venv/bin/pip install --upgrade pip && \
    /build/venv/bin/pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime GDAL shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /build/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY ml/           ./ml/
COPY backend/      ./backend/

# Create runtime directories
RUN mkdir -p /runtime/uploads /runtime/results /runtime/checkpoints

# Environment defaults (overridable via docker-compose or -e flags)
ENV RUNTIME_DIR=/runtime \
    UPLOAD_DIR=/runtime/uploads \
    RESULTS_DIR=/runtime/results \
    DATABASE_URL=sqlite:////runtime/app.db \
    MODEL_CHECKPOINT=/runtime/checkpoints/best_model.pt \
    PATCH_SIZE=256 \
    MC_PASSES=10 \
    BACKEND_HOST=0.0.0.0 \
    BACKEND_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
