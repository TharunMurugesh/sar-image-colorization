"""
backend/app/main.py
FastAPI application entry point.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.db import create_tables
from backend.app.api import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup:
    1. Create runtime directories.
    2. Create SQLite tables.
    3. Log checkpoint availability (do NOT load model — it's lazy-loaded on first request).
    """
    # Ensure runtime directories exist
    for d in (settings.runtime_dir, settings.upload_dir, settings.results_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    # Create database schema
    create_tables()
    logger.info("Database tables ready — %s", settings.database_url)

    # Checkpoint status
    ckpt = Path(settings.model_checkpoint)
    if ckpt.exists():
        logger.info("Checkpoint found: %s", ckpt)
    else:
        logger.warning(
            "⚠  No checkpoint at '%s'. "
            "POST /api/colorize will return HTTP 503 until a checkpoint is placed there.",
            ckpt,
        )

    yield  # Server is running

    logger.info("Shutting down SAR Colorization backend.")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="SAR Image Colorization API",
        description=(
            "SAR-Structure-Guided U-Net (SSG-UNet) colorization service. "
            "Upload a SAR image (GeoTIFF / PNG / JPG) and receive a colorized RGB "
            "image along with an MC-Dropout uncertainty heatmap."
        ),
        version="1.0.0-day3",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API routes ────────────────────────────────────────────────────────────
    app.include_router(api_router)

    # ── Static file serving for result PNGs ──────────────────────────────────
    # Serves /api/results/<filename> from the results directory.
    # Created lazily so it doesn't fail if results_dir doesn't exist yet.
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/api/results",
        StaticFiles(directory=str(results_dir)),
        name="results",
    )

    return app


app = create_app()
