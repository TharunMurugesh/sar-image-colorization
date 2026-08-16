"""
backend/app/schemas/colorize.py
Pydantic v2 request/response schemas for the colorization API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Job creation (internal only — triggered by file upload) ──────────────────

class JobCreate(BaseModel):
    filename: str
    sar_channels: Optional[int] = None
    sar_height: Optional[int] = None
    sar_width: Optional[int] = None


# ── Job status values ─────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    """Lightweight status-only view of a job (used for polling)."""
    id: str
    status: str  # pending | running | done | error
    error_message: Optional[str] = None

    model_config = {"from_attributes": True}


# ── Full job response ─────────────────────────────────────────────────────────

class JobResponse(BaseModel):
    """Full representation of a colorization job."""
    id: str
    filename: str
    status: str
    created_at: datetime
    updated_at: datetime
    # URLs served by the backend; None until job finishes
    result_url: Optional[str] = Field(None, description="URL to download colorized PNG")
    uncertainty_url: Optional[str] = Field(None, description="URL to download uncertainty heatmap PNG")
    uncertainty_mean: Optional[float] = Field(None, description="Mean pixel uncertainty (0–1)")
    error_message: Optional[str] = None
    # SAR input metadata
    sar_channels: Optional[int] = None
    sar_height: Optional[int] = None
    sar_width: Optional[int] = None

    model_config = {"from_attributes": True}


# ── Health check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str                   # "ok" | "degraded"
    model_loaded: bool
    checkpoint_path: str
    checkpoint_exists: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    version: str = "day3"


# ── History list ──────────────────────────────────────────────────────────────

class HistoryResponse(BaseModel):
    total: int
    jobs: list[JobResponse]
