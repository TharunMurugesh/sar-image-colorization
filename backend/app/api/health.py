"""
backend/app/api/health.py
GET /api/health — model and system status.
"""
from __future__ import annotations

import torch
from fastapi import APIRouter

from backend.app.config import settings
from backend.app.schemas.colorize import HealthResponse
from backend.app.services.model_loader import is_model_loaded, is_checkpoint_available

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System and model health check",
    tags=["system"],
)
def health_check() -> HealthResponse:
    """
    Returns the operational status of the backend.

    - **model_loaded**: True once the checkpoint has been loaded into VRAM/RAM.
    - **checkpoint_exists**: True if the .pt file is present on disk.
    - **status**: 'ok' when a checkpoint is available; 'degraded' otherwise.
    """
    gpu_available = torch.cuda.is_available()
    gpu_name: str | None = (
        torch.cuda.get_device_name(0) if gpu_available else None
    )
    checkpoint_exists = is_checkpoint_available()

    return HealthResponse(
        status="ok" if checkpoint_exists else "degraded",
        model_loaded=is_model_loaded(),
        checkpoint_path=str(settings.model_checkpoint),
        checkpoint_exists=checkpoint_exists,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
    )
