"""
backend/app/api/history.py
GET /api/history — paginated list of past colorization jobs.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.db.models import ColorizationJob
from backend.app.schemas.colorize import HistoryResponse, JobResponse
from backend.app.api.colorize import _job_to_response

router = APIRouter()


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="List past colorization jobs",
    tags=["history"],
)
def list_history(
    limit: int = Query(default=20, ge=1, le=100, description="Max results to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    status: Optional[str] = Query(default=None, description="Filter by status: pending|running|done|error"),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    """
    Returns a paginated list of colorization jobs, ordered by creation time
    (most recent first).

    Supports optional filtering by **status** and pagination via **limit**/**offset**.
    """
    query = db.query(ColorizationJob).order_by(ColorizationJob.created_at.desc())

    if status is not None:
        query = query.filter(ColorizationJob.status == status)

    total = query.count()
    jobs = query.offset(offset).limit(limit).all()

    return HistoryResponse(
        total=total,
        jobs=[_job_to_response(j) for j in jobs],
    )
