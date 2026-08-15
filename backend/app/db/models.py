"""
backend/app/db/models.py
SQLAlchemy ORM models.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from backend.app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ColorizationJob(Base):
    """Represents one colorization request (upload → inference → result)."""

    __tablename__ = "colorization_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # Original uploaded filename
    filename = Column(String(255), nullable=False)
    # pending | running | done | error
    status = Column(String(20), nullable=False, default="pending")
    # ISO-8601 timestamps stored as UTC
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)
    # Absolute paths to saved output files (None until job completes)
    result_path = Column(Text, nullable=True)
    uncertainty_path = Column(Text, nullable=True)
    # Scalar summary of uncertainty map (mean pixel value)
    uncertainty_mean = Column(Float, nullable=True)
    # Human-readable error message if status == 'error'
    error_message = Column(Text, nullable=True)
    # SAR input metadata
    sar_channels = Column(Integer, nullable=True)
    sar_height = Column(Integer, nullable=True)
    sar_width = Column(Integer, nullable=True)
