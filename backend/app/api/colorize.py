"""
backend/app/api/colorize.py
POST  /api/colorize        — upload SAR file → create job → run inference
GET   /api/colorize/{id}   — poll job status / retrieve result URLs
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from fastapi import File as FastAPIFile
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db import get_db
from backend.app.db.models import ColorizationJob
from backend.app.schemas.colorize import JobResponse, JobStatus
from backend.app.services.colorize_service import (
    ALLOWED_EXTENSIONS,
    UnsupportedFileError,
    run_colorization,
)
from backend.app.services.model_loader import ModelNotReadyError

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _job_to_response(job: ColorizationJob, request_base: str = "") -> JobResponse:
    """Convert ORM object → Pydantic response, generating public URLs."""
    result_url: str | None = None
    uncertainty_url: str | None = None

    if job.result_path:
        fname = Path(job.result_path).name
        result_url = f"{request_base}/api/results/{fname}"
    if job.uncertainty_path:
        fname = Path(job.uncertainty_path).name
        uncertainty_url = f"{request_base}/api/results/{fname}"

    return JobResponse(
        id=job.id,
        filename=job.filename,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result_url=result_url,
        uncertainty_url=uncertainty_url,
        uncertainty_mean=job.uncertainty_mean,
        error_message=job.error_message,
        sar_channels=job.sar_channels,
        sar_height=job.sar_height,
        sar_width=job.sar_width,
    )


# ── Background worker ─────────────────────────────────────────────────────────

def _run_job_background(job_id: str, upload_path: Path) -> None:
    """
    Executed in a background thread by FastAPI BackgroundTasks.
    Opens its own DB session (BackgroundTasks runs after response is sent).
    """
    from backend.app.db.session import SessionLocal

    db = SessionLocal()
    try:
        job = db.query(ColorizationJob).filter(ColorizationJob.id == job_id).first()
        if job is None:
            logger.error("Background job %s: record not found in DB", job_id)
            return

        # Mark as running
        job.status = "running"
        db.commit()

        result = run_colorization(upload_path, job_id)

        # Mark as done
        job.status = "done"
        job.result_path = str(result.result_path)
        job.uncertainty_path = str(result.uncertainty_path)
        job.uncertainty_mean = result.uncertainty_mean
        job.sar_channels = result.sar_channels
        job.sar_height = result.sar_height
        job.sar_width = result.sar_width
        db.commit()
        logger.info("Job %s completed successfully.", job_id)

    except ModelNotReadyError as exc:
        logger.warning("Job %s failed: no checkpoint — %s", job_id, exc)
        if job := db.query(ColorizationJob).filter(ColorizationJob.id == job_id).first():
            job.status = "error"
            job.error_message = (
                "No trained model checkpoint is available. "
                "Train SSGUNet first and place the checkpoint at the configured path. "
                f"Detail: {exc}"
            )
            db.commit()

    except UnsupportedFileError as exc:
        logger.warning("Job %s: unsupported file — %s", job_id, exc)
        if job := db.query(ColorizationJob).filter(ColorizationJob.id == job_id).first():
            job.status = "error"
            job.error_message = str(exc)
            db.commit()

    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s: unexpected error during inference", job_id)
        if job := db.query(ColorizationJob).filter(ColorizationJob.id == job_id).first():
            job.status = "error"
            job.error_message = f"Inference error: {type(exc).__name__}: {exc}"
            db.commit()

    finally:
        db.close()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/colorize",
    response_model=JobResponse,
    status_code=202,
    summary="Upload a SAR image and start colorization",
    tags=["colorization"],
)
async def create_colorization_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = FastAPIFile(..., description="SAR image: GeoTIFF / PNG / JPG / JPEG"),
    db: Session = Depends(get_db),
) -> JobResponse:
    """
    Upload a SAR image and enqueue a colorization job.

    Accepted formats: **.tif**, **.tiff**, **.png**, **.jpg**, **.jpeg**

    Returns HTTP 202 Accepted immediately with a job record.
    Poll `GET /api/colorize/{id}` to check completion.

    Raises HTTP 400 if the file type is unsupported.
    Raises HTTP 503 if no trained checkpoint is available
    (checked before accepting the file to give fast feedback).
    """
    # ── Validate extension before touching the file ───────────────────────────
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File type '{ext}' is not supported. "
                f"Accepted formats: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    # ── Early checkpoint check (fast feedback before saving the file) ─────────
    from backend.app.services.model_loader import is_checkpoint_available
    if not is_checkpoint_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model checkpoint is available. "
                "Train SSGUNet first (see ml/training/train.py) and place the "
                ".pt file at the path configured by MODEL_CHECKPOINT."
            ),
        )

    # ── Save upload to disk ───────────────────────────────────────────────────
    job_id = str(uuid.uuid4())
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"{job_id}{ext}"

    try:
        with upload_path.open("wb") as out_f:
            shutil.copyfileobj(file.file, out_f)
    finally:
        await file.close()

    # ── Create DB record ──────────────────────────────────────────────────────
    job = ColorizationJob(id=job_id, filename=filename, status="pending")
    db.add(job)
    db.commit()
    db.refresh(job)

    # ── Enqueue background inference ──────────────────────────────────────────
    background_tasks.add_task(_run_job_background, job_id, upload_path)

    logger.info("Job %s created for file '%s'", job_id, filename)
    return _job_to_response(job)


@router.get(
    "/colorize/{job_id}",
    response_model=JobResponse,
    summary="Get colorization job status and result",
    tags=["colorization"],
)
def get_colorization_job(
    job_id: str,
    db: Session = Depends(get_db),
) -> JobResponse:
    """
    Retrieve the current status and result of a colorization job.

    - **pending** / **running**: inference in progress.
    - **done**: `result_url` and `uncertainty_url` are populated.
    - **error**: `error_message` contains the reason.
    """
    job = db.query(ColorizationJob).filter(ColorizationJob.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _job_to_response(job)
