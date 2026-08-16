"""
backend/app/services/colorize_service.py
End-to-end SAR colorization pipeline.

Responsibilities:
1. Load the uploaded file (GeoTIFF / PNG / JPG / JPEG) using the existing
   ml.data.dataset.load_any loader — no duplication of ML logic.
2. Adapt to the model's expected channel count (3 channels).
3. Normalize to [0, 1].
4. Crop/pad to the configured patch size.
5. Run MC-Dropout inference via ml.evaluation.uncertainty.mc_dropout_inference.
6. Apply trust-gated rendering via ml.evaluation.uncertainty.trust_gated_rendering.
7. Persist colorized PNG and uncertainty heatmap PNG to results_dir.
8. Return statistics (uncertainty_mean) to the caller.

This module does NOT manage database state — that is the router's responsibility.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

# ── Reuse existing ML modules ────────────────────────────────────────────────
from ml.data.dataset import load_any, _crop_or_pad
from ml.evaluation.uncertainty import mc_dropout_inference, trust_gated_rendering

from backend.app.config import settings
from backend.app.services.model_loader import get_model, get_model_device

logger = logging.getLogger(__name__)

# Supported upload extensions — validated before writing to disk
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass
class ColorizationResult:
    """Return value from run_colorization()."""
    result_path: Path         # absolute path to colorized output PNG
    uncertainty_path: Path    # absolute path to uncertainty heatmap PNG
    uncertainty_mean: float   # mean pixel uncertainty (scalar in [0, ∞))
    sar_channels: int
    sar_height: int
    sar_width: int


class UnsupportedFileError(ValueError):
    """Raised when the uploaded file has an unsupported extension."""
    pass


# ── Channel adaptation ───────────────────────────────────────────────────────

def _adapt_channels(arr: np.ndarray, target_channels: int = 3) -> np.ndarray:
    """
    Adapt a (C, H, W) float32 array to `target_channels` channels.

    Rules:
    - C == target_channels → return as-is.
    - C == 1 → repeat to (target_channels, H, W).
    - C == 2 → stack as (VV, VH, VV) pattern for 3-channel target.
    - C > target_channels → take first `target_channels` bands.
    """
    c = arr.shape[0]
    if c == target_channels:
        return arr
    if c == 1:
        return np.repeat(arr, target_channels, axis=0)
    if c == 2 and target_channels == 3:
        # (VV, VH, VV) — common dual-pol SAR convention
        return np.stack([arr[0], arr[1], arr[0]], axis=0)
    if c > target_channels:
        return arr[:target_channels]
    # c < target_channels (e.g. 2 → 4): repeat last channel
    extra = target_channels - c
    return np.concatenate([arr, np.repeat(arr[-1:], extra, axis=0)], axis=0)


# ── Per-channel min-max normalisation ────────────────────────────────────────

def _normalize(arr: np.ndarray) -> np.ndarray:
    """
    Per-channel min-max normalise to [0, 1].
    Clips to handle outliers in SAR amplitude data.

    For GeoTIFF SAR data the raw values can be large float32 amplitudes;
    we use the 2nd–98th percentile per channel to avoid outlier stretching.
    """
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        band = arr[i]
        lo = np.percentile(band, 2)
        hi = np.percentile(band, 98)
        if hi - lo < 1e-9:
            out[i] = np.zeros_like(band)
        else:
            out[i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


# ── Uncertainty → heatmap PNG ────────────────────────────────────────────────

def _uncertainty_to_heatmap(uncertainty: torch.Tensor) -> Image.Image:
    """
    Convert a (1, H, W) uncertainty tensor to a colour-mapped PIL Image.
    Uses a green → yellow → red viridis-like mapping for intuitive reading.
    """
    u = uncertainty.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
    # Normalize to [0, 255]
    u_norm = (u - u.min()) / (u.max() - u.min() + 1e-9)
    u8 = (u_norm * 255).astype(np.uint8)
    # Simple thermal colourmap: cold (blue) → hot (red)
    h, w = u8.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = u8                          # Red increases with uncertainty
    rgb[:, :, 1] = (255 - u8) // 2            # Green decreases
    rgb[:, :, 2] = (255 - u8)                 # Blue high when certain
    return Image.fromarray(rgb, mode="RGB")


from ml.inference.pipeline import run_pipeline as ml_run_pipeline

def run_colorization(upload_path: Path, job_id: str) -> ColorizationResult:
    ext = upload_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileError(
            f"File type '{ext}' is not supported. "
            f"Accepted: {sorted(ALLOWED_EXTENSIONS)}"
        )

    # Check model readiness first
    get_model()

    res = ml_run_pipeline(raw_sar_path=upload_path, job_id=job_id)

    return ColorizationResult(
        result_path=res.result_path,
        uncertainty_path=res.uncertainty_path,
        uncertainty_mean=res.uncertainty_mean,
        sar_channels=res.sar_channels,
        sar_height=res.sar_height,
        sar_width=res.sar_width,
    )
