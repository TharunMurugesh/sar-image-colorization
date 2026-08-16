"""
ml/inference/pipeline.py
Day 5 — Canonical ML inference pipeline service for SSG-U-Net.

Exposes run_pipeline(raw_sar_image_path, job_id, ...) used by:
  - FastAPI backend (/api/colorize)
  - Evaluation & reporting scripts
  - Interactive CLI / notebook testing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from ml.data.dataset import load_any, _crop_or_pad
from ml.evaluation.uncertainty import mc_dropout_inference, trust_gated_rendering
from backend.app.config import settings
from backend.app.services.model_loader import get_model, get_model_device

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass
class PipelineResult:
    """Canonical result object returned by run_pipeline()."""
    result_path: Path         # Path to trust-gated output PNG
    uncertainty_path: Path    # Path to uncertainty heatmap PNG
    raw_colorized_path: Path  # Path to raw colorized RGB PNG
    trust_score: float        # Relative trust score in [0, 100] (100 * mean(alpha))
    uncertainty_mean: float   # Mean pixel uncertainty
    processing_time: float    # Duration in seconds
    sar_channels: int
    sar_height: int
    sar_width: int
    model_path: str


def _adapt_channels(arr: np.ndarray, target_channels: int = 3) -> np.ndarray:
    """Adapt (C, H, W) array to target_channels."""
    c = arr.shape[0]
    if c == target_channels:
        return arr
    if c == 1:
        return np.repeat(arr, target_channels, axis=0)
    if c == 2 and target_channels == 3:
        return np.stack([arr[0], arr[1], arr[0]], axis=0)
    if c > target_channels:
        return arr[:target_channels]
    extra = target_channels - c
    return np.concatenate([arr, np.repeat(arr[-1:], extra, axis=0)], axis=0)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Percentile normalization per channel to [0, 1]."""
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


def _uncertainty_to_heatmap(uncertainty: torch.Tensor) -> Image.Image:
    """Convert uncertainty tensor to RGB heat map PIL image."""
    u = uncertainty.squeeze(0).squeeze(0).cpu().numpy()
    u_norm = (u - u.min()) / (u.max() - u.min() + 1e-9)
    u8 = (u_norm * 255).astype(np.uint8)
    h, w = u8.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = u8
    rgb[:, :, 1] = (255 - u8) // 2
    rgb[:, :, 2] = (255 - u8)
    return Image.fromarray(rgb, mode="RGB")


def run_pipeline(
    raw_sar_path: Union[str, Path],
    job_id: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> PipelineResult:
    """
    Execute end-to-end ML inference path:
      SAR preprocessing -> SAR structure map -> SSG-U-Net -> RGB prediction
      -> MC-Dropout uncertainty -> trust-gated output
    """
    start_time = time.time()
    path = Path(raw_sar_path)
    job_id = job_id or f"infer_{int(start_time)}"
    out_dir = Path(output_dir or settings.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Loading SAR image: %s", job_id, path)
    arr = load_any(path)
    orig_c, orig_h, orig_w = arr.shape

    # Preprocessing
    arr = _adapt_channels(arr, target_channels=settings.model_in_channels)
    arr = _normalize(arr)
    patch_sz = settings.patch_size
    arr = _crop_or_pad(arr, patch_sz, patch_sz)

    device = get_model_device()
    x = torch.from_numpy(arr.copy()).unsqueeze(0).to(device)

    # MC-Dropout inference
    model = get_model()
    mean_pred, uncertainty, _edges = mc_dropout_inference(
        model, x, num_samples=settings.mc_passes
    )

    # Trust-gated rendering
    gated = trust_gated_rendering(mean_pred, x, uncertainty, tau=settings.trust_tau)

    # Calculate trust score: 100 * mean(alpha)
    # alpha = exp(-uncertainty / tau)
    alpha = torch.exp(-uncertainty / settings.trust_tau)
    trust_score = float((100.0 * alpha.mean()).cpu().item())
    uncertainty_mean = float(uncertainty.mean().cpu().item())

    # Save outputs
    result_path = out_dir / f"{job_id}_colorized.png"
    uncertainty_path = out_dir / f"{job_id}_uncertainty.png"
    raw_colorized_path = out_dir / f"{job_id}_raw.png"

    # Trust-gated composite PNG
    gated_np = gated.squeeze(0).permute(1, 2, 0).cpu().numpy()
    gated_u8 = (np.clip(gated_np, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(gated_u8, mode="RGB").save(result_path)

    # Raw predicted colorized PNG
    raw_np = mean_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    raw_u8 = (np.clip(raw_np, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(raw_u8, mode="RGB").save(raw_colorized_path)

    # Uncertainty heatmap PNG
    _uncertainty_to_heatmap(uncertainty).save(uncertainty_path)

    proc_time = time.time() - start_time

    return PipelineResult(
        result_path=result_path,
        uncertainty_path=uncertainty_path,
        raw_colorized_path=raw_colorized_path,
        trust_score=trust_score,
        uncertainty_mean=uncertainty_mean,
        processing_time=proc_time,
        sar_channels=orig_c,
        sar_height=orig_h,
        sar_width=orig_w,
        model_path=str(settings.model_checkpoint),
    )
