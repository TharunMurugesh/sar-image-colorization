"""
ml/evaluation/metrics.py
Day 3 — Quantitative evaluation metrics for SAR colorization.

Metrics when ground-truth RGB reference exists:
  - PSNR  (Peak Signal-to-Noise Ratio, dB)
  - SSIM  (Structural Similarity Index)
  - Mean DeltaE76 (perceptual color difference in CIE Lab space)

Baselines:
  1. Grayscale baseline  — SAR intensity replicated to R, G, B
  2. Colormap baseline   — SAR intensity mapped through a fixed colormap (viridis)

Usage:
    from ml.evaluation.metrics import evaluate_dataset, compute_psnr, compute_ssim, compute_delta_e
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Per-image metrics (operate on numpy arrays in [0, 1])
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    Args:
        pred:   (H, W, 3) or (3, H, W) float in [0, 1]
        target: same shape
    Returns: PSNR in dB (higher is better; inf if identical)
    """
    pred, target = _to_hwc(pred), _to_hwc(target)
    mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Structural Similarity Index (mean over channels).
    Args:
        pred:   (H, W, 3) or (3, H, W) float in [0, 1]
        target: same shape
    Returns: SSIM in [-1, 1] (1 = identical)
    """
    pred, target = _to_hwc(pred), _to_hwc(target)
    ssim_vals = [_ssim_channel(pred[..., c], target[..., c]) for c in range(pred.shape[-1])]
    return float(np.mean(ssim_vals))


def compute_delta_e(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Mean DeltaE76 (Euclidean distance in CIE Lab space).
    Args:
        pred:   (H, W, 3) or (3, H, W) float in [0, 1] — RGB
        target: same shape — RGB reference
    Returns: mean DeltaE76 (lower is better; <1 indistinguishable, ~2 just noticeable)
    """
    pred, target = _to_hwc(pred), _to_hwc(target)
    lab_pred   = _rgb_to_lab(pred)
    lab_target = _rgb_to_lab(target)
    delta = np.sqrt(np.sum((lab_pred - lab_target) ** 2, axis=-1))
    return float(np.mean(delta))


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def grayscale_baseline(sar: np.ndarray) -> np.ndarray:
    """
    Baseline 1: replicate normalized SAR intensity into R, G, B.
    Args:
        sar: (C, H, W) or (H, W) float array
    Returns: (H, W, 3) float in [0, 1]
    """
    if sar.ndim == 3:
        gray = sar.mean(axis=0)    # (H, W)
    else:
        gray = sar
    gray = _normalize_01(gray)
    return np.stack([gray, gray, gray], axis=-1)   # (H, W, 3)


def colormap_baseline(sar: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """
    Baseline 2: map normalized SAR intensity through a fixed matplotlib colormap.
    Args:
        sar:  (C, H, W) or (H, W) float array
        cmap: matplotlib colormap name
    Returns: (H, W, 3) float in [0, 1]
    """
    import matplotlib.pyplot as plt  # lazy import so metrics module stays lightweight
    if sar.ndim == 3:
        gray = sar.mean(axis=0)
    else:
        gray = sar
    gray = _normalize_01(gray)
    cmap_fn = plt.get_cmap(cmap)
    rgba = cmap_fn(gray)     # (H, W, 4)
    return rgba[..., :3].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mc_passes: int = 1,
) -> dict:
    """
    Run the model on all batches in `loader` and return averaged metrics.

    Returns dict:
        psnr_mean, psnr_std,
        ssim_mean, ssim_std,
        delta_e_mean, delta_e_std,
        baseline_gray_psnr, baseline_gray_ssim, baseline_gray_delta_e,
        baseline_cmap_psnr, baseline_cmap_ssim, baseline_cmap_delta_e,
        n_samples
    """
    from ml.evaluation.uncertainty import mc_dropout_inference

    model.eval()

    psnr_list, ssim_list, de_list = [], [], []
    gray_psnr_list, gray_ssim_list, gray_de_list = [], [], []
    cmap_psnr_list, cmap_ssim_list, cmap_de_list = [], [], []

    with torch.no_grad():
        for batch in loader:
            sar_t   = batch["sar"].to(device)     # (B, C, H, W)
            tgt_t   = batch["target_rgb"]         # (B, 3, H, W)  CPU

            if tgt_t is None:
                continue  # inference-only batch, skip

            if mc_passes > 1:
                pred_t, _, _ = mc_dropout_inference(model, sar_t, num_samples=mc_passes)
            else:
                pred_t, _ = model(sar_t)

            pred_t = pred_t.cpu()

            B = pred_t.shape[0]
            for i in range(B):
                pred_np = pred_t[i].permute(1, 2, 0).numpy()       # (H, W, 3)
                tgt_np  = tgt_t[i].permute(1, 2, 0).numpy()        # (H, W, 3)
                sar_np  = sar_t[i].cpu().numpy()                    # (C, H, W)

                # Model metrics
                psnr_list.append(compute_psnr(pred_np, tgt_np))
                ssim_list.append(compute_ssim(pred_np, tgt_np))
                de_list.append(compute_delta_e(pred_np, tgt_np))

                # Grayscale baseline
                gray = grayscale_baseline(sar_np)
                gray_psnr_list.append(compute_psnr(gray, tgt_np))
                gray_ssim_list.append(compute_ssim(gray, tgt_np))
                gray_de_list.append(compute_delta_e(gray, tgt_np))

                # Colormap baseline
                cmap = colormap_baseline(sar_np)
                cmap_psnr_list.append(compute_psnr(cmap, tgt_np))
                cmap_ssim_list.append(compute_ssim(cmap, tgt_np))
                cmap_de_list.append(compute_delta_e(cmap, tgt_np))

    def _safe_mean(lst):
        return float(np.mean(lst)) if lst else float("nan")
    def _safe_std(lst):
        return float(np.std(lst)) if lst else float("nan")

    return {
        "n_samples":         len(psnr_list),
        "psnr_mean":         _safe_mean(psnr_list),
        "psnr_std":          _safe_std(psnr_list),
        "ssim_mean":         _safe_mean(ssim_list),
        "ssim_std":          _safe_std(ssim_list),
        "delta_e_mean":      _safe_mean(de_list),
        "delta_e_std":       _safe_std(de_list),
        "baseline_gray_psnr":  _safe_mean(gray_psnr_list),
        "baseline_gray_ssim":  _safe_mean(gray_ssim_list),
        "baseline_gray_de":    _safe_mean(gray_de_list),
        "baseline_cmap_psnr":  _safe_mean(cmap_psnr_list),
        "baseline_cmap_ssim":  _safe_mean(cmap_ssim_list),
        "baseline_cmap_de":    _safe_mean(cmap_de_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_hwc(arr: np.ndarray) -> np.ndarray:
    """Convert (C, H, W) → (H, W, C); (H, W) → (H, W, 1); (H, W, C) unchanged."""
    if arr.ndim == 2:
        return arr[..., np.newaxis]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
        return arr.transpose(1, 2, 0)
    return arr


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _ssim_channel(pred: np.ndarray, target: np.ndarray, win: int = 11) -> float:
    """Single-channel SSIM using sliding window (simplified, non-GPU)."""
    from scipy.ndimage import uniform_filter
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    p = pred.astype(np.float64)
    t = target.astype(np.float64)
    mu1 = uniform_filter(p, win)
    mu2 = uniform_filter(t, win)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = uniform_filter(p * p, win) - mu1_sq
    s2 = uniform_filter(t * t, win) - mu2_sq
    s12 = uniform_filter(p * t, win) - mu12
    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return float(np.mean(num / (den + 1e-12)))


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) float [0,1] RGB → CIE Lab."""
    from skimage.color import rgb2lab
    return rgb2lab(np.clip(rgb, 0, 1).astype(np.float32))
