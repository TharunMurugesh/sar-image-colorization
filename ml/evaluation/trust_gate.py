"""
ml/evaluation/trust_gate.py
Day 4 — Trust gating with percentile-based thresholds.

Computes low and high confidence thresholds from the validation set,
and attenuates low-confidence regions toward SAR grayscale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from ml.evaluation.uncertainty import mc_dropout_inference


def compute_validation_thresholds(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    low_percentile: float = 10.0,
    high_percentile: float = 90.0,
    mc_passes: int = 10,
) -> Tuple[float, float]:
    """
    Compute low and high uncertainty thresholds based on validation set percentiles.
    Note: We are working with uncertainty (variance), so we'll flip the logic:
    Low uncertainty = high confidence.
    """
    model.eval()
    uncertainties = []

    print("[trust_gate] Computing validation uncertainty thresholds...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val Thresholds"):
            sar_t = batch["sar"].to(device)
            # Run MC dropout
            _, unc_t, _ = mc_dropout_inference(model, sar_t, num_samples=mc_passes)
            uncertainties.append(unc_t.cpu().numpy().flatten())

    if not uncertainties:
        return 0.0, 1.0

    all_unc = np.concatenate(uncertainties)
    
    # We want to map:
    # low_unc (high confidence) -> alpha = 1
    # high_unc (low confidence) -> alpha = 0
    # Because 'confidence' is inversely related to 'uncertainty' (variance), 
    # we use the percentiles of uncertainty directly.
    # The lowest uncertainty values are the top percentiles of confidence.
    # We map unc <= low_thresh to alpha=1 (trusted)
    # We map unc >= high_thresh to alpha=0 (untrusted)
    
    low_thresh = float(np.percentile(all_unc, low_percentile))
    high_thresh = float(np.percentile(all_unc, high_percentile))
    
    # If there's no variance, avoid div by zero
    if high_thresh - low_thresh < 1e-9:
        high_thresh = low_thresh + 1e-9

    print(f"[trust_gate] Uncertainty thresholds: low={low_thresh:.6f}, high={high_thresh:.6f}")
    return low_thresh, high_thresh


def trust_gated_rendering_percentile(
    pred_rgb: Tensor,
    sar_input: Tensor,
    uncertainty: Tensor,
    low_thresh: float,
    high_thresh: float,
) -> Tuple[Tensor, float]:
    """
    Attenuates low-confidence regions toward SAR grayscale using percentile thresholds.
    
    Args:
        pred_rgb: Predicted color image (B, 3, H, W)
        sar_input: Original SAR input (B, C, H, W)
        uncertainty: Estimated uncertainty map (B, 1, H, W)
        low_thresh: Uncertainty below this is fully trusted (alpha=1)
        high_thresh: Uncertainty above this is fully untrusted (alpha=0)
        
    Returns:
        gated_pred: The blended image
        trust_score: Scalar 100 * mean(alpha)
    """
    # Convert SAR input to grayscale
    sar_gray = sar_input.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    
    # Compute alpha map:
    # alpha = 1 when uncertainty <= low_thresh
    # alpha = 0 when uncertainty >= high_thresh
    # linear interpolation in between
    alpha = 1.0 - (uncertainty - low_thresh) / (high_thresh - low_thresh)
    alpha = torch.clamp(alpha, 0.0, 1.0)
    
    gated_pred = alpha * pred_rgb + (1 - alpha) * sar_gray
    trust_score = float((100.0 * alpha.mean()).cpu().item())
    
    return gated_pred, trust_score


def load_thresholds(path: Path) -> Tuple[float, float]:
    if not path.exists():
        return 0.001, 0.010  # default fallbacks
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("low_thresh", 0.001), data.get("high_thresh", 0.010)


def save_thresholds(path: Path, low_thresh: float, high_thresh: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"low_thresh": low_thresh, "high_thresh": high_thresh}, f, indent=2)
