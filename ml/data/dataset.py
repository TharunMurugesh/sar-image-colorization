"""
ml/data/dataset.py
Canonical paired SAR/RGB dataset class for SIH1733.

Design rules (from build plan §2):
  - Each SAR file must map unambiguously to one reference via explicit ID rules.
  - Raises an error for ambiguous or missing pairs.
  - Returns SAR tensor + RGB target tensor + metadata dict.
  - Applies the same crop/pad to SAR and target.
  - target_rgb=None is allowed only for inference-only mode.
  - Never applies SAR-specific transforms to the reference.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ── Optional imports ─────────────────────────────────────────────────────────
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ─────────────────────────────────────────────────────────────────────────────
# Pair discovery
# ─────────────────────────────────────────────────────────────────────────────

RASTER_EXT = {".tif", ".tiff", ".geotiff"}
IMAGE_EXT   = {".png", ".jpg", ".jpeg"}
NUMPY_EXT   = {".npy", ".npz"}
ALL_EXT     = RASTER_EXT | IMAGE_EXT | NUMPY_EXT

# Filename markers — heuristic; override via subclass or custom matcher
SAR_PATTERN = re.compile(r"(sar|vv|vh|s1|sigma|backscatter)", re.IGNORECASE)
OPT_PATTERN = re.compile(r"(optical|rgb|color|colour|ref|target|gt|vis)", re.IGNORECASE)


def _extract_id(path: Path) -> Optional[str]:
    """Return a canonical pair ID from the filename stem.

    Strategy:
      1. If the stem contains a numeric block, use it.
      2. Otherwise use the full stem as the ID (minus any role suffix).
    """
    stem = path.stem
    m = re.search(r"[^\d](\d+)[^\d]?", stem)
    if m:
        return m.group(1).lstrip("0") or "0"
    # Strip common role suffixes and return remaining stem
    cleaned = re.sub(SAR_PATTERN.pattern + r"|" + OPT_PATTERN.pattern, "", stem, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\._\-]+", "_", cleaned).strip("_")
    return cleaned if cleaned else stem


def discover_pairs(
    data_root: Path,
    sar_pattern: Optional[re.Pattern] = None,
    opt_pattern: Optional[re.Pattern] = None,
) -> List[Dict[str, Path]]:
    """Scan *data_root* and return a list of ``{"sar": Path, "target": Path}`` dicts.

    Rules:
    - Files matching *sar_pattern* (default: SAR_PATTERN) → SAR role.
    - Files matching *opt_pattern* (default: OPT_PATTERN) → reference role.
    - IDs are extracted with _extract_id(); both must share the same ID.
    - Raises ``ValueError`` if the same ID has multiple SAR or multiple reference files.
    - Raises ``ValueError`` if no pairs are found.
    """
    sar_pat = sar_pattern or SAR_PATTERN
    opt_pat = opt_pattern or OPT_PATTERN

    sar_by_id: Dict[str, List[Path]] = {}
    opt_by_id: Dict[str, List[Path]] = {}
    no_role: List[Path] = []

    for p in sorted(data_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in ALL_EXT:
            continue
        pid = _extract_id(p)
        is_sar = bool(sar_pat.search(p.name))
        is_opt = bool(opt_pat.search(p.name))

        if is_sar and not is_opt:
            sar_by_id.setdefault(pid, []).append(p)
        elif is_opt and not is_sar:
            opt_by_id.setdefault(pid, []).append(p)
        else:
            no_role.append(p)

    # If heuristics failed, fall back to folder-based pairing:
    # images in a "sar" subfolder vs "optical"/"rgb" subfolder.
    if not sar_by_id and not opt_by_id:
        for p in sorted(data_root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in ALL_EXT:
                continue
            parts_lower = [part.lower() for part in p.parts]
            if any(k in " ".join(parts_lower) for k in ("sar", "vv", "vh", "s1")):
                pid = _extract_id(p)
                sar_by_id.setdefault(pid, []).append(p)
            elif any(k in " ".join(parts_lower) for k in ("optical", "rgb", "color", "ref", "target", "gt")):
                pid = _extract_id(p)
                opt_by_id.setdefault(pid, []).append(p)

    # Validate: no duplicate SAR or reference per ID
    errors = []
    for pid, paths in sar_by_id.items():
        if len(paths) > 1:
            errors.append(f"Ambiguous SAR files for ID '{pid}': {[str(p) for p in paths]}")
    for pid, paths in opt_by_id.items():
        if len(paths) > 1:
            errors.append(f"Ambiguous reference files for ID '{pid}': {[str(p) for p in paths]}")
    if errors:
        raise ValueError("Pairing ambiguity:\n" + "\n".join(errors))

    # Build pairs
    pairs = []
    all_ids = sorted(set(sar_by_id) & set(opt_by_id))
    if not all_ids:
        # Try by matched order if both sets have same length (fallback)
        sar_paths = sorted(sar_by_id.get(k, v)[0] if k in sar_by_id else v[0]
                           for k, v in sar_by_id.items())
        # Just report the issue clearly
        raise ValueError(
            f"No matched SAR/reference pairs found in {data_root}.\n"
            f"SAR IDs: {sorted(sar_by_id)[:10]}\n"
            f"Reference IDs: {sorted(opt_by_id)[:10]}\n"
            f"Files without a clear role: {[p.name for p in no_role[:10]]}\n\n"
            "Check that your dataset folder structure or filenames identify SAR and optical files."
        )

    for pid in all_ids:
        pairs.append({"sar": sar_by_id[pid][0], "target": opt_by_id[pid][0], "id": pid})

    return pairs


def discover_sar_only(data_root: Path) -> List[Dict[str, Path]]:
    """Return SAR-only entries for inference-only mode (no reference)."""
    entries = []
    for p in sorted(data_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in ALL_EXT:
            entries.append({"sar": p, "target": None, "id": _extract_id(p)})
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Low-level loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_raster_as_float32(path: Path) -> np.ndarray:
    """Load a GeoTIFF and return a float32 array of shape (C, H, W)."""
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required to load GeoTIFF files. Install with: pip install rasterio")
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # (C, H, W)
    return arr


def _load_image_as_float32(path: Path) -> np.ndarray:
    """Load a PNG/JPEG/BMP and return float32 (C, H, W) in [0, 1]."""
    if HAS_PIL:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        return arr.transpose(2, 0, 1)  # → (3, H, W)
    raise ImportError("Pillow is required to load image files. Install with: pip install Pillow")


def _load_numpy_as_float32(path: Path) -> np.ndarray:
    """Load .npy / .npz and return float32. Shape must be (C,H,W) or (H,W,C) or (H,W)."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        npz = np.load(path)
        key = list(npz.keys())[0]
        arr = npz[key].astype(np.float32)

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]          # → (1, H, W)
    elif arr.ndim == 3 and arr.shape[2] <= 4:
        arr = arr.transpose(2, 0, 1)        # (H, W, C) → (C, H, W)
    # else assume already (C, H, W)
    return arr


def load_any(path: Path) -> np.ndarray:
    """Load a file by extension into float32 (C, H, W)."""
    ext = path.suffix.lower()
    if ext in RASTER_EXT:
        return _load_raster_as_float32(path)
    if ext in IMAGE_EXT:
        return _load_image_as_float32(path)
    if ext in NUMPY_EXT:
        return _load_numpy_as_float32(path)
    raise ValueError(f"Unsupported file extension: {ext}")


# ─────────────────────────────────────────────────────────────────────────────
# Crop / pad to fixed patch size
# ─────────────────────────────────────────────────────────────────────────────

def _crop_or_pad(arr: np.ndarray, target_h: int, target_w: int, seed: Optional[int] = None) -> np.ndarray:
    """Deterministically centre-crop or centre-pad a (C, H, W) array."""
    _, h, w = arr.shape

    # Pad if smaller
    if h < target_h or w < target_w:
        ph = max(0, target_h - h)
        pw = max(0, target_w - w)
        arr = np.pad(
            arr,
            ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
            mode="reflect",
        )
        _, h, w = arr.shape

    # Random crop (same crop for SAR and target — caller must pass the same rng)
    top  = (h - target_h) // 2
    left = (w - target_w) // 2
    arr  = arr[:, top: top + target_h, left: left + target_w]
    return arr


def apply_crop_pad(
    sar: np.ndarray,
    target: Optional[np.ndarray],
    patch_h: int,
    patch_w: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply identical crop/pad to SAR and (optionally) target.

    Both arrays must have the same spatial dimensions before this call.
    """
    if target is not None:
        if sar.shape[1:] != target.shape[1:]:
            raise ValueError(
                f"SAR and target spatial dimensions differ: SAR={sar.shape[1:]}, target={target.shape[1:]}"
            )
    sar_out = _crop_or_pad(sar, patch_h, patch_w)
    target_out = _crop_or_pad(target, patch_h, patch_w) if target is not None else None
    return sar_out, target_out


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class SARColorizationDataset(Dataset):
    """Paired SAR / optical-RGB dataset for SIH1733.

    Args:
        pairs: list of dicts with keys ``sar``, ``target`` (Path), ``id`` (str).
               Obtain from ``discover_pairs()`` or ``discover_sar_only()``.
        patch_size: (H, W) tuple for centre-crop / pad. Default (256, 256).
        sar_transform: callable applied to the SAR float32 array (C, H, W)
                       **after** crop/pad. E.g. normalization, Lee filter.
        target_transform: callable applied to the target float32 array (3, H, W)
                          **after** crop/pad. Only RGB operations.
        joint_transform: callable applied to ``(sar, target)`` tuple for shared
                         geometric augmentations (flips, rotations).
        inference_only: if True, target_rgb will be None and no error is raised
                        for missing targets.
    """

    def __init__(
        self,
        pairs: List[Dict],
        patch_size: Tuple[int, int] = (256, 256),
        sar_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None,
        inference_only: bool = False,
    ):
        self.pairs = pairs
        self.patch_h, self.patch_w = patch_size
        self.sar_transform = sar_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.inference_only = inference_only

        # Validate
        if not inference_only:
            missing = [p for p in pairs if p.get("target") is None]
            if missing:
                raise ValueError(
                    f"{len(missing)} samples have no reference target but inference_only=False. "
                    f"Example: {missing[0]['sar']}. "
                    "Set inference_only=True for unpaired data."
                )

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str, None]]:
        entry = self.pairs[idx]
        sar_path    = Path(entry["sar"])
        target_path = Path(entry["target"]) if entry.get("target") else None
        pair_id     = entry.get("id", str(idx))

        # ── Load raw arrays ──────────────────────────────────────────────────
        sar_arr = load_any(sar_path)  # (C_sar, H, W) float32

        target_arr: Optional[np.ndarray] = None
        if target_path is not None:
            target_arr = load_any(target_path)  # (3, H, W) float32
            if target_arr.shape[0] != 3:
                # Attempt to keep only first 3 channels or convert grayscale → RGB
                if target_arr.shape[0] > 3:
                    target_arr = target_arr[:3]
                elif target_arr.shape[0] == 1:
                    target_arr = np.repeat(target_arr, 3, axis=0)
                else:
                    raise ValueError(
                        f"Reference image has {target_arr.shape[0]} channel(s); expected 3 RGB channels: {target_path}"
                    )

        # ── Align spatial dims then crop/pad ─────────────────────────────────
        sar_arr, target_arr = apply_crop_pad(sar_arr, target_arr, self.patch_h, self.patch_w)

        # ── Shared geometric augmentation ────────────────────────────────────
        if self.joint_transform is not None:
            sar_arr, target_arr = self.joint_transform(sar_arr, target_arr)

        # ── SAR-specific transform (normalization, despeckling, …) ───────────
        if self.sar_transform is not None:
            sar_arr = self.sar_transform(sar_arr)

        # ── RGB-specific transform (normalization, color jitter, …) ──────────
        if self.target_transform is not None and target_arr is not None:
            target_arr = self.target_transform(target_arr)

        # ── Validate no NaN / Inf ────────────────────────────────────────────
        if np.any(~np.isfinite(sar_arr)):
            raise RuntimeError(f"SAR array contains NaN/Inf after preprocessing: {sar_path}")
        if target_arr is not None and np.any(~np.isfinite(target_arr)):
            raise RuntimeError(f"Target array contains NaN/Inf after preprocessing: {target_path}")

        # ── Convert to tensors ───────────────────────────────────────────────
        sar_tensor    = torch.from_numpy(sar_arr.copy())
        target_tensor = torch.from_numpy(target_arr.copy()) if target_arr is not None else None

        metadata = {
            "pair_id":   pair_id,
            "sar_path":  str(sar_path),
            "target_path": str(target_path) if target_path else None,
            "sar_shape": list(sar_tensor.shape),
        }

        return {
            "sar":        sar_tensor,        # (C_sar, H, W)  float32
            "target_rgb": target_tensor,     # (3, H, W)      float32  or None
            "metadata":   metadata,
        }

    # ── Convenience helpers ──────────────────────────────────────────────────

    @classmethod
    def from_directory(
        cls,
        data_root: Union[str, Path],
        split_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> "SARColorizationDataset":
        """Construct from a directory, filtering by pair IDs if provided.

        Args:
            data_root: directory containing paired SAR + reference files.
            split_ids: if given, only pairs whose ``id`` is in this list are used.
            **kwargs: forwarded to ``__init__``.
        """
        root = Path(data_root)
        try:
            pairs = discover_pairs(root)
        except ValueError as e:
            # If no paired data found and caller passes inference_only=True, fall back
            if kwargs.get("inference_only", False):
                pairs = discover_sar_only(root)
            else:
                raise

        if split_ids is not None:
            split_set = set(str(s) for s in split_ids)
            pairs = [p for p in pairs if str(p["id"]) in split_set]

        return cls(pairs, **kwargs)

    def sar_band_count(self) -> int:
        """Return the number of SAR channels in the first sample (fast check)."""
        entry = self.pairs[0]
        arr = load_any(Path(entry["sar"]))
        return arr.shape[0]
