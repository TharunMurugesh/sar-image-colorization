"""
scripts/ingest_sen12.py
Ingest SEN1-2 or Augmented-SEN1-2 paired patches into the project pipeline.

Expected input layout (Augmented-SEN1-2 from Kaggle):
  <data-root>/
    s1/   — Sentinel-1 SAR patches (PNG or TIFF, named by tile ID)
    s2/   — Sentinel-2 optical patches (PNG or TIFF, same tile IDs)

OR full SEN1-2 layout:
  <data-root>/
    SEN1-2/
      ROIs<N>_<season>/
        s1_<N>_<season>/   — SAR
        s2_<N>_<season>/   — optical

Outputs:
  data/sen12/sar/     — validated, renamed SAR patches
  data/sen12/rgb/     — validated optical patches
  data/sen12_manifest.csv   — full manifest with scene/split/traceability

Scene-disjoint split strategy:
  - Groups patches by their ROI / region prefix in the filename
  - Never mixes patches from the same ROI across train/val/test
  - 70% train / 15% val / 15% test by ROI count

Cloud filtering:
  - Optical patches with mean brightness > 210 (on 0-255 scale) are 
    likely cloud-dominated → skipped
  - Optical patches with std < 5 (nearly uniform → fog/water) can 
    optionally be kept (flag --keep-uniform)

Usage:
    kaggle datasets download shambac/augmented-sentinel-1-2 -p data/raw/
    unzip data/raw/augmented-sentinel-1-2.zip -d data/raw/sen12/
    python scripts/ingest_sen12.py --data-root data/raw/sen12 --max-patches 5000

    # For full SEN1-2 (one season):
    python scripts/ingest_sen12.py --data-root data/raw/SEN1-2/ROIs1970_fall \\
        --max-patches 8000 --layout fullsen12
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PATCH_SIZE = 256


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_as_rgb_uint8(path: Path) -> np.ndarray | None:
    """Load any image as (H, W, 3) uint8. Returns None on failure."""
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception:
        return None


def _load_as_gray_float(path: Path) -> np.ndarray | None:
    """Load any image as (H, W) float32 in [0,1]. Returns None on failure."""
    try:
        if path.suffix.lower() in {".tif", ".tiff"}:
            try:
                import rasterio
                with rasterio.open(path) as src:
                    arr = src.read().astype(np.float32).mean(axis=0)
            except ImportError:
                arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
        else:
            arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        if hi - lo < 1e-9:
            return None
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    except Exception:
        return None


# ── Cloud / quality filter ────────────────────────────────────────────────────

def _is_cloudy(rgb: np.ndarray, brightness_thresh: int = 210) -> bool:
    """True if optical patch is likely cloud-dominated (too bright)."""
    return float(rgb.mean()) > brightness_thresh


def _is_too_dark(sar: np.ndarray, dark_thresh: float = 0.02) -> bool:
    """True if SAR patch is nearly all zero (no-data region)."""
    return float(sar.mean()) < dark_thresh


# ── Pair discovery ────────────────────────────────────────────────────────────

def _find_pairs_flat(data_root: Path) -> List[Tuple[Path, Path, str]]:
    """Discover (sar_path, rgb_path, tile_id) from a flat s1/ s2/ layout."""
    s1_dir = data_root / "s1"
    s2_dir = data_root / "s2"
    if not s1_dir.exists() or not s2_dir.exists():
        # Try top-level
        s1_dir = data_root
        s2_dir = data_root

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    s1_files: Dict[str, Path] = {}
    s2_files: Dict[str, Path] = {}

    for p in sorted(s1_dir.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            s1_files[p.stem] = p

    for p in sorted(s2_dir.rglob("*")):
        if p.suffix.lower() in exts and p.is_file() and p.stem not in s1_files:
            s2_files[p.stem] = p

    # If same directory, try to separate by name pattern
    if not s2_files:
        all_files = {p.stem: p for p in sorted(data_root.rglob("*"))
                     if p.suffix.lower() in exts and p.is_file()}
        s1_files = {k: v for k, v in all_files.items() if "s1" in k.lower()}
        s2_files = {k: v for k, v in all_files.items() if "s2" in k.lower()}

    common = set(s1_files) & set(s2_files)
    if not common:
        # Match by stem (same name in different dirs)
        s1_stems = {p.stem: p for p in sorted(s1_dir.rglob("*"))
                    if p.suffix.lower() in exts and p.is_file()}
        s2_stems = {p.stem: p for p in sorted(s2_dir.rglob("*"))
                    if p.suffix.lower() in exts and p.is_file()}
        common = set(s1_stems) & set(s2_stems)
        return [(s1_stems[k], s2_stems[k], k) for k in sorted(common)]

    return [(s1_files[k], s2_files[k], k) for k in sorted(common)]


def _find_pairs_fullsen12(data_root: Path) -> List[Tuple[Path, Path, str]]:
    """Discover pairs from full SEN1-2 ROI directory layout."""
    pairs = []
    for s1_dir in sorted(data_root.rglob("s1_*")):
        if not s1_dir.is_dir():
            continue
        s2_name = s1_dir.name.replace("s1_", "s2_")
        s2_dir  = s1_dir.parent / s2_name
        if not s2_dir.is_dir():
            continue
        exts = {".png", ".tif", ".tiff"}
        s1_fs = {p.stem: p for p in sorted(s1_dir.iterdir())
                 if p.suffix.lower() in exts}
        s2_fs = {p.stem: p for p in sorted(s2_dir.iterdir())
                 if p.suffix.lower() in exts}
        for stem in sorted(set(s1_fs) & set(s2_fs)):
            region = s1_dir.name   # e.g. s1_1970_fall
            tile_id = f"{region}__{stem}"
            pairs.append((s1_fs[stem], s2_fs[stem], tile_id))
    return pairs


# ── Region extraction (for scene-disjoint split) ──────────────────────────────

def _extract_region(tile_id: str) -> str:
    """Extract a region/scene prefix from tile_id for scene-disjoint split."""
    # SEN1-2 tile IDs often contain ROI numbers: e.g. ROIs1970_fall__p001
    import re
    m = re.match(r"(ROI[sS]?\d+[_a-z]*)", tile_id, re.IGNORECASE)
    if m:
        return m.group(1)
    # Augmented SEN1-2: often "season_NNNN" → use first 3 digits as region
    m = re.search(r"(\d{3})", tile_id)
    if m:
        return m.group(1)
    return tile_id[:8]   # fallback: first 8 chars


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest SEN1-2 / Augmented-SEN1-2 patches into the project."
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root of downloaded SEN1-2 data")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sen12"),
                        help="Where to place processed patches")
    parser.add_argument("--manifest-out", type=Path, default=Path("data/sen12_manifest.csv"))
    parser.add_argument("--max-patches", type=int, default=6000,
                        help="Maximum number of patches to ingest (after quality filter)")
    parser.add_argument("--layout", choices=["flat", "fullsen12"], default="flat",
                        help="'flat' for Kaggle augmented, 'fullsen12' for ROI layout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac",   type=float, default=0.15)
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify the data layout without copying files")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_root = (PROJECT_ROOT / args.data_root).resolve()
    out_dir   = (PROJECT_ROOT / args.output_dir).resolve()
    out_sar   = out_dir / "sar"
    out_rgb   = out_dir / "rgb"

    if not args.verify_only:
        out_sar.mkdir(parents=True, exist_ok=True)
        out_rgb.mkdir(parents=True, exist_ok=True)

    print(f"[ingest_sen12] Scanning {data_root} (layout={args.layout}) ...")
    if args.layout == "fullsen12":
        pairs = _find_pairs_fullsen12(data_root)
    else:
        pairs = _find_pairs_flat(data_root)

    print(f"[ingest_sen12] Found {len(pairs)} candidate pairs.")

    if args.verify_only:
        print(f"[ingest_sen12] --verify-only: first 5 pairs:")
        for s, r, tid in pairs[:5]:
            print(f"  SAR: {s.name}  RGB: {r.name}  ID: {tid}")
        return

    # Scene-disjoint split by region
    regions: Dict[str, list] = {}
    for s, r, tid in pairs:
        reg = _extract_region(tid)
        regions.setdefault(reg, []).append((s, r, tid))

    region_list = sorted(regions.keys())
    rng.shuffle(region_list)
    n = len(region_list)
    n_train = max(1, round(n * args.train_frac))
    n_val   = max(1, round(n * args.val_frac))
    n_test  = n - n_train - n_val
    if n_test < 1:
        n_val  -= 1
        n_test  = n - n_train - n_val

    train_regions = set(region_list[:n_train])
    val_regions   = set(region_list[n_train:n_train + n_val])
    test_regions  = set(region_list[n_train + n_val:])

    print(f"[ingest_sen12] Regions: {n_train} train, {n_val} val, {n_test} test")

    manifest_rows = []
    accepted = skipped_cloud = skipped_dark = skipped_shape = 0
    split_counts = {"train": 0, "val": 0, "test": 0}

    # Process all pairs
    all_with_split = []
    for s_path, r_path, tid in pairs:
        reg = _extract_region(tid)
        if reg in train_regions:
            split = "train"
        elif reg in val_regions:
            split = "val"
        elif reg in test_regions:
            split = "test"
        else:
            continue
        all_with_split.append((s_path, r_path, tid, split))

    # Shuffle so max-patch limit samples evenly from all splits
    rng.shuffle(all_with_split)

    for s_path, r_path, tid, split in all_with_split:
        if accepted >= args.max_patches:
            break

        sar_arr = _load_as_gray_float(s_path)
        rgb_arr = _load_as_rgb_uint8(r_path)

        if sar_arr is None or rgb_arr is None:
            skipped_shape += 1
            continue

        # Size check
        if sar_arr.shape[0] < PATCH_SIZE or sar_arr.shape[1] < PATCH_SIZE:
            skipped_shape += 1
            continue

        # Quality filters
        if _is_cloudy(rgb_arr):
            skipped_cloud += 1
            continue
        if _is_too_dark(sar_arr):
            skipped_dark += 1
            continue

        # Resize to 256×256 if needed
        h, w = sar_arr.shape[:2]
        if h != PATCH_SIZE or w != PATCH_SIZE:
            sar_img  = Image.fromarray((sar_arr * 255).astype(np.uint8), "L")
            rgb_img  = Image.fromarray(rgb_arr, "RGB")
            sar_arr  = np.array(sar_img.resize((PATCH_SIZE, PATCH_SIZE))) / 255.0
            rgb_arr  = np.array(rgb_img.resize((PATCH_SIZE, PATCH_SIZE)))

        # Save
        safe_tid = tid.replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_sar_path = out_sar / f"sen12_{safe_tid}.png"
        out_rgb_path = out_rgb / f"sen12_{safe_tid}.png"
        Image.fromarray((sar_arr * 255).astype(np.uint8), "L").save(out_sar_path)
        Image.fromarray(rgb_arr, "RGB").save(out_rgb_path)

        rel_sar = out_sar_path.relative_to(PROJECT_ROOT).as_posix()
        rel_rgb = out_rgb_path.relative_to(PROJECT_ROOT).as_posix()
        manifest_rows.append({
            "scene_id":       _extract_region(tid),
            "source_dataset": "sen12",
            "split":          split,
            "sar_path":       rel_sar,
            "rgb_path":       rel_rgb,
            "patch_x":        0,
            "patch_y":        0,
            "patch_size":     PATCH_SIZE,
        })
        split_counts[split] += 1
        accepted += 1

    print(f"\n[ingest_sen12] Accepted: {accepted}")
    print(f"  Train: {split_counts['train']}, Val: {split_counts['val']}, Test: {split_counts['test']}")
    print(f"  Skipped — cloud: {skipped_cloud}, dark: {skipped_dark}, shape: {skipped_shape}")

    # Write manifest
    manifest_path = (PROJECT_ROOT / args.manifest_out).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scene_id", "source_dataset", "split",
            "sar_path", "rgb_path", "patch_x", "patch_y", "patch_size",
        ])
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"[ingest_sen12] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
