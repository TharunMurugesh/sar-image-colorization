"""
scripts/inspect_dataset.py
Day 1 — Task 2: Dataset audit

Recursively inspects the data root, detects file types, reports dimensions,
numeric statistics, pairing, and determines whether SAR values appear to be
in dB. Saves the results to reports/day1_dataset_audit.md.

Usage:
    python scripts/inspect_dataset.py --data-root data/raw --output reports/day1_dataset_audit.md
"""

import argparse
import os
import re
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Optional geospatial support ──────────────────────────────────────────────
try:
    import rasterio
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ── Optional image support ───────────────────────────────────────────────────
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RASTER_EXTENSIONS = {".tif", ".tiff", ".geotiff"}
IMAGE_EXTENSIONS  = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
NUMPY_EXTENSIONS  = {".npy", ".npz"}
ALL_EXTENSIONS    = RASTER_EXTENSIONS | IMAGE_EXTENSIONS | NUMPY_EXTENSIONS

# Typical dB range for Sentinel-1 amplitude-in-dB or calibrated power-in-dB
DB_LOWER, DB_UPPER = -40.0, 10.0

# SAR-ish filename patterns (heuristic)
SAR_PATTERNS  = re.compile(r"(sar|vv|vh|s1|sigma|backscatter)", re.IGNORECASE)
OPT_PATTERNS  = re.compile(r"(optical|rgb|color|colour|ref|target|gt|label|vis)", re.IGNORECASE)

MAX_SAMPLE_FILES = 20   # number of files to fully inspect
MAX_ARRAY_ELEMENTS = 10_000_000  # skip per-pixel stats for huge arrays


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def collect_files(root: Path) -> List[Path]:
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in ALL_EXTENSIONS:
            files.append(p)
    return files


def classify_extension(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in RASTER_EXTENSIONS:
        return "GeoTIFF/raster"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in NUMPY_EXTENSIONS:
        return "numpy"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# File loaders
# ─────────────────────────────────────────────────────────────────────────────

def _safe_stats(arr: np.ndarray) -> Dict:
    """Return min/max/mean/std, handling large arrays by subsampling."""
    flat = arr.ravel().astype(np.float64)
    if flat.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None, "nan_count": 0, "inf_count": 0}
    if flat.size > MAX_ARRAY_ELEMENTS:
        idx = np.random.choice(flat.size, MAX_ARRAY_ELEMENTS, replace=False)
        sample = flat[idx]
    else:
        sample = flat
    finite = sample[np.isfinite(sample)]
    return {
        "min":       float(np.min(finite)) if finite.size else None,
        "max":       float(np.max(finite)) if finite.size else None,
        "mean":      float(np.mean(finite)) if finite.size else None,
        "std":       float(np.std(finite)) if finite.size else None,
        "nan_count": int(np.sum(np.isnan(sample))),
        "inf_count": int(np.sum(np.isinf(sample))),
    }


def load_raster(path: Path) -> Optional[Dict]:
    if not HAS_RASTERIO:
        return None
    try:
        with rasterio.open(path) as src:
            data = src.read()  # (bands, H, W)
            crs  = src.crs
            transform = src.transform
            nodata = src.nodata
        stats = _safe_stats(data)
        return {
            "shape": data.shape,   # (C, H, W)
            "dtype": str(data.dtype),
            "crs":   str(crs) if crs else None,
            "transform_is_default": (transform == rasterio.transform.IDENTITY),
            "nodata": nodata,
            **stats,
        }
    except Exception as e:
        return {"error": str(e)}


def load_image(path: Path) -> Optional[Dict]:
    # Try PIL first
    if HAS_PIL:
        try:
            img = Image.open(path)
            arr = np.array(img)
            stats = _safe_stats(arr)
            return {
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "mode":  img.mode,
                "crs":   None,
                **stats,
            }
        except Exception:
            pass
    # Fallback to OpenCV
    if HAS_CV2:
        try:
            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                return {"error": "cv2.imread returned None"}
            # Convert from BGR
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            stats = _safe_stats(arr)
            return {"shape": arr.shape, "dtype": str(arr.dtype), "crs": None, **stats}
        except Exception as e:
            return {"error": str(e)}
    return None


def load_numpy(path: Path) -> Optional[Dict]:
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(path, allow_pickle=False)
            stats = _safe_stats(arr)
            return {"shape": arr.shape, "dtype": str(arr.dtype), **stats}
        else:  # npz
            npz = np.load(path, allow_pickle=False)
            results = {}
            for k, arr in npz.items():
                s = _safe_stats(arr)
                results[k] = {"shape": arr.shape, "dtype": str(arr.dtype), **s}
            return {"npz_keys": results}
    except Exception as e:
        return {"error": str(e)}


def inspect_file(path: Path) -> Dict:
    kind = classify_extension(path)
    info: Dict = {"path": str(path), "kind": kind, "size_bytes": path.stat().st_size}
    if kind == "GeoTIFF/raster":
        details = load_raster(path) or {"error": "rasterio unavailable"}
    elif kind == "image":
        details = load_image(path) or {"error": "no image library available"}
    elif kind == "numpy":
        details = load_numpy(path) or {"error": "numpy load failed"}
    else:
        details = {}
    info.update(details)
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic analysis
# ─────────────────────────────────────────────────────────────────────────────

def guess_sar_or_optical(path: Path, info: Dict) -> str:
    name = path.name
    if SAR_PATTERNS.search(name):
        return "SAR"
    if OPT_PATTERNS.search(name):
        return "optical/reference"
    # Check by channel count + value range
    shape = info.get("shape")
    if shape is not None:
        channels = shape[0] if len(shape) == 3 and shape[0] <= 4 else (shape[2] if len(shape) == 3 else 1)
        if channels in (1, 2):
            return "likely SAR (1-2 ch)"
        if channels == 3:
            return "likely optical RGB (3 ch)"
    return "unknown"


def guess_value_domain(info: Dict) -> str:
    mn = info.get("min")
    mx = info.get("max")
    if mn is None or mx is None:
        return "unknown"
    if mn >= 0 and mx <= 1.0 + 1e-3:
        return "normalized [0,1]"
    if mn >= 0 and mx <= 255 + 1e-3:
        return "uint8-like [0,255]"
    if mn < -20 and mx < 30:
        return "likely dB (negative range)"
    if mn >= 0 and mx < 1e6:
        return "amplitude/power (positive range)"
    return f"raw numeric [{mn:.2f}, {mx:.2f}]"


def looks_like_db(info: Dict) -> bool:
    mn = info.get("min")
    mx = info.get("max")
    if mn is None or mx is None:
        return False
    return mn < DB_UPPER and mn > -100 and mx < DB_UPPER


def extract_pair_id(path: Path) -> Optional[str]:
    """Try to extract a numeric or alphanumeric pair/scene ID from filename."""
    stem = path.stem
    # Try common patterns: _001, _S01, _T1, etc.
    m = re.search(r"[\._\-]([a-zA-Z]?\d+)[\._\-]?", stem)
    if m:
        return m.group(1)
    # Try just the first numeric block
    m = re.search(r"(\d+)", stem)
    return m.group(1) if m else None


def infer_pairing(files: List[Path]) -> Dict:
    """Group files by inferred pair/scene ID and check SAR+optical presence."""
    groups: Dict[str, List[Path]] = defaultdict(list)
    ungrouped = []
    for f in files:
        pid = extract_pair_id(f)
        if pid:
            groups[pid].append(f)
        else:
            ungrouped.append(f)
    # Count groups that have ≥2 files (potential SAR+optical pair)
    paired_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    return {
        "total_groups": len(groups),
        "paired_groups": len(paired_groups),
        "ungrouped_files": len(ungrouped),
        "sample_pairs": {k: [str(v) for v in vs] for k, vs in list(paired_groups.items())[:5]},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def fmt_shape(shape) -> str:
    if shape is None:
        return "N/A"
    return " × ".join(str(d) for d in shape)


def write_report(
    root: Path,
    files: List[Path],
    inspected: List[Dict],
    pairing: Dict,
    output: Path,
) -> None:
    ext_counts = Counter(f.suffix.lower() for f in files)
    role_counts = Counter(guess_sar_or_optical(f, i) for f, i in zip(files[:MAX_SAMPLE_FILES], inspected))

    lines = []
    a = lines.append

    a(f"# Day 1 Dataset Audit Report")
    a(f"")
    a(f"Generated: {datetime.utcnow().isoformat()} UTC")
    a(f"Data root: `{root}`")
    a(f"")
    a(f"---")
    a(f"")
    a(f"## 1. File Inventory")
    a(f"")
    a(f"| Extension | Count |")
    a(f"|-----------|------:|")
    for ext, cnt in sorted(ext_counts.items()):
        a(f"| `{ext}` | {cnt} |")
    a(f"| **Total** | **{len(files)}** |")
    a(f"")

    a(f"## 2. Detected Roles (heuristic)")
    a(f"")
    a(f"| Inferred role | Count |")
    a(f"|---------------|------:|")
    for role, cnt in role_counts.items():
        a(f"| {role} | {cnt} |")
    a(f"")

    a(f"## 3. Representative File Inspection")
    a(f"")
    a(f"> Showing up to {MAX_SAMPLE_FILES} files.")
    a(f"")
    for info in inspected:
        p = Path(info["path"])
        role = guess_sar_or_optical(p, info)
        domain = guess_value_domain(info)
        in_db = looks_like_db(info)
        a(f"### `{p.name}`")
        a(f"")
        a(f"| Field | Value |")
        a(f"|-------|-------|")
        a(f"| Path | `{info['path']}` |")
        a(f"| Kind | {info.get('kind', 'unknown')} |")
        a(f"| Size | {info.get('size_bytes', 0):,} bytes |")
        if "error" in info:
            a(f"| **Error** | {info['error']} |")
        else:
            a(f"| Shape (C×H×W or H×W×C) | {fmt_shape(info.get('shape'))} |")
            a(f"| Dtype | `{info.get('dtype', 'N/A')}` |")
            a(f"| CRS | `{info.get('crs', 'none')}` |")
            a(f"| Min | {info.get('min', 'N/A')} |")
            a(f"| Max | {info.get('max', 'N/A')} |")
            a(f"| Mean | {info.get('mean', 'N/A')} |")
            a(f"| Std | {info.get('std', 'N/A')} |")
            a(f"| NaN count | {info.get('nan_count', 0)} |")
            a(f"| Inf count | {info.get('inf_count', 0)} |")
            a(f"| Inferred role | {role} |")
            a(f"| Value domain | {domain} |")
            a(f"| Likely dB? | {'yes' if in_db else 'no'} |")
        a(f"")

    a(f"## 4. Pairing Analysis")
    a(f"")
    a(f"| Item | Value |")
    a(f"|------|-------|")
    a(f"| Total inferred groups | {pairing['total_groups']} |")
    a(f"| Groups with ≥2 files (potential pairs) | {pairing['paired_groups']} |")
    a(f"| Ungrouped files | {pairing['ungrouped_files']} |")
    a(f"")
    if pairing["sample_pairs"]:
        a(f"### Sample pairs detected")
        a(f"")
        for gid, members in pairing["sample_pairs"].items():
            a(f"**Group `{gid}`:**")
            for m in members:
                a(f"- `{Path(m).name}`")
            a(f"")

    a(f"## 5. Data Contract Summary")
    a(f"")
    a(f"| Item | Detected |")
    a(f"|------|----------|")

    # Determine dominant file type
    dom_ext = ext_counts.most_common(1)[0][0] if ext_counts else "unknown"
    if dom_ext in RASTER_EXTENSIONS:
        fmt = "GeoTIFF"
    elif dom_ext in IMAGE_EXTENSIONS:
        fmt = f"Image ({dom_ext})"
    elif dom_ext in NUMPY_EXTENSIONS:
        fmt = f"NumPy ({dom_ext})"
    else:
        fmt = dom_ext

    # Band count from first inspected SAR file
    sar_bands = "unknown"
    sar_domain = "unknown"
    has_reference = "unknown"
    for info in inspected:
        role = guess_sar_or_optical(Path(info["path"]), info)
        shape = info.get("shape")
        if "SAR" in role and shape:
            sar_bands = str(shape[0]) if len(shape) == 3 else "1"
            sar_domain = guess_value_domain(info)
            break
    for info in inspected:
        role = guess_sar_or_optical(Path(info["path"]), info)
        if "optical" in role or "RGB" in role:
            has_reference = "yes"
            break

    a(f"| SAR format | {fmt} |")
    a(f"| SAR bands | {sar_bands} |")
    a(f"| SAR numeric domain | {sar_domain} |")
    a(f"| Reference/optical found | {has_reference} |")
    a(f"| Pairing method | filename-based ID inference |")
    a(f"| Spatial alignment | not confirmed — verify manually |")
    a(f"")
    a(f"## 6. Action Items")
    a(f"")
    a(f"- [ ] Confirm SAR and optical files are spatially co-registered.")
    a(f"- [ ] Confirm that SAR numeric domain is correctly identified above.")
    a(f"- [ ] If reference is not genuine optical imagery, update the project plan (see §2 Data Contract).")
    a(f"- [ ] Run `scripts/make_splits.py` to generate scene-safe train/val/test splits.")
    a(f"")
    a(f"---")
    a(f"*Generated by `scripts/inspect_dataset.py` — SIH1733 Day 1 audit*")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[inspect_dataset] Report saved to: {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Day 1 — SIH1733 dataset audit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python scripts/inspect_dataset.py --data-root data/raw
              python scripts/inspect_dataset.py --data-root data/raw --output reports/audit.md
        """),
    )
    parser.add_argument("--data-root", required=True, type=Path, help="Root directory of the dataset.")
    parser.add_argument("--output", default=Path("reports/day1_dataset_audit.md"), type=Path)
    parser.add_argument("--max-files", default=MAX_SAMPLE_FILES, type=int,
                        help="Maximum files to fully inspect (default: 20).")
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    root = args.data_root.resolve()

    if not root.exists():
        print(f"[inspect_dataset] ERROR: data root does not exist: {root}", file=sys.stderr)
        print(f"[inspect_dataset] Place the SIH1733 dataset under {root} and re-run.")
        sys.exit(1)

    print(f"[inspect_dataset] Scanning: {root}")
    files = collect_files(root)
    print(f"[inspect_dataset] Found {len(files)} files with known extensions.")

    if not files:
        print("[inspect_dataset] No image/raster/numpy files found. Check --data-root.")
        sys.exit(1)

    # Sample up to max_files for detailed inspection
    np.random.shuffle(files_arr := np.array(files, dtype=object))
    sample_files = list(files_arr[: args.max_files])

    print(f"[inspect_dataset] Inspecting {len(sample_files)} files ...")
    inspected = []
    for p in sample_files:
        print(f"  -> {p.name}")
        inspected.append(inspect_file(p))

    pairing = infer_pairing(files)

    write_report(root, files, inspected, pairing, args.output)

    # Print data contract summary to stdout
    print("\n[inspect_dataset] --- DATA CONTRACT SUMMARY ---")
    for info in inspected[:3]:
        p = Path(info["path"])
        role = guess_sar_or_optical(p, info)
        domain = guess_value_domain(info)
        shape = info.get("shape", "?")
        print(f"  {p.name:40s}  role={role:20s}  shape={str(shape):20s}  domain={domain}")
    print("[inspect_dataset] Full report:", args.output)


if __name__ == "__main__":
    main()
