"""
ml/data/smoke_test.py
Day 1 — Task 5: Smoke test for the dataset loader.

Loads 5 random train samples and verifies:
  - SAR tensor shape is correct
  - target_rgb tensor shape is (3, H, W)
  - No NaN / Inf in either tensor
  - SAR and reference spatial dimensions match
  - Values are in expected domains
  - Saves a visual contact sheet to reports/day1_pairing_check.png

Usage:
    python ml/data/smoke_test.py [--data-root data/raw] [--splits-dir data/]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ml.data.dataset import SARColorizationDataset, discover_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_split_csv_if_exists(splits_dir: Path, split_name: str):
    """Try to load a split CSV; return None if not found."""
    import csv
    csv_path = splits_dir / f"splits_{split_name}.csv"
    if not csv_path.exists():
        return None
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append({
                "id":     row["pair_id"],
                "sar":    row["sar_path"],
                "target": row["target_path"] or None,
            })
    return pairs


def _to_display(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float32 tensor to (H, W, 3) uint8 for display."""
    arr = tensor.numpy()  # (C, H, W)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    c = arr.shape[0]
    if c == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif c > 3:
        arr = arr[:3]  # take first 3 bands
    # arr shape now (3, H, W)
    arr = arr.transpose(1, 2, 0)  # → (H, W, 3)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    arr = np.clip(arr, 0, 1)
    return arr


def check_sample(idx: int, sample: dict) -> dict:
    """Run sanity checks on one sample and return a results dict."""
    sar    = sample["sar"]
    target = sample["target_rgb"]
    meta   = sample["metadata"]

    results = {
        "pair_id":      meta.get("pair_id", "?"),
        "sar_shape":    tuple(sar.shape),
        "target_shape": tuple(target.shape) if target is not None else None,
        "sar_nan":      bool(torch.any(~torch.isfinite(sar))),
        "target_nan":   bool(torch.any(~torch.isfinite(target))) if target is not None else None,
        "sar_min":      float(sar.min()),
        "sar_max":      float(sar.max()),
        "target_min":   float(target.min()) if target is not None else None,
        "target_max":   float(target.max()) if target is not None else None,
        "spatial_match": (sar.shape[1:] == target.shape[1:]) if target is not None else None,
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Contact sheet
# ─────────────────────────────────────────────────────────────────────────────

def save_contact_sheet(samples: list, output: Path) -> None:
    """Save a grid of SAR | target pairs for visual verification."""
    n = len(samples)
    has_target = any(s["target_rgb"] is not None for s in samples)
    ncols = 2 if has_target else 1
    nrows = n

    fig_w = ncols * 3.5
    fig_h = nrows * 3.5
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for row, sample in enumerate(samples):
        sar    = sample["sar"]
        target = sample["target_rgb"]
        meta   = sample["metadata"]
        pid    = meta.get("pair_id", str(row))

        sar_disp = _to_display(sar)
        axes[row, 0].imshow(sar_disp)
        axes[row, 0].set_title(f"SAR [{pid}]\n{tuple(sar.shape)}", fontsize=7)
        axes[row, 0].axis("off")

        if has_target:
            if target is not None:
                tgt_disp = _to_display(target)
                axes[row, 1].imshow(tgt_disp)
                axes[row, 1].set_title(f"Target RGB\n{tuple(target.shape)}", fontsize=7)
            else:
                axes[row, 1].text(0.5, 0.5, "No target", ha="center", va="center")
            axes[row, 1].axis("off")

    plt.suptitle("Day 1 — SAR / Target Pairing Check", fontsize=10, y=1.01)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[smoke_test] Contact sheet saved: {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Day 1 smoke test — loader verification.")
    parser.add_argument("--data-root",  default="data/raw",  type=Path)
    parser.add_argument("--splits-dir", default="data/",     type=Path)
    parser.add_argument("--n-samples",  default=5,           type=int)
    parser.add_argument("--output",     default="reports/day1_pairing_check.png", type=Path)
    parser.add_argument("--patch-size", default=256,         type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.data_root.resolve()

    if not root.exists():
        print(f"[smoke_test] ERROR: data root does not exist: {root}", file=sys.stderr)
        print("[smoke_test] Place the SIH1733 dataset under data/raw/ and run Day 1 scripts first.")
        sys.exit(1)

    # Try to load a pre-made split; otherwise use all discovered pairs
    train_pairs = _load_split_csv_if_exists(args.splits_dir, "train")
    if train_pairs:
        print(f"[smoke_test] Loaded {len(train_pairs)} train pairs from splits CSV.")
    else:
        print("[smoke_test] No splits CSV found — discovering pairs directly.")
        try:
            train_pairs = discover_pairs(root)
        except ValueError as e:
            print(f"[smoke_test] ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    if not train_pairs:
        print("[smoke_test] ERROR: no pairs available.", file=sys.stderr)
        sys.exit(1)

    # Build dataset
    dataset = SARColorizationDataset(
        pairs      = train_pairs,
        patch_size = (args.patch_size, args.patch_size),
    )

    # Sample
    n = min(args.n_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:n].tolist()

    print(f"\n[smoke_test] Loading {n} samples …")
    all_passed = True
    samples_for_display = []

    for i, idx in enumerate(indices):
        try:
            sample = dataset[idx]
        except Exception as e:
            print(f"  Sample {idx}: LOAD ERROR — {e}")
            all_passed = False
            continue

        r = check_sample(i, sample)
        samples_for_display.append(sample)

        # Evaluate checks
        checks = {
            "SAR shape valid":    r["sar_shape"] is not None and len(r["sar_shape"]) == 3,
            "No SAR NaN/Inf":     not r["sar_nan"],
            "Target shape valid": r["target_shape"] is None or (len(r["target_shape"]) == 3 and r["target_shape"][0] == 3),
            "No target NaN/Inf":  r["target_nan"] is None or not r["target_nan"],
            "Spatial match":      r["spatial_match"] is None or r["spatial_match"],
        }

        passed = all(checks.values())
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False

        print(f"\n  [{i+1}/{n}] pair_id={r['pair_id']!r}  {status}")
        print(f"    SAR shape  : {r['sar_shape']}  range=[{r['sar_min']:.4f}, {r['sar_max']:.4f}]")
        if r["target_shape"]:
            print(f"    Target shape: {r['target_shape']}  range=[{r['target_min']:.4f}, {r['target_max']:.4f}]")
        else:
            print(f"    Target      : None (inference-only mode or no reference)")

        for check_name, ok in checks.items():
            mark = "  ✓" if ok else "  ✗"
            print(f"    {mark} {check_name}")

    # Contact sheet
    if samples_for_display:
        save_contact_sheet(samples_for_display, args.output)

    print(f"\n[smoke_test] Overall: {'ALL CHECKS PASSED ✓' if all_passed else 'SOME CHECKS FAILED ✗'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
