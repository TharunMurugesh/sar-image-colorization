"""
scripts/make_splits.py
Day 1 — Task 4: Scene-safe train / validation / test split generator.

Split strategy (from build plan §Day1 Task 4):
  - Preferred split unit: scene / location ID.
  - Patches from the same scene must NOT appear in both train and test.
  - Split: 70% train / 15% val / 15% test.
  - Output: CSV files so every future run uses exactly the same samples.

Usage:
    python scripts/make_splits.py --data-root data/raw --output-dir data/
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path so ml.data can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.dataset import discover_pairs, discover_sar_only


# ─────────────────────────────────────────────────────────────────────────────
# Scene ID extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_scene_id(path: Path) -> str:
    """Extract a scene / location identifier from a file path.

    Heuristic priority:
      1. Parent directory name (most reliable scene grouping).
      2. Leading alphanumeric prefix of the filename stem.
      3. Fall back to pair ID (the file stem itself).

    This function is intentionally forgiving — pair IDs remain unique even
    if two files share the same parent directory.
    """
    parent = path.parent.name.lower()
    # If parent is a meaningful name (not "sar", "raw", "data"), use it
    generic = {"sar", "raw", "data", "optical", "rgb", "color", "ref", "target", "images", "imgs"}
    if parent not in generic and parent != ".":
        return parent

    # Try leading scene prefix from filename, e.g. "scene_003_sar.tif" → "scene_003"
    stem = path.stem
    m = re.match(r"([a-zA-Z]+[_\-]?\d+)[_\-]", stem)
    if m:
        return m.group(1).lower()

    # Fall back to first numeric run as scene
    m = re.search(r"(\d{2,})", stem)
    if m:
        return m.group(1)

    return stem.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Split logic
# ─────────────────────────────────────────────────────────────────────────────

def scene_safe_split(
    pairs: List[Dict],
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    seed: int         = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split pairs into train / val / test without scene leakage.

    Groups pairs by scene ID, shuffles scenes (not pairs), then assigns
    whole scenes to splits.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Group by scene
    scene_to_pairs: Dict[str, List[Dict]] = {}
    for p in pairs:
        scene_id = extract_scene_id(Path(p["sar"]))
        scene_to_pairs.setdefault(scene_id, []).append(p)

    scenes = sorted(scene_to_pairs.keys())
    rng.shuffle(scenes)

    n_scenes   = len(scenes)
    n_train    = max(1, round(n_scenes * train_frac))
    n_val      = max(1, round(n_scenes * val_frac))
    n_test     = n_scenes - n_train - n_val
    if n_test < 1:
        # Redistribute to ensure at least 1 in each split
        if n_val > 1:
            n_val -= 1
        elif n_train > 2:
            n_train -= 1
        n_test = n_scenes - n_train - n_val

    train_scenes = scenes[:n_train]
    val_scenes   = scenes[n_train: n_train + n_val]
    test_scenes  = scenes[n_train + n_val:]

    train_pairs = [p for s in train_scenes for p in scene_to_pairs[s]]
    val_pairs   = [p for s in val_scenes   for p in scene_to_pairs[s]]
    test_pairs  = [p for s in test_scenes  for p in scene_to_pairs[s]]

    # Verify no leakage
    train_ids = {p["id"] for p in train_pairs}
    val_ids   = {p["id"] for p in val_pairs}
    test_ids  = {p["id"] for p in test_pairs}
    assert not (train_ids & test_ids),  "LEAKAGE: same ID in train and test!"
    assert not (train_ids & val_ids),   "LEAKAGE: same ID in train and val!"
    assert not (val_ids   & test_ids),  "LEAKAGE: same ID in val and test!"

    return train_pairs, val_pairs, test_pairs


def id_split(
    pairs: List[Dict],
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    seed: int         = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Fallback: split by pair/sample ID when scene IDs are unavailable."""
    rng = random.Random(seed)
    pairs_shuffled = pairs[:]
    rng.shuffle(pairs_shuffled)

    n = len(pairs_shuffled)
    n_train = max(1, round(n * train_frac))
    n_val   = max(1, round(n * val_frac))
    n_test  = n - n_train - n_val
    if n_test < 1:
        n_val  = max(1, n_val - 1)
        n_test = n - n_train - n_val

    train_pairs = pairs_shuffled[:n_train]
    val_pairs   = pairs_shuffled[n_train: n_train + n_val]
    test_pairs  = pairs_shuffled[n_train + n_val:]
    return train_pairs, val_pairs, test_pairs


# ─────────────────────────────────────────────────────────────────────────────
# CSV I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_split_csv(pairs: List[Dict], path: Path, split_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "pair_id", "sar_path", "target_path"])
        writer.writeheader()
        for p in pairs:
            writer.writerow({
                "split":       split_name,
                "pair_id":     p.get("id", ""),
                "sar_path":    str(p.get("sar", "")),
                "target_path": str(p.get("target", "")),
            })
    print(f"  [splits] Saved {len(pairs):4d} pairs → {path}")


def load_split_csv(path: Path) -> List[Dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append({
                "id":     row["pair_id"],
                "sar":    row["sar_path"],
                "target": row["target_path"] or None,
            })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Day 1 — Generate scene-safe train/val/test splits."
    )
    parser.add_argument("--data-root",   required=True, type=Path, help="Root of the dataset.")
    parser.add_argument("--output-dir",  default=Path("data"),    type=Path, help="Output directory for CSVs.")
    parser.add_argument("--train-frac",  default=0.70,  type=float)
    parser.add_argument("--val-frac",    default=0.15,  type=float)
    parser.add_argument("--seed",        default=42,    type=int)
    parser.add_argument("--no-scene-split", action="store_true",
                        help="Skip scene grouping; split by pair ID instead (documents limitation).")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.data_root.resolve()

    if not root.exists():
        print(f"[splits] ERROR: data root does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"[splits] Discovering pairs in {root} …")
    try:
        pairs = discover_pairs(root)
        print(f"[splits] Found {len(pairs)} SAR/reference pairs.")
    except ValueError as e:
        print(f"[splits] WARNING: {e}")
        print("[splits] Falling back to SAR-only discovery (inference mode).")
        pairs = discover_sar_only(root)
        print(f"[splits] Found {len(pairs)} SAR-only files.")
        if not pairs:
            print("[splits] No files found at all. Check --data-root.")
            sys.exit(1)

    if len(pairs) < 3:
        print(f"[splits] WARNING: only {len(pairs)} pairs — splits will have ≤1 sample each.")

    # Determine split strategy
    use_scene = not args.no_scene_split
    if use_scene:
        scenes = {extract_scene_id(Path(p["sar"])) for p in pairs}
        if len(scenes) < 3:
            print(f"[splits] Only {len(scenes)} unique scene(s) detected — falling back to ID split.")
            use_scene = False

    if use_scene:
        print("[splits] Using scene-based split (no scene leakage).")
        train_pairs, val_pairs, test_pairs = scene_safe_split(
            pairs, args.train_frac, args.val_frac, args.seed
        )
        split_note = "scene-based (no scene leakage)"
    else:
        print("[splits] Using pair-ID-based split (LIMITATION: possible scene leakage — document this).")
        train_pairs, val_pairs, test_pairs = id_split(
            pairs, args.train_frac, args.val_frac, args.seed
        )
        split_note = "pair-ID-based (scene leakage possible)"

    out = args.output_dir.resolve()
    save_split_csv(train_pairs, out / "splits_train.csv", "train")
    save_split_csv(val_pairs,   out / "splits_val.csv",   "val")
    save_split_csv(test_pairs,  out / "splits_test.csv",  "test")

    # Human-readable summary
    total = len(train_pairs) + len(val_pairs) + len(test_pairs)
    print(f"\n[splits] Split summary ({split_note}):")
    print(f"  Train : {len(train_pairs):4d} pairs  ({100*len(train_pairs)/total:.1f}%)")
    print(f"  Val   : {len(val_pairs):4d} pairs  ({100*len(val_pairs)/total:.1f}%)")
    print(f"  Test  : {len(test_pairs):4d} pairs  ({100*len(test_pairs)/total:.1f}%)")
    print(f"  Total : {total:4d} pairs")
    print(f"\n  Seed  : {args.seed}")
    print(f"  Method: {split_note}")

    # Write a small metadata file
    meta_path = out / "splits_meta.txt"
    meta_path.write_text(
        f"split_method={split_note}\n"
        f"seed={args.seed}\n"
        f"train={len(train_pairs)}\n"
        f"val={len(val_pairs)}\n"
        f"test={len(test_pairs)}\n"
        f"total={total}\n"
    )
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
