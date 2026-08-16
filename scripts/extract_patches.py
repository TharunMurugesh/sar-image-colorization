"""
scripts/extract_patches.py
Tile SAR-Optical image pairs into 256x256 patches for training.

Outputs:
  data/processed/patches/sar/  — SAR patch PNGs
  data/processed/patches/rgb/  — Optical patch PNGs
  data/splits_{train,val,test}.csv
  data/manifest_v2.csv

Training patches get 6 augmentations (orig, fliph, flipv, rot90/180/270).
Val/test patches are extracted without overlap or augmentation.

Usage:
    python scripts/extract_patches.py
    python scripts/extract_patches.py --stride 128
    python scripts/extract_patches.py --data-root data/raw/sih1733 --source-dataset sih1733
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataset import discover_pairs, load_any


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _load_sar_gray(path: Path) -> np.ndarray:
    arr = load_any(path).mean(axis=0)
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _save_gray(arr: np.ndarray, path: Path) -> None:
    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(path)


def _save_rgb(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr, mode="RGB").save(path)


def _augment(sar: np.ndarray, rgb: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    return [
        ("orig",   sar,                  rgb),
        ("fliph",  np.fliplr(sar),       np.fliplr(rgb)),
        ("flipv",  np.flipud(sar),       np.flipud(rgb)),
        ("rot90",  np.rot90(sar, k=1),   np.rot90(rgb, k=1)),
        ("rot180", np.rot90(sar, k=2),   np.rot90(rgb, k=2)),
        ("rot270", np.rot90(sar, k=3),   np.rot90(rgb, k=3)),
    ]


def extract_patches(
    sar_path: Path, rgb_path: Path, pair_id: str,
    out_sar: Path, out_rgb: Path,
    patch_h: int = 256, patch_w: int = 256, stride: int = 128,
    augment: bool = True, source_dataset: str = "sih1733", split: str = "train",
) -> list[dict]:
    sar = _load_sar_gray(sar_path)
    rgb = _load_rgb(rgb_path)
    h = min(sar.shape[0], rgb.shape[0])
    w = min(sar.shape[1], rgb.shape[1])
    sar, rgb = sar[:h, :w], rgb[:h, :w]

    records = []
    for row in range(0, h - patch_h + 1, stride):
        for col in range(0, w - patch_w + 1, stride):
            sp = sar[row:row + patch_h, col:col + patch_w]
            rp = rgb[row:row + patch_h, col:col + patch_w]
            variants = _augment(sp, rp) if augment else [("orig", sp, rp)]
            for suffix, s, r in variants:
                pid = f"{pair_id}_r{row:04d}_c{col:04d}_{suffix}"
                sar_out = out_sar / f"{pid}.png"
                rgb_out = out_rgb / f"{pid}.png"
                _save_gray(s, sar_out)
                _save_rgb(r, rgb_out)
                records.append({
                    "scene_id": pair_id, "source_dataset": source_dataset, "split": split,
                    "sar": sar_out, "target": rgb_out,
                    "patch_x": col, "patch_y": row, "patch_size": patch_h,
                    "id": pid,
                })

    print(f"  [{pair_id}] {h}x{w} -> {len(records)} patches (stride={stride})")
    return records


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_split_csv(records: list[dict], path: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "pair_id", "sar_path", "target_path"])
        w.writeheader()
        for r in records:
            w.writerow({"split": split, "pair_id": r["id"], "sar_path": _rel(r["sar"]), "target_path": _rel(r["target"])})
    print(f"  {len(records)} records -> {path}")


def write_manifest(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scene_id", "source_dataset", "split", "sar_path", "rgb_path", "patch_x", "patch_y", "patch_size"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({
                "scene_id": r["scene_id"], "source_dataset": r["source_dataset"], "split": r["split"],
                "sar_path": _rel(r["sar"]), "rgb_path": _rel(r["target"]),
                "patch_x": r["patch_x"], "patch_y": r["patch_y"], "patch_size": r["patch_size"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 256x256 patches from SAR-Optical pairs.")
    parser.add_argument("--data-root",      type=Path, default=Path("data/raw/sih1733"))
    parser.add_argument("--output-dir",     type=Path, default=Path("data/processed/patches"))
    parser.add_argument("--splits-dir",     type=Path, default=Path("data"))
    parser.add_argument("--patch-size",     type=int,  default=256)
    parser.add_argument("--stride",         type=int,  default=128)
    parser.add_argument("--no-augment",     action="store_true")
    parser.add_argument("--source-dataset", type=str,  default="sih1733")
    parser.add_argument("--manifest-only",  action="store_true",
                        help="Regenerate CSVs from existing patches without re-extracting")
    args = parser.parse_args()

    data_root  = (PROJECT_ROOT / args.data_root).resolve()
    out_dir    = (PROJECT_ROOT / args.output_dir).resolve()
    splits_dir = (PROJECT_ROOT / args.splits_dir).resolve()
    out_sar    = out_dir / "sar"
    out_rgb    = out_dir / "rgb"
    out_sar.mkdir(parents=True, exist_ok=True)
    out_rgb.mkdir(parents=True, exist_ok=True)

    pairs = sorted(discover_pairs(data_root), key=lambda p: str(p["id"]))
    print(f"[extract_patches] {len(pairs)} pairs found in {data_root}")

    n = len(pairs)
    n_val   = max(1, round(n * 0.15))
    n_test  = max(1, round(n * 0.15)) if n >= 3 else 0
    n_train = n - n_val - n_test

    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:n_train + n_val]
    test_pairs  = pairs[n_train + n_val:]
    print(f"[extract_patches] Split: {len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test scenes\n")

    train_records, val_records, test_records = [], [], []

    for pair in train_pairs:
        train_records.extend(extract_patches(
            Path(pair["sar"]), Path(pair["target"]), str(pair["id"]),
            out_sar, out_rgb, args.patch_size, args.patch_size, args.stride,
            augment=not args.no_augment, source_dataset=args.source_dataset, split="train",
        ))
    for pair in val_pairs:
        val_records.extend(extract_patches(
            Path(pair["sar"]), Path(pair["target"]), str(pair["id"]),
            out_sar, out_rgb, args.patch_size, args.patch_size, args.patch_size,
            augment=False, source_dataset=args.source_dataset, split="val",
        ))
    for pair in test_pairs:
        test_records.extend(extract_patches(
            Path(pair["sar"]), Path(pair["target"]), str(pair["id"]),
            out_sar, out_rgb, args.patch_size, args.patch_size, args.patch_size,
            augment=False, source_dataset=args.source_dataset, split="test",
        ))

    print(f"\n[extract_patches] Total: {len(train_records)} train / {len(val_records)} val / {len(test_records)} test")

    write_split_csv(train_records, splits_dir / "splits_train.csv", "train")
    write_split_csv(val_records,   splits_dir / "splits_val.csv",   "val")
    write_split_csv(test_records,  splits_dir / "splits_test.csv",  "test")

    all_records = train_records + val_records + test_records
    write_manifest(all_records, splits_dir / "manifest_v2.csv")

    (splits_dir / "splits_meta.txt").write_text(
        f"patch_size={args.patch_size}\nstride={args.stride}\n"
        f"augment={not args.no_augment}\nsource={args.source_dataset}\n"
        f"train={len(train_records)}\nval={len(val_records)}\ntest={len(test_records)}\n"
    )
    print(f"\n[extract_patches] Done. Splits -> {splits_dir}")


if __name__ == "__main__":
    main()
