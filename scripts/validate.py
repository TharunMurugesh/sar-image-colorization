"""
scripts/validate.py
Full model validation, baseline comparison, and evaluation script.

1. Loads the specified model checkpoint (and optionally a baseline checkpoint).
2. Evaluates on the test/val set using metrics from `ml/evaluation/metrics.py`.
3. Computes percentiles for the trust gate from `ml/evaluation/trust_gate.py`.
4. Generates a side-by-side qualitative grid comparing the old and new models.
5. Saves metrics to `reports/metrics_v2.json`.
6. Saves qualitative grid to `reports/comparison_grid_v2.png`.
7. Saves a markdown report to `reports/validation_report_v2.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataset import SARColorizationDataset
from ml.evaluation.metrics import colormap_baseline, evaluate_dataset
from ml.evaluation.trust_gate import compute_validation_thresholds, save_thresholds
from ml.models.unet import SSGUNet
from ml.training.train import sar_transform, target_transform
from ml.evaluation.uncertainty import mc_dropout_inference


def load_manifest_split(csv_path: Path, split_name: str) -> list[dict]:
    """Helper to load pairs from either a legacy split CSV or a v2 manifest CSV."""
    import csv
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "split" in row and row["split"] != split_name:
                continue
            
            sar_p = Path(row["sar_path"])
            tgt_p = Path(row["rgb_path"]) if "rgb_path" in row else Path(row.get("target_path", ""))
            
            if not sar_p.is_absolute():
                sar_p = (PROJECT_ROOT / sar_p).resolve()
            if tgt_p and str(tgt_p) and not tgt_p.is_absolute():
                tgt_p = (PROJECT_ROOT / tgt_p).resolve()

            pairs.append({
                "id": row.get("scene_id", row.get("pair_id", "")),
                "sar": sar_p,
                "target": tgt_p if str(tgt_p) else None,
            })
    return pairs


def plot_qualitative_grid(
    model: torch.nn.Module, 
    baseline_model: torch.nn.Module | None,
    loader: DataLoader, 
    device: torch.device, 
    out_path: Path, 
    num_samples: int = 10
):
    model.eval()
    if baseline_model:
        baseline_model.eval()
        
    samples_plotted = 0
    cols = 5 if baseline_model else 4
    
    fig, axes = plt.subplots(num_samples, cols, figsize=(4 * cols, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
        
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    col_titles = ["SAR Input", "Colormap (Viridis)"]
    if baseline_model:
        col_titles.append("Old SSG-U-Net")
    col_titles.extend(["New Model", "Ground Truth RGB"])
    
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=14, pad=10)

    with torch.no_grad():
        for batch in loader:
            if samples_plotted >= num_samples:
                break
                
            sar_t = batch["sar"].to(device)
            tgt_t = batch["target_rgb"]

            pred_t, _ = model(sar_t)
            pred_t = pred_t.cpu()
            
            base_pred_t = None
            if baseline_model:
                base_pred_t, _ = baseline_model(sar_t)
                base_pred_t = base_pred_t.cpu()

            sar_cpu = sar_t.cpu()
            B = sar_t.size(0)
            
            for i in range(B):
                if samples_plotted >= num_samples:
                    break

                sar_np = sar_cpu[i].numpy()
                if sar_np.ndim == 3:
                    sar_gray = sar_np.mean(axis=0)
                else:
                    sar_gray = sar_np
                
                cmap_img = colormap_baseline(sar_np, cmap="viridis")
                
                pred_img = pred_t[i].permute(1, 2, 0).numpy()
                pred_img = np.clip(pred_img, 0, 1)
                
                base_img = None
                if base_pred_t is not None:
                    base_img = base_pred_t[i].permute(1, 2, 0).numpy()
                    base_img = np.clip(base_img, 0, 1)

                tgt_img = None
                if tgt_t is not None:
                    tgt_img = tgt_t[i].permute(1, 2, 0).numpy()
                    tgt_img = np.clip(tgt_img, 0, 1)

                row_axes = axes[samples_plotted]
                
                row_axes[0].imshow(sar_gray, cmap="gray")
                row_axes[0].axis("off")
                
                row_axes[1].imshow(cmap_img)
                row_axes[1].axis("off")

                idx = 2
                if base_img is not None:
                    row_axes[idx].imshow(base_img)
                    row_axes[idx].axis("off")
                    idx += 1
                    
                row_axes[idx].imshow(pred_img)
                row_axes[idx].axis("off")
                idx += 1

                if tgt_img is not None:
                    row_axes[idx].imshow(tgt_img)
                row_axes[idx].axis("off")

                samples_plotted += 1

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[validate] Saved qualitative grid to {out_path}")


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    print(f"[validate] Loading model from {ckpt_path.name}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = checkpoint.get("config", {"in_channels": 3, "out_channels": 3})
    model = SSGUNet(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"], mc_dropout=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate and validate the SAR colorization model.")
    parser.add_argument("--checkpoint", type=str, default="runtime/checkpoints/best_model_sar_optical_v2.pt")
    parser.add_argument("--baseline-checkpoint", type=str, default="runtime/checkpoints/best_model_sih1733_only.pt")
    parser.add_argument("--data-manifest", type=Path, default=None, help="Path to v2 manifest CSV")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[validate] Using device: {device}")

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Load Main Model
    ckpt_path = PROJECT_ROOT / args.checkpoint
    if not ckpt_path.exists():
        print(f"[validate] ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)
    model = load_model(ckpt_path, device)

    # Load Baseline Model (optional)
    baseline_model = None
    base_ckpt_path = PROJECT_ROOT / args.baseline_checkpoint
    if base_ckpt_path.exists():
        baseline_model = load_model(base_ckpt_path, device)
    else:
        print(f"[validate] Note: Baseline checkpoint not found at {base_ckpt_path}")

    # Load Dataset
    data_dir = PROJECT_ROOT / "data"
    
    if args.data_manifest and args.data_manifest.exists():
        eval_csv = args.data_manifest
        val_pairs = load_manifest_split(eval_csv, "val")
        test_pairs = load_manifest_split(eval_csv, "test")
    else:
        test_csv = data_dir / "splits_test.csv"
        val_csv = data_dir / "splits_val.csv"
        val_pairs = load_manifest_split(val_csv, "val") if val_csv.exists() else []
        test_pairs = load_manifest_split(test_csv, "test") if test_csv.exists() else []
        eval_csv = test_csv if test_pairs else val_csv

    # Decide eval set (prefer test, fallback to val)
    if len(test_pairs) > 0:
        eval_pairs = test_pairs
        split_name = "Test"
    elif len(val_pairs) > 0:
        eval_pairs = val_pairs
        split_name = "Validation"
    else:
        print("[validate] ERROR: No test or val splits found.")
        sys.exit(1)

    print(f"[validate] Using {split_name} split ({len(eval_pairs)} pairs)")
    eval_pairs = [p for p in eval_pairs if Path(p["sar"]).exists()]

    eval_ds = SARColorizationDataset(
        eval_pairs, patch_size=(256, 256),
        sar_transform=sar_transform, target_transform=target_transform,
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    if len(val_pairs) > 0:
        val_pairs = [p for p in val_pairs if Path(p["sar"]).exists()]
        val_ds = SARColorizationDataset(
            val_pairs, patch_size=(256, 256),
            sar_transform=sar_transform, target_transform=target_transform
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = eval_loader

    # Compute Metrics
    print(f"\n[validate] Computing metrics on {split_name} set for MAIN model...")
    metrics = evaluate_dataset(model, eval_loader, device, mc_passes=1)

    if baseline_model:
        print(f"[validate] Computing metrics on {split_name} set for BASELINE model...")
        base_metrics = evaluate_dataset(baseline_model, eval_loader, device, mc_passes=1)
        metrics["baseline_model_psnr"] = base_metrics["psnr_mean"]
        metrics["baseline_model_ssim"] = base_metrics["ssim_mean"]
        metrics["baseline_model_de"]   = base_metrics["delta_e_mean"]

    metrics_path = reports_dir / "metrics_v2.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[validate] Saved metrics to {metrics_path}")

    # Compute Trust Gate Thresholds (always from val loader)
    print("\n[validate] Computing trust gate thresholds from Validation set...")
    low_thresh, high_thresh = compute_validation_thresholds(
        model, val_loader, device, low_percentile=10.0, high_percentile=90.0, mc_passes=10
    )
    thresh_path = PROJECT_ROOT / "runtime" / "checkpoints" / "trust_thresholds.json"
    save_thresholds(thresh_path, low_thresh, high_thresh)

    # Generate Qualitative Grid
    grid_path = reports_dir / "comparison_grid_v2.png"
    print(f"\n[validate] Generating qualitative grid (up to 10 samples)...")
    plot_qualitative_grid(model, baseline_model, eval_loader, device, grid_path, num_samples=min(10, len(eval_ds)))

    # Generate Markdown Report
    report_path = reports_dir / "validation_report_v2.md"
    with open(report_path, "w") as f:
        f.write("# SAR Image Colorization — Validation Report (v2)\n\n")
        f.write(f"**Split evaluated**: {split_name} ({metrics['n_samples']} samples)\n\n")
        
        f.write("## 1. Quantitative Metrics\n\n")
        f.write("| Metric | New Model | Old SSG-U-Net | Grayscale Base | Colormap Base |\n")
        f.write("|--------|-----------|---------------|----------------|---------------|\n")
        
        base_psnr = metrics.get('baseline_model_psnr', float('nan'))
        base_ssim = metrics.get('baseline_model_ssim', float('nan'))
        base_de   = metrics.get('baseline_model_de', float('nan'))
        
        f.write(f"| PSNR (dB) | **{metrics['psnr_mean']:.2f}** ± {metrics['psnr_std']:.2f} | {base_psnr:.2f} | {metrics['baseline_gray_psnr']:.2f} | {metrics['baseline_cmap_psnr']:.2f} |\n")
        f.write(f"| SSIM      | **{metrics['ssim_mean']:.3f}** ± {metrics['ssim_std']:.3f} | {base_ssim:.3f} | {metrics['baseline_gray_ssim']:.3f} | {metrics['baseline_cmap_ssim']:.3f} |\n")
        f.write(f"| DeltaE76  | **{metrics['delta_e_mean']:.2f}** ± {metrics['delta_e_std']:.2f} | {base_de:.2f} | {metrics['baseline_gray_de']:.2f} | {metrics['baseline_cmap_de']:.2f} |\n\n")
        
        f.write("## 2. Uncertainty & Trust Gate\n\n")
        f.write(f"- **Trust Gate Thresholds** (Validation 10th / 90th percentiles):\n")
        f.write(f"  - Low Uncertainty (100% Trust): {low_thresh:.6f}\n")
        f.write(f"  - High Uncertainty (0% Trust): {high_thresh:.6f}\n\n")

        f.write("## 3. Qualitative Evaluation\n\n")
        f.write("See `comparison_grid_v2.png` for a side-by-side visual comparison.\n")
        
    print(f"[validate] Saved markdown report to {report_path}")
    print("\n[validate] All validation tasks completed successfully!")

if __name__ == "__main__":
    main()
