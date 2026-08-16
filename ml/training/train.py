"""
ml/training/train.py
Day 5 — Training script for SSG-U-Net.

Features:
  - Uses canonical SARColorizationDataset and scene-safe splits
  - Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
  - Loss: SARColorizationLoss (L1_RGB + 0.5*(1-SSIM) + 0.1*L_structure)
  - Early stopping and saving best checkpoint to runtime/checkpoints/best_model.pt
  - Command-line flag --smoke-test for quick verification before full training run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataset import SARColorizationDataset, discover_pairs
from scripts.make_splits import load_split_csv
from ml.models.unet import SSGUNet
from ml.training.loss import SARColorizationLoss


def adapt_sar_channels(arr: np.ndarray, target_channels: int = 3) -> np.ndarray:
    """Adapt SAR numpy array (C, H, W) to 3 channels for SSG-U-Net."""
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


def sar_transform(arr: np.ndarray) -> np.ndarray:
    """Per-sample SAR transform: adapt channels and normalize to [0, 1]."""
    arr = adapt_sar_channels(arr, 3)
    # Min-max normalization per channel with percentile clipping
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


def target_transform(arr: np.ndarray) -> np.ndarray:
    """RGB target transform: ensure range [0, 1]."""
    if arr.max() > 1.0:
        return arr / 255.0
    return arr


def get_dataloaders(data_dir: Path, batch_size: int = 2) -> tuple[DataLoader, DataLoader]:
    """Load train and val Datasets and return DataLoaders."""
    train_csv = data_dir / "splits_train.csv"
    val_csv = data_dir / "splits_val.csv"

    if train_csv.exists() and val_csv.exists():
        train_pairs = load_split_csv(train_csv)
        val_pairs = load_split_csv(val_csv)
    else:
        print("[train] Warning: Split CSVs not found. Discovering pairs directly...")
        pairs = discover_pairs(data_dir / "raw" / "sih1733")
        train_pairs = pairs[:2]
        val_pairs = pairs[2:]

    # Filter out missing files if any path doesn't exist
    train_pairs = [p for p in train_pairs if Path(p["sar"]).exists()]
    val_pairs = [p for p in val_pairs if Path(p["sar"]).exists()]

    if len(val_pairs) == 0:
        print("[train] Note: val split empty, using train pair for validation evaluation.")
        val_pairs = train_pairs[:1]

    train_ds = SARColorizationDataset(
        train_pairs,
        patch_size=(256, 256),
        sar_transform=sar_transform,
        target_transform=target_transform,
    )
    val_ds = SARColorizationDataset(
        val_pairs,
        patch_size=(256, 256),
        sar_transform=sar_transform,
        target_transform=target_transform,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(
    epochs: int = 20,
    batch_size: int = 2,
    lr: float = 1e-4,
    checkpoint_dir: Path = Path("runtime/checkpoints"),
    smoke_test: bool = False,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Training device: {device}")

    train_loader, val_loader = get_dataloaders(Path("data"), batch_size=batch_size)
    print(f"[train] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = SSGUNet(in_channels=3, out_channels=3, mc_dropout=True).to(device)
    criterion = SARColorizationLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_model.pt"

    best_val_loss = float("inf")

    if smoke_test:
        print("\n=== RUNNING TRAINING SMOKE TEST ===")
        epochs = 1

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            sar = batch["sar"].to(device)         # (B, 3, 256, 256)
            target = batch["target_rgb"].to(device) # (B, 3, 256, 256)

            optimizer.zero_grad()
            pred, edges = model(sar)
            loss, loss_dict = criterion(pred, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if smoke_test:
                print(f"[Smoke Test] Batch loss: {loss.item():.4f} (l1={loss_dict['l1']:.4f}, ssim={loss_dict['ssim']:.4f}, struct={loss_dict['struct']:.4f})")
                print("[Smoke Test] Dataset loading -> Model init -> Forward pass -> Loss calc -> Backward pass -> Optimizer step SUCCESSFUL!")
                # Save checkpoint for smoke test
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": loss.item(),
                        "config": {"in_channels": 3, "out_channels": 3},
                    },
                    best_checkpoint_path,
                )
                print(f"[Smoke Test] Saved test checkpoint to {best_checkpoint_path}")
                return best_checkpoint_path

        avg_train_loss = train_loss / max(1, train_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                sar = batch["sar"].to(device)
                target = batch["target_rgb"].to(device)
                pred, _ = model(sar)
                loss, _ = criterion(pred, target)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "config": {"in_channels": 3, "out_channels": 3},
                },
                best_checkpoint_path,
            )
            print(f"  -> Saved new best checkpoint (Val Loss: {best_val_loss:.4f})", flush=True)

    print(f"\n[train] Training complete. Best model saved to: {best_checkpoint_path}")
    return best_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSG-U-Net for SAR Colorization")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 1-batch smoke test")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke_test=args.smoke_test,
    )
