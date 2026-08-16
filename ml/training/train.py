"""
ml/training/train.py
SSG-U-Net training script for SAR-to-Optical reconstruction.

Usage:
    python ml/training/train.py
    python ml/training/train.py --epochs 50 --batch-size 4 --patience 15
    python ml/training/train.py --data-manifest data/manifest_v2.csv --checkpoint-name run2.pt
    python ml/training/train.py --smoke-test
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataset import SARColorizationDataset, discover_pairs
from scripts.make_splits import load_split_csv
from ml.models.unet import SSGUNet
from ml.training.loss import SARColorizationLoss


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def adapt_sar_channels(arr: np.ndarray, target_channels: int = 3) -> np.ndarray:
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
    arr = adapt_sar_channels(arr, 3)
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        band = arr[i]
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        out[i] = np.zeros_like(band) if hi - lo < 1e-9 else np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def target_transform(arr: np.ndarray) -> np.ndarray:
    return arr / 255.0 if arr.max() > 1.0 else arr


def _load_manifest_split(csv_path: Path, split_name: str) -> list[dict]:
    import csv
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") != split_name:
                continue
            sar_p = Path(row["sar_path"])
            rgb_p = Path(row["rgb_path"])
            if not sar_p.is_absolute():
                sar_p = (PROJECT_ROOT / sar_p).resolve()
            if not rgb_p.is_absolute():
                rgb_p = (PROJECT_ROOT / rgb_p).resolve()
            pairs.append({"id": row["scene_id"], "sar": sar_p, "target": rgb_p})
    return pairs


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 4,
    manifest: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    if manifest is not None and manifest.exists():
        train_pairs = _load_manifest_split(manifest, "train")
        val_pairs   = _load_manifest_split(manifest, "val")
        print(f"[train] Manifest: {len(train_pairs)} train, {len(val_pairs)} val")
    else:
        train_csv = data_dir / "splits_train.csv"
        val_csv   = data_dir / "splits_val.csv"
        if train_csv.exists() and val_csv.exists():
            train_pairs = load_split_csv(train_csv)
            val_pairs   = load_split_csv(val_csv)
        else:
            pairs = discover_pairs(data_dir / "raw" / "sih1733")
            train_pairs, val_pairs = pairs[:2], pairs[2:]

    train_pairs = [p for p in train_pairs if Path(p["sar"]).exists()]
    val_pairs   = [p for p in val_pairs   if Path(p["sar"]).exists()]

    if not val_pairs:
        val_pairs = train_pairs[:max(1, len(train_pairs) // 10)]

    ds_kwargs = dict(patch_size=(256, 256), sar_transform=sar_transform, target_transform=target_transform)
    train_ds = SARColorizationDataset(train_pairs, **ds_kwargs)
    val_ds   = SARColorizationDataset(val_pairs,   **ds_kwargs)

    loader_kwargs = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs),
    )


def train_model(
    epochs: int = 75,
    batch_size: int = 4,
    lr: float = 1e-4,
    patience: int = 15,
    seed: int = 42,
    checkpoint_dir: Path = Path("runtime/checkpoints"),
    checkpoint_name: str = "best_model.pt",
    data_manifest: Path | None = None,
    smoke_test: bool = False,
) -> Path:
    set_seed(seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"[train] device={device}  amp={'on' if use_amp else 'off'}")

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_loader, val_loader = get_dataloaders(Path("data"), batch_size, data_manifest)
    print(f"[train] {len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples")

    model     = SSGUNet(in_channels=3, out_channels=3, mc_dropout=True).to(device)
    criterion = SARColorizationLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = checkpoint_dir / checkpoint_name

    config = {
        "checkpoint_name": checkpoint_name,
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "patience": patience, "seed": seed,
        "optimizer": "AdamW", "weight_decay": 1e-4, "grad_clip": 1.0,
        "loss": "L1+SSIM+Lab+Struct",
        "model": "SSGUNet(resnet18, mc_dropout=True)",
        "amp": use_amp, "device": str(device),
    }
    config_path = checkpoint_dir / checkpoint_name.replace(".pt", "_config.json")
    config_path.write_text(json.dumps(config, indent=2))

    best_val   = float("inf")
    patience_n = 0
    stopped_early = False
    history = {"train_loss": [], "val_loss": [], "train_l1": [], "train_ssim": [], "train_lab": [], "train_struct": []}

    if smoke_test:
        epochs = 1

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_batches = 0.0, 0
        comp = {"l1": 0.0, "ssim": 0.0, "lab": 0.0, "struct": 0.0}

        for batch in train_loader:
            sar, target = batch["sar"].to(device), batch["target_rgb"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred, _ = model(sar)
                loss, loss_dict = criterion(pred, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()
            t_batches += 1
            for k in comp:
                comp[k] += loss_dict.get(k, 0.0)

            if smoke_test:
                torch.save(
                    {"model": model.state_dict(), "epoch": 1, "val_loss": loss.item(),
                     "config": {"in_channels": 3, "out_channels": 3}},
                    best_ckpt,
                )
                print(f"[smoke-test] loss={loss.item():.4f}  checkpoint saved -> {best_ckpt}")
                return best_ckpt

        avg_train = t_loss / max(1, t_batches)
        avg_comp  = {k: v / max(1, t_batches) for k, v in comp.items()}

        model.eval()
        v_loss, v_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                sar, target = batch["sar"].to(device), batch["target_rgb"].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred, _ = model(sar)
                    loss, _ = criterion(pred, target)
                v_loss += loss.item()
                v_batches += 1
        avg_val = v_loss / max(1, v_batches)

        print(
            f"Epoch [{epoch:03d}/{epochs}]  "
            f"train={avg_train:.4f} (l1={avg_comp['l1']:.3f} ssim={avg_comp['ssim']:.3f} "
            f"lab={avg_comp['lab']:.3f} str={avg_comp['struct']:.3f})  val={avg_val:.4f}",
            flush=True,
        )

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        for k in ["l1", "ssim", "lab", "struct"]:
            history[f"train_{k}"].append(avg_comp[k])

        if avg_val < best_val:
            best_val   = avg_val
            patience_n = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": best_val,
                 "config": {"in_channels": 3, "out_channels": 3}, "training_config": config},
                best_ckpt,
            )
            print(f"  -> best checkpoint saved (val={best_val:.4f})", flush=True)
        else:
            patience_n += 1
            print(f"  -> no improvement ({patience_n}/{patience})", flush=True)
            if patience_n >= patience:
                stopped_early = True
                break

    n_ep = len(history["train_loss"])
    if stopped_early:
        print(f"\n[train] Early stopping at epoch {n_ep}/{epochs}. Best: {best_ckpt}")
    else:
        print(f"\n[train] Completed {n_ep} epochs. Best: {best_ckpt}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = range(1, n_ep + 1)
    axes[0].plot(xs, history["train_loss"], label="Train")
    axes[0].plot(xs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)
    for k, lbl in [("l1", "L1"), ("ssim", "SSIM"), ("lab", "Lab"), ("struct", "Struct")]:
        axes[1].plot(xs, history[f"train_{k}"], label=lbl)
    axes[1].set_title("Components (Train)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(reports_dir / checkpoint_name.replace(".pt", "_training_curve.png"), dpi=120)
    plt.close()

    hist_path = reports_dir / checkpoint_name.replace(".pt", "_history.json")
    with open(hist_path, "w") as f:
        json.dump({**history, "stopped_early": stopped_early, "best_val_loss": best_val, "config": config}, f, indent=2)

    return best_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSG-U-Net for SAR-to-Optical Reconstruction")
    parser.add_argument("--epochs",          type=int,   default=75)
    parser.add_argument("--batch-size",      type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--patience",        type=int,   default=15)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--checkpoint-name", type=str,   default="best_model.pt")
    parser.add_argument("--data-manifest",   type=Path,  default=None)
    parser.add_argument("--smoke-test",      action="store_true")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        checkpoint_name=args.checkpoint_name,
        data_manifest=args.data_manifest,
        smoke_test=args.smoke_test,
    )
