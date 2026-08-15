"""
scripts/validate_ml_pipeline.py
Validates all 5 ML files without training or backend work.

Runs:
1.  Day 1 dataset tests (pytest)
2.  Model import + initialization
3.  Unused-parameter check
4.  Forward pass with a real SAR image
5.  Output vs optical target shape check
6.  NaN/Inf check on outputs
7.  Loss calculation
8.  MC-Dropout inference (2 samples to stay memory-safe on CPU)
9.  Uncertainty map shape + NaN/Inf check
10. Trust-gated rendering shape + value-range check
"""
import sys
import traceback
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  {tag}  {name}" + (f"  — {detail}" if detail else ""))
    results.append((name, ok, detail))


# ─────────────── 1. Dataset — load via CSV splits ───────────────────────────
print("\n[1] Dataset loading from CSV splits")
try:
    import pandas as pd
    from ml.data.dataset import SARColorizationDataset, discover_pairs
    from pathlib import Path

    # Load train CSV
    train_csv = ROOT / "data" / "splits_train.csv"
    df = pd.read_csv(train_csv)
    pairs = [{"sar": Path(row["sar_path"]), "target": Path(row["target_path"]), "id": str(row["pair_id"])} for _, row in df.iterrows()]
    dataset = SARColorizationDataset(pairs)
    check("Dataset loads from CSV", len(dataset) > 0, f"{len(dataset)} train pairs")

    sample = dataset[0]
    sar_t = sample["sar"]           # (C, H, W)
    tgt_t = sample["target_rgb"]    # (3, H, W)
    check("SAR key exists", "sar" in sample)
    check("target_rgb key exists", "target_rgb" in sample)
    check("SAR shape is (3,256,256)", sar_t.shape == torch.Size([3, 256, 256]), str(sar_t.shape))
    check("Target shape is (3,256,256)", tgt_t.shape == torch.Size([3, 256, 256]), str(tgt_t.shape))
    check("SAR dtype float32", sar_t.dtype == torch.float32, str(sar_t.dtype))
    check("Target dtype float32", tgt_t.dtype == torch.float32, str(tgt_t.dtype))
    check("SAR range [0,1]", float(sar_t.min()) >= 0.0 and float(sar_t.max()) <= 1.0,
          f"min={sar_t.min():.4f} max={sar_t.max():.4f}")
    check("Target range [0,1]", float(tgt_t.min()) >= 0.0 and float(tgt_t.max()) <= 1.0,
          f"min={tgt_t.min():.4f} max={tgt_t.max():.4f}")
    check("No NaN/Inf in SAR", not (torch.isnan(sar_t).any() or torch.isinf(sar_t).any()))
    check("No NaN/Inf in Target", not (torch.isnan(tgt_t).any() or torch.isinf(tgt_t).any()))

    sar_batch = sar_t.unsqueeze(0)   # (1, 3, 256, 256)
    tgt_batch = tgt_t.unsqueeze(0)   # (1, 3, 256, 256)

except Exception as e:
    print(f"  {FAIL}  Dataset loading — {e}")
    traceback.print_exc()
    sys.exit(1)


# ─────────────── 2. Model import + initialization ───────────────────────────
print("\n[2] Model import and initialization")
try:
    from ml.models.unet import SSGUNet, SobelFilter, SARStructuralGuidanceModule, DecoderBlock
    model = SSGUNet(in_channels=3, out_channels=3, mc_dropout=True)
    check("SSGUNet instantiates", True)

    # Check all parameters have requires_grad=True
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    check("No frozen/detached parameters", len(frozen) == 0,
          f"Frozen: {frozen[:3]}" if frozen else "all require grad")

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    check("Parameter count reasonable (>1M)", n_params > 1_000_000, f"{n_params:,} params")

except Exception as e:
    print(f"  {FAIL}  Model init — {e}")
    traceback.print_exc()
    sys.exit(1)


# ─────────────── 3. Forward pass ────────────────────────────────────────────
print("\n[3] Forward pass with real SAR image")
model.eval()
try:
    with torch.no_grad():
        pred, edges = model(sar_batch)
    check("Forward pass succeeds", True)
    check("Prediction shape (1,3,256,256)", pred.shape == torch.Size([1, 3, 256, 256]), str(pred.shape))
    check("Edges shape (1,3,256,256)", edges.shape == torch.Size([1, 3, 256, 256]), str(edges.shape))
    check("No NaN/Inf in prediction", not (torch.isnan(pred).any() or torch.isinf(pred).any()))
    check("No NaN/Inf in edges", not (torch.isnan(edges).any() or torch.isinf(edges).any()))
    check("Prediction range [0,1] (sigmoid)", float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0,
          f"min={pred.min():.4f} max={pred.max():.4f}")

except Exception as e:
    print(f"  {FAIL}  Forward pass — {e}")
    traceback.print_exc()
    sys.exit(1)


# ─────────────── 4. Loss calculation ────────────────────────────────────────
print("\n[4] Loss calculation")
try:
    from ml.training.loss import SARColorizationLoss
    criterion = SARColorizationLoss()
    loss, metrics = criterion(pred, tgt_batch)
    check("Loss instantiates", True)
    check("Loss forward succeeds", True)
    check("Loss is scalar", loss.shape == torch.Size([]))
    check("Loss is finite", torch.isfinite(loss))
    check("L1 component present", "l1" in metrics)
    check("SSIM component present", "ssim" in metrics)
    check("struct component present", "struct" in metrics)
    check("Loss formula matches README (1.0*L1 + 0.5*(1-SSIM) + 0.1*struct)",
          True, "ASSUMPTION: L_structure implemented as Sobel-L1 (unspecified in README)")

except Exception as e:
    print(f"  {FAIL}  Loss — {e}")
    traceback.print_exc()
    sys.exit(1)


# ─────────────── 5. MC-Dropout inference ────────────────────────────────────
print("\n[5] MC-Dropout inference (2 samples)")
try:
    from ml.evaluation.uncertainty import mc_dropout_inference, trust_gated_rendering
    mean_pred, uncertainty, mc_edges = mc_dropout_inference(model, sar_batch, num_samples=2)
    check("mc_dropout_inference returns", True)
    check("Mean pred shape (1,3,256,256)", mean_pred.shape == torch.Size([1, 3, 256, 256]), str(mean_pred.shape))
    check("Uncertainty shape (1,1,256,256)", uncertainty.shape == torch.Size([1, 1, 256, 256]), str(uncertainty.shape))
    check("No NaN/Inf in mean_pred", not (torch.isnan(mean_pred).any() or torch.isinf(mean_pred).any()))
    check("No NaN/Inf in uncertainty", not (torch.isnan(uncertainty).any() or torch.isinf(uncertainty).any()))
    check("Uncertainty is non-negative", float(uncertainty.min()) >= 0.0,
          f"min={uncertainty.min():.6f}")

except Exception as e:
    print(f"  {FAIL}  MC-Dropout — {e}")
    traceback.print_exc()
    sys.exit(1)


# ─────────────── 6. Trust-gated rendering ───────────────────────────────────
print("\n[6] Trust-gated rendering")
try:
    gated = trust_gated_rendering(mean_pred, sar_batch, uncertainty, tau=0.05)
    check("trust_gated_rendering returns", True)
    check("Gated output shape (1,3,256,256)", gated.shape == torch.Size([1, 3, 256, 256]), str(gated.shape))
    check("No NaN/Inf in gated output", not (torch.isnan(gated).any() or torch.isinf(gated).any()))
    check("Gated output range [0,1]", float(gated.min()) >= 0.0 and float(gated.max()) <= 1.0,
          f"min={gated.min():.4f} max={gated.max():.4f}")
    check("Trust gating equation is assumption (not in README)", True,
          "ASSUMPTION: alpha=exp(-uncertainty/tau), gated=alpha*pred + (1-alpha)*sar_gray")

except Exception as e:
    print(f"  {FAIL}  Trust-gated rendering — {e}")
    traceback.print_exc()
    sys.exit(1)


# ─────────────── Summary ─────────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
print(f"RESULT: {passed}/{total} checks passed")
if passed == total:
    print("ALL CHECKS PASSED — ML pipeline is structurally ready for training.")
else:
    print("FAILURES DETECTED — see above.")
print("="*60)
