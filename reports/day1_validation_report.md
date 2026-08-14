# Day 1 Dataset Validation Report — SIH1733

**Generated:** 2026-08-14  
**Project:** SIH1733 — SAR Image Colorization for Comprehensive Insight using Deep Learning (ISRO/SAC)  
**Data root:** `data/raw/`

---

## Executive Summary

The official SIH1733 dataset, as currently supplied, contains **3 SAR/optical image pairs** distributed across three named subdirectories (`Pair-1`, `Pair-2`, `Pair-3`). All three pairs load cleanly, pass shape and NaN checks, and their filenames pair correctly.

However, the dataset in its current form is **not sufficient to train a robust deep-learning colorization model by itself.** The 3 pairs constitute a demonstration corpus — not a training corpus. Supervised deep learning requires substantially more scene-diverse paired data.

**Verdict: DAY 1 CONDITIONAL — DATA VALID BUT ADDITIONAL TRAINING DATA REQUIRED**

---

## Dataset Inventory

| Pair | SAR file | Optical file | SAR dimensions (H×W) | Optical dimensions (H×W) | SAR channels | Optical channels |
|------|----------|--------------|----------------------|--------------------------|--------------|-----------------|
| 1 | `Pair-1/SAR-Image-1.jpg` | `Pair-1/Optical-Image-1.jpg` | 1733 × 2500 | 1733 × 2500 | 3 (all identical — effectively 1) | 3 (distinct R/G/B) |
| 2 | `Pair-2/SAR-Image-2.jpg` | `Pair-2/Optical-Image-2.jpg` | 1733 × 2500 | 1733 × 2500 | 3 (all identical — effectively 1) | 3 (distinct R/G/B) |
| 3 | `Pair-3/SAR-Image-3.jpg` | `Pair-3/Optical-Image-3.jpg` | 1733 × 2500 | 1733 × 2500 | 3 (all identical — effectively 1) | 3 (distinct R/G/B) |

All files are JPEG, `uint8`, no GeoTIFF CRS/transform, no EXIF metadata.

---

## Numeric Domain

### SAR images

| Statistic | Pair 1 | Pair 2 | Pair 3 |
|-----------|--------|--------|--------|
| Dtype | uint8 | uint8 | uint8 |
| Min | 0 | 0 | 0 |
| Max | 255 | 255 | 255 |
| Mean | 82.36 | 54.61 | 49.60 |
| Std | 46.10 | 46.25 | 41.89 |
| NaN | 0 | 0 | 0 |
| Inf | 0 | 0 | 0 |
| R=G=B (grayscale-in-RGB)? | **YES** | **YES** | **YES** |

**Critical finding:** All three SAR images are **grayscale data stored in a 3-channel JPEG**. The R, G, and B channel values are byte-for-byte identical across all pixels in all pairs. This confirms the SAR files are effectively single-band (1-channel) imagery — the JPEG container has simply duplicated the grayscale values into all three colour channels.

**Physical domain:** `UNKNOWN — requires manual verification.`

The values span `[0, 255]` as `uint8`. This is a display-ready normalized encoding, but the original physical domain (amplitude, power, or dB) is not recoverable from the JPEG alone. JPEG compression is lossy and strips geospatial metadata; the raw physical backscatter values are not available in these files.

The dB detection heuristic (`min < −20`) correctly returns `False` for these files — the range is display-normalized, not raw dB. The heuristic is trustworthy for GeoTIFF sources; for uint8 JPEGs it is uninformative and should be disregarded.

**Implication for preprocessing:** The SAR preprocessing pipeline (`ml/preprocessing/sar.py`, Day 2) must be configured for a **display-normalized [0, 255] → [0, 1] rescaling** path, not the dB-domain log-transform path. The `value_domain` parameter in `prepare_sar()` should be set to `"normalized"` or `"uint8"`.

### Optical / reference images

| Statistic | Pair 1 | Pair 2 | Pair 3 |
|-----------|--------|--------|--------|
| Dtype | uint8 | uint8 | uint8 |
| Min | 0 | 0 | 0 |
| Max | 255 | 255 | 255 |
| Mean (R / G / B) | 169.1 / 150.0 / 104.2 | 199.4 / 173.5 / 118.3 | 172.8 / 143.3 / 95.2 |
| Std (R / G / B) | 51.3 / 44.9 / 53.3 | 51.0 / 34.6 / 40.4 | 49.4 / 39.9 / 44.4 |
| R=G=B? | **NO** | **NO** | **NO** |

All optical images have genuinely distinct R, G, B channels consistent with real colour imagery. This confirms they are valid optical RGB references, not grayscale or false-colour SAR products. The higher mean luminance and warm bias (R > G > B) is consistent with natural land/vegetation optical imagery.

---

## Pairing and Alignment

### Pair 1

| Check | Result |
|-------|--------|
| Pairing (filename ID) | **PASS** — ID `1` extracted from both files |
| SAR dimensions | 1733 × 2500 |
| Optical dimensions | 1733 × 2500 |
| Spatial dimensions match | **YES** |
| SAR channels (effective) | 1 (grayscale-in-RGB) |
| Optical channels | 3 (distinct RGB) |
| Channel compatibility for loader | **YES** — loader handles 3-ch SAR, model must treat as 1-ch or use only first channel |
| Spatial co-registration evidence | **UNKNOWN** — dimensions match but no GCP, CRS, transform, or EXIF to confirm pixel-level alignment |
| Limitation | JPEG encoding, display-normalized, no geospatial metadata |

### Pair 2

| Check | Result |
|-------|--------|
| Pairing | **PASS** |
| Spatial dimensions match | **YES** (1733 × 2500) |
| Effective SAR channels | 1 |
| Optical channels | 3 |
| Co-registration | **UNKNOWN** |

### Pair 3

| Check | Result |
|-------|--------|
| Pairing | **PASS** |
| Spatial dimensions match | **YES** (1733 × 2500) |
| Effective SAR channels | 1 |
| Optical channels | 3 |
| Co-registration | **UNKNOWN** |

**Important caveat on co-registration:** Matching pixel dimensions alone does not prove spatial co-registration. The SAR and optical images could be at the same pixel size but cover different geographic extents, or be offset by a systematic translation. Since the files are JPEGs with no CRS, no transform, no EXIF GPS, and no GCP metadata, spatial alignment cannot be verified programmatically. Manual visual inspection or knowledge of the ISRO/SAC dataset preparation pipeline is required before claiming that pixel `(r, c)` in the SAR image corresponds to pixel `(r, c)` in the optical image.

---

## Metadata

| Source | CRS | Geotransform | EXIF | Geolocation | Notes |
|--------|-----|-------------|------|-------------|-------|
| All 6 JPEG files | None | None | None | None | Pure image data, no geospatial or sensor metadata retained |

No CRS, no coordinate reference, no EXIF GPS tags, no acquisition timestamps, no sensor band IDs. The JPEG container format does not support geospatial metadata; all such information was stripped during dataset preparation.

---

## Dataset Size Limitation

> **There are exactly 3 independent SAR/optical pairs in the current SIH1733 dataset.**

This is a critical constraint for the planned supervised deep-learning system.

**Why 3 pairs is insufficient as a standalone training corpus:**

1. **Statistical learning requires scene diversity.** A U-Net with a ResNet-18 encoder has millions of parameters. Training on 3 scenes guarantees severe overfitting — the model will memorize the three training images rather than learning a generalizable SAR→RGB mapping.

2. **Patch extraction does not create independent samples.** Extracting 256×256 patches from the same 1733×2500 image creates spatially correlated tiles. These patches share lighting conditions, scene content, and SAR characteristics. They are **not independent training examples** and must not be counted as independent pairs.

3. **Train/val/test split is statistically meaningless at 3 pairs.** Even a 2/0.5/0.5 split (in integer pairs: 2 train, 0 val, 0 test, or 1 train, 1 val, 1 test) leaves each split with effectively 1 scene. Metrics computed on a 1-scene test set have no statistical power and cannot support any generalization claim.

4. **Scene leakage is guaranteed if patches are treated as independent.** If 256×256 patches from the same full image are split between train and test, the model trivially sees test-scene content during training.

**The split generator (`scripts/make_splits.py`) was intentionally NOT run** because the implementation correctly detects only 3 scenes, which is below the minimum needed for a meaningful split. Running it would produce a 2-train / 1-val / 0-test or similar degenerate partition that cannot support valid evaluation.

---

## Test Results

| Test | Command | Result | Details |
|------|---------|--------|---------|
| Dataset audit | `python scripts/inspect_dataset.py --data-root data/raw` | **PASS** | 6 files discovered, all 6 inspected, report saved |
| Smoke test | `python ml/data/smoke_test.py --data-root data/raw --n-samples 3` | **PASS** | 3/3 pairs loaded, all shape/NaN/spatial checks passed |
| Unit tests | `pytest tests/test_dataset.py -v` | **PASS** | 17/17 tests passed |
| Split generation | NOT RUN | **SKIPPED (intentional)** | Only 3 independent scenes — split would be statistically meaningless |

**Generated artifacts:**
- `reports/day1_dataset_audit.md` — full per-file audit table
- `reports/day1_pairing_check.png` — visual contact sheet (SAR | Optical for all 3 pairs)

**Bug fixed during validation:**
- `ml/data/smoke_test.py`: removed fragile `scripts.make_splits` module import; replaced with self-contained inline CSV reader (no production logic changed)
- `scripts/inspect_dataset.py`: replaced Unicode `→` and `…` print characters with ASCII equivalents for Windows cp1252 compatibility

---

## SIH1733 Compliance Assessment

| Requirement | Status | Evidence |
|-------------|--------|---------|
| SAR input | **PASS** | 3 single-band (grayscale-in-RGB JPEG) SAR images confirmed |
| Optical reference | **PASS** | 3 genuine multi-channel optical RGB images confirmed; R≠G≠B verified |
| SAR–Optical filename pairing | **PASS** | Numeric ID `1/2/3` extracted from filenames; all 3 pairs matched correctly |
| Spatial alignment (pixel-level) | **UNKNOWN** | Dimensions match (1733×2500 all pairs), but no CRS/EXIF/GCP to confirm registration |
| SAR physical domain known | **UNKNOWN** | uint8 JPEG — display-normalized encoding; original amplitude/power/dB lost |
| Suitable for supervised DL (loader level) | **CONDITIONAL** | Loader works; SAR must be treated as 1-channel (deduplicate grayscale-in-RGB) |
| Sufficient data for training by itself | **NO** | 3 scenes is insufficient; severe overfitting guaranteed without additional data |
| GeoTIFF / geospatial metadata | **FAIL** | JPEG format, no CRS, no transform, no EXIF |

---

## Decision Gate

```
DAY 1 CONDITIONAL — DATA VALID BUT ADDITIONAL TRAINING DATA REQUIRED
```

The loader, pairing logic, and smoke test all pass. The dataset is technically readable and correctly paired. However, **3 image pairs cannot support supervised deep-learning training or held-out evaluation** on their own. The SIH1733 dataset functions as a demonstration corpus (visual examples for the problem statement), not as a training dataset.

---

## Recommended Next Action

**Identify and integrate additional paired SAR/optical data before beginning model training.**

Specifically:

1. **Check the full official SIH1733 ZIP** for any additional data beyond the 3 pairs currently placed under `data/raw/`. The ISRO/SAC download page describes a ~10 MB dataset — the 3 JPEGs account for ~7.2 MB; additional files may exist.

2. **If the full dataset is confirmed to be only 3 pairs**, supplement with a compatible publicly available paired SAR/optical corpus (e.g., SEN12MS, SpaceNet-6, or manually co-registered Sentinel-1/Sentinel-2 tiles) before training. Document the supplementary source in `reports/day1_dataset_audit.md` and confirm that its licence is compatible with SIH submission.

3. **Do not begin Day 2 model implementation** until the training data question is resolved.

4. **Do not adjust patch size or extraction strategy** to artificially inflate the apparent sample count from the 3 existing pairs.

---

*Report generated by Day 1 validation run — SIH1733 SAR Image Colorization project*
