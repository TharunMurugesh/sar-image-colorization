# SAR Image Colorization — Validation Report (v2)

**Split evaluated**: Test (54 samples)

## 1. Quantitative Metrics

| Metric | New Model | Old SSG-U-Net | Grayscale Base | Colormap Base |
|--------|-----------|---------------|----------------|---------------|
| PSNR (dB) | **14.73** ± 0.93 | 13.64 | 8.01 | 8.53 |
| SSIM      | **0.330** ± 0.090 | 0.280 | -0.078 | -0.033 |
| DeltaE76  | **16.77** ± 1.87 | 19.47 | 48.51 | 69.67 |

## 2. Uncertainty & Trust Gate

- **Trust Gate Thresholds** (Validation 10th / 90th percentiles):
  - Low Uncertainty (100% Trust): 0.000116
  - High Uncertainty (0% Trust): 0.002555

## 3. Qualitative Evaluation

See `comparison_grid_v2.png` for a side-by-side visual comparison.
