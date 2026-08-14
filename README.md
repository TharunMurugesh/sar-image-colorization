# SAR Image Colorization — SIH1733

**Physically-informed SAR-to-color image translation using a SAR-Structure-Guided U-Net (SSG-U-Net) with MC-Dropout uncertainty estimation.**

> ISRO SIH1733 — *SAR Image Colorization for Comprehensive Insight using Deep Learning Model (h)*

---

## Overview

This system learns to colorize SAR (Synthetic Aperture Radar) imagery using paired SAR/optical training data.

**Core capabilities:**
1. SAR-Structure-Guided U-Net (ResNet-18 encoder + lightweight SAR Structural Guidance Module)
2. Joint color + structure training objective: `L = 1.0·L1_RGB + 0.5·(1−SSIM) + 0.1·L_structure`
3. MC-Dropout uncertainty estimation (relative confidence proxy — **not calibrated probability**)
4. Trust-gated rendering that attenuates low-confidence regions toward SAR grayscale
5. React + FastAPI full-stack application with SQLite history

---

## Repository Structure

```
sar-colorization/
├── frontend/               # React + Vite UI
├── backend/                # FastAPI application
├── ml/                     # ML subsystem (preprocessing / model / training / inference / evaluation)
│   ├── data/               # Dataset class and smoke test
│   ├── preprocessing/      # SAR, RGB, transforms, structure
│   ├── models/             # SSG-U-Net
│   ├── training/           # Train script, loss
│   ├── inference/          # Shared inference pipeline
│   └── evaluation/         # Metrics, uncertainty, trust gating
├── data/
│   ├── raw/                # Original dataset files (gitignored)
│   └── processed/          # Processed tensors / cached patches (gitignored)
├── scripts/                # One-off utilities
├── tests/                  # pytest test suite
├── notebooks/              # Analysis notebooks
├── reports/                # Audit, metrics, qualitative outputs
├── runtime/                # Generated artifacts (gitignored)
│   ├── uploads/
│   ├── results/
│   └── app.db
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10 or 3.11
- CUDA-capable GPU recommended (CPU fallback supported)
- Node.js ≥ 18 (for frontend)

### 1. Create virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

For CUDA (adjust the wheel for your CUDA version):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Place dataset

Copy the official SIH1733 dataset to `data/raw/`.

### 4. Day 1 — Dataset audit

```bash
python scripts/inspect_dataset.py --data-root data/raw --output reports/day1_dataset_audit.md
```

### 5. Generate splits

```bash
python scripts/make_splits.py --data-root data/raw --output-dir data/
```

### 6. Smoke test

```bash
python ml/data/smoke_test.py
```

---

## Development — native

### Backend

```bash
cd backend
pip install -r requirements.txt  # if separate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Docker (quick start)

```bash
docker compose up --build
```

Open `http://localhost:5173` in your browser.

---

## Uncertainty caveat

> The uncertainty-derived trust score is a **relative confidence proxy** from stochastic model predictions. It is **not** a calibrated probability of colour correctness.

## Colour caveat

> Predicted RGB is a learned visual representation conditioned on SAR/optical training pairs. It is **not** recovered optical ground truth.

---

## Project: SIH1733 — ISRO/SAC
