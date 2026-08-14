"""
tests/test_dataset.py
Unit tests for ml.data.dataset — Day 1 verification.

These tests use synthetic in-memory data so they run without the SIH1733 dataset.
They verify the shape contract, NaN detection, pairing validation, and crop/pad.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

from ml.data.dataset import (
    SARColorizationDataset,
    _crop_or_pad,
    apply_crop_pad,
    _extract_id,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — synthetic data on disk
# ─────────────────────────────────────────────────────────────────────────────

def _make_npy(arr: np.ndarray, path: Path) -> Path:
    np.save(path, arr)
    return path


@pytest.fixture
def synthetic_pairs(tmp_path):
    """Create 6 synthetic SAR+target .npy pairs in a temp directory."""
    pairs = []
    for i in range(6):
        sar_arr = np.random.rand(2, 300, 300).astype(np.float32)  # 2-band SAR
        tgt_arr = np.random.rand(3, 300, 300).astype(np.float32)  # RGB target
        sar_path = tmp_path / f"sar_pair{i:03d}.npy"
        tgt_path = tmp_path / f"optical_pair{i:03d}.npy"
        np.save(sar_path, sar_arr)
        np.save(tgt_path, tgt_arr)
        pairs.append({"sar": sar_path, "target": tgt_path, "id": str(i)})
    return pairs


@pytest.fixture
def synthetic_sar_only(tmp_path):
    """Create SAR-only entries for inference-only testing."""
    pairs = []
    for i in range(3):
        sar_arr = np.random.rand(2, 300, 300).astype(np.float32)
        sar_path = tmp_path / f"sar_{i:03d}.npy"
        np.save(sar_path, sar_arr)
        pairs.append({"sar": sar_path, "target": None, "id": str(i)})
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Tests — _extract_id
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractId:
    def test_numeric_id(self):
        p = Path("sar_001.npy")
        assert _extract_id(p) == "1"

    def test_optical_numeric_id(self):
        p = Path("optical_003_ref.tif")
        assert _extract_id(p) == "3"

    def test_no_numeric(self):
        p = Path("somefile.tif")
        # Should return something non-empty
        result = _extract_id(p)
        assert isinstance(result, str) and len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests — crop / pad
# ─────────────────────────────────────────────────────────────────────────────

class TestCropPad:
    def test_centre_crop(self):
        arr = np.zeros((2, 300, 300), dtype=np.float32)
        out = _crop_or_pad(arr, 256, 256)
        assert out.shape == (2, 256, 256)

    def test_pad_small(self):
        arr = np.zeros((2, 100, 100), dtype=np.float32)
        out = _crop_or_pad(arr, 256, 256)
        assert out.shape == (2, 256, 256)

    def test_exact_size(self):
        arr = np.zeros((1, 256, 256), dtype=np.float32)
        out = _crop_or_pad(arr, 256, 256)
        assert out.shape == (1, 256, 256)

    def test_apply_crop_pad_same_result(self):
        sar = np.random.rand(2, 300, 300).astype(np.float32)
        tgt = np.random.rand(3, 300, 300).astype(np.float32)
        sar_out, tgt_out = apply_crop_pad(sar, tgt, 256, 256)
        assert sar_out.shape == (2, 256, 256)
        assert tgt_out.shape == (3, 256, 256)

    def test_spatial_mismatch_raises(self):
        sar = np.zeros((2, 300, 300), dtype=np.float32)
        tgt = np.zeros((3, 200, 200), dtype=np.float32)  # different spatial size
        with pytest.raises(ValueError, match="spatial dimensions differ"):
            apply_crop_pad(sar, tgt, 256, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Tests — SARColorizationDataset
# ─────────────────────────────────────────────────────────────────────────────

class TestSARColorizationDataset:
    def test_len(self, synthetic_pairs):
        ds = SARColorizationDataset(synthetic_pairs, patch_size=(256, 256))
        assert len(ds) == 6

    def test_item_shapes(self, synthetic_pairs):
        ds = SARColorizationDataset(synthetic_pairs, patch_size=(256, 256))
        sample = ds[0]
        assert sample["sar"].shape == (2, 256, 256)
        assert sample["target_rgb"].shape == (3, 256, 256)

    def test_item_dtype(self, synthetic_pairs):
        ds = SARColorizationDataset(synthetic_pairs, patch_size=(256, 256))
        sample = ds[0]
        assert sample["sar"].dtype == torch.float32
        assert sample["target_rgb"].dtype == torch.float32

    def test_no_nan(self, synthetic_pairs):
        ds = SARColorizationDataset(synthetic_pairs, patch_size=(256, 256))
        sample = ds[0]
        assert torch.all(torch.isfinite(sample["sar"]))
        assert torch.all(torch.isfinite(sample["target_rgb"]))

    def test_metadata_keys(self, synthetic_pairs):
        ds = SARColorizationDataset(synthetic_pairs, patch_size=(256, 256))
        sample = ds[0]
        meta = sample["metadata"]
        assert "pair_id" in meta
        assert "sar_path" in meta
        assert "target_path" in meta

    def test_inference_only_no_target(self, synthetic_sar_only):
        ds = SARColorizationDataset(synthetic_sar_only, patch_size=(256, 256), inference_only=True)
        sample = ds[0]
        assert sample["target_rgb"] is None

    def test_missing_target_raises_without_inference_only(self, synthetic_sar_only):
        with pytest.raises(ValueError, match="inference_only"):
            SARColorizationDataset(synthetic_sar_only, patch_size=(256, 256), inference_only=False)

    def test_nan_in_sar_raises(self, tmp_path):
        sar_arr = np.full((2, 300, 300), np.nan, dtype=np.float32)
        tgt_arr = np.random.rand(3, 300, 300).astype(np.float32)
        sar_path = tmp_path / "sar_bad.npy"
        tgt_path = tmp_path / "optical_bad.npy"
        np.save(sar_path, sar_arr)
        np.save(tgt_path, tgt_arr)
        ds = SARColorizationDataset(
            [{"sar": sar_path, "target": tgt_path, "id": "bad"}],
            patch_size=(256, 256),
        )
        with pytest.raises(RuntimeError, match="NaN/Inf"):
            _ = ds[0]

    def test_sar_transform_applied(self, synthetic_pairs):
        """SAR transform should be applied to SAR tensor but not target."""
        def multiply_by_2(arr: np.ndarray) -> np.ndarray:
            return arr * 2.0

        ds_base   = SARColorizationDataset(synthetic_pairs, patch_size=(256, 256))
        ds_trans  = SARColorizationDataset(synthetic_pairs, patch_size=(256, 256), sar_transform=multiply_by_2)

        s_base  = ds_base[0]
        s_trans = ds_trans[0]

        np.testing.assert_allclose(
            s_trans["sar"].numpy(),
            s_base["sar"].numpy() * 2.0,
            rtol=1e-5,
        )
        # Target should be unaffected
        np.testing.assert_allclose(
            s_trans["target_rgb"].numpy(),
            s_base["target_rgb"].numpy(),
            rtol=1e-5,
        )
