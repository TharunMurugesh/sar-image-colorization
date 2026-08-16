"""
backend/app/services/model_loader.py
Singleton model loader for SSGUNet.

Design principles:
- The model is loaded ONCE and cached for the lifetime of the process.
- If the checkpoint file does not exist, ModelNotReadyError is raised — 
  random weights are NEVER used.
- Thread-safe via threading.Lock (multiple worker threads may call load()).
- Device selection: CUDA if available, otherwise CPU.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch

from ml.models.unet import SSGUNet
from backend.app.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model: SSGUNet | None = None
_device: torch.device | None = None


class ModelNotReadyError(RuntimeError):
    """Raised when inference is requested but the checkpoint is unavailable."""
    pass


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path | None = None) -> SSGUNet:
    """
    Load and return the SSGUNet model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
                         Defaults to settings.model_checkpoint.

    Returns:
        The loaded model in eval() mode on the selected device.

    Raises:
        ModelNotReadyError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint is malformed.
    """
    global _model, _device

    ckpt_path = Path(checkpoint_path or settings.model_checkpoint)

    with _lock:
        if _model is not None:
            # Already loaded — return cached model
            return _model

        if not ckpt_path.exists():
            raise ModelNotReadyError(
                f"No trained checkpoint found at '{ckpt_path}'. "
                "Train the SSGUNet first (see ml/training/train.py) and place "
                "the resulting .pt file at the configured MODEL_CHECKPOINT path. "
                "Random weights are not used for inference."
            )

        logger.info("Loading SSGUNet checkpoint from %s …", ckpt_path)
        device = get_device()

        model = SSGUNet(
            in_channels=settings.model_in_channels,
            out_channels=settings.model_out_channels,
            mc_dropout=True,
        )

        state = torch.load(ckpt_path, map_location=device, weights_only=True)

        # Support both raw state_dict and checkpoint dicts ({"model": state_dict, …})
        if isinstance(state, dict) and "model" in state:
            state_dict = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        _model = model
        _device = device
        logger.info("SSGUNet loaded on %s (%s)", device, ckpt_path.name)
        return _model


def get_model() -> SSGUNet:
    """
    Return the cached model. Calls load_model() on first use.

    Raises:
        ModelNotReadyError: If the checkpoint is not present.
    """
    if _model is None:
        return load_model()
    return _model


def get_model_device() -> torch.device:
    """Return the device the model is currently on."""
    global _device
    if _device is None:
        _device = get_device()
    return _device


def is_model_loaded() -> bool:
    """True if the model has been successfully loaded into memory."""
    return _model is not None


def is_checkpoint_available() -> bool:
    """True if the checkpoint file exists on disk (not necessarily loaded yet)."""
    return Path(settings.model_checkpoint).exists()


def reset_model() -> None:
    """
    Unload the model and clear the cache.
    Primarily for testing — do not call in production.
    """
    global _model, _device
    with _lock:
        _model = None
        _device = None
