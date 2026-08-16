"""
backend/app/config.py
Application configuration via pydantic-settings.

All values can be overridden with environment variables or a .env file.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ─────────────────────────────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ── Storage ────────────────────────────────────────────────────────────────
    runtime_dir: Path = Path("runtime")
    upload_dir: Path = Path("runtime/uploads")
    results_dir: Path = Path("runtime/results")
    database_url: str = "sqlite:///./runtime/app.db"

    # ── ML ─────────────────────────────────────────────────────────────────────
    model_checkpoint: Path = Path("runtime/checkpoints/best_model.pt")
    patch_size: int = 256
    mc_passes: int = 10
    # Trust-gate temperature (τ in the uncertainty paper section)
    trust_tau: float = 0.05
    # Model architecture
    model_in_channels: int = 3
    model_out_channels: int = 3

    # ── Derived helpers ────────────────────────────────────────────────────────
    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @field_validator("upload_dir", "results_dir", "runtime_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: str | Path) -> Path:
        return Path(v)


# Module-level singleton — import this everywhere
settings = Settings()
