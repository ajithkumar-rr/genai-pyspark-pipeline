"""Shared paths and logging configuration for the e-commerce pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure consistent console logging for command-line pipeline jobs.

    Args:
        level: Minimum severity to emit. Defaults to ``logging.INFO``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_project_directories() -> None:
    """Create the raw and processed data directories if they do not exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
