"""Shared test fixtures: a tiny but complete pipeline built once per test session.

Plain functions (not pytest fixtures) so the suite runs under both
``pytest`` and ``python -m unittest``.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from demandmesh.config import Settings
from demandmesh.data_generator import generate_dataset
from demandmesh.data_preprocessing import clean_data, save_history
from demandmesh.model_training import train

TINY_CITIES = ("SRT", "LDH")


@lru_cache(maxsize=1)
def tiny_environment() -> Settings:
    """Generate -> clean -> train a small model into a temporary directory."""
    root = Path(tempfile.mkdtemp(prefix="demandmesh-test-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    settings = Settings(
        raw_dir=root / "raw",
        processed_dir=root / "processed",
        model_dir=root / "models",
        reports_dir=root / "reports",
        routing_time_limit_s=0.5,
    )
    settings.ensure_dirs()
    demand, zones = generate_dataset(start="2024-06-01", end="2025-12-31", seed=7, cities=TINY_CITIES)
    zones.to_csv(settings.zones_path, index=False)
    history, _ = clean_data(demand)
    save_history(history, settings.history_path)
    train(history, zones, settings.model_dir, params={"max_iter": 40}, seed=7)
    return settings
