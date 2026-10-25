"""Fit the Stuff+ model and write grades, figures, and the app tables.

Expects ``data/raw/pitches_YYYY.parquet`` from ``scripts/pull_statcast.py``.

    python scripts/run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stuffplus.pipeline import run  # noqa: E402


if __name__ == "__main__":
    run()
