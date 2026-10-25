"""Rebuild data/app from data/processed without refitting.

    python scripts/export_app_bundle.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stuffplus.app_bundle import export_app_bundle  # noqa: E402


if __name__ == "__main__":
    manifest = export_app_bundle()
    print(json.dumps(manifest, indent=2))
