"""Write the compact tables the Streamlit app ships with.

A Community Cloud clone does not have the gitignored Statcast caches or
``models/stuff_models.joblib``. Grades are already on the pitcher-pitch
rows, so the app never needs the fitted model. This module copies those
grades, the reliability tables, and the suggestion text, and keeps a
short random cloud of pitches for the movement and release charts.

Every pitcher in ``pitcher_season.parquet`` stays in the bundle. The
pitch cloud keeps every pitcher-season that the full plot sample has,
thinned until ``data/app`` is well under 25 MB.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from stuffplus.config import PROCESSED_DIR, RANDOM_SEED, ROOT

APP_DIR = ROOT / "data" / "app"
# 25 MB is the hard ceiling. Stay under 20 MB so the clone is obviously smaller.
MAX_BYTES = 25 * 1024 * 1024
TARGET_BYTES = 20 * 1024 * 1024
# Thinner clouds first-fit under the target. 48 is already a downsample of the
# pipeline's 80-pitch draw, and it is enough to see shape and release.
PER_PITCH_OPTIONS = (48, 32, 20, 12)

COPIED_FILES = (
    "arsenal.parquet",
    "pitcher_season.parquet",
    "reliability.json",
    "suggestions.json",
    "memo.json",
    "metrics.json",
)

PLOT_COLUMNS = (
    "pitcher",
    "game_year",
    "pitch_type",
    "release_speed",
    "ivb",
    "hb_arm",
    "release_pos_x",
    "release_pos_z",
)


def _directory_size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def downsample_plot_sample(sample: pd.DataFrame, per_pitch: int) -> pd.DataFrame:
    """Keep at most ``per_pitch`` rows for each pitcher, season, and pitch type.

    The source sample is already a random draw. Shuffling again with the
    project seed makes a thinner subset deterministic.
    """
    missing = [column for column in PLOT_COLUMNS if column not in sample.columns]
    if missing:
        raise ValueError(f"plot sample is missing columns the app charts: {missing}")
    frame = sample.loc[:, list(PLOT_COLUMNS)].copy()
    rng = np.random.default_rng(RANDOM_SEED)
    frame["_draw"] = rng.random(len(frame))
    frame = frame.sort_values(["pitcher", "game_year", "pitch_type", "_draw"], kind="mergesort")
    frame["_rn"] = frame.groupby(["pitcher", "game_year", "pitch_type"], observed=False).cumcount()
    kept = frame.loc[frame["_rn"] < per_pitch].drop(columns=["_draw", "_rn"])
    kept["pitcher"] = kept["pitcher"].astype("int32")
    kept["game_year"] = kept["game_year"].astype("int16")
    for column in ("release_speed", "ivb", "hb_arm", "release_pos_x", "release_pos_z"):
        kept[column] = pd.to_numeric(kept[column], errors="coerce").astype("float32")
    kept["pitch_type"] = kept["pitch_type"].astype("category")
    return kept.reset_index(drop=True)


def _pitcher_seasons(frame: pd.DataFrame, season_column: str) -> set[tuple[int, int]]:
    pairs = zip(frame["pitcher"].astype(int), frame[season_column].astype(int), strict=True)
    return set(pairs)


def export_app_bundle(
    processed_dir: Path | None = None,
    app_dir: Path | None = None,
) -> dict:
    """Copy the app tables into ``data/app`` and return a small manifest."""
    source = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR
    destination = Path(app_dir) if app_dir is not None else APP_DIR
    required = [source / name for name in (*COPIED_FILES, "plot_sample.parquet")]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "App bundle needs the pipeline outputs "
            f"{missing}. Run python scripts/pull_statcast.py and python scripts/run_pipeline.py."
        )

    destination.mkdir(parents=True, exist_ok=True)
    for name in COPIED_FILES:
        shutil.copy2(source / name, destination / name)

    full_sample = pd.read_parquet(source / "plot_sample.parquet")
    source_pairs = _pitcher_seasons(full_sample, "game_year")
    chosen = None
    slim = None
    for per_pitch in PER_PITCH_OPTIONS:
        slim = downsample_plot_sample(full_sample, per_pitch)
        slim_pairs = _pitcher_seasons(slim, "game_year")
        dropped = source_pairs - slim_pairs
        if dropped:
            raise RuntimeError(f"Downsample dropped {len(dropped)} pitcher-seasons from the pitch cloud")
        slim.to_parquet(destination / "plot_sample.parquet", index=False, compression="zstd")
        size = _directory_size(destination)
        chosen = per_pitch
        if size <= TARGET_BYTES:
            break
    if slim is None or chosen is None:
        raise RuntimeError("No pitch-cloud sample was written")

    pitchers = pd.read_parquet(destination / "pitcher_season.parquet", columns=["pitcher", "season", "player_name"])
    offered = _pitcher_seasons(pitchers, "season")
    offered_ids = {pitcher for pitcher, _season in offered}
    sample_ids = set(slim["pitcher"].astype(int))
    missing_ids = sorted(offered_ids - sample_ids)
    if missing_ids:
        raise RuntimeError(
            f"{len(missing_ids)} pitchers on the desk have no pitch cloud. First ids: {missing_ids[:8]}"
        )

    size = _directory_size(destination)
    if size >= MAX_BYTES:
        raise RuntimeError(f"data/app is {size} bytes, which is not under 25 MB")

    manifest = {
        "source": "pipeline outputs in data/processed",
        "per_pitch_cap": chosen,
        "bytes": size,
        "pitcher_seasons": len(offered),
        "pitchers": len(offered_ids),
        "pitch_cloud_rows": int(len(slim)),
        "files": sorted({*(path.name for path in destination.iterdir() if path.is_file()), "manifest.json"}),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["bytes"] = _directory_size(destination)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["bytes"] >= MAX_BYTES:
        raise RuntimeError(f"data/app is {manifest['bytes']} bytes, which is not under 25 MB")
    return manifest
