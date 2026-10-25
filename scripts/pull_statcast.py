"""Download regular-season Statcast pitches and cache them as parquet.

The pull uses ``pybaseball.statcast``, which reads Baseball Savant's
pitch-level CSV. Savant will not return a full season in one call, so
pybaseball requests one day at a time. This script walks the regular
season in small parallel batches, keeps the columns the model needs,
and writes one parquet file per day so a failed run can resume.

Raw day files live in ``data/raw/chunks/`` and are gitignored. The
script then builds one season file, ``data/raw/pitches_YYYY.parquet``,
which is the cache the rest of the pipeline reads.

Usage:
    python scripts/pull_statcast.py
    python scripts/pull_statcast.py --season 2026
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stuffplus.config import (  # noqa: E402
    CHUNK_DIR,
    FLOAT_COLUMNS,
    INT_COLUMNS,
    KEEP_COLUMNS,
    RAW_DIR,
    SEASONS,
)

# A browser user agent and a hard timeout. pybaseball's own request sets
# timeout=None, which will hang a multi-hour pull on one stalled day.
_ORIGINAL_GET = requests.get


def _get_with_limits(*args, **kwargs):
    headers = dict(kwargs.get("headers") or {})
    headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (compatible; stuffplus-research/1.0; +https://baseballsavant.mlb.com)",
    )
    kwargs["headers"] = headers
    kwargs["timeout"] = kwargs.get("timeout") or 90
    return _ORIGINAL_GET(*args, **kwargs)


requests.get = _get_with_limits


def _daterange(start: str, end: str) -> list[str]:
    cursor = datetime.strptime(start, "%Y-%m-%d").date()
    last = datetime.strptime(end, "%Y-%m-%d").date()
    days = []
    while cursor <= last:
        days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


def _typed_empty() -> pd.DataFrame:
    """A zero-row frame with the same dtypes as a real daily file.

    Off days have to be cached too, or a resumed pull downloads them again.
    An untyped empty frame would also break the season merge.
    """
    frame = pd.DataFrame({column: pd.Series(dtype="object") for column in KEEP_COLUMNS})
    for column in FLOAT_COLUMNS:
        frame[column] = frame[column].astype("float32")
    for column in INT_COLUMNS:
        frame[column] = frame[column].astype("Int32")
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    return frame[KEEP_COLUMNS]


def slim_statcast(df: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-season rows and the columns the model uses."""
    if df is None or len(df) == 0:
        return _typed_empty()
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    if "pitch_type" not in df.columns or "game_type" not in df.columns:
        return _typed_empty()
    df = df.loc[df["game_type"].astype(str) == "R"].copy()
    for column in KEEP_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    df = df[KEEP_COLUMNS]
    for column in FLOAT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float32")
    for column in INT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int32")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_pk", "pitcher", "at_bat_number", "pitch_number"])
    df = df.drop_duplicates(["game_pk", "at_bat_number", "pitch_number", "pitcher"])
    return df


def fetch_day(day: str) -> pd.DataFrame:
    """One calendar day through pybaseball. The same date is start and end."""
    from pybaseball import statcast

    last_error = None
    for attempt in range(3):
        try:
            raw = statcast(start_dt=day, end_dt=day, verbose=False, parallel=False)
            return slim_statcast(raw)
        except Exception as exc:  # network and the occasional empty-file parse
            last_error = exc
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"{day} failed after 3 tries: {last_error}")


def pull_season(year: int, workers: int = 4) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    start, end = SEASONS[year]
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    days = _daterange(start, end)
    pending = [day for day in days if not (CHUNK_DIR / f"{day}.parquet").exists()]
    print(f"[pull] {year}: {len(days)} days, {len(pending)} still to fetch", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_day, day): day for day in pending}
        for future in as_completed(futures):
            day = futures[future]
            frame = future.result()
            frame.to_parquet(CHUNK_DIR / f"{day}.parquet", index=False, compression="zstd")
            done += 1
            if done % 10 == 0 or done == len(pending):
                print(f"[pull] {year}: cached {done}/{len(pending)} new days", flush=True)


def merge_season(year: int) -> dict:
    """Stack the daily files for one season into a single parquet."""
    import duckdb

    start, _ = SEASONS[year]
    # Daily files are named by date, so the season prefix selects the year.
    pattern = str(CHUNK_DIR / f"{year}-*.parquet")
    destination = RAW_DIR / f"pitches_{year}.parquet"
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT * FROM read_parquet('{pattern}')
            WHERE game_year = {int(year)}
        ) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    counts = con.execute(
        f"""
        SELECT COUNT(*) AS pitches,
               COUNT(DISTINCT pitcher) AS pitchers,
               MIN(game_date) AS first_date,
               MAX(game_date) AS last_date
        FROM read_parquet('{destination}')
        """
    ).fetchdf()
    summary = counts.iloc[0].to_dict()
    summary["season"] = year
    summary["path"] = str(destination)
    summary["first_date"] = str(summary["first_date"])
    summary["last_date"] = str(summary["last_date"])
    summary["pitches"] = int(summary["pitches"])
    summary["pitchers"] = int(summary["pitchers"])
    print(
        f"[pull] {year}: {summary['pitches']:,} pitches, "
        f"{summary['pitchers']:,} pitchers, "
        f"{summary['first_date']} to {summary['last_date']}",
        flush=True,
    )
    if summary["pitches"] < 500_000:
        print(f"[pull] warning: {year} has fewer than 500,000 pitches", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Statcast pitches via pybaseball")
    parser.add_argument("--season", type=int, action="append", dest="seasons")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    years = args.seasons or list(SEASONS)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Importing pybaseball is slow. Do it once, after the requests patch.
    import pybaseball

    pybaseball.cache.enable()
    summaries = []
    for year in years:
        if year not in SEASONS:
            raise SystemExit(f"No date window configured for {year}")
        if not args.merge_only:
            pull_season(year, workers=args.workers)
        summaries.append(merge_season(year))
    manifest = {
        "source": "Baseball Savant via pybaseball.statcast",
        "game_type": "R",
        "seasons": summaries,
        "pulled_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[pull] wrote data/raw/manifest.json", flush=True)


if __name__ == "__main__":
    main()
