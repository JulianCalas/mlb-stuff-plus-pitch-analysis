"""Season ERA and FIP from the MLB Stats API.

Statcast does not record earned runs, so ERA cannot be built from the
pitch table. The Stats API season line is the official counting stat.
FIP is computed from home runs, walks, hit batsmen, and strikeouts, with
the constant set so that league FIP matches league ERA in that season.
That is the usual FIP constant, estimated here instead of copied from a
workbook.

Innings pitched in the API look like ``162.1``, which means 162 and one
third, not 162.1 decimal innings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

MLB_STATS = "https://statsapi.mlb.com/api/v1/stats"


def parse_innings(value) -> float:
    """Convert a baseball innings string to decimal innings."""
    text = str(value).strip()
    if text in {"", "None", "nan", "--"}:
        return float("nan")
    if "." not in text:
        return float(text)
    whole, fraction = text.split(".", 1)
    thirds = int(fraction[0]) if fraction else 0
    if thirds not in (0, 1, 2):
        return float(text)
    return int(whole) + thirds / 3.0


def fetch_season_pitching(season: int, timeout: int = 60) -> pd.DataFrame:
    """One row per pitcher for a regular season. Stints are summed."""
    rows = []
    offset = 0
    limit = 5000
    while True:
        response = requests.get(
            MLB_STATS,
            params={
                "stats": "season",
                "group": "pitching",
                "season": season,
                "playerPool": "All",
                "limit": limit,
                "offset": offset,
                "sportId": 1,
                "gameType": "R",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        splits = response.json()["stats"][0]["splits"]
        if not splits:
            break
        for split in splits:
            stat = split["stat"]
            player = split["player"]
            rows.append({
                "season": season,
                "pitcher": int(player["id"]),
                "mlb_name": player.get("fullName"),
                "ip": parse_innings(stat.get("inningsPitched")),
                "er": float(stat.get("earnedRuns") or 0),
                "k": float(stat.get("strikeOuts") or 0),
                "bb": float(stat.get("baseOnBalls") or 0),
                "hbp": float(stat.get("hitBatsmen") or 0),
                "hr": float(stat.get("homeRuns") or 0),
                "bf": float(stat.get("battersFaced") or 0),
                "era_reported": float(stat["era"]) if stat.get("era") not in (None, ".---") else np.nan,
            })
        if len(splits) < limit:
            break
        offset += limit
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    summed = frame.groupby(["season", "pitcher"], as_index=False).agg({
        "mlb_name": "first",
        "ip": "sum",
        "er": "sum",
        "k": "sum",
        "bb": "sum",
        "hbp": "sum",
        "hr": "sum",
        "bf": "sum",
    })
    summed = summed.loc[summed["ip"] > 0].copy()
    summed["era"] = summed["er"] / summed["ip"] * 9.0
    summed["k_pct"] = np.where(summed["bf"] > 0, summed["k"] / summed["bf"], np.nan)
    return add_fip(summed)


def add_fip(season_frame: pd.DataFrame) -> pd.DataFrame:
    """Add FIP, with the constant fit separately inside each season."""
    pieces = []
    for _, group in season_frame.groupby("season"):
        block = group.copy()
        innings = block["ip"].sum()
        raw = (
            13.0 * block["hr"].sum()
            + 3.0 * (block["bb"].sum() + block["hbp"].sum())
            - 2.0 * block["k"].sum()
        ) / innings
        league_era = block["er"].sum() / innings * 9.0
        constant = league_era - raw
        block["fip_constant"] = constant
        block["fip"] = (
            13.0 * block["hr"]
            + 3.0 * (block["bb"] + block["hbp"])
            - 2.0 * block["k"]
        ) / block["ip"] + constant
        block["league_era"] = league_era
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)
