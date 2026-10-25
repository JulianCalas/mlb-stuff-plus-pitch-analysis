"""Features measured against the pitcher's own fastball.

A slider at 85 mph is a different pitch under a 98 mph four-seamer than
under a 92 mph sinker. Stuff models call this the fastball gap: velocity
gap, vertical-break gap, and horizontal-break gap. The primary fastball
is the most-thrown of four-seam, sinker, and cutter, with at least 50
pitches. Ties go to the four-seamer, then the sinker, then the cutter.
Pitchers without that fastball keep missing gaps, and LightGBM is allowed
to treat the gap as missing rather than inventing a fastball.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stuffplus.config import FASTBALL_TIEBREAK, FASTBALL_TYPES


def primary_fastball_table(df: pd.DataFrame, min_pitches: int = 50) -> pd.DataFrame:
    """One row per pitcher-season: which fastball, and its average shape."""
    keys = ["pitcher", "game_year"]
    fb = df.loc[df["pitch_type"].isin(FASTBALL_TYPES), keys + [
        "pitch_type", "release_speed", "ivb", "hb_arm", "vaa"
    ]]
    if fb.empty:
        return pd.DataFrame(columns=keys + [
            "primary_fb", "fb_velo", "fb_ivb", "fb_hb", "fb_vaa", "fb_n"
        ])
    counts = (
        fb.groupby(keys + ["pitch_type"], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )
    counts["tiebreak"] = counts["pitch_type"].map(FASTBALL_TIEBREAK).fillna(9)
    counts = counts.sort_values(
        keys + ["n", "tiebreak"], ascending=[True, True, False, True]
    )
    primary = counts.drop_duplicates(keys, keep="first")
    primary = primary.loc[primary["n"] >= min_pitches].copy()
    means = (
        fb.groupby(keys + ["pitch_type"], observed=False)[["release_speed", "ivb", "hb_arm", "vaa"]]
        .mean()
        .reset_index()
    )
    primary = primary.merge(means, on=keys + ["pitch_type"], how="left")
    primary = primary.rename(columns={
        "pitch_type": "primary_fb",
        "release_speed": "fb_velo",
        "ivb": "fb_ivb",
        "hb_arm": "fb_hb",
        "vaa": "fb_vaa",
        "n": "fb_n",
    })
    return primary[keys + ["primary_fb", "fb_velo", "fb_ivb", "fb_hb", "fb_vaa", "fb_n"]]


def add_fastball_gaps(df: pd.DataFrame, min_fastball_pitches: int = 50) -> pd.DataFrame:
    """Attach primary-fastball gaps. The fastball's own gaps are near zero."""
    out = df.copy()
    primary = primary_fastball_table(out, min_pitches=min_fastball_pitches)
    out = out.merge(primary, on=["pitcher", "game_year"], how="left")
    out["velo_diff"] = out["release_speed"] - out["fb_velo"]
    out["ivb_diff"] = out["ivb"] - out["fb_ivb"]
    out["hb_diff"] = out["hb_arm"] - out["fb_hb"]
    out["is_primary_fb"] = out["pitch_type"].eq(out["primary_fb"]) & out["primary_fb"].notna()
    return out


def stuff_plus_from_run_value(pred_rv, league_mean, league_sd, sd_points: float = 10.0) -> np.ndarray:
    """Scale a predicted run value so 100 is average and higher is better.

    ``pred_rv`` is batter-side run value, so lower is better for the pitcher.
    A prediction one league standard deviation better than average scores 110.
    ``league_mean`` and ``league_sd`` must already be the reference group
    (usually the same pitch type and season, among qualified pitchers).
    """
    pred = np.asarray(pred_rv, dtype=float)
    sd = float(league_sd)
    if not np.isfinite(sd) or sd <= 1e-8:
        return np.full(pred.shape, np.nan)
    return 100.0 + sd_points * (float(league_mean) - pred) / sd
