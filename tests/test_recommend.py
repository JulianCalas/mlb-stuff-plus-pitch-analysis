"""Suggestion rules on a hand-built arsenal, with no model fit."""

import pandas as pd

from stuffplus.recommend import suggest_for_pitcher, usage_shift_runs


def _league() -> pd.DataFrame:
    rows = []
    # The pitcher we are advising. Slider is a drag; no changeup.
    rows.append(_row(1, "FF", 0.55, 110, 96, 16, 8, cluster=3, pitches=600))
    rows.append(_row(1, "SL", 0.45, 85, 84, 2, -2, cluster=3, pitches=500))
    # Peers in the same fastball cluster. Most of them throw a good changeup,
    # and their sliders in this velocity band grade as sweepers.
    for pitcher in range(2, 18):
        rows.append(_row(pitcher, "FF", 0.50, 108, 96, 16, 8, cluster=3, pitches=500))
        rows.append(_row(pitcher, "SL", 0.25, 112, 85, 0, -8, cluster=3, pitches=250))
        if pitcher <= 12:
            rows.append(_row(pitcher, "CH", 0.25, 106, 86, 6, 14, cluster=3, pitches=250))
        else:
            rows.append(_row(pitcher, "CU", 0.25, 101, 80, -8, -4, cluster=3, pitches=250))
    return pd.DataFrame(rows)


def _row(pitcher, pitch_type, usage, stuff, velo, ivb, hb, cluster, pitches):
    return {
        "pitcher": pitcher,
        "season": 2026,
        "p_throws": "R",
        "pitch_type": pitch_type,
        "usage": usage,
        "stuff_plus": stuff,
        "velo": velo,
        "ivb": ivb,
        "hb_arm": hb,
        "pred_rv": -0.002 * (stuff - 100),
        "actual_rv": -0.002 * (stuff - 100),
        "pitches": pitches,
        "fb_cluster": cluster,
        "loc_sd": 0.8,
        "loc_sd_league": 0.7,
    }


def test_usage_shift_math():
    # 0.03 runs per pitch, on 10 percent of pitches, per 100 pitches.
    assert abs(usage_shift_runs(0.02, -0.01, share=0.10) - 0.30) < 1e-9


def test_suggestions_cover_usage_shape_and_a_missing_pitch():
    league = _league()
    advice = suggest_for_pitcher(league.loc[league.pitcher == 1], league)
    kinds = {item["kind"] for item in advice}
    assert "usage" in kinds
    assert "shape" in kinds
    assert "add_pitch" in kinds
    added = next(item for item in advice if item["kind"] == "add_pitch")
    assert added["pitch_type"] == "CH"
    shaped = next(item for item in advice if item["kind"] == "shape")
    assert shaped["pitch_type"] == "SL"
    assert shaped["evidence"]["hb_gap"] < -1.5
