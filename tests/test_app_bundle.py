"""The Cloud bundle keeps every pitcher and only the columns the charts use."""

import numpy as np
import pandas as pd

from stuffplus.app_bundle import PLOT_COLUMNS, downsample_plot_sample


def test_downsample_keeps_every_pitcher_season_and_respects_the_cap():
    rows = []
    for pitcher in (1, 2):
        for year in (2025, 2026):
            for pitch_type in ("FF", "SL"):
                for i in range(10):
                    rows.append({
                        "pitcher": pitcher,
                        "game_year": year,
                        "pitch_type": pitch_type,
                        "release_speed": 90 + i,
                        "ivb": float(i),
                        "hb_arm": float(-i),
                        "release_pos_x": -1.5,
                        "release_pos_z": 6.0,
                        "unused": 1,
                    })
    sample = pd.DataFrame(rows)
    kept = downsample_plot_sample(sample, per_pitch=3)
    assert list(kept.columns) == list(PLOT_COLUMNS)
    pairs = set(zip(kept["pitcher"].astype(int), kept["game_year"].astype(int), kept["pitch_type"].astype(str)))
    assert pairs == {(1, 2025, "FF"), (1, 2025, "SL"), (1, 2026, "FF"), (1, 2026, "SL"), (2, 2025, "FF"), (2, 2025, "SL"), (2, 2026, "FF"), (2, 2026, "SL")}
    counts = kept.groupby(["pitcher", "game_year", "pitch_type"], observed=True).size()
    assert counts.max() == 3
    assert counts.min() == 3
    assert kept["release_speed"].dtype == np.float32
