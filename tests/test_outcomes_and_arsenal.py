"""Outcome buckets, fastball gaps, and the Stuff+ scale."""

import numpy as np
import pandas as pd

from stuffplus.arsenal import add_fastball_gaps, stuff_plus_from_run_value
from stuffplus.outcomes import (
    assign_count_neutral_run_value,
    assign_outcome,
    is_swing,
    is_whiff,
    outcome_run_values,
)


def _pitches():
    return pd.DataFrame({
        "description": [
            "swinging_strike",
            "foul_tip",
            "foul",
            "called_strike",
            "ball",
            "ball",
            "hit_into_play",
            "hit_into_play",
            "missed_bunt",
            "pitchout",
        ],
        "events": [
            None,
            "strikeout",
            None,
            None,
            None,
            "walk",
            "single",
            "field_out",
            "sac_bunt",
            None,
        ],
    })


def test_outcome_buckets_and_whiffs():
    pitches = _pitches()
    outcome = assign_outcome(pitches["description"], pitches["events"])
    assert list(outcome) == [
        "whiff",
        "whiff",
        "foul",
        "called_strike",
        "ball",
        "ball",  # a walk is still a ball; the count is not a feature
        "single",
        "in_play_out",
        "other",
        "other",
    ]
    assert list(is_whiff(pitches["description"]))[:2] == [True, True]
    assert bool(is_swing(pd.Series(["foul"]))[0]) is True
    assert bool(is_swing(pd.Series(["called_strike"]))[0]) is False


def test_count_neutral_values_come_only_from_the_training_set():
    train = pd.DataFrame({
        "outcome": ["whiff", "whiff", "home_run", "home_run"],
        "delta_run_exp": [-0.20, -0.40, 1.20, 1.40],
    })
    means = outcome_run_values(train, min_pitches=1)
    test = pd.DataFrame({
        "outcome": ["whiff", "home_run", "other"],
        "delta_run_exp": [-0.05, 2.50, 0.10],
    })
    assigned = assign_count_neutral_run_value(test, means)
    assert np.isclose(assigned.iloc[0], -0.30)
    assert np.isclose(assigned.iloc[1], 1.30)
    assert np.isnan(assigned.iloc[2])


def test_primary_fastball_is_the_workhorse_and_gaps_are_zero_there():
    rows = []
    for pitch_type, speed, n in [("FF", 95, 40), ("SI", 93, 80), ("SL", 85, 60)]:
        for i in range(n):
            rows.append({
                "pitcher": 1,
                "game_year": 2026,
                "pitch_type": pitch_type,
                "release_speed": speed,
                "ivb": 16 if pitch_type == "FF" else (8 if pitch_type == "SI" else 0),
                "hb_arm": 8 if pitch_type != "SL" else -6,
                "vaa": -5,
            })
    # Same usage of FF and SI on a second pitcher: the tie goes to the four-seamer.
    for pitch_type, n in [("FF", 50), ("SI", 50)]:
        for _ in range(n):
            rows.append({
                "pitcher": 2,
                "game_year": 2026,
                "pitch_type": pitch_type,
                "release_speed": 94 if pitch_type == "FF" else 92,
                "ivb": 15,
                "hb_arm": 7,
                "vaa": -5,
            })
    graded = add_fastball_gaps(pd.DataFrame(rows), min_fastball_pitches=50)
    sinker = graded.loc[(graded.pitcher == 1) & (graded.pitch_type == "SI")].iloc[0]
    slider = graded.loc[(graded.pitcher == 1) & (graded.pitch_type == "SL")].iloc[0]
    assert sinker.primary_fb == "SI"
    assert np.isclose(sinker.velo_diff, 0)
    assert np.isclose(slider.velo_diff, 85 - 93)
    tied = graded.loc[graded.pitcher == 2, "primary_fb"].iloc[0]
    assert tied == "FF"


def test_stuff_plus_centers_at_100_and_rewards_lower_run_value():
    # Run value is batter-side, so a more negative prediction is a better pitch.
    average = stuff_plus_from_run_value([-0.01], league_mean=-0.01, league_sd=0.005)
    better = stuff_plus_from_run_value([-0.015], league_mean=-0.01, league_sd=0.005)
    worse = stuff_plus_from_run_value([-0.005], league_mean=-0.01, league_sd=0.005)
    assert np.isclose(average[0], 100)
    assert np.isclose(better[0], 110)
    assert np.isclose(worse[0], 90)
