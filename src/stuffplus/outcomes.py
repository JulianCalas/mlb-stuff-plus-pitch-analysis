"""Count-neutral pitch run value.

``delta_run_exp`` is the change in batting-team run expectancy on that
one pitch. It is the right currency, and it is also full of count and
base-out noise: the same hanging slider is worth more on 3-1 than on 0-2.
Stuff is a property of the pitch, not of the count it happened to be
thrown in.

The target used here is the league-average ``delta_run_exp`` of the
pitch's outcome bucket, estimated on the training seasons only and then
applied forward. A whiff in 2026 is worth the average whiff from the
training seasons, not whatever the count happened to be. Walks stay
inside the ball bucket and strikeouts stay inside whiffs or called
strikes, so the average already includes the fact that some balls become
walks and some strikes become strikeouts. The model never sees the count.

Run value is from the batter's side. Negative is good for the pitcher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
FOUL_DESCRIPTIONS = {"foul"}
BALL_DESCRIPTIONS = {"ball", "blocked_ball"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}


def assign_outcome(description: pd.Series, events: pd.Series) -> pd.Series:
    """Map each pitch to a coarse outcome. Bunts and pitchouts become ``other``."""
    desc = description.fillna("").astype(str)
    ev = events.fillna("").astype(str)
    out = pd.Series("other", index=desc.index, dtype="object")
    out = out.mask(desc.isin(WHIFF_DESCRIPTIONS), "whiff")
    out = out.mask(desc.isin(FOUL_DESCRIPTIONS), "foul")
    out = out.mask(desc.eq("called_strike"), "called_strike")
    out = out.mask(desc.isin(BALL_DESCRIPTIONS), "ball")
    out = out.mask(desc.eq("hit_by_pitch") | ev.eq("hit_by_pitch"), "hit_by_pitch")

    in_play = desc.str.startswith("hit_into_play")
    out = out.mask(in_play & ev.eq("single"), "single")
    out = out.mask(in_play & ev.eq("double"), "double")
    out = out.mask(in_play & ev.eq("triple"), "triple")
    out = out.mask(in_play & ev.eq("home_run"), "home_run")
    out = out.mask(in_play & ev.eq("field_error"), "field_error")
    played = HIT_EVENTS | {"field_error", ""}
    out = out.mask(in_play & ~ev.isin(played), "in_play_out")

    bunt = desc.str.contains("bunt") | ev.str.contains("bunt")
    out = out.mask(bunt | desc.eq("pitchout"), "other")
    return out


def is_swing(description: pd.Series) -> pd.Series:
    """A swing is a whiff, a foul, or contact. Takes and pitchouts are not swings."""
    desc = description.fillna("").astype(str)
    swing_exact = WHIFF_DESCRIPTIONS | FOUL_DESCRIPTIONS
    return desc.isin(swing_exact) | desc.str.startswith("hit_into_play")


def is_whiff(description: pd.Series) -> pd.Series:
    """Swinging strike, blocked swinging strike, or a caught foul tip."""
    desc = description.fillna("").astype(str)
    return desc.isin(WHIFF_DESCRIPTIONS)


def outcome_run_values(
    train: pd.DataFrame,
    run_value_col: str = "delta_run_exp",
    min_pitches: int = 200,
) -> pd.Series:
    """Mean run value of each outcome on the training seasons.

    Buckets with fewer than ``min_pitches`` pitches are dropped. Two
    hundred is the production floor so a stray label cannot set a run
    value. Tests pass a smaller floor.
    """
    usable = train.loc[train[run_value_col].notna() & train["outcome"].ne("other")]
    grouped = usable.groupby("outcome", observed=False)[run_value_col].agg(["mean", "size"])
    grouped = grouped.loc[grouped["size"] >= min_pitches, "mean"]
    return grouped


def assign_count_neutral_run_value(
    df: pd.DataFrame,
    training_means: pd.Series,
    run_value_col: str = "delta_run_exp",
) -> pd.Series:
    """Look up the training-set value of each pitch's outcome.

    Pitches with no usable outcome, or with a missing raw run value, get
    NaN and are left out of fitting and scoring.
    """
    mapped = df["outcome"].map(training_means)
    mapped = mapped.where(df["outcome"].ne("other") & df[run_value_col].notna())
    return mapped.astype(float)
