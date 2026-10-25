"""Split-half reliability and the Spearman-Brown cross-check."""

import numpy as np
import pandas as pd

from stuffplus.reliability import (
    reliability_curve,
    sample_size_for_reliability,
    single_observation_reliability,
    smallest_stable_n,
    year_over_year,
)


def test_spearman_brown_round_trip():
    r1 = single_observation_reliability(0.5, 100)
    n = sample_size_for_reliability(r1, 0.5)
    assert abs(n - 100) < 1e-6
    # A higher target needs a larger sample.
    assert sample_size_for_reliability(r1, 0.7) > 100


def test_split_half_is_one_when_pitchers_never_vary():
    pitcher_ids = np.repeat([1, 2, 3, 4], 40)
    # Each pitcher is a constant, and the constants differ.
    values = np.repeat([10.0, 20.0, 30.0, 40.0], 40)
    curve = reliability_curve(
        pitcher_ids, values, sample_sizes=(10,), repeats=5, seed=7, min_pitchers=4
    )
    assert curve[0]["pearson"] > 0.99
    assert smallest_stable_n(curve, target=0.7) == 10


def test_year_over_year_correlation():
    left = pd.Series([1.0, 2.0, 3.0, 4.0], index=[10, 11, 12, 13])
    right = pd.Series([1.1, 1.9, 3.2, 3.8], index=[10, 11, 12, 13])
    result = year_over_year(left, right, min_pairs=4)
    assert result["n"] == 4
    assert result["pearson"] > 0.99
