"""How many pitches it takes before a grade stops bouncing around.

Split-half reliability: among pitchers with at least ``2n`` observations,
draw two disjoint samples of size ``n``, average the metric in each
sample, and correlate the two averages across pitchers. That correlation
is the reliability of a sample of size ``n``. We repeat the draw and
average the correlations so one lucky split does not write the chart.

Year-over-year reliability is the same idea with a full season on each
side. ERA and FIP are season stats, so they only get the year-over-year
comparison. Stuff+ and whiff rate get both.

The Spearman-Brown formula converts one measured reliability into the
sample size that would be expected to reach a target reliability, under
the assumption that each extra pitch adds independent information. The
empirical curve is the result we trust. The formula is the cross-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def single_observation_reliability(r_at_n: float, n: int) -> float:
    """Back out the reliability of one observation from a sample of size ``n``.

    If ``r`` is the reliability of a mean of ``n`` observations, the
    classical formula is ``r = n r1 / (1 + (n - 1) r1)``. Solving for
    ``r1`` gives the value returned here.
    """
    if n <= 0 or not np.isfinite(r_at_n) or r_at_n <= 0 or r_at_n >= 1:
        return float("nan")
    denom = n - r_at_n * (n - 1)
    if denom <= 0:
        return float("nan")
    return float(r_at_n / denom)


def sample_size_for_reliability(r1: float, r_target: float) -> float:
    """Sample size whose reliability would equal ``r_target``, given ``r1``.

    Rearrangement of the Spearman-Brown prophecy formula. ``r1`` is the
    reliability of a single observation, from ``single_observation_reliability``.
    """
    if not np.isfinite(r1) or not np.isfinite(r_target):
        return float("nan")
    if r1 <= 0 or r1 >= 1 or r_target <= 0 or r_target >= 1:
        return float("nan")
    return float(r_target * (1.0 - r1) / (r1 * (1.0 - r_target)))


def _safe_corr(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if len(left) < 3 or np.nanstd(left) == 0 or np.nanstd(right) == 0:
        return float("nan"), float("nan")
    pearson = float(pearsonr(left, right).statistic)
    spearman = float(spearmanr(left, right).statistic)
    return pearson, spearman


def split_half_correlation(
    pitcher_ids: np.ndarray,
    values: np.ndarray,
    n: int,
    rng: np.random.Generator,
    min_pitchers: int = 40,
) -> dict:
    """One split-half correlation for samples of size ``n``.

    Pitchers with fewer than ``2n`` finite observations are skipped.
    Returns Pearson and Spearman correlations and the pitcher count.
    """
    pitcher_ids = np.asarray(pitcher_ids)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    pitcher_ids = pitcher_ids[ok]
    values = values[ok]
    left_means = []
    right_means = []
    # Group indices without a pandas groupby so the draw stays obvious.
    order = np.argsort(pitcher_ids, kind="mergesort")
    pitcher_ids = pitcher_ids[order]
    values = values[order]
    breaks = np.flatnonzero(np.r_[True, pitcher_ids[1:] != pitcher_ids[:-1], True])
    for start, stop in zip(breaks[:-1], breaks[1:]):
        if stop - start < 2 * n:
            continue
        take = rng.choice(stop - start, size=2 * n, replace=False)
        sample = values[start:stop][take]
        left_means.append(sample[:n].mean())
        right_means.append(sample[n:].mean())
    used = len(left_means)
    if used < min_pitchers:
        return {"n": n, "pitchers": used, "pearson": None, "spearman": None}
    pearson, spearman = _safe_corr(np.asarray(left_means), np.asarray(right_means))
    return {
        "n": n,
        "pitchers": used,
        "pearson": None if not np.isfinite(pearson) else pearson,
        "spearman": None if not np.isfinite(spearman) else spearman,
    }


def reliability_curve(
    pitcher_ids: np.ndarray,
    values: np.ndarray,
    sample_sizes: tuple[int, ...] | list[int],
    repeats: int = 20,
    seed: int = 7,
    min_pitchers: int = 40,
) -> list[dict]:
    """Average split-half correlations over several random splits."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in sample_sizes:
        pears = []
        spears = []
        pitchers = 0
        for _ in range(repeats):
            one = split_half_correlation(
                pitcher_ids, values, int(n), rng, min_pitchers=min_pitchers
            )
            pitchers = one["pitchers"]
            if one["pearson"] is not None:
                pears.append(one["pearson"])
                spears.append(one["spearman"])
        rows.append({
            "n": int(n),
            "pitchers": int(pitchers),
            "repeats_used": len(pears),
            "pearson": float(np.mean(pears)) if pears else None,
            "spearman": float(np.mean(spears)) if spears else None,
        })
    return rows


def smallest_stable_n(curve: list[dict], target: float = 0.70) -> int | None:
    """Smallest tested sample size whose Pearson correlation reaches ``target``."""
    for row in sorted(curve, key=lambda r: r["n"]):
        r = row["pearson"]
        if r is not None and r >= target:
            return int(row["n"])
    return None


def year_over_year(left: pd.Series, right: pd.Series, min_pairs: int = 30) -> dict:
    """Pearson and Spearman correlation between two aligned season series."""
    both = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if len(both) < min_pairs or both["left"].std() == 0 or both["right"].std() == 0:
        return {"n": int(len(both)), "pearson": None, "spearman": None}
    pearson, spearman = _safe_corr(both["left"].to_numpy(), both["right"].to_numpy())
    return {
        "n": int(len(both)),
        "pearson": None if not np.isfinite(pearson) else float(pearson),
        "spearman": None if not np.isfinite(spearman) else float(spearman),
    }
