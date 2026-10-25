"""Group similar pitch shapes, across pitchers and across official labels.

A slider and a sweeper are labels. The shape is velocity, induced vertical
break, arm-side break, and spin. A Gaussian mixture model puts a soft
boundary on that space. We choose the number of groups by BIC on a
stratified sample, then assign every pitch.

The clusters are not a new pitch-classification product. They answer a
player-development question: which pitchers with a fastball like this one
also throw something this pitcher does not, and what does that something
look like?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from stuffplus.config import RANDOM_SEED

CLUSTER_FEATURES = ["release_speed", "ivb", "hb_arm", "release_spin_rate"]


def _complete(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df[CLUSTER_FEATURES].notna().all(axis=1)].copy()


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Up to ``n`` pitches, spread across official pitch types."""
    usable = _complete(df)
    types = usable["pitch_type"].nunique()
    per_type = max(int(np.ceil(n / max(types, 1))), 1)
    parts = []
    for _, group in usable.groupby("pitch_type", observed=False):
        take = min(len(group), per_type)
        parts.append(group.sample(n=take, random_state=seed))
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)
    return sample


def fit_mixture(
    df: pd.DataFrame,
    k_min: int = 8,
    k_max: int = 12,
    sample_size: int = 60000,
    seed: int = RANDOM_SEED,
) -> dict:
    """Fit a diagonal Gaussian mixture and keep the k with the best BIC.

    Diagonal covariance is the independent-features version of the mixture.
    It is fast enough to refit while we compare k, and the clusters are
    still easy to describe as a center in velocity and movement.
    """
    sample = stratified_sample(df, sample_size, seed)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(sample[CLUSTER_FEATURES].to_numpy(dtype=float))
    records = []
    best = None
    for k in range(k_min, k_max + 1):
        mixture = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=seed,
            n_init=2,
            max_iter=200,
        )
        mixture.fit(matrix)
        bic = float(mixture.bic(matrix))
        records.append({"k": k, "bic": bic})
        if best is None or bic < best["bic"]:
            best = {"k": k, "bic": bic, "model": mixture}
    labels = best["model"].predict(matrix)
    ari = float(adjusted_rand_score(sample["pitch_type"].astype(str), labels))
    return {
        "scaler": scaler,
        "model": best["model"],
        "k": int(best["k"]),
        "bic": best["bic"],
        "bic_curve": records,
        "adjusted_rand_vs_pitch_type": ari,
        "sample_size": int(len(sample)),
        "features": list(CLUSTER_FEATURES),
    }


def assign_clusters(df: pd.DataFrame, fitted: dict, chunk: int = 250000) -> np.ndarray:
    """Cluster id for every row. Rows with missing shape get -1."""
    labels = np.full(len(df), -1, dtype=int)
    usable = df[CLUSTER_FEATURES].notna().all(axis=1).to_numpy()
    index = np.flatnonzero(usable)
    matrix = fitted["scaler"].transform(df.loc[usable, CLUSTER_FEATURES].to_numpy(dtype=float))
    predicted = np.empty(len(matrix), dtype=int)
    for start in range(0, len(matrix), chunk):
        stop = start + chunk
        predicted[start:stop] = fitted["model"].predict(matrix[start:stop])
    labels[index] = predicted
    return labels


def cluster_profiles(df: pd.DataFrame, stuff_col: str = "stuff_plus") -> pd.DataFrame:
    """What each cluster actually is: shape center and the label it mostly contains."""
    usable = df.loc[df["cluster"] >= 0].copy()
    rows = []
    for cluster_id, group in usable.groupby("cluster"):
        mix = group["pitch_type"].value_counts(normalize=True)
        row = {
            "cluster": int(cluster_id),
            "pitches": int(len(group)),
            "majority_pitch": str(mix.index[0]),
            "majority_share": float(mix.iloc[0]),
            "velo": float(group["release_speed"].mean()),
            "ivb": float(group["ivb"].mean()),
            "hb_arm": float(group["hb_arm"].mean()),
            "spin": float(group["release_spin_rate"].mean()),
        }
        if stuff_col in group.columns and group[stuff_col].notna().any():
            row["stuff_plus"] = float(group[stuff_col].mean())
        rows.append(row)
    profiles = pd.DataFrame(rows).sort_values(["majority_pitch", "velo"], ascending=[True, False])
    return profiles.reset_index(drop=True)
