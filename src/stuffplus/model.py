"""Baselines, gradient boosting, and the metrics we are willing to quote.

The target is count-neutral run value (see outcomes.py). Three predictors
are fit on the same rows:

1. League mean. This is the "every pitch is average" floor.
2. Pitch-type mean. This uses the official pitch label, which the stuff
   model is not allowed to see. Beating it overall means shape knows
   something the label does not. The fairer test is within a pitch type,
   where the label is a constant and this baseline has nothing to say.
3. Ridge regression on the physical features. A linear reading of velocity,
   break, release, and fastball gaps.
4. LightGBM on the same features. Trees can learn that velocity gap matters
   more on a sweeper than on a four-seamer.

Fifteen percent of the training rows are used only to decide how many
trees to keep. The test season is not in that decision. We do not refit
on those rows afterward, so the saved model is the one the metrics describe.

Pitch-level R-squared on run value is supposed to be small. One pitch
outcome is mostly the hitter, the count, and luck. The grade earns its
keep when pitches are averaged up to a pitcher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stuffplus.config import (
    EARLY_STOP_FRACTION,
    EARLY_STOPPING_ROUNDS,
    FEATURES,
    LGBM_PARAMS,
    RANDOM_SEED,
    RIDGE_ALPHA,
)


def feature_matrix(df: pd.DataFrame, features: list[str] | tuple[str, ...] = FEATURES) -> np.ndarray:
    """Float matrix. Missing tracking values stay NaN for the trees."""
    frame = df.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    return frame.to_numpy(dtype=np.float64)


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, R-squared, and Pearson correlation on finite pairs."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return {"n": 0, "rmse": None, "mae": None, "r2": None, "corr": None}
    residual = y - p
    corr = None
    if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0:
        corr = float(np.corrcoef(y, p)[0, 1])
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(r2_score(y, p)),
        "corr": corr,
    }


def metrics_by_pitch_type(df: pd.DataFrame, pred_col: str, actual_col: str) -> dict:
    """Within-type fit. This is the test of shape, holding the label fixed."""
    out = {}
    for pitch_type, group in df.groupby("pitch_type", observed=False):
        out[str(pitch_type)] = regression_metrics(group[actual_col], group[pred_col])
    return out


def aggregated_correlation(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    min_pitches: int = 100,
) -> dict:
    """Correlation after averaging to pitcher and pitch type.

    Noise that cancels across pitches should shrink here. If this
    correlation is much larger than the pitch-level correlation, the
    model is picking up a real pitcher trait and not just luck.
    """
    grouped = (
        df.groupby(["pitcher", "pitch_type"], observed=False)
        .agg(pred=(pred_col, "mean"), actual=(actual_col, "mean"), n=(actual_col, "size"))
        .reset_index()
    )
    qualified = grouped.loc[grouped["n"] >= min_pitches]
    metrics = regression_metrics(qualified["actual"], qualified["pred"])
    metrics["groups"] = int(len(qualified))
    return metrics


def calibration_table(y_true, y_pred, bins: int = 10) -> list[dict]:
    """Mean predicted value against mean actual value by prediction decile."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    frame = pd.DataFrame({"y": y[mask], "p": p[mask]})
    if frame.empty:
        return []
    try:
        frame["bin"] = pd.qcut(frame["p"], bins, duplicates="drop")
    except ValueError:
        return []
    rows = []
    for interval, group in frame.groupby("bin", observed=True):
        rows.append({
            "bin": str(interval),
            "n": int(len(group)),
            "predicted": float(group["p"].mean()),
            "actual": float(group["y"].mean()),
        })
    return rows


def expected_calibration_error(y_true, y_prob, bins: int = 10) -> float | None:
    """Weighted absolute gap between predicted probability and event rate."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    total = len(y)
    for i in range(bins):
        if i == bins - 1:
            inside = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            inside = (p >= edges[i]) & (p < edges[i + 1])
        if inside.sum() == 0:
            continue
        error += (inside.sum() / total) * abs(y[inside].mean() - p[inside].mean())
    return float(error)


def classification_metrics(y_true, y_prob) -> dict:
    """AUC, Brier score, log loss, and calibration error for a whiff model."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"n": int(len(y)), "auc": None, "brier": None, "log_loss": None, "ece": None}
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "base_rate": float(y.mean()),
    }


def _fit_indices(n_rows: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.arange(n_rows)
    rng.shuffle(order)
    cut = int(round(n_rows * (1.0 - fraction)))
    cut = min(max(cut, 1), n_rows - 1)
    return order[:cut], order[cut:]


def _lgbm_params(task: str) -> dict:
    params = dict(LGBM_PARAMS)
    if task == "regression":
        params.update(objective="regression", metric="rmse")
    else:
        # No class reweighting. We want probabilities, not a boosted whiff rate.
        params.update(objective="binary", metric="binary_logloss")
    return params


def _fit_booster(X, y, task: str, seed: int):
    """Fit LightGBM and stop when the inner validation score stops improving."""
    fit_idx, valid_idx = _fit_indices(len(y), EARLY_STOP_FRACTION, seed)
    params = _lgbm_params(task)
    model_cls = LGBMRegressor if task == "regression" else LGBMClassifier
    model = model_cls(**params)
    callbacks = [early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)]
    try:
        model.fit(
            X[fit_idx],
            y[fit_idx],
            eval_X=X[valid_idx],
            eval_y=y[valid_idx],
            callbacks=callbacks,
        )
    except TypeError:
        # Older LightGBM releases take eval_set instead of eval_X / eval_y.
        model.fit(
            X[fit_idx],
            y[fit_idx],
            eval_set=[(X[valid_idx], y[valid_idx])],
            callbacks=callbacks,
        )
    except Exception:
        # Deterministic mode can refuse some sampling settings. Refit once
        # without it rather than failing the whole pipeline.
        params["deterministic"] = False
        model = model_cls(**params)
        try:
            model.fit(
                X[fit_idx],
                y[fit_idx],
                eval_X=X[valid_idx],
                eval_y=y[valid_idx],
                callbacks=callbacks,
            )
        except TypeError:
            model.fit(
                X[fit_idx],
                y[fit_idx],
                eval_set=[(X[valid_idx], y[valid_idx])],
                callbacks=callbacks,
            )
    best_iteration = int(getattr(model, "best_iteration_", None) or params["n_estimators"])
    return model, best_iteration, int(len(fit_idx)), int(len(valid_idx))


def fit_ridge(X, y) -> Pipeline:
    """Linear baseline. Median fill is only for the linear model."""
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=RIDGE_ALPHA)),
    ])
    pipeline.fit(X, y)
    return pipeline


def pitch_type_means(pitch_types: pd.Series, y: np.ndarray) -> tuple[pd.Series, float]:
    """Training-set mean run value by the official pitch label, plus the grand mean."""
    frame = pd.DataFrame({"pitch_type": pitch_types.to_numpy(), "y": y})
    means = frame.groupby("pitch_type", observed=False)["y"].mean()
    return means, float(frame["y"].mean())


def predict_pitch_type_mean(pitch_types: pd.Series, means: pd.Series, grand: float) -> np.ndarray:
    predicted = pitch_types.map(means).astype(float)
    return predicted.fillna(grand).to_numpy(dtype=float)


class RunValueModels:
    """The ship-season comparison: label baseline, ridge, and LightGBM."""

    def __init__(self, features: list[str] | tuple[str, ...] = FEATURES):
        self.features = list(features)
        self.booster = None
        self.ridge = None
        self.type_means = None
        self.grand_mean = None
        self.best_iteration = None
        self.fit_rows = None
        self.valid_rows = None

    def fit(self, train: pd.DataFrame, target_col: str) -> "RunValueModels":
        usable = train.loc[train[target_col].notna()].copy()
        X = feature_matrix(usable, self.features)
        y = usable[target_col].to_numpy(dtype=float)
        self.type_means, self.grand_mean = pitch_type_means(usable["pitch_type"], y)
        # Ridge sees every training row. The tree model holds out 15 percent
        # inside this function, only to pick the number of trees.
        self.ridge = fit_ridge(X, y)
        self.booster, self.best_iteration, self.fit_rows, self.valid_rows = _fit_booster(
            X, y, task="regression", seed=RANDOM_SEED
        )
        return self

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = feature_matrix(df, self.features)
        return {
            "lightgbm": self.booster.predict(X),
            "ridge": self.ridge.predict(X),
            "pitch_type_mean": predict_pitch_type_mean(df["pitch_type"], self.type_means, self.grand_mean),
            "league_mean": np.full(len(df), self.grand_mean, dtype=float),
        }


class WhiffModel:
    """Probability of a whiff given a swing, from the same physical features.

    This is the calibration check. Stuff+ itself comes from run value,
    because a pitch can be effective by inducing weak contact too.
    """

    def __init__(self, features: list[str] | tuple[str, ...] = FEATURES):
        self.features = list(features)
        self.booster = None
        self.best_iteration = None

    def fit(self, swings: pd.DataFrame) -> "WhiffModel":
        usable = swings.loc[swings["is_whiff"].notna()].copy()
        X = feature_matrix(usable, self.features)
        y = usable["is_whiff"].astype(int).to_numpy()
        self.booster, self.best_iteration, _, _ = _fit_booster(
            X, y, task="classification", seed=RANDOM_SEED + 1
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = feature_matrix(df, self.features)
        return self.booster.predict_proba(X)[:, 1]
