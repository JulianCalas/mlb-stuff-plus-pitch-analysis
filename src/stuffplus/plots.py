"""Figures for the memo, the README, and the notebook.

Every chart states a comparison: a grade against 100, a prediction against
an outcome, or one sample against another. Colors match the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stuffplus.config import (
    CARDINAL,
    CREAM,
    FEATURE_LABELS,
    GOOD,
    INK,
    NAVY,
    PITCH_COLORS,
    PITCH_NAMES,
    WARN,
)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E7E5E4",
        "axes.labelcolor": INK,
        "axes.titlecolor": NAVY,
        "axes.titleweight": "semibold",
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
    })


def _save(fig: plt.Figure, path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_calibration(table: list[dict], path, title: str, xlabel: str, ylabel: str) -> None:
    apply_style()
    if not table:
        return
    pred = [row["predicted"] for row in table]
    actual = [row["actual"] for row in table]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(pred, actual, color=NAVY, marker="o", linewidth=2, label="Deciles")
    low = min(pred + actual)
    high = max(pred + actual)
    pad = (high - low) * 0.08 or 0.01
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="#A8A29E", linestyle="--", label="Perfect")
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_shap_bars(mean_abs: dict[str, float], path, title: str) -> None:
    apply_style()
    items = sorted(mean_abs.items(), key=lambda item: item[1])
    labels = [FEATURE_LABELS.get(key, key) for key, _ in items]
    values = [value for _, value in items]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.barh(labels, values, color=NAVY)
    ax.set_xlabel("Mean |SHAP| on predicted run value")
    ax.set_title(title)
    _save(fig, path)


def plot_reliability_curves(curves: dict[str, list[dict]], path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = {"Stuff+": NAVY, "Whiff rate": CARDINAL, "Actual run value": "#0E8A8A"}
    for name, rows in curves.items():
        xs = [row["n"] for row in rows if row["pearson"] is not None]
        ys = [row["pearson"] for row in rows if row["pearson"] is not None]
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", color=colors.get(name, INK), label=name, linewidth=2)
    ax.axhline(0.70, color=WARN, linestyle="--", linewidth=1, label="0.70 reliability")
    ax.set_xlabel("Pitches in each half (swings, for whiff rate)")
    ax.set_ylabel("Split-half correlation")
    ax.set_ylim(0, 1.02)
    ax.set_title("How fast each measure settles down")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_yoy_bars(correlations: dict[str, float | None], path, subtitle: str) -> None:
    apply_style()
    labels = list(correlations)
    values = [0 if correlations[label] is None else correlations[label] for label in labels]
    colors = [NAVY if correlations[label] is not None else "#D6D3D1" for label in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    positions = np.arange(len(labels))
    ax.bar(positions, values, color=colors)
    ax.set_xticks(positions, labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Year-over-year Pearson correlation")
    ax.set_title("2025 to 2026 reliability")
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8, color="#57534E")
    _save(fig, path)


def plot_stuff_vs_results(pitcher_season: pd.DataFrame, path, highlight_id: int | None, highlight_name: str) -> None:
    apply_style()
    frame = pitcher_season.dropna(subset=["stuff_plus", "runs_prevented_per_100"]).copy()
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.scatter(
        frame["stuff_plus"],
        frame["runs_prevented_per_100"],
        s=18,
        c="#D6D3D1",
        alpha=0.9,
        linewidths=0,
        label="Qualified pitchers",
    )
    marlins = frame.loc[frame["team"] == "MIA"]
    if len(marlins):
        ax.scatter(
            marlins["stuff_plus"],
            marlins["runs_prevented_per_100"],
            s=36,
            c=CARDINAL,
            label="Marlins",
            zorder=3,
        )
    if highlight_id is not None:
        one = frame.loc[frame["pitcher"] == highlight_id]
        if len(one):
            ax.scatter(
                one["stuff_plus"],
                one["runs_prevented_per_100"],
                s=80,
                c=NAVY,
                zorder=4,
                label=highlight_name,
            )
    ax.axvline(100, color="#A8A29E", linestyle="--", linewidth=1)
    ax.axhline(0, color="#A8A29E", linestyle="--", linewidth=1)
    ax.set_xlabel("Arsenal Stuff+  (100 = league average)")
    ax.set_ylabel("Runs prevented per 100 pitches vs league")
    ax.set_title("Stuff against results, 2026")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)


def plot_leaderboard(board: pd.DataFrame, path, pitch_name: str) -> None:
    apply_style()
    frame = board.head(15).iloc[::-1]
    colors = [GOOD if value >= 100 else CARDINAL for value in frame["stuff_plus"]]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.barh(frame["player_name"], frame["stuff_plus"], color=colors)
    ax.axvline(100, color=INK, linewidth=0.8)
    ax.set_xlabel("Stuff+")
    ax.set_title(f"{pitch_name} Stuff+, 2026  ·  minimum 200 pitches")
    _save(fig, path)


def plot_movement(sample: pd.DataFrame, path, title: str) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    for pitch_type, group in sample.groupby("pitch_type", observed=True):
        if group.empty:
            continue
        ax.scatter(
            group["hb_arm"],
            group["ivb"],
            s=16,
            alpha=0.55,
            linewidths=0,
            color=PITCH_COLORS.get(pitch_type, "#78716C"),
            label=PITCH_NAMES.get(pitch_type, pitch_type),
        )
    ax.axhline(0, color="#D6D3D1", linewidth=1)
    ax.axvline(0, color="#D6D3D1", linewidth=1)
    ax.set_xlabel("Arm-side break (inches)  →")
    ax.set_ylabel("Induced vertical break (inches)  →")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="best")
    _save(fig, path)


def plot_release(sample: pd.DataFrame, path, title: str) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for pitch_type, group in sample.groupby("pitch_type", observed=True):
        if group.empty:
            continue
        ax.scatter(
            group["release_pos_x"],
            group["release_pos_z"],
            s=16,
            alpha=0.55,
            linewidths=0,
            color=PITCH_COLORS.get(pitch_type, "#78716C"),
            label=PITCH_NAMES.get(pitch_type, pitch_type),
        )
    ax.set_xlabel("Horizontal release (ft), catcher's view  → first base")
    ax.set_ylabel("Release height (ft)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def plot_pitch_grades(arsenal: pd.DataFrame, path, title: str) -> None:
    apply_style()
    frame = arsenal.sort_values("stuff_plus")
    labels = [PITCH_NAMES.get(pitch, pitch) for pitch in frame["pitch_type"]]
    colors = [PITCH_COLORS.get(pitch, NAVY) for pitch in frame["pitch_type"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.barh(labels, frame["stuff_plus"], color=colors)
    ax.axvline(100, color=INK, linewidth=0.8)
    ax.set_xlabel("Stuff+")
    ax.set_title(title)
    _save(fig, path)


def plot_cluster_centers(profiles: pd.DataFrame, path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    colors = [PITCH_COLORS.get(pitch, "#78716C") for pitch in profiles["majority_pitch"]]
    ax.scatter(profiles["hb_arm"], profiles["ivb"], s=profiles["pitches"] / 80, c=colors, alpha=0.9)
    for row in profiles.itertuples(index=False):
        ax.annotate(
            f"{row.cluster}:{row.majority_pitch}",
            (row.hb_arm, row.ivb),
            fontsize=7,
            color=INK,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.axhline(0, color="#D6D3D1", linewidth=1)
    ax.axvline(0, color="#D6D3D1", linewidth=1)
    ax.set_xlabel("Arm-side break (inches)")
    ax.set_ylabel("Induced vertical break (inches)")
    ax.set_title("Shape clusters, sized by pitches")
    _save(fig, path)


def plot_vaa(summary: pd.DataFrame, path) -> None:
    """summary columns: pitch_type, vaa_mean, vaa_p25, vaa_p75."""
    apply_style()
    frame = summary.sort_values("vaa_mean")
    labels = [PITCH_NAMES.get(pitch, pitch) for pitch in frame["pitch_type"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    y = np.arange(len(frame))
    ax.hlines(y, frame["vaa_p25"], frame["vaa_p75"], color="#D6D3D1", linewidth=6)
    ax.scatter(frame["vaa_mean"], y, color=NAVY, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Vertical approach angle (degrees, more negative = steeper)")
    ax.set_title("Approach angle by pitch type")
    _save(fig, path)


def plot_whiff_calibration(table: list[dict], path) -> None:
    plot_calibration(
        table,
        path,
        title="Whiff model calibration, 2026 swings",
        xlabel="Predicted whiff probability",
        ylabel="Actual whiff rate",
    )
