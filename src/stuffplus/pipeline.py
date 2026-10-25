"""End-to-end Stuff+ fit.

Train on earlier seasons, grade a later season, then measure how stable
the grade is and which pitchers it would actually change a plan for.
Run ``python scripts/run_pipeline.py`` rather than importing this by hand.
"""

from __future__ import annotations

import datetime
import gc
import json
import traceback
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from stuffplus.arsenal import add_fastball_gaps
from stuffplus.cluster import assign_clusters, cluster_profiles, fit_mixture
from stuffplus.config import (
    ARSENAL_MIN_PITCHES,
    CORE_PITCH_TYPES,
    FEATURES,
    FIGURE_DIR,
    FOCUS_TEAM,
    LEADERBOARD_MIN_PITCHES,
    MODEL_DIR,
    PITCH_NAMES,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    RELIABILITY_TARGET,
    SEASONS,
    SHIP_TEST_YEAR,
    SHIP_TRAIN_YEARS,
    SPLIT_HALF_MIN_PITCHERS,
    SPLIT_HALF_REPEATS,
    SPLIT_HALF_SIZES,
    SQL_DIR,
    STUFF_MIN_PITCHES,
    STUFF_SD_POINTS,
    YOY_SCORE_YEARS,
    YOY_TRAIN_YEARS,
)
from stuffplus.model import (
    WhiffModel,
    aggregated_correlation,
    calibration_table,
    classification_metrics,
    metrics_by_pitch_type,
    regression_metrics,
)
from stuffplus.model import RunValueModels
from stuffplus.outcomes import (
    assign_count_neutral_run_value,
    assign_outcome,
    is_swing,
    is_whiff,
    outcome_run_values,
)
from stuffplus.physics import add_physics_features, pitching_team, pretty_name
from stuffplus.plots import (
    plot_calibration,
    plot_cluster_centers,
    plot_leaderboard,
    plot_movement,
    plot_pitch_grades,
    plot_release,
    plot_reliability_curves,
    plot_shap_bars,
    plot_stuff_vs_results,
    plot_vaa,
    plot_whiff_calibration,
    plot_yoy_bars,
)
from stuffplus.recommend import suggest_all
from stuffplus.reliability import (
    reliability_curve,
    sample_size_for_reliability,
    single_observation_reliability,
    smallest_stable_n,
    year_over_year,
)
from stuffplus.season_stats import fetch_season_pitching

RAW_GLOB = str(RAW_DIR / "pitches_*.parquet")


def _log(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime.date, datetime.datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2))


def load_named_sql() -> dict[str, str]:
    blocks: dict[str, list[str]] = {}
    name = None
    for line in (SQL_DIR / "analysis.sql").read_text().splitlines():
        if line.startswith("-- name:"):
            name = line.split(":", 1)[1].strip()
            blocks[name] = []
        elif name is not None:
            blocks[name].append(line)
    return {key: "\n".join(lines) for key, lines in blocks.items()}


def run_sql(query: str, **kwargs) -> pd.DataFrame:
    statement = query.format(**kwargs)
    return duckdb.connect().execute(statement).fetchdf()


def load_modeling_frame() -> pd.DataFrame:
    """Read the cached seasons through DuckDB and keep model columns only."""
    types = ", ".join(f"'{pitch}'" for pitch in CORE_PITCH_TYPES)
    frame = duckdb.connect().execute(
        f"""
        SELECT
            pitch_type,
            game_year::INTEGER AS game_year,
            pitcher::INTEGER AS pitcher,
            player_name,
            p_throws,
            description,
            COALESCE(events, '') AS events,
            release_speed::FLOAT AS release_speed,
            release_spin_rate::FLOAT AS release_spin_rate,
            spin_axis::FLOAT AS spin_axis,
            release_pos_x::FLOAT AS release_pos_x,
            release_pos_y::FLOAT AS release_pos_y,
            release_pos_z::FLOAT AS release_pos_z,
            release_extension::FLOAT AS release_extension,
            pfx_x::FLOAT AS pfx_x,
            pfx_z::FLOAT AS pfx_z,
            plate_x::FLOAT AS plate_x,
            plate_z::FLOAT AS plate_z,
            vx0::FLOAT AS vx0,
            vy0::FLOAT AS vy0,
            vz0::FLOAT AS vz0,
            ax::FLOAT AS ax,
            ay::FLOAT AS ay,
            az::FLOAT AS az,
            zone::FLOAT AS zone,
            delta_run_exp::FLOAT AS delta_run_exp,
            arm_angle::FLOAT AS arm_angle,
            api_break_x_arm::FLOAT AS api_break_x_arm,
            home_team,
            away_team,
            inning_topbot
        FROM read_parquet('{RAW_GLOB}')
        WHERE pitch_type IN ({types})
          AND p_throws IN ('L', 'R')
          AND release_speed BETWEEN 64 AND 104
          AND pfx_x IS NOT NULL AND pfx_z IS NOT NULL
          AND abs(pfx_x) < 2.6 AND abs(pfx_z) < 2.6
          AND vx0 IS NOT NULL AND vy0 IS NOT NULL AND vz0 IS NOT NULL
          AND ax IS NOT NULL AND ay IS NOT NULL AND az IS NOT NULL
          AND release_pos_y IS NOT NULL
          AND delta_run_exp IS NOT NULL
          AND description IS NOT NULL
        """
    ).fetchdf()
    frame = frame.reset_index(drop=True)
    return frame


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Physics, fastball gaps, outcome labels. Drops pitches with no break."""
    _log(f"engineering features on {len(frame):,} pitches")
    frame["outcome"] = assign_outcome(frame["description"], frame["events"])
    frame["is_swing"] = is_swing(frame["description"]).to_numpy()
    frame["is_whiff"] = is_whiff(frame["description"]).to_numpy()
    frame["team"] = pitching_team(
        frame["inning_topbot"].to_numpy(),
        frame["home_team"].to_numpy(),
        frame["away_team"].to_numpy(),
    )
    frame["player_name"] = frame["player_name"].map(pretty_name)
    frame = add_physics_features(frame)
    frame = add_fastball_gaps(frame)
    frame["in_zone"] = frame["zone"].between(1, 9).fillna(False).to_numpy()
    keep = frame["ivb"].notna() & frame["hb_arm"].notna() & frame["release_speed"].notna()
    frame = frame.loc[keep].reset_index(drop=True)
    drop = [
        "events", "description", "pfx_x", "pfx_z", "spin_axis", "vx0", "vy0", "vz0",
        "ax", "ay", "az", "home_team", "away_team", "inning_topbot",
    ]
    return frame.drop(columns=[column for column in drop if column in frame.columns])


def _assert_run_value_signs(means: pd.Series) -> None:
    """A home run must be worth more to the offense than a whiff."""
    if means["home_run"] <= 0 or means["whiff"] >= 0 or means["home_run"] <= means["whiff"]:
        raise RuntimeError(f"Run-value signs look wrong: {means.to_dict()}")


def _evaluate(frame: pd.DataFrame, target_col: str, predictions: dict[str, np.ndarray]) -> dict:
    actual = frame[target_col].to_numpy()
    scored = frame.copy()
    report = {}
    for name, predicted in predictions.items():
        predicted = np.asarray(predicted)
        if len(predicted) != len(frame):
            raise RuntimeError(
                f"{name} returned {len(predicted)} predictions for {len(frame)} pitches"
            )
        scored[name] = predicted
        report[name] = regression_metrics(actual, predicted)
        report[name]["by_pitch_type"] = metrics_by_pitch_type(scored, name, target_col)
        report[name]["aggregated_min_100"] = aggregated_correlation(scored, name, target_col, min_pitches=100)
    report["lightgbm_vs_raw_run_exp"] = regression_metrics(frame["delta_run_exp"], predictions["lightgbm"])
    report["calibration"] = calibration_table(actual, predictions["lightgbm"])
    return report


def _reference_table(arsenal: pd.DataFrame) -> pd.DataFrame:
    qualified = arsenal.loc[arsenal["pitches"] >= STUFF_MIN_PITCHES]
    reference = (
        qualified.groupby(["game_year", "pitch_type"])["pred_rv"]
        .agg(lg_mean="mean", lg_sd="std", lg_n="count")
        .reset_index()
    )
    thin = reference["lg_n"] < 8
    reference.loc[thin, ["lg_mean", "lg_sd"]] = np.nan
    return reference


def _apply_stuff_scale(arsenal: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    graded = arsenal.merge(reference, on=["game_year", "pitch_type"], how="left")
    graded["stuff_plus"] = 100.0 + STUFF_SD_POINTS * (
        graded["lg_mean"] - graded["pred_rv"]
    ) / graded["lg_sd"]
    return graded


def _mode_fastball_cluster(frame: pd.DataFrame) -> pd.DataFrame:
    fastballs = frame.loc[frame["is_primary_fb"].fillna(False) & frame["cluster"].ge(0)]
    counts = (
        fastballs.groupby(["pitcher", "game_year", "cluster"], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )
    if counts.empty:
        return pd.DataFrame(columns=["pitcher", "game_year", "fb_cluster"])
    counts = counts.sort_values(["pitcher", "game_year", "n"], ascending=[True, True, False])
    picked = counts.drop_duplicates(["pitcher", "game_year"])
    return picked.rename(columns={"cluster": "fb_cluster"})[["pitcher", "game_year", "fb_cluster"]]


def build_arsenal(frame: pd.DataFrame, pred_col: str, rv_col: str, years: tuple[int, ...]) -> pd.DataFrame:
    """One row per pitcher, season, and pitch type."""
    sub = frame.loc[
        frame["game_year"].isin(years) & frame[pred_col].notna() & frame[rv_col].notna()
    ].copy()
    sub["whiff_flag"] = sub["is_whiff"].astype(float)
    sub["swing_flag"] = sub["is_swing"].astype(float)
    arsenal = (
        sub.groupby(["pitcher", "game_year", "pitch_type"], observed=False)
        .agg(
            pitches=(pred_col, "size"),
            velo=("release_speed", "mean"),
            ivb=("ivb", "mean"),
            hb_arm=("hb_arm", "mean"),
            vaa=("vaa", "mean"),
            haa_arm=("haa_arm", "mean"),
            spin=("release_spin_rate", "mean"),
            extension=("release_extension", "mean"),
            arm_angle=("arm_angle", "mean"),
            release_pos_x=("release_pos_x", "mean"),
            release_pos_z=("release_pos_z", "mean"),
            pred_rv=(pred_col, "mean"),
            actual_rv=(rv_col, "mean"),
            actual_raw_rv=("delta_run_exp", "mean"),
            whiffs=("whiff_flag", "sum"),
            swings=("swing_flag", "sum"),
            zone_rate=("in_zone", "mean"),
            plate_x_var=("plate_x", "var"),
            plate_z_var=("plate_z", "var"),
            p_throws=("p_throws", "first"),
            primary_fb=("primary_fb", "first"),
            player_name=("player_name", "first"),
            team=("team", "first"),
        )
        .reset_index()
    )
    # Majority team, in case the first row was the road half of a trade.
    team_counts = (
        sub.groupby(["pitcher", "game_year", "team"], observed=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["pitcher", "game_year", "n"], ascending=[True, True, False])
        .drop_duplicates(["pitcher", "game_year"])
        .drop(columns="n")
    )
    arsenal = arsenal.drop(columns="team").merge(team_counts, on=["pitcher", "game_year"], how="left")
    arsenal["whiff_rate"] = np.where(arsenal["swings"] > 0, arsenal["whiffs"] / arsenal["swings"], np.nan)
    arsenal["loc_sd"] = np.sqrt(arsenal["plate_x_var"].clip(lower=0) + arsenal["plate_z_var"].clip(lower=0))
    totals = arsenal.groupby(["pitcher", "game_year"])["pitches"].transform("sum")
    arsenal["usage"] = arsenal["pitches"] / totals
    reference = _reference_table(arsenal)
    arsenal = _apply_stuff_scale(arsenal, reference)
    return arsenal, reference


def _pitcher_season(arsenal: pd.DataFrame) -> pd.DataFrame:
    usable = arsenal.dropna(subset=["stuff_plus", "pred_rv", "actual_rv"]).copy()
    usable["w_stuff"] = usable["stuff_plus"] * usable["pitches"]
    usable["w_pred"] = usable["pred_rv"] * usable["pitches"]
    usable["w_actual"] = usable["actual_rv"] * usable["pitches"]
    rolled = usable.groupby(["pitcher", "season"], as_index=False).agg(
        player_name=("player_name", "first"),
        team=("team", "first"),
        p_throws=("p_throws", "first"),
        pitches=("pitches", "sum"),
        whiffs=("whiffs", "sum"),
        swings=("swings", "sum"),
        w_stuff=("w_stuff", "sum"),
        w_pred=("w_pred", "sum"),
        w_actual=("w_actual", "sum"),
        pitch_types=("pitch_type", "nunique"),
    )
    rolled["stuff_plus"] = rolled["w_stuff"] / rolled["pitches"]
    rolled["pred_rv"] = rolled["w_pred"] / rolled["pitches"]
    rolled["actual_rv"] = rolled["w_actual"] / rolled["pitches"]
    rolled["whiff_rate"] = np.where(rolled["swings"] > 0, rolled["whiffs"] / rolled["swings"], np.nan)
    if "stuff_plus_frozen" in usable.columns:
        frozen = usable.dropna(subset=["stuff_plus_frozen"]).copy()
        frozen["w"] = frozen["stuff_plus_frozen"] * frozen["pitches"]
        frozen_roll = frozen.groupby(["pitcher", "season"], as_index=False).agg(
            w=("w", "sum"), n=("pitches", "sum")
        )
        frozen_roll["stuff_plus_frozen"] = frozen_roll["w"] / frozen_roll["n"]
        rolled = rolled.merge(
            frozen_roll[["pitcher", "season", "stuff_plus_frozen"]],
            on=["pitcher", "season"],
            how="left",
        )
    return rolled


def _add_results_axis(rolled: pd.DataFrame) -> pd.DataFrame:
    """Runs prevented per 100 pitches versus the pitch-weighted league mean."""
    out = rolled.copy()
    pieces = []
    for _, group in out.groupby("season"):
        block = group.copy()
        league = np.average(block["actual_rv"], weights=block["pitches"])
        block["runs_prevented_per_100"] = (league - block["actual_rv"]) * 100.0
        block["league_actual_rv"] = league
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)


def _weighted_metric(arsenal: pd.DataFrame, column: str) -> pd.Series:
    usable = arsenal.dropna(subset=[column]).copy()
    usable["w"] = usable[column] * usable["pitches"]
    grouped = usable.groupby(["pitcher", "season"]).agg(w=("w", "sum"), n=("pitches", "sum"))
    grade = grouped["w"] / grouped["n"]
    grade.name = column
    return grade


def choose_focus_pitcher(rolled: pd.DataFrame, arsenal: pd.DataFrame) -> pd.DataFrame:
    """Rank Marlins pitchers by how much a development memo would have to say."""
    candidates = rolled.loc[(rolled["team"] == FOCUS_TEAM) & (rolled["season"] == SHIP_TEST_YEAR)].copy()
    depth = (
        arsenal.loc[(arsenal["season"] == SHIP_TEST_YEAR) & (arsenal["pitches"] >= 50)]
        .groupby("pitcher")["pitch_type"]
        .nunique()
    )
    drags = set(
        arsenal.loc[
            (arsenal["season"] == SHIP_TEST_YEAR)
            & (arsenal["stuff_plus"] <= 92)
            & (arsenal["usage"] >= 0.12)
            & (arsenal["pitches"] >= 80),
            "pitcher",
        ]
    )
    candidates["pitch_types_50"] = candidates["pitcher"].map(depth).fillna(0)
    candidates["has_drag_pitch"] = candidates["pitcher"].isin(drags)
    candidates["gap"] = (candidates["actual_rv"] - candidates["pred_rv"]) * 100.0
    candidates["score"] = (
        (candidates["stuff_plus"] - 100).abs()
        + candidates["gap"].abs() * 8.0
        + candidates["has_drag_pitch"].astype(float) * 12.0
        + (candidates["pitch_types_50"] >= 3).astype(float) * 5.0
    )
    qualified = candidates.loc[(candidates["pitches"] >= 700) & (candidates["pitch_types_50"] >= 3)]
    pool = qualified if len(qualified) else candidates.loc[candidates["pitches"] >= 400]
    return pool.sort_values("score", ascending=False)


def _shap_report(model: RunValueModels, frame: pd.DataFrame, sample_size: int = 4000) -> dict:
    import shap

    usable = frame.loc[frame["game_year"] == SHIP_TEST_YEAR, FEATURES + ["pitch_type"]].dropna(subset=["release_speed", "ivb", "hb_arm"])
    if len(usable) > sample_size:
        usable = usable.sample(n=sample_size, random_state=RANDOM_SEED)
    matrix = usable.loc[:, FEATURES]
    explainer = shap.TreeExplainer(model.booster)
    values = explainer.shap_values(matrix)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    mean_abs = {
        feature: float(np.mean(np.abs(values[:, index])))
        for index, feature in enumerate(FEATURES)
    }
    by_type = {}
    for pitch_type, index in usable.reset_index(drop=True).groupby("pitch_type").groups.items():
        rows = values[list(index)]
        by_type[str(pitch_type)] = {
            feature: float(np.mean(np.abs(rows[:, feature_index])))
            for feature_index, feature in enumerate(FEATURES)
        }
    return {"mean_abs_shap": mean_abs, "mean_abs_shap_by_pitch_type": by_type, "sample_size": int(len(usable))}


def _reliability_block(frame_2026: pd.DataFrame) -> dict:
    stuff = reliability_curve(
        frame_2026["pitcher"].to_numpy(),
        frame_2026["stuff_plus"].to_numpy(),
        SPLIT_HALF_SIZES,
        repeats=SPLIT_HALF_REPEATS,
        seed=RANDOM_SEED,
        min_pitchers=SPLIT_HALF_MIN_PITCHERS,
    )
    swings = frame_2026.loc[frame_2026["is_swing"]]
    whiff = reliability_curve(
        swings["pitcher"].to_numpy(),
        swings["is_whiff"].astype(float).to_numpy(),
        SPLIT_HALF_SIZES,
        repeats=SPLIT_HALF_REPEATS,
        seed=RANDOM_SEED + 3,
        min_pitchers=SPLIT_HALF_MIN_PITCHERS,
    )
    actual = reliability_curve(
        frame_2026["pitcher"].to_numpy(),
        frame_2026["delta_run_exp"].to_numpy(),
        SPLIT_HALF_SIZES,
        repeats=SPLIT_HALF_REPEATS,
        seed=RANDOM_SEED + 5,
        min_pitchers=SPLIT_HALF_MIN_PITCHERS,
    )
    curves = {"Stuff+": stuff, "Whiff rate": whiff, "Actual run value": actual}
    prophecy = {}
    for name, curve in curves.items():
        anchor = next((row for row in curve if row["n"] == 100 and row["pearson"] is not None), None)
        if anchor is None:
            prophecy[name] = None
            continue
        r1 = single_observation_reliability(anchor["pearson"], 100)
        prophecy[name] = {
            "r_at_100": anchor["pearson"],
            "single_observation_reliability": r1,
            "n_for_0_70": sample_size_for_reliability(r1, RELIABILITY_TARGET),
            "empirical_n_for_0_70": smallest_stable_n(curve, RELIABILITY_TARGET),
        }
    return {"curves": curves, "spearman_brown": prophecy}


def _yoy_block(arsenal: pd.DataFrame, stats: pd.DataFrame) -> dict:
    """Frozen-model Stuff+ against whiff rate, strikeout rate, FIP, and ERA."""
    frozen = arsenal.dropna(subset=["stuff_plus_frozen"]).copy()
    grades = _weighted_metric(frozen, "stuff_plus_frozen").rename("stuff").reset_index()
    whiff = (
        arsenal.groupby(["pitcher", "season"], as_index=False)
        .agg(whiffs=("whiffs", "sum"), swings=("swings", "sum"), pitches=("pitches", "sum"))
    )
    whiff["whiff_rate"] = np.where(whiff["swings"] > 0, whiff["whiffs"] / whiff["swings"], np.nan)

    def _pair(frame: pd.DataFrame, column: str, min_2025, min_2026, count_column: str) -> dict:
        left = frame.loc[(frame["season"] == 2025) & (frame[count_column] >= min_2025), ["pitcher", column]]
        right = frame.loc[(frame["season"] == 2026) & (frame[count_column] >= min_2026), ["pitcher", column]]
        merged = left.merge(right, on="pitcher", suffixes=("_2025", "_2026"))
        return year_over_year(merged[f"{column}_2025"], merged[f"{column}_2026"])

    stuff_pairs = grades.copy()
    stuff_counts = arsenal.groupby(["pitcher", "season"])["pitches"].sum().rename("pitches").reset_index()
    stuff_pairs = stuff_pairs.merge(stuff_counts, on=["pitcher", "season"])
    report = {
        "stuff_plus_min_200": _pair(stuff_pairs, "stuff", 200, 200, "pitches"),
        "whiff_rate_min_150_swings": _pair(whiff, "whiff_rate", 150, 150, "swings"),
    }
    if not stats.empty:
        for minimum in (40, 80):
            era = _pair(stats, "era", minimum, minimum, "ip")
            fip = _pair(stats, "fip", minimum, minimum, "ip")
            strikeouts = _pair(stats, "k_pct", minimum, minimum, "ip")
            report[f"era_min_{minimum}_ip"] = era
            report[f"fip_min_{minimum}_ip"] = fip
            report[f"k_pct_min_{minimum}_ip"] = strikeouts
    return report


def _plot_sample(frame: pd.DataFrame, per_pitch: int = 80) -> pd.DataFrame:
    """A draw of individual pitches so movement plots do not need the full table."""
    columns = [
        "pitcher", "game_year", "player_name", "team", "p_throws", "pitch_type",
        "release_speed", "ivb", "hb_arm", "vaa", "haa_arm", "release_pos_x", "release_pos_z",
        "release_extension", "arm_angle", "stuff_plus", "plate_x", "plate_z", "cluster",
        "pred_ship", "pred_whiff",
    ]
    present = [column for column in columns if column in frame.columns]
    sample_frame = frame.loc[frame["game_year"].isin(YOY_SCORE_YEARS), present].copy()
    sample_frame = sample_frame.loc[sample_frame["stuff_plus"].notna()]
    rng = np.random.default_rng(RANDOM_SEED)
    sample_frame["_draw"] = rng.random(len(sample_frame))
    sample_frame = sample_frame.sort_values(["pitcher", "game_year", "pitch_type", "_draw"])
    sample_frame["_rn"] = sample_frame.groupby(["pitcher", "game_year", "pitch_type"], observed=False).cumcount()
    kept = sample_frame.loc[sample_frame["_rn"] < per_pitch].drop(columns=["_draw", "_rn"])
    return kept.reset_index(drop=True)


def _assign_pitch_stuff(frame: pd.DataFrame, reference: pd.DataFrame, pred_col: str, year: int) -> pd.Series:
    """Pitch-level Stuff+ on the same scale as the pitcher-pitch-season grade."""
    lookup = reference.loc[reference["game_year"] == year, ["pitch_type", "lg_mean", "lg_sd"]]
    merged = frame.loc[frame["game_year"] == year, ["pitch_type", pred_col]].merge(
        lookup, on="pitch_type", how="left"
    )
    # merge may reorder. Use a map instead so the series aligns to the frame index.
    means = lookup.set_index("pitch_type")["lg_mean"]
    sds = lookup.set_index("pitch_type")["lg_sd"]
    mu = frame.loc[frame["game_year"] == year, "pitch_type"].map(means)
    sd = frame.loc[frame["game_year"] == year, "pitch_type"].map(sds)
    pred = frame.loc[frame["game_year"] == year, pred_col]
    return 100.0 + STUFF_SD_POINTS * (mu - pred) / sd


def run() -> dict:
    """Fit, grade, and write every artifact the app and the memo read."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_named_sql()
    _log("season inventory")
    inventory = run_sql(queries["season_inventory"], raw_glob=RAW_GLOB)
    whiff_sql = run_sql(queries["whiff_by_pitch_type"], raw_glob=RAW_GLOB)
    inventory.to_csv(PROCESSED_DIR / "season_inventory.csv", index=False)
    whiff_sql.to_csv(PROCESSED_DIR / "whiff_by_pitch_type.csv", index=False)

    frame = prepare_features(load_modeling_frame())
    _log(f"modeling frame {len(frame):,} pitches")
    both = frame[["arm_angle", "arm_angle_proxy"]].dropna()
    arm_corr = float(np.corrcoef(both["arm_angle"], both["arm_angle_proxy"])[0, 1]) if len(both) > 100 else None
    arm_mae = float(np.mean(np.abs(both["arm_angle"] - both["arm_angle_proxy"]))) if len(both) else None
    break_check = frame[["hb_arm", "api_break_x_arm"]].dropna()
    break_corr = None
    if len(break_check) > 100:
        break_corr = float(np.corrcoef(break_check["hb_arm"] / 12.0, break_check["api_break_x_arm"])[0, 1])

    yoy_means = outcome_run_values(frame.loc[frame["game_year"].isin(YOY_TRAIN_YEARS)])
    ship_means = outcome_run_values(frame.loc[frame["game_year"].isin(SHIP_TRAIN_YEARS)])
    _assert_run_value_signs(ship_means)
    frame["rv_yoy"] = assign_count_neutral_run_value(frame, yoy_means)
    frame["rv_ship"] = assign_count_neutral_run_value(frame, ship_means)

    _log("fitting 2024-only model for year-over-year grades")
    yoy_model = RunValueModels().fit(frame.loc[frame["game_year"].isin(YOY_TRAIN_YEARS)], "rv_yoy")
    _log(f"2024 model kept {yoy_model.best_iteration} trees")
    frozen_index = frame.index[frame["game_year"].isin(YOY_SCORE_YEARS)]
    frozen_pred = yoy_model.predict(frame.loc[frozen_index])
    frame["pred_frozen"] = np.nan
    frame.loc[frozen_index, "pred_frozen"] = frozen_pred["lightgbm"]
    metrics_2025 = _evaluate(frame.loc[frozen_index].loc[lambda d: d["game_year"] == 2025], "rv_yoy", {
        name: frozen_pred[name][frame.loc[frozen_index, "game_year"].to_numpy() == 2025]
        for name in frozen_pred
    })
    del frozen_pred
    gc.collect()

    _log("fitting 2024-2025 model for 2026 grades")
    ship_model = RunValueModels().fit(frame.loc[frame["game_year"].isin(SHIP_TRAIN_YEARS)], "rv_ship")
    _log(f"ship model kept {ship_model.best_iteration} trees")
    ship_index = frame.index[frame["game_year"] == SHIP_TEST_YEAR]
    ship_pred = ship_model.predict(frame.loc[ship_index])
    frame["pred_ship"] = np.nan
    frame.loc[ship_index, "pred_ship"] = ship_pred["lightgbm"]
    metrics_2026 = _evaluate(frame.loc[ship_index], "rv_ship", ship_pred)
    del ship_pred
    gc.collect()

    _log("fitting whiff model")
    swing_train = frame.loc[frame["game_year"].isin(SHIP_TRAIN_YEARS) & frame["is_swing"]]
    whiff_model = WhiffModel().fit(swing_train)
    swing_2026 = frame.index[(frame["game_year"] == SHIP_TEST_YEAR) & frame["is_swing"]]
    frame["pred_whiff"] = np.nan
    frame.loc[swing_2026, "pred_whiff"] = whiff_model.predict_proba(frame.loc[swing_2026])
    whiff_metrics = classification_metrics(
        frame.loc[swing_2026, "is_whiff"].astype(int),
        frame.loc[swing_2026, "pred_whiff"],
    )
    whiff_calibration = calibration_table(
        frame.loc[swing_2026, "is_whiff"].astype(float),
        frame.loc[swing_2026, "pred_whiff"],
    )
    _log(f"whiff AUC {whiff_metrics.get('auc')}")

    _log("clustering pitch shapes")
    fitted = fit_mixture(frame)
    frame["cluster"] = assign_clusters(frame, fitted)
    fastball_cluster = _mode_fastball_cluster(frame)
    _log(f"mixture k={fitted['k']} adjusted Rand vs pitch type {fitted['adjusted_rand_vs_pitch_type']:.3f}")

    _log("aggregating arsenals")
    frozen_arsenal, frozen_ref = build_arsenal(frame, "pred_frozen", "rv_yoy", YOY_SCORE_YEARS)
    ship_arsenal, ship_ref = build_arsenal(frame, "pred_ship", "rv_ship", (SHIP_TEST_YEAR,))
    frame.loc[frame["game_year"] == SHIP_TEST_YEAR, "stuff_plus"] = _assign_pitch_stuff(
        frame, ship_ref, "pred_ship", SHIP_TEST_YEAR
    )
    # 2025 pitch-level grades use the frozen scale so the plot sample matches that season.
    frame.loc[frame["game_year"] == 2025, "stuff_plus"] = _assign_pitch_stuff(
        frame, frozen_ref, "pred_frozen", 2025
    )

    ship_out = ship_arsenal.rename(columns={"game_year": "season"})
    frozen_out = frozen_arsenal.rename(columns={"game_year": "season", "stuff_plus": "stuff_plus_frozen"})
    published = pd.concat([
        frozen_out.loc[frozen_out["season"] == 2025].drop(columns=["lg_mean", "lg_sd", "lg_n"], errors="ignore"),
    ], ignore_index=True)
    # 2025 headline grade is the frozen grade. Rename it back to stuff_plus.
    published = published.rename(columns={"stuff_plus_frozen": "stuff_plus"})
    published["stuff_plus_frozen"] = published["stuff_plus"]
    ship_keep = ship_out.drop(columns=["lg_mean", "lg_sd", "lg_n"], errors="ignore")
    frozen_2026 = frozen_out.loc[
        frozen_out["season"] == 2026, ["pitcher", "season", "pitch_type", "stuff_plus_frozen"]
    ]
    ship_keep = ship_keep.merge(frozen_2026, on=["pitcher", "season", "pitch_type"], how="left")
    published = pd.concat([published, ship_keep], ignore_index=True)
    published = published.merge(
        fastball_cluster.rename(columns={"game_year": "season"}),
        on=["pitcher", "season"],
        how="left",
    )
    location = (
        published.loc[published["pitches"] >= STUFF_MIN_PITCHES]
        .groupby(["season", "pitch_type"])["loc_sd"]
        .median()
        .rename("loc_sd_league")
        .reset_index()
    )
    published = published.merge(location, on=["season", "pitch_type"], how="left")

    # Whiff grade on the 2026 ship model, higher probability is better.
    whiff_rows = (
        frame.loc[frame["pred_whiff"].notna()]
        .groupby(["pitcher", "game_year", "pitch_type"], observed=False)["pred_whiff"]
        .mean()
        .rename("pred_whiff_rate")
        .reset_index()
        .rename(columns={"game_year": "season"})
    )
    published = published.merge(whiff_rows, on=["pitcher", "season", "pitch_type"], how="left")
    qual_whiff = published.loc[
        (published["season"] == SHIP_TEST_YEAR) & (published["pitches"] >= STUFF_MIN_PITCHES)
    ]
    whiff_ref = qual_whiff.groupby("pitch_type")["pred_whiff_rate"].agg(whiff_mean="mean", whiff_sd="std")
    published = published.merge(whiff_ref, on="pitch_type", how="left")
    published["whiff_grade"] = np.where(
        published["season"].eq(SHIP_TEST_YEAR),
        100.0 + STUFF_SD_POINTS * (published["pred_whiff_rate"] - published["whiff_mean"]) / published["whiff_sd"],
        np.nan,
    )
    published = published.drop(columns=["whiff_mean", "whiff_sd"])

    rolled = _add_results_axis(_pitcher_season(published))
    focus_table = choose_focus_pitcher(rolled, published)
    focus = focus_table.iloc[0]
    focus_id = int(focus["pitcher"])
    focus_name = str(focus["player_name"])
    _log(f"memo pitcher {focus_name} ({focus_id}), Stuff+ {focus['stuff_plus']:.1f}")

    _log("reliability")
    reliability_frame = frame.loc[
        (frame["game_year"] == SHIP_TEST_YEAR) & frame["stuff_plus"].notna(),
        ["pitcher", "stuff_plus", "is_swing", "is_whiff", "delta_run_exp", "pitch_type"],
    ]
    reliability = _reliability_block(reliability_frame)
    try:
        stats = fetch_season_pitching(2025)
        stats = pd.concat([stats, fetch_season_pitching(2026)], ignore_index=True)
        stats.to_parquet(PROCESSED_DIR / "season_pitching.parquet", index=False)
    except Exception as exc:
        _log(f"MLB season stats unavailable: {exc}")
        stats = pd.DataFrame()
    yoy = _yoy_block(published, stats)

    _log("suggestions")
    advice_input = published.loc[published.groupby(["pitcher", "season"])["pitches"].transform("sum") >= ARSENAL_MIN_PITCHES]
    advice_input = advice_input.dropna(subset=["stuff_plus", "pred_rv", "actual_rv"])
    suggestions = suggest_all(advice_input)

    _log("SHAP")
    try:
        shap_report = _shap_report(ship_model, frame)
    except Exception as exc:
        _log(f"SHAP failed, using gain importance: {exc}")
        gain = ship_model.booster.feature_importances_
        shap_report = {
            "mean_abs_shap": {feature: float(value) for feature, value in zip(FEATURES, gain)},
            "mean_abs_shap_by_pitch_type": {},
            "sample_size": 0,
            "fallback": "lightgbm_gain",
            "error": str(exc),
        }

    grouped_shape = frame.groupby("pitch_type", observed=False)
    shapes = grouped_shape.agg(
        pitches=("release_speed", "size"),
        velo=("release_speed", "mean"),
        ivb=("ivb", "mean"),
        hb_arm=("hb_arm", "mean"),
        vaa_mean=("vaa", "mean"),
        arm_angle=("arm_angle", "mean"),
    ).reset_index()
    quantiles = grouped_shape["vaa"].quantile([0.25, 0.75]).unstack()
    quantiles = quantiles.rename(columns={0.25: "vaa_p25", 0.75: "vaa_p75"}).reset_index()
    shapes = shapes.merge(quantiles, on="pitch_type", how="left")
    profiles = cluster_profiles(frame.loc[frame["game_year"] == SHIP_TEST_YEAR].assign(
        stuff_plus=frame.loc[frame["game_year"] == SHIP_TEST_YEAR, "stuff_plus"]
    ))
    sample = _plot_sample(frame)
    sample_2026_focus = sample.loc[
        (sample["pitcher"] == focus_id) & (sample["game_year"] == SHIP_TEST_YEAR)
    ]

    _log("writing tables")
    published.to_parquet(PROCESSED_DIR / "arsenal.parquet", index=False)
    rolled.to_parquet(PROCESSED_DIR / "pitcher_season.parquet", index=False)
    sample.to_parquet(PROCESSED_DIR / "plot_sample.parquet", index=False)
    shapes.to_csv(PROCESSED_DIR / "pitch_type_shapes.csv", index=False)
    profiles.to_csv(PROCESSED_DIR / "cluster_profiles.csv", index=False)
    focus_table.to_csv(PROCESSED_DIR / "marlins_candidates.csv", index=False)
    _write_json(PROCESSED_DIR / "suggestions.json", suggestions)
    _write_json(PROCESSED_DIR / "shap.json", shap_report)
    _write_json(PROCESSED_DIR / "reliability.json", {"split_half": reliability, "year_over_year": yoy})
    _write_json(PROCESSED_DIR / "outcome_run_values.json", {
        "train_2024": yoy_means.to_dict(),
        "train_2024_2025": ship_means.to_dict(),
    })

    con = duckdb.connect()
    con.register("arsenal", published)
    leaderboard = con.execute(
        queries["leaderboard"].format(test_season=SHIP_TEST_YEAR, min_pitches=LEADERBOARD_MIN_PITCHES)
    ).fetchdf()
    focus_club = con.execute(
        queries["focus_club"].format(focus_team=FOCUS_TEAM, test_season=SHIP_TEST_YEAR)
    ).fetchdf()
    leaderboard.to_parquet(PROCESSED_DIR / "leaderboard.parquet", index=False)
    focus_club.to_csv(PROCESSED_DIR / "marlins_arsenal.csv", index=False)

    focus_arsenal = published.loc[
        (published["pitcher"] == focus_id) & (published["season"] == SHIP_TEST_YEAR)
    ]
    _log("writing figures")
    plot_whiff_calibration(whiff_calibration, FIGURE_DIR / "calibration_whiff.png")
    plot_calibration(
        metrics_2026["calibration"],
        FIGURE_DIR / "calibration_rv.png",
        title="Run-value model calibration, 2026",
        xlabel="Predicted count-neutral run value",
        ylabel="Actual count-neutral run value",
    )
    plot_shap_bars(
        shap_report["mean_abs_shap"],
        FIGURE_DIR / "shap_summary.png",
        title="What moves the 2026 run-value prediction",
    )
    plot_reliability_curves(reliability["curves"], FIGURE_DIR / "reliability_curves.png")
    yoy_plot = {
        "Stuff+": yoy.get("stuff_plus_min_200", {}).get("pearson"),
        "Whiff%": yoy.get("whiff_rate_min_150_swings", {}).get("pearson"),
        "K%": yoy.get("k_pct_min_40_ip", {}).get("pearson"),
        "FIP": yoy.get("fip_min_40_ip", {}).get("pearson"),
        "ERA": yoy.get("era_min_40_ip", {}).get("pearson"),
    }
    plot_yoy_bars(
        yoy_plot,
        FIGURE_DIR / "reliability_yoy.png",
        subtitle="Stuff+ uses one model trained on 2024 only. ERA and FIP need 40 innings both years.",
    )
    scatter = rolled.loc[(rolled["season"] == SHIP_TEST_YEAR) & (rolled["pitches"] >= 400)]
    plot_stuff_vs_results(scatter, FIGURE_DIR / "stuff_vs_results.png", focus_id, focus_name)
    for pitch_type in ("FF", "SI", "SL", "ST", "CH", "CU"):
        board = leaderboard.loc[leaderboard["pitch_type"] == pitch_type].head(15)
        if len(board):
            plot_leaderboard(
                board,
                FIGURE_DIR / f"leaderboard_{pitch_type.lower()}.png",
                PITCH_NAMES[pitch_type],
            )
    if len(sample_2026_focus):
        plot_movement(
            sample_2026_focus,
            FIGURE_DIR / "marlins_movement.png",
            title=f"{focus_name} movement, 2026",
        )
        plot_release(
            sample_2026_focus,
            FIGURE_DIR / "marlins_release.png",
            title=f"{focus_name} release, catcher's view, 2026",
        )
    if len(focus_arsenal):
        plot_pitch_grades(
            focus_arsenal,
            FIGURE_DIR / "marlins_stuff.png",
            title=f"{focus_name} Stuff+ by pitch, 2026",
        )
    if len(profiles):
        plot_cluster_centers(profiles, FIGURE_DIR / "cluster_movement.png")
    plot_vaa(shapes, FIGURE_DIR / "vaa_by_pitch_type.png")

    memo = {
        "pitcher": focus_id,
        "player_name": focus_name,
        "team": FOCUS_TEAM,
        "season": SHIP_TEST_YEAR,
        "stuff_plus": float(focus["stuff_plus"]),
        "pitches": int(focus["pitches"]),
        "whiff_rate": None if pd.isna(focus["whiff_rate"]) else float(focus["whiff_rate"]),
        "runs_prevented_per_100": float(focus["runs_prevented_per_100"]),
        "pred_rv_per_100": float(focus["pred_rv"] * 100),
        "actual_rv_per_100": float(focus["actual_rv"] * 100),
        "gap_per_100": float((focus["actual_rv"] - focus["pred_rv"]) * 100),
        "arsenal": focus_arsenal.to_dict(orient="records"),
        "suggestions": suggestions.get(f"{SHIP_TEST_YEAR}:{focus_id}", []),
        "candidates": focus_table.head(8).to_dict(orient="records"),
    }
    _write_json(PROCESSED_DIR / "memo.json", memo)

    stuff_whiff_corr = None
    both_grades = published.loc[
        (published["season"] == SHIP_TEST_YEAR)
        & published["stuff_plus"].notna()
        & published["whiff_grade"].notna()
        & (published["pitches"] >= STUFF_MIN_PITCHES)
    ]
    if len(both_grades) > 30 and both_grades["stuff_plus"].std() > 0:
        stuff_whiff_corr = float(np.corrcoef(both_grades["stuff_plus"], both_grades["whiff_grade"])[0, 1])

    metrics = {
        "inventory": inventory.to_dict(orient="records"),
        "modeling_pitches": int(len(frame)),
        "modeling_pitches_by_season": {
            str(int(year)): int(n) for year, n in frame.groupby("game_year").size().items()
        },
        "features": FEATURES,
        "diagnostics": {
            "arm_angle_proxy_correlation": arm_corr,
            "arm_angle_proxy_mae_degrees": arm_mae,
            "arm_side_break_vs_savant_correlation": break_corr,
            "outcome_run_values_2024_2025": ship_means.to_dict(),
        },
        "holdout_2025_trained_on_2024": {
            "best_iteration": yoy_model.best_iteration,
            "fit_rows": yoy_model.fit_rows,
            "valid_rows": yoy_model.valid_rows,
            "metrics": metrics_2025,
        },
        "holdout_2026_trained_on_2024_2025": {
            "best_iteration": ship_model.best_iteration,
            "fit_rows": ship_model.fit_rows,
            "valid_rows": ship_model.valid_rows,
            "metrics": metrics_2026,
        },
        "whiff_model_2026": whiff_metrics,
        "whiff_calibration": whiff_calibration,
        "stuff_plus_vs_whiff_grade_correlation": stuff_whiff_corr,
        "clustering": {
            "k": fitted["k"],
            "bic": fitted["bic"],
            "bic_curve": fitted["bic_curve"],
            "adjusted_rand_vs_pitch_type": fitted["adjusted_rand_vs_pitch_type"],
            "sample_size": fitted["sample_size"],
        },
        "reliability": {"split_half": reliability, "year_over_year": yoy},
        "shap": shap_report,
        "focus_pitcher": {"id": focus_id, "name": focus_name, "season": SHIP_TEST_YEAR},
        "pitch_type_shapes": shapes.to_dict(orient="records"),
    }
    _write_json(PROCESSED_DIR / "metrics.json", metrics)
    joblib.dump(
        {"ship": ship_model, "yoy": yoy_model, "whiff": whiff_model, "cluster": fitted},
        MODEL_DIR / "stuff_models.joblib",
    )
    _log("done")
    return metrics
