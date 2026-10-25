"""Stuff+ desk.

Pick a pitcher and read the arsenal the way a player-development group
would: shape, grade, and a suggestion that cites the comparison it came
from. Grades are precomputed. This file does not refit the model.

    streamlit run app/streamlit_app.py --server.port 8741
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stuffplus.config import (  # noqa: E402
    FOCUS_TEAM,
    NAVY,
    PITCH_COLORS,
    PITCH_NAMES,
    PROCESSED_DIR,
    SHIP_TEST_YEAR,
)

st.set_page_config(page_title="Stuff+ desk", layout="wide")


@st.cache_data
def load_tables():
    arsenal = pd.read_parquet(PROCESSED_DIR / "arsenal.parquet")
    pitchers = pd.read_parquet(PROCESSED_DIR / "pitcher_season.parquet")
    reliability = json.loads((PROCESSED_DIR / "reliability.json").read_text())
    suggestions = json.loads((PROCESSED_DIR / "suggestions.json").read_text())
    memo = json.loads((PROCESSED_DIR / "memo.json").read_text())
    metrics = json.loads((PROCESSED_DIR / "metrics.json").read_text())
    return arsenal, pitchers, reliability, suggestions, memo, metrics


def pitcher_pitches(pitcher: int, season: int) -> pd.DataFrame:
    path = str(PROCESSED_DIR / "plot_sample.parquet")
    return duckdb.connect().execute(
        """
        SELECT * FROM read_parquet(?)
        WHERE pitcher = ? AND game_year = ?
        """,
        [path, int(pitcher), int(season)],
    ).fetchdf()


def _fmt_stuff(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.0f}"


def movement_figure(sample: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for pitch_type, group in sample.groupby("pitch_type"):
        figure.add_trace(go.Scatter(
            x=group["hb_arm"],
            y=group["ivb"],
            mode="markers",
            name=PITCH_NAMES.get(pitch_type, pitch_type),
            marker=dict(color=PITCH_COLORS.get(pitch_type, "#78716C"), size=8, opacity=0.75),
            hovertemplate=(
                "%{text}<br>Arm-side %{x:.1f} in<br>IVB %{y:.1f} in"
                "<br>%{customdata[0]:.1f} mph<extra></extra>"
            ),
            text=[PITCH_NAMES.get(pitch_type, pitch_type)] * len(group),
            customdata=group[["release_speed"]],
        ))
    figure.add_hline(y=0, line_color="#D6D3D1")
    figure.add_vline(x=0, line_color="#D6D3D1")
    figure.update_layout(
        template="plotly_white",
        height=520,
        legend_title_text="",
        xaxis_title="Arm-side break (inches) →",
        yaxis_title="Induced vertical break (inches) →",
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def release_figure(sample: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for pitch_type, group in sample.groupby("pitch_type"):
        figure.add_trace(go.Scatter(
            x=group["release_pos_x"],
            y=group["release_pos_z"],
            mode="markers",
            name=PITCH_NAMES.get(pitch_type, pitch_type),
            marker=dict(color=PITCH_COLORS.get(pitch_type, "#78716C"), size=8, opacity=0.75),
            showlegend=False,
        ))
    figure.update_layout(
        template="plotly_white",
        height=420,
        xaxis_title="Horizontal release (ft), catcher's view → first base",
        yaxis_title="Release height (ft)",
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def render_arsenal(arsenal, pitchers, suggestions, season, pitcher_id):
    rows = arsenal.loc[(arsenal["season"] == season) & (arsenal["pitcher"] == pitcher_id)].copy()
    summary = pitchers.loc[(pitchers["season"] == season) & (pitchers["pitcher"] == pitcher_id)]
    if rows.empty or summary.empty:
        st.info("No graded pitches for this pitcher in this season.")
        return
    summary = summary.iloc[0]
    rows = rows.sort_values("usage", ascending=False)
    model_note = (
        "Out of sample. The 2026 grade was fit on 2024–2025."
        if season == SHIP_TEST_YEAR
        else "Out of sample. The 2025 grade was fit on 2024 only, so it can be compared with 2026 on one model (the frozen grade)."
    )
    st.caption(model_note)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Arsenal Stuff+", _fmt_stuff(summary["stuff_plus"]))
    c2.metric("Pitches", f"{int(summary['pitches']):,}")
    c3.metric("Whiff rate", f"{summary['whiff_rate'] * 100:.1f}%" if pd.notna(summary["whiff_rate"]) else "—")
    c4.metric(
        "Runs prevented / 100 vs league",
        f"{summary['runs_prevented_per_100']:+.2f}",
    )
    if season == SHIP_TEST_YEAR and pd.notna(summary.get("stuff_plus_frozen")):
        st.caption(
            f"Same season on the 2024-only model: Stuff+ {summary['stuff_plus_frozen']:.0f}. "
            "Use that number, not the headline grade, when you line 2026 up against 2025."
        )

    left, right = st.columns([1.3, 1])
    sample = pitcher_pitches(pitcher_id, season)
    with left:
        st.subheader("Movement")
        if sample.empty:
            st.write("No pitch sample stored for this pitcher.")
        else:
            st.plotly_chart(movement_figure(sample), use_container_width=True)
            st.caption("Positive horizontal break is arm-side run. Glove-side sweep is negative. Both hands use the same axis.")
    with right:
        st.subheader("Release")
        if not sample.empty:
            st.plotly_chart(release_figure(sample), use_container_width=True)
            st.caption("Catcher's view. First base is to the right. This is location of the ball, not the official arm angle.")

    display = rows.copy()
    display["Pitch"] = display["pitch_type"].map(lambda code: PITCH_NAMES.get(code, code))
    display["Usage"] = (display["usage"] * 100).round(1)
    display["Velo"] = display["velo"].round(1)
    display["IVB"] = display["ivb"].round(1)
    display["Arm-side"] = display["hb_arm"].round(1)
    display["VAA"] = display["vaa"].round(2)
    display["Whiff%"] = (display["whiff_rate"] * 100).round(1)
    display["Stuff+"] = display["stuff_plus"].round(1)
    display["Whiff grade"] = display["whiff_grade"].round(1)
    display["Actual RV / 100"] = (display["actual_rv"] * 100).round(2)
    display["Model RV / 100"] = (display["pred_rv"] * 100).round(2)
    st.subheader("Arsenal")
    st.dataframe(
        display[[
            "Pitch", "Usage", "Velo", "IVB", "Arm-side", "VAA", "Whiff%",
            "Stuff+", "Whiff grade", "Model RV / 100", "Actual RV / 100", "pitches",
        ]].rename(columns={"pitches": "Pitches"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Stuff+ is 100 for an average pitcher-season of that pitch, and 10 points is one standard deviation. "
        "Run value is count-neutral and from the batter's side, so a negative model RV is a better pitch. "
        "Whiff grade is the same scale on predicted whiff rate and is blank for 2025."
    )

    st.subheader("Development notes")
    notes = suggestions.get(f"{int(season)}:{int(pitcher_id)}", [])
    if not notes:
        st.write("Not enough pitches for a suggestion, or this season was not scored.")
    for note in notes:
        st.markdown(f"**{note['title']}**")
        st.write(note["detail"])


def render_leaderboard(arsenal, season):
    board = arsenal.loc[
        (arsenal["season"] == season)
        & (arsenal["pitches"] >= 200)
        & arsenal["stuff_plus"].notna()
    ].copy()
    pitch_type = st.selectbox(
        "Pitch",
        options=list(PITCH_NAMES),
        format_func=lambda code: PITCH_NAMES[code],
        index=0,
    )
    view = board.loc[board["pitch_type"] == pitch_type].sort_values("stuff_plus", ascending=False)
    view["Pitcher"] = view["player_name"]
    view["Throws"] = view["p_throws"]
    view["Team"] = view["team"]
    view["Stuff+"] = view["stuff_plus"].round(1)
    view["Velo"] = view["velo"].round(1)
    view["IVB"] = view["ivb"].round(1)
    view["Arm-side"] = view["hb_arm"].round(1)
    view["Usage %"] = (view["usage"] * 100).round(1)
    view["Whiff %"] = (view["whiff_rate"] * 100).round(1)
    st.dataframe(
        view[["Pitcher", "Team", "Throws", "Stuff+", "Velo", "IVB", "Arm-side", "Usage %", "Whiff %", "pitches"]].head(40)
        .rename(columns={"pitches": "Pitches"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Minimum 200 pitches. 100 is average for this pitch type in this season, not average across all pitches.")


def render_reliability(reliability, metrics):
    st.subheader("Split-half, 2026 grades")
    st.write(
        "Each point is the correlation between two random samples of that many pitches "
        "from the same pitcher. Stuff+ is a description of the ball, so it settles "
        "sooner than a result."
    )
    curves = reliability["split_half"]["curves"]
    rows = []
    for name, curve in curves.items():
        for point in curve:
            if point["pearson"] is None:
                continue
            rows.append({"Measure": name, "Sample size": point["n"], "Correlation": round(point["pearson"], 3), "Pitchers": point["pitchers"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    prophecy = reliability["split_half"]["spearman_brown"]
    st.markdown("**Smallest tested sample at a 0.70 correlation**")
    for name, block in prophecy.items():
        if not block:
            continue
        empirical = block["empirical_n_for_0_70"]
        formula = block["n_for_0_70"]
        formula_text = f"{formula:.0f}" if formula is not None else "—"
        empirical_text = str(empirical) if empirical is not None else "not reached in the grid"
        st.write(f"{name}: empirical {empirical_text}. Spearman-Brown, anchored at 100 observations, says about {formula_text}.")

    st.subheader("Year over year, 2025 to 2026")
    yoy = reliability["year_over_year"]
    yoy_rows = []
    labels = {
        "stuff_plus_min_200": "Stuff+ (frozen 2024 model, 200 pitches)",
        "whiff_rate_min_150_swings": "Whiff rate (150 swings)",
        "k_pct_min_40_ip": "Strikeout rate (40 innings)",
        "fip_min_40_ip": "FIP (40 innings)",
        "era_min_40_ip": "ERA (40 innings)",
        "k_pct_min_80_ip": "Strikeout rate (80 innings)",
        "fip_min_80_ip": "FIP (80 innings)",
        "era_min_80_ip": "ERA (80 innings)",
    }
    for key, label in labels.items():
        block = yoy.get(key) or {}
        if not block:
            continue
        yoy_rows.append({
            "Measure": label,
            "Pitchers": block.get("n"),
            "Pearson": None if block.get("pearson") is None else round(block["pearson"], 3),
            "Spearman": None if block.get("spearman") is None else round(block["spearman"], 3),
        })
    st.dataframe(pd.DataFrame(yoy_rows), hide_index=True, use_container_width=True)
    holdout = metrics["holdout_2026_trained_on_2024_2025"]["metrics"]["lightgbm"]
    st.caption(
        f"Pitch-level R² of the 2026 run-value model is {holdout['r2']:.3f}. "
        "That is expected. The year-over-year number above is the one that says whether the grade is a trait."
    )


def render_method(metrics):
    st.subheader("What is and is not in the grade")
    st.write(
        "The model sees velocity, spin, induced vertical break, arm-side break, "
        "release height and arm-side release, extension, measured arm angle, "
        "vertical and horizontal approach angle, spin-axis sine and cosine, "
        "movement per 1,000 rpm, and the velocity and movement gaps versus "
        "the pitcher's primary fastball. It does not see pitch type, location, "
        "count, or the hitter."
    )
    ship = metrics["holdout_2026_trained_on_2024_2025"]["metrics"]
    rows = []
    for name in ("league_mean", "pitch_type_mean", "ridge", "lightgbm"):
        block = ship[name]
        rows.append({
            "Model": name,
            "R²": round(block["r2"], 4),
            "RMSE": round(block["rmse"], 4),
            "Pitch-level corr": None if block["corr"] is None else round(block["corr"], 3),
            "Pitcher-pitch corr": None if block["aggregated_min_100"]["corr"] is None else round(block["aggregated_min_100"]["corr"], 3),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    cluster = metrics["clustering"]
    st.write(
        f"Shape clustering used a diagonal Gaussian mixture with k = {cluster['k']}, "
        f"chosen by BIC. Adjusted Rand index against the official pitch label: "
        f"{cluster['adjusted_rand_vs_pitch_type']:.3f}. "
        "A low number means the clusters are not just renaming the label. "
        "A high number means they mostly rediscovered it, and the useful part "
        "is the within-label split."
    )
    diag = metrics["diagnostics"]
    st.write(
        f"The arm-angle proxy correlates {diag['arm_angle_proxy_correlation']:.2f} "
        f"with Statcast's measured arm angle (MAE {diag['arm_angle_proxy_mae_degrees']:.1f}°). "
        "The model uses the measurement. Mirrored horizontal break correlates "
        f"{diag['arm_side_break_vs_savant_correlation']:.3f} with Savant's arm-side break field."
    )


def main():
    needed = PROCESSED_DIR / "arsenal.parquet"
    if not needed.exists():
        st.error("Grades are not built yet. Run `python scripts/pull_statcast.py` and then `python scripts/run_pipeline.py`.")
        return
    arsenal, pitchers, reliability, suggestions, memo, metrics = load_tables()
    st.title("Stuff+")
    st.markdown(
        "Pitch quality from the flight of the ball. **100 is league average** for that pitch type. "
        "Built as a player-development desk: shape first, results second."
    )
    arsenal_tab, board_tab, reliability_tab, method_tab = st.tabs(
        ["Arsenal", "Leaderboard", "Reliability", "Method"]
    )
    seasons = sorted(pitchers["season"].unique())
    with arsenal_tab:
        with st.sidebar:
            st.header("Pitcher")
            season = st.selectbox("Season", seasons, index=len(seasons) - 1)
            season_pitchers = pitchers.loc[pitchers["season"] == season].sort_values("player_name")
            teams = ["All teams"] + sorted(season_pitchers["team"].dropna().unique())
            default_team_index = teams.index(FOCUS_TEAM) if FOCUS_TEAM in teams else 0
            team = st.selectbox("Team", teams, index=default_team_index)
            pool = season_pitchers if team == "All teams" else season_pitchers.loc[season_pitchers["team"] == team]
            names = pool["player_name"].tolist()
            ids = pool["pitcher"].tolist()
            default_index = 0
            if season == memo["season"] and memo["pitcher"] in ids:
                default_index = ids.index(memo["pitcher"])
            choice = st.selectbox("Pitcher", options=list(range(len(ids))), format_func=lambda i: names[i], index=default_index)
            st.caption("Stuff+ uses physical traits only. Location shows up in the notes when results and stuff disagree.")
        render_arsenal(arsenal, pitchers, suggestions, int(season), int(ids[choice]))
    with board_tab:
        season_board = st.selectbox("Leaderboard season", seasons, index=len(seasons) - 1, key="board_season")
        render_leaderboard(arsenal, int(season_board))
    with reliability_tab:
        render_reliability(reliability, metrics)
    with method_tab:
        render_method(metrics)


if __name__ == "__main__":
    main()
else:
    main()
