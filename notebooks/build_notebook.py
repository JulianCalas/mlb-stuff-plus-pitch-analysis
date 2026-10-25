"""Create the story notebook. Run after the pipeline so the paths exist."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip()))

def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip()))

md("""
# What makes a pitch effective?

A Stuff+ model for player development. The grade uses only the flight of the ball: velocity, spin, movement, release, extension, approach angles, and the gap from the pitcher's own fastball. Location, count, and the hitter are left out on purpose. Those are command and sequencing. This is the shape.

100 is league average for that pitch type. Ten points is one standard deviation of pitcher-season means. Higher is better for the pitcher.

The sections below are the interview walkthrough: the data, the physics, the holdout, the grade, how fast it stabilizes, and what a Marlins development memo would actually say.
""")

code("""
from pathlib import Path
import json
import sys

import duckdb
import pandas as pd
from IPython.display import Image, display, Markdown

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from stuffplus.config import FEATURE_LABELS, PITCH_NAMES, RAW_DIR, FIGURE_DIR, PROCESSED_DIR
from stuffplus.physics import approach_angles, arm_side_break, induced_vertical_break

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 120)
metrics = json.loads((PROCESSED_DIR / "metrics.json").read_text())
print("Modeling pitches:", f"{metrics['modeling_pitches']:,}")
print("By season:", metrics["modeling_pitches_by_season"])
""")

md("""
## The data

Regular-season Statcast, pulled with `pybaseball.statcast` from Baseball Savant and cached as parquet. The inventory query is plain SQL in DuckDB. Spring training and the postseason are not in the file. The model trains on 2024 and 2025 and is graded on 2026. A second model trains on 2024 only, so 2025 and 2026 can be compared without either year having been in the fit.
""")

code("""
inventory = duckdb.connect().execute(f'''
    SELECT game_year AS season,
           COUNT(*) AS pitches,
           COUNT(DISTINCT pitcher) AS pitchers,
           MIN(game_date) AS first_date,
           MAX(game_date) AS last_date
    FROM read_parquet('{(RAW_DIR / "pitches_*.parquet").as_posix()}')
    GROUP BY 1
    ORDER BY 1
''').fetchdf()
inventory
""")

md("""
## One pitch, from the tracking fit to a grade

Statcast publishes the acceleration fit (`vx0, vy0, vz0, ax, ay, az`) and the movement relative to a spinless pitch (`pfx_x`, `pfx_z`). Induced vertical break is `pfx_z` in inches. Horizontal break is mirrored so positive means arm side for a left-hander and a right-hander. Approach angle is the direction of the velocity vector at the front of the plate, which is 17 inches from the coordinate origin. Savant's `attack_angle` column is the batter's swing, not the pitch, so it is not used.

The cell below recomputes break and vertical approach angle for one tracked pitch and shows the same functions the model uses.
""")

code("""
example = duckdb.connect().execute(f'''
    SELECT pitch_type, p_throws, release_speed, pfx_x, pfx_z,
           vx0, vy0, vz0, ax, ay, az, release_pos_y, arm_angle
    FROM read_parquet('{(RAW_DIR / "pitches_2026.parquet").as_posix()}')
    WHERE pitch_type = 'FF' AND p_throws = 'R' AND release_speed BETWEEN 95 AND 97
    LIMIT 1
''').fetchdf().iloc[0]

ivb = float(induced_vertical_break(example.pfx_z))
hb = float(arm_side_break(example.pfx_x, example.p_throws))
_, vaa, _ = approach_angles(
    example.vx0, example.vy0, example.vz0,
    example.ax, example.ay, example.az, example.release_pos_y,
)
print(f"{example.p_throws}HP {example.pitch_type} at {example.release_speed:.1f} mph")
print(f"Induced vertical break {ivb:.1f} in, arm-side break {hb:.1f} in")
print(f"Vertical approach angle {float(vaa):.2f} degrees (negative means downhill)")
print(f"Measured arm angle {example.arm_angle:.1f} degrees")
diag = metrics["diagnostics"]
print(
    f"On the full sample, this arm-side break matches Savant at r = "
    f"{diag['arm_side_break_vs_savant_correlation']:.3f}."
)
print(
    f"A shoulder-free arm-angle proxy correlates "
    f"{diag['arm_angle_proxy_correlation']:.2f} with the measurement "
    f"(MAE {diag['arm_angle_proxy_mae_degrees']:.1f} degrees), so the model uses the measurement."
)
""")

md("""
## Did shape beat the pitch label?

Three baselines sit under the gradient-boosted model. The league mean says every pitch is average. The pitch-type mean uses the official label, which Stuff+ is not allowed to see. Ridge regression is the linear reading of the same physical features. LightGBM is the grade.

Pitch-level R² on run value is supposed to be small. A single pitch outcome is the hitter, the count, and a lot of luck. The honest test is whether the prediction still lines up after it is averaged to a pitcher and a pitch type, and whether it lines up inside one pitch type, where the label has nothing left to say.

Run value here is count-neutral. Each outcome (whiff, called strike, foul, ball, single, home run, and so on) is worth the training-set average of `delta_run_exp` for that outcome. The count the pitch happened to be thrown in does not enter the target.
""")

code("""
def line(block, name):
    agg = block["aggregated_min_100"]
    return {
        "model": name,
        "pitch_r2": round(block["r2"], 4),
        "pitch_rmse": round(block["rmse"], 4),
        "pitcher_pitch_r": None if agg["corr"] is None else round(agg["corr"], 3),
        "groups": agg["groups"],
    }

for label, key in [
    ("2025, trained on 2024", "holdout_2025_trained_on_2024"),
    ("2026, trained on 2024-2025", "holdout_2026_trained_on_2024_2025"),
]:
    block = metrics[key]
    print(label, "trees", block["best_iteration"])
    rows = [line(block["metrics"][name], name) for name in ("league_mean", "pitch_type_mean", "ridge", "lightgbm")]
    display(pd.DataFrame(rows))

whiff = metrics["whiff_model_2026"]
print(
    f"Whiff model on 2026 swings: AUC {whiff['auc']:.3f}, "
    f"Brier {whiff['brier']:.4f}, log loss {whiff['log_loss']:.4f}, "
    f"calibration error {whiff['ece']:.4f}, base rate {whiff['base_rate']:.3f}."
)
print(
    "Correlation of run-value Stuff+ and the whiff grade, "
    f"pitcher-pitch-seasons with 100+ pitches: {metrics['stuff_plus_vs_whiff_grade_correlation']:.3f}."
)
""")

md("""
## Calibration

If the model says a bin of pitches should be worth a given run value, the bin should come in near that number. The same check on the whiff model is a probability: predicted whiff rate against actual whiff rate. A model can rank pitches and still be miscalibrated. Both charts are the 2026 holdout.
""")

code("""
for name in ["calibration_rv.png", "calibration_whiff.png"]:
    display(Markdown(f"**{name}**"))
    display(Image(filename=str(FIGURE_DIR / name)))
""")

md("""
## What the model uses

SHAP values are the contribution of each feature to a pitch's predicted run value, in run-value units. Mean absolute SHAP is the average size of that contribution on a 2026 sample. Positive run value is good for the hitter, so a feature that lowers predicted run value is a better pitch. The fastball gaps are in the model because an 85 mph slider is a different pitch under a 98 mph four-seamer than under a 92 mph sinker.
""")

code("""
shap_table = (
    pd.Series(metrics["shap"]["mean_abs_shap"])
    .rename("mean_abs_shap")
    .sort_values(ascending=False)
    .rename(index=FEATURE_LABELS)
    .to_frame()
)
shap_table["mean_abs_shap"] = shap_table["mean_abs_shap"].round(5)
display(shap_table)
display(Image(filename=str(FIGURE_DIR / "shap_summary.png")))
""")

md("""
## Clusters

Official pitch labels are a codebook. The mixture model only sees velocity, induced vertical break, arm-side break, and spin, and it chooses the number of groups by BIC. The adjusted Rand index against the official label says how much of the codebook the shape space rediscovers. Peer recommendations use the fastball's cluster: pitchers with a similar fastball, and the secondary pitches they throw that this pitcher does not.
""")

code("""
cluster = metrics["clustering"]
print(f"k = {cluster['k']}, BIC-selected. Adjusted Rand vs pitch type = {cluster['adjusted_rand_vs_pitch_type']:.3f}.")
display(pd.read_csv(PROCESSED_DIR / "cluster_profiles.csv").round(2).head(15))
display(Image(filename=str(FIGURE_DIR / "cluster_movement.png")))
display(Image(filename=str(FIGURE_DIR / "vaa_by_pitch_type.png")))
""")

md("""
## Reliability

Split-half correlation is the agreement between two random samples of the same size from the same pitcher. That is the reliability of a sample that size. Year-over-year correlation is the same idea with a season on each side. ERA and FIP only exist as season lines, so they get the second test. Stuff+ for that test comes from the model trained only on 2024, applied to both 2025 and 2026, so neither grade was fit on its own season.

The Spearman-Brown formula is a cross-check. It assumes each extra pitch adds independent information. The empirical curve is the result to quote. The formula is there so an interview can show the assumption.
""")

code("""
reliability = json.loads((PROCESSED_DIR / "reliability.json").read_text())
rows = []
for name, curve in reliability["split_half"]["curves"].items():
    for point in curve:
        if point["pearson"] is None:
            continue
        rows.append({"measure": name, "n": point["n"], "pearson": round(point["pearson"], 3), "pitchers": point["pitchers"]})
display(pd.DataFrame(rows))
print("Spearman-Brown cross-check")
display(pd.DataFrame(reliability["split_half"]["spearman_brown"]).T)
yoy = pd.DataFrame(reliability["year_over_year"]).T
display(yoy)
display(Image(filename=str(FIGURE_DIR / "reliability_curves.png")))
display(Image(filename=str(FIGURE_DIR / "reliability_yoy.png")))
""")

md("""
## Player development

The scatter is arsenal Stuff+ against runs prevented per 100 pitches versus the league. Up and to the right is good shape and good results. Right and down is the development list: the ball is better than the line. The memo pitcher is marked. Suggestions in the app are the same function as the memo: a usage shift with a run estimate, a shape target taken from same-handed pitches in the same velocity band that grade 110 or better, or a pitch type that peer fastballs usually carry and this arsenal does not.
""")

code("""
memo = json.loads((PROCESSED_DIR / "memo.json").read_text())
print(memo["player_name"], memo["season"], "Stuff+", round(memo["stuff_plus"], 1), "pitches", memo["pitches"])
print(
    f"Model RV/100 {memo['pred_rv_per_100']:.2f}, actual RV/100 {memo['actual_rv_per_100']:.2f}, "
    f"gap (actual minus model) {memo['gap_per_100']:.2f}"
)
arsenal = pd.DataFrame(memo["arsenal"])
show = arsenal[["pitch_type", "pitches", "usage", "velo", "ivb", "hb_arm", "vaa", "whiff_rate", "stuff_plus", "pred_rv", "actual_rv"]].copy()
show["pitch"] = show["pitch_type"].map(PITCH_NAMES)
show["usage"] = (show["usage"] * 100).round(1)
show["whiff_rate"] = (show["whiff_rate"] * 100).round(1)
show["stuff_plus"] = show["stuff_plus"].round(1)
show["pred_rv_per_100"] = (show["pred_rv"] * 100).round(2)
show["actual_rv_per_100"] = (show["actual_rv"] * 100).round(2)
display(show[["pitch", "pitches", "usage", "velo", "ivb", "hb_arm", "vaa", "whiff_rate", "stuff_plus", "pred_rv_per_100", "actual_rv_per_100"]].round(2))
for note in memo["suggestions"]:
    print("\\n" + note["title"])
    print(note["detail"])
display(Image(filename=str(FIGURE_DIR / "stuff_vs_results.png")))
display(Image(filename=str(FIGURE_DIR / "marlins_movement.png")))
display(Image(filename=str(FIGURE_DIR / "marlins_stuff.png")))
""")

md("""
## What this does not say

The grade is not a contract recommendation. It does not know health, it does not know the hitter, and a shape target is a comparison with pitches that already grade well, not an experiment. Closing a two-inch gap in sweep does not add a promised number of Stuff+ points. The usage-shift run estimate ignores the counts where the weaker pitch is already the right pitch. Sample size is on the reliability chart: below the stabilization point, the grade is a description of the pitches in hand, not a season talent number.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
path = Path(__file__).resolve().parent / "pitch_quality_story.ipynb"
nbf.write(nb, path)
print(path)
