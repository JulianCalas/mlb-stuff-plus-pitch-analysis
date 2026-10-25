"""Paths, pitch groups, and model settings.

Nothing in here is estimated from the data. Thresholds that define a
grade (100 = average, 10 points = one standard deviation) live here so
the memo, the app, and the tests all use the same definition.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNK_DIR = RAW_DIR / "chunks"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURE_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"
SQL_DIR = ROOT / "sql"
REPORT_DIR = ROOT / "reports"

# Regular-season windows. pybaseball also returns spring training and
# the postseason; the loader keeps game_type == "R" only.
SEASONS = {
    2024: ("2024-03-18", "2024-10-02"),
    2025: ("2025-03-18", "2025-10-02"),
    2026: ("2026-03-18", "2026-10-02"),
}

# Train on the two earlier seasons, grade the latest one. A second model
# trains on 2024 only so 2025 and 2026 can be compared with neither year
# inside that model's fit. See pipeline.train_and_score.
SHIP_TRAIN_YEARS = (2024, 2025)
SHIP_TEST_YEAR = 2026
YOY_TRAIN_YEARS = (2024,)
YOY_SCORE_YEARS = (2025, 2026)

# Pitch types the model is allowed to grade. Knuckleballs, eephuses, and
# pitchouts are too rare to support a within-type average.
CORE_PITCH_TYPES = ("FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS")

# A pitcher's "primary fastball" is the most-used of these. Cutters count
# so a cutter-first pitcher still gets a velocity gap on the slider.
FASTBALL_TYPES = ("FF", "SI", "FC")
FASTBALL_TIEBREAK = {"FF": 0, "SI": 1, "FC": 2}

PITCH_NAMES = {
    "FF": "Four-seam",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CU": "Curveball",
    "KC": "Knuckle curve",
    "CH": "Changeup",
    "FS": "Splitter",
}

# Statcast coordinates: +x is the catcher's right (first-base side).
# Arm-side break is -pfx_x for a righty and +pfx_x for a lefty.
# Front of home plate is 17 inches from the coordinate origin.
PLATE_Y_FT = 17.0 / 12.0
INCHES_PER_FOOT = 12.0
# Historical sidearm reference height used by pitch-physics worksheets.
ECKERSLEY_LINE_FT = 2.85

# Rows outside these ranges are bad tracks, not real pitches.
MIN_RELEASE_SPEED = 64.0
MAX_RELEASE_SPEED = 104.0
MAX_BREAK_INCHES = 30.0

# Stuff+ scale. 100 is the average qualified pitcher-pitch-season.
# 10 points is one standard deviation of those season means.
STUFF_SD_POINTS = 10.0
STUFF_MIN_PITCHES = 100
LEADERBOARD_MIN_PITCHES = 200
ARSENAL_MIN_PITCHES = 200
SPLIT_HALF_SIZES = (25, 50, 75, 100, 150, 200, 300, 500, 800, 1000)
SPLIT_HALF_REPEATS = 20
SPLIT_HALF_MIN_PITCHERS = 40
RELIABILITY_TARGET = 0.70

# Recommendation thresholds. Documented in recommend.py.
USAGE_DRAG_STUFF = 92.0
USAGE_DRAG_SHARE = 0.12
USAGE_KEEP_STUFF = 103.0
SHAPE_STUFF_MAX = 98.0
SHAPE_TARGET_STUFF = 110.0
SHAPE_GAP_INCHES = 1.5
ADD_PITCH_PEER_RATE = 0.35
ADD_PITCH_PEER_STUFF = 100.0
STUFF_AHEAD_GRADE = 104.0
STUFF_AHEAD_RV_GAP = 0.15  # runs per 100 pitches, results worse than stuff

FEATURES = [
    "release_speed",
    "release_spin_rate",
    "ivb",
    "hb_arm",
    "release_x_arm",
    "release_pos_z",
    "release_extension",
    "arm_angle",
    "vaa",
    "haa_arm",
    "velo_diff",
    "ivb_diff",
    "hb_diff",
    "spin_sin",
    "spin_cos",
    "movement_per_1k_spin",
]

FEATURE_LABELS = {
    "release_speed": "Velocity (mph)",
    "release_spin_rate": "Spin rate (rpm)",
    "ivb": "Induced vertical break (in)",
    "hb_arm": "Arm-side break (in)",
    "release_x_arm": "Horizontal release, arm side (ft)",
    "release_pos_z": "Release height (ft)",
    "release_extension": "Extension (ft)",
    "arm_angle": "Arm angle (deg)",
    "vaa": "Vertical approach angle (deg)",
    "haa_arm": "Horizontal approach angle, arm side (deg)",
    "velo_diff": "Velocity vs primary fastball (mph)",
    "ivb_diff": "IVB vs primary fastball (in)",
    "hb_diff": "Arm-side break vs primary fastball (in)",
    "spin_sin": "Spin axis, sine (mirrored)",
    "spin_cos": "Spin axis, cosine (mirrored)",
    "movement_per_1k_spin": "Movement per 1,000 rpm (in)",
}

LGBM_PARAMS = {
    "n_estimators": 1200,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": 7,
    "n_jobs": 4,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}

EARLY_STOPPING_ROUNDS = 50
EARLY_STOP_FRACTION = 0.15
RIDGE_ALPHA = 10.0
RANDOM_SEED = 7

# Focus club for the player-development memo.
FOCUS_TEAM = "MIA"
FOCUS_TEAM_NAME = "Miami Marlins"

KEEP_COLUMNS = [
    "pitch_type",
    "pitch_name",
    "game_date",
    "game_year",
    "game_pk",
    "game_type",
    "player_name",
    "pitcher",
    "p_throws",
    "stand",
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "description",
    "events",
    "type",
    "zone",
    "delta_run_exp",
    "delta_pitcher_run_exp",
    "arm_angle",
    "api_break_x_arm",
    "api_break_z_with_gravity",
    "home_team",
    "away_team",
    "inning_topbot",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "at_bat_number",
    "pitch_number",
]

FLOAT_COLUMNS = [
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "delta_run_exp",
    "delta_pitcher_run_exp",
    "arm_angle",
    "api_break_x_arm",
    "api_break_z_with_gravity",
]

INT_COLUMNS = [
    "game_year",
    "game_pk",
    "pitcher",
    "balls",
    "strikes",
    "zone",
    "outs_when_up",
    "inning",
    "at_bat_number",
    "pitch_number",
]

PITCH_COLORS = {
    "FF": "#C41E3A",
    "SI": "#E07A3D",
    "FC": "#7B2D8E",
    "SL": "#2E6BFF",
    "ST": "#0E8A8A",
    "SV": "#6B4C9A",
    "CU": "#2E8B57",
    "KC": "#1F6B4A",
    "CH": "#C59214",
    "FS": "#8C4A1F",
}

NAVY = "#0C2340"
CARDINAL = "#BA0C2F"
CREAM = "#F7F4EF"
INK = "#1C1917"
GOOD = "#1F7A4D"
WARN = "#A15C07"
