"""Physical features from a Statcast pitch record.

Statcast coordinates, from the catcher's view:

- ``+x`` is the catcher's right, the first-base side
- ``+y`` points back at the pitcher (the rubber is near y = 60.5 ft)
- ``+z`` is up
- the origin is the back tip of home plate

``pfx_x`` and ``pfx_z`` are movement in feet relative to a spinless
pitch that started with the same velocity. Gravity is already removed
from ``pfx_z``, so ``pfx_z`` is induced vertical break. ``pfx_x`` is
horizontal break in the catcher's frame, which means a left-hander's
slider and a right-hander's slider have opposite signs. Every feature
that has a left-right direction is mirrored so that positive means
arm side for both hands. The model can then learn "more sweep" as one
idea instead of two.

Approach angles are not in the pitch table. ``attack_angle`` on Savant
is the batter's swing path, not the pitch. We integrate the published
acceleration fit from release to the front of the plate (y = 17/12 ft)
and read the velocity direction there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stuffplus.config import ECKERSLEY_LINE_FT, INCHES_PER_FOOT, PLATE_Y_FT


def to_arm_side(values, p_throws) -> np.ndarray:
    """Flip a catcher-frame x measurement so positive is arm side.

    Right-handers' arm side is the third-base side, which is negative x.
    Left-handers' arm side is the first-base side, which is positive x.
    Multiplying a right-hander's value by -1 puts both hands on one scale.
    This is the same sign convention as Savant's ``api_break_x_arm``.
    """
    values = np.asarray(values, dtype=float)
    throws = np.asarray(p_throws)
    sign = np.where(throws == "R", -1.0, 1.0)
    return values * sign


def induced_vertical_break(pfx_z_ft) -> np.ndarray:
    """Induced vertical break in inches. Positive is ride, negative is drop."""
    return np.asarray(pfx_z_ft, dtype=float) * INCHES_PER_FOOT


def arm_side_break(pfx_x_ft, p_throws) -> np.ndarray:
    """Horizontal break in inches. Positive is arm-side run, negative is sweep."""
    return to_arm_side(pfx_x_ft, p_throws) * INCHES_PER_FOOT


def arm_angle_proxy(release_pos_z, release_pos_x) -> np.ndarray:
    """Release-point angle above a sidearm reference line, in degrees.

    Savant's ``arm_angle`` is the angle from the throwing shoulder to the
    ball. The public pitch table does not include the shoulder. This proxy
    uses the Eckersley line (2.85 ft, a historical sidearm height) as the
    vertex and the absolute horizontal release as the width. It is a
    geometry sketch, not a replacement for the measured arm angle. The
    model uses Savant's column and we report how the two compare.
    """
    height = np.asarray(release_pos_z, dtype=float) - ECKERSLEY_LINE_FT
    width = np.abs(np.asarray(release_pos_x, dtype=float))
    return np.degrees(np.arctan2(height, width))


def time_to_plate(y0, vy0, ay, y_plate: float = PLATE_Y_FT) -> np.ndarray:
    """Seconds from release until the ball reaches the front of home plate.

    Position in the y direction is ``y(t) = y0 + vy0 t + 0.5 ay t^2``.
    ``vy0`` is negative because the ball travels toward the plate. When
    drag acceleration is essentially zero we use the linear solution.
    Roots that are not a real flight time (about 0.4 seconds) become NaN
    so a bad track cannot quietly become a wild approach angle.
    """
    y0 = np.asarray(y0, dtype=float)
    vy0 = np.asarray(vy0, dtype=float)
    ay = np.asarray(ay, dtype=float)
    a = 0.5 * ay
    b = vy0
    c = y0 - y_plate
    disc = b * b - 4.0 * a * c
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    denom = 2.0 * a
    linear = np.abs(a) < 1e-6
    safe_vy = np.where(np.abs(vy0) < 1e-6, np.nan, vy0)
    t_linear = (y_plate - y0) / safe_vy
    safe_denom = np.where(np.abs(denom) < 1e-8, np.nan, denom)
    t1 = (-b - sqrt_disc) / safe_denom
    t2 = (-b + sqrt_disc) / safe_denom
    t1 = np.where(t1 > 0, t1, np.inf)
    t2 = np.where(t2 > 0, t2, np.inf)
    t = np.minimum(t1, t2)
    t = np.where(np.isfinite(t), t, np.nan)
    t = np.where(disc < 0, np.nan, t)
    t = np.where(linear, t_linear, t)
    t = np.where((t > 0.25) & (t < 0.80), t, np.nan)
    return t


def approach_angles(vx0, vy0, vz0, ax, ay, az, release_pos_y):
    """Vertical and horizontal approach angles at the front of the plate.

    Returns ``(time, vaa, haa)``. VAA is negative when the ball is moving
    down. A typical four-seamer is near -5 degrees; a curveball is near
    -10. HAA is still in the catcher's frame (positive = toward first
    base). Callers mirror it with ``to_arm_side``.
    """
    t = time_to_plate(release_pos_y, vy0, ay)
    vx = np.asarray(vx0, dtype=float) + np.asarray(ax, dtype=float) * t
    vy = np.asarray(vy0, dtype=float) + np.asarray(ay, dtype=float) * t
    vz = np.asarray(vz0, dtype=float) + np.asarray(az, dtype=float) * t
    # -vy is positive because the ball is still traveling toward the plate.
    vaa = np.degrees(np.arctan2(vz, -vy))
    haa = np.degrees(np.arctan2(vx, -vy))
    return t, vaa, haa


def mirror_spin_axis(spin_axis, p_throws) -> np.ndarray:
    """Reflect a left-hander's spin axis into a right-handed frame.

    Statcast measures spin axis in degrees in the movement plane. A
    left-hander's fastball near 140 degrees is the mirror of a
    right-hander's fastball near 220. ``(360 - axis)`` is that reflection.
    Sine and cosine of the mirrored axis are the model features, so the
    0/360 boundary does not look like a huge jump.
    """
    axis = np.asarray(spin_axis, dtype=float)
    throws = np.asarray(p_throws)
    mirrored = np.where(throws == "L", np.mod(360.0 - axis, 360.0), axis)
    return mirrored


def movement_per_thousand_rpm(ivb, hb_arm, spin_rate) -> np.ndarray:
    """Inches of total break per 1,000 rpm.

    A rough stand-in for how much of the spin is "active" (it moves the
    ball) rather than gyro spin (it doesn't). It is not Alan Nathan's
    spin efficiency. Similar movement at much higher spin usually means
    more gyro.
    """
    ivb = np.asarray(ivb, dtype=float)
    hb = np.asarray(hb_arm, dtype=float)
    spin = np.asarray(spin_rate, dtype=float)
    total = np.sqrt(ivb * ivb + hb * hb)
    per = total / (spin / 1000.0)
    per = np.where(spin > 0, per, np.nan)
    return per


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add break, approach, release, and spin features. Does not drop rows."""
    out = df.copy()
    throws = out["p_throws"].to_numpy()
    out["ivb"] = induced_vertical_break(out["pfx_z"].to_numpy())
    out["hb_arm"] = arm_side_break(out["pfx_x"].to_numpy(), throws)
    out["release_x_arm"] = to_arm_side(out["release_pos_x"].to_numpy(), throws)
    out["plate_x_arm"] = to_arm_side(out["plate_x"].to_numpy(), throws)
    out["arm_angle_proxy"] = arm_angle_proxy(
        out["release_pos_z"].to_numpy(), out["release_pos_x"].to_numpy()
    )
    t, vaa, haa = approach_angles(
        out["vx0"].to_numpy(),
        out["vy0"].to_numpy(),
        out["vz0"].to_numpy(),
        out["ax"].to_numpy(),
        out["ay"].to_numpy(),
        out["az"].to_numpy(),
        out["release_pos_y"].to_numpy(),
    )
    out["flight_time"] = t
    out["vaa"] = vaa
    out["haa_arm"] = to_arm_side(haa, throws)
    spin_m = mirror_spin_axis(out["spin_axis"].to_numpy(), throws)
    out["spin_axis_mirrored"] = spin_m
    radians = np.deg2rad(spin_m)
    out["spin_sin"] = np.sin(radians)
    out["spin_cos"] = np.cos(radians)
    out["movement_per_1k_spin"] = movement_per_thousand_rpm(
        out["ivb"].to_numpy(), out["hb_arm"].to_numpy(), out["release_spin_rate"].to_numpy()
    )
    return out


def pitching_team(inning_topbot, home_team, away_team) -> np.ndarray:
    """The team that is pitching.

    The top of the inning is the home defense. The bottom is the road
    defense. Statcast does not carry a pitcher-team column.
    """
    top = np.asarray(inning_topbot).astype(str) == "Top"
    home = np.asarray(home_team).astype(str)
    away = np.asarray(away_team).astype(str)
    return np.where(top, home, away)


def pretty_name(name: str) -> str:
    """Turn ``'Alcantara, Sandy'`` into ``'Sandy Alcantara'``."""
    if not isinstance(name, str) or "," not in name:
        return name
    last, first = name.split(",", 1)
    return f"{first.strip()} {last.strip()}"
