"""Geometry tests that do not need Statcast on disk."""

import numpy as np
import pandas as pd

from stuffplus.physics import (
    approach_angles,
    arm_side_break,
    induced_vertical_break,
    mirror_spin_axis,
    pitching_team,
    pretty_name,
    time_to_plate,
    to_arm_side,
)


def test_arm_side_break_mirrors_left_and_right():
    # +pfx_x is the catcher's right. That is glove side for a righty
    # and arm side for a lefty. Half a foot is 6 inches.
    assert arm_side_break(-0.5, "R") == 6
    assert arm_side_break(0.5, "L") == 6
    assert arm_side_break(0.5, "R") == -6
    mirrored = to_arm_side([-1.5, 1.5], ["R", "L"])
    assert np.allclose(mirrored, [1.5, 1.5])


def test_induced_vertical_break_is_pfx_z_in_inches():
    assert induced_vertical_break(1.5) == 18
    assert induced_vertical_break(-0.25) == -3


def test_time_to_plate_with_no_acceleration():
    # 54 ft release, no y acceleration, 130 ft/s toward the plate.
    y_plate = 17 / 12
    expected = (54 - y_plate) / 130
    t = time_to_plate(54, -130, 0)
    assert np.isclose(t, expected, atol=1e-6)
    assert 0.35 < t < 0.50


def test_vertical_approach_angle_is_negative_when_the_ball_is_dropping():
    # Constant velocities: vy = -130 ft/s, vz = -10 ft/s, no acceleration.
    t, vaa, haa = approach_angles(
        vx0=0, vy0=-130, vz0=-10, ax=0, ay=0, az=0, release_pos_y=54
    )
    assert np.isclose(vaa, np.degrees(np.arctan2(-10, 130)), atol=1e-6)
    assert vaa < 0
    assert np.isclose(haa, 0, atol=1e-6)
    assert np.isfinite(t)


def test_spin_axis_mirror_matches_a_right_handed_fastball():
    # A lefty axis of 140 is the reflection of a righty axis of 220.
    assert mirror_spin_axis(140, "L") == 220
    assert mirror_spin_axis(220, "R") == 220
    both = mirror_spin_axis([140, 220], ["L", "R"])
    assert np.allclose(both, [220, 220])


def test_pitching_team_follows_the_half_inning():
    teams = pitching_team(
        ["Top", "Bot"],
        ["MIA", "MIA"],
        ["ATL", "ATL"],
    )
    assert list(teams) == ["MIA", "ATL"]


def test_pretty_name():
    assert pretty_name("Alcantara, Sandy") == "Sandy Alcantara"
    assert pretty_name("Sandy Alcantara") == "Sandy Alcantara"
