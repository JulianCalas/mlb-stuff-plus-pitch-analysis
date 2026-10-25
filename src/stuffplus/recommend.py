"""Player-development suggestions from grades, shapes, and peer arsenals.

Every suggestion is a comparison the model already made, written as
something a coach can try in a bullpen. None of them is a causal claim.
Moving two inches of sweep does not guarantee ten points of Stuff+,
because we did not randomly assign the sweep. The wording keeps that
distinction, and the thresholds below are deliberate so a one-point
wobble never becomes a recommendation.

Thresholds live in config.py and are restated here in the function
docstrings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stuffplus.config import (
    ADD_PITCH_PEER_RATE,
    ADD_PITCH_PEER_STUFF,
    PITCH_NAMES,
    SHAPE_GAP_INCHES,
    SHAPE_STUFF_MAX,
    SHAPE_TARGET_STUFF,
    STUFF_AHEAD_GRADE,
    STUFF_AHEAD_RV_GAP,
    USAGE_DRAG_SHARE,
    USAGE_DRAG_STUFF,
    USAGE_KEEP_STUFF,
)


def pitch_label(pitch_type: str) -> str:
    return PITCH_NAMES.get(str(pitch_type), str(pitch_type))


def usage_shift_runs(rv_from: float, rv_to: float, share: float = 0.10) -> float:
    """Runs saved per 100 pitches by moving ``share`` of pitches to a better one.

    Run values are per pitch, batter-side, so a positive difference
    ``rv_from - rv_to`` means the pitch being reduced allows more runs.
    The result is on a per-100-pitches scale. It ignores count, platoon,
    and the fact that the pitcher already chose when to throw each pitch.
    """
    return float(share * (rv_from - rv_to) * 100.0)


def _target_pool(arsenal: pd.DataFrame, pitch_type: str, hand: str, velo: float, pitcher: int) -> pd.DataFrame:
    """High-grade versions of this pitch, same hand, within 2 mph, other pitchers."""
    pool = arsenal.loc[
        (arsenal["pitch_type"] == pitch_type)
        & (arsenal["p_throws"] == hand)
        & (arsenal["pitcher"] != pitcher)
        & (arsenal["stuff_plus"] >= SHAPE_TARGET_STUFF)
        & (arsenal["pitches"] >= 100)
        & ((arsenal["velo"] - velo).abs() <= 2.0)
    ]
    if len(pool) < 8:
        pool = arsenal.loc[
            (arsenal["pitch_type"] == pitch_type)
            & (arsenal["p_throws"] == hand)
            & (arsenal["pitcher"] != pitcher)
            & (arsenal["stuff_plus"] >= SHAPE_TARGET_STUFF)
            & (arsenal["pitches"] >= 100)
            & ((arsenal["velo"] - velo).abs() <= 4.0)
        ]
    return pool


def suggest_for_pitcher(pitcher_rows: pd.DataFrame, league: pd.DataFrame) -> list[dict]:
    """Suggestions for one pitcher-season.

    ``pitcher_rows`` is that pitcher's arsenal table (one row per pitch).
    ``league`` is every pitcher-season-pitch row in the same season, with
    the same columns. ``fb_cluster`` is the shape cluster of each pitcher's
    primary fastball, repeated on every row of that pitcher.
    """
    rows = pitcher_rows.dropna(subset=["stuff_plus", "pred_rv", "actual_rv"]).copy()
    if rows.empty:
        return []
    rows = rows.sort_values("usage", ascending=False)
    hand = rows["p_throws"].iloc[0]
    pitcher = int(rows["pitcher"].iloc[0])
    season = int(rows["season"].iloc[0])
    suggestions: list[dict] = []

    total_pitches = float(rows["pitches"].sum())
    if total_pitches <= 0:
        return []
    arsenal_stuff = float(np.average(rows["stuff_plus"], weights=rows["pitches"]))
    arsenal_pred = float(np.average(rows["pred_rv"], weights=rows["pitches"]))
    arsenal_actual = float(np.average(rows["actual_rv"], weights=rows["pitches"]))
    results_gap = (arsenal_actual - arsenal_pred) * 100.0

    if arsenal_stuff >= STUFF_AHEAD_GRADE and results_gap >= STUFF_AHEAD_RV_GAP:
        fastball = rows.loc[rows["pitch_type"].isin(["FF", "SI", "FC"])]
        location_note = ""
        has_location = (
            "loc_sd" in fastball.columns
            and "loc_sd_league" in fastball.columns
            and fastball["loc_sd"].notna().any()
            and fastball["loc_sd_league"].notna().any()
        )
        if len(fastball) and has_location:
            fb = fastball.sort_values("pitches", ascending=False).iloc[0]
            location_note = (
                f" The {pitch_label(fb.pitch_type).lower()} location scatter "
                f"is {fb.loc_sd:.2f} ft against a league {fb.loc_sd_league:.2f} ft "
                f"for that pitch type."
            )
        suggestions.append({
            "kind": "stuff_over_results",
            "pitch_type": None,
            "title": "The arsenal grades ahead of the results",
            "detail": (
                f"Usage-weighted Stuff+ is {arsenal_stuff:.0f}, but results were "
                f"{results_gap:.2f} runs per 100 pitches worse than the shape model expected. "
                f"That gap is command, sequencing, or contact quality, not a missing tick of velocity."
                f"{location_note} Bullpen priority: competitive misses and first-pitch strikes, "
                f"not a new grip."
            ),
            "evidence": {
                "stuff_plus": round(arsenal_stuff, 1),
                "results_minus_stuff_per_100": round(results_gap, 3),
                "season": season,
            },
        })
    elif arsenal_stuff <= (100 - (STUFF_AHEAD_GRADE - 100)) and -results_gap >= STUFF_AHEAD_RV_GAP:
        suggestions.append({
            "kind": "results_over_stuff",
            "pitch_type": None,
            "title": "Results are ahead of the shape grade",
            "detail": (
                f"Usage-weighted Stuff+ is {arsenal_stuff:.0f}, and results were "
                f"{-results_gap:.2f} runs per 100 pitches better than the shape model expected. "
                f"Treat the ERA as fragile until the locations or the pitch mix explain it. "
                f"Do not redesign a pitch that is already carrying the staff line."
            ),
            "evidence": {
                "stuff_plus": round(arsenal_stuff, 1),
                "results_minus_stuff_per_100": round(results_gap, 3),
            },
        })

    drags = rows.loc[
        (rows["stuff_plus"] <= USAGE_DRAG_STUFF)
        & (rows["usage"] >= USAGE_DRAG_SHARE)
        & (rows["pitches"] >= 80)
    ]
    keeps = rows.loc[
        (rows["stuff_plus"] >= USAGE_KEEP_STUFF)
        & (rows["usage"] >= 0.05)
        & (rows["pitches"] >= 80)
    ]
    if len(drags) and len(keeps):
        drag = drags.sort_values("stuff_plus").iloc[0]
        keep = keeps.sort_values("stuff_plus", ascending=False).iloc[0]
        saved = usage_shift_runs(drag.pred_rv, keep.pred_rv, share=0.10)
        suggestions.append({
            "kind": "usage",
            "pitch_type": str(drag.pitch_type),
            "title": (
                f"Trim the {pitch_label(drag.pitch_type).lower()}, "
                f"lean on the {pitch_label(keep.pitch_type).lower()}"
            ),
            "detail": (
                f"The {pitch_label(drag.pitch_type).lower()} is {drag.usage * 100:.0f}% of the mix "
                f"at Stuff+ {drag.stuff_plus:.0f} ({drag.velo:.1f} mph, {drag.ivb:.1f} in IVB, "
                f"{drag.hb_arm:.1f} in arm-side break). The {pitch_label(keep.pitch_type).lower()} "
                f"is Stuff+ {keep.stuff_plus:.0f} at {keep.usage * 100:.0f}% usage. "
                f"Shifting 10 pitches out of every 100 from the "
                f"{pitch_label(drag.pitch_type).lower()} to the {pitch_label(keep.pitch_type).lower()} "
                f"is worth about {saved:.2f} runs per 100 pitches on the model's scale. "
                f"That is a ceiling. It ignores platoon splits and the counts where the "
                f"weaker pitch is already the right choice."
            ),
            "evidence": {
                "from_pitch": str(drag.pitch_type),
                "to_pitch": str(keep.pitch_type),
                "from_stuff": round(float(drag.stuff_plus), 1),
                "to_stuff": round(float(keep.stuff_plus), 1),
                "from_usage": round(float(drag.usage), 3),
                "runs_per_100_if_shift_10pct": round(saved, 3),
            },
        })

    shape_candidates = []
    for pitch in rows.itertuples(index=False):
        if pitch.stuff_plus >= SHAPE_STUFF_MAX or pitch.pitches < 80 or not np.isfinite(pitch.stuff_plus):
            continue
        pool = _target_pool(league, pitch.pitch_type, hand, float(pitch.velo), pitcher)
        if len(pool) < 8:
            continue
        target_ivb = float(pool["ivb"].median())
        target_hb = float(pool["hb_arm"].median())
        target_velo = float(pool["velo"].median())
        target_stuff = float(pool["stuff_plus"].median())
        ivb_gap = target_ivb - float(pitch.ivb)
        hb_gap = target_hb - float(pitch.hb_arm)
        if max(abs(ivb_gap), abs(hb_gap)) < SHAPE_GAP_INCHES:
            continue
        shape_candidates.append((max(abs(ivb_gap), abs(hb_gap)), pitch, {
            "target_velo": target_velo,
            "target_ivb": target_ivb,
            "target_hb": target_hb,
            "target_stuff": target_stuff,
            "peers": int(len(pool)),
            "ivb_gap": ivb_gap,
            "hb_gap": hb_gap,
        }))
    shape_candidates.sort(key=lambda item: item[0], reverse=True)
    for _, pitch, target in shape_candidates[:2]:
        move_bits = []
        if abs(target["ivb_gap"]) >= SHAPE_GAP_INCHES:
            direction = "more ride" if target["ivb_gap"] > 0 else "more depth"
            move_bits.append(f"{abs(target['ivb_gap']):.1f} in {direction}")
        if abs(target["hb_gap"]) >= SHAPE_GAP_INCHES:
            direction = "more arm-side run" if target["hb_gap"] > 0 else "more glove-side sweep"
            move_bits.append(f"{abs(target['hb_gap']):.1f} in {direction}")
        suggestions.append({
            "kind": "shape",
            "pitch_type": str(pitch.pitch_type),
            "title": f"Shape target for the {pitch_label(pitch.pitch_type).lower()}",
            "detail": (
                f"This {pitch_label(pitch.pitch_type).lower()} averages {pitch.velo:.1f} mph, "
                f"{pitch.ivb:.1f} in IVB, {pitch.hb_arm:.1f} in arm-side break, Stuff+ {pitch.stuff_plus:.0f}. "
                f"Same-handed versions within 2 to 4 mph that grade {SHAPE_TARGET_STUFF:.0f} or better "
                f"({target['peers']} pitcher-seasons) sit near {target['target_velo']:.1f} mph, "
                f"{target['target_ivb']:.1f} in IVB, {target['target_hb']:.1f} in arm-side break, "
                f"median Stuff+ {target['target_stuff']:.0f}. The gap is {' and '.join(move_bits)}. "
                f"That is a bullpen target, not a promise that closing the gap adds "
                f"{target['target_stuff'] - pitch.stuff_plus:.0f} points."
            ),
            "evidence": {
                "pitch_type": str(pitch.pitch_type),
                "velo": round(float(pitch.velo), 2),
                "ivb": round(float(pitch.ivb), 2),
                "hb_arm": round(float(pitch.hb_arm), 2),
                "stuff_plus": round(float(pitch.stuff_plus), 1),
                **{key: round(float(value), 2) if isinstance(value, float) else value for key, value in target.items()},
            },
        })

    established = set(rows.loc[rows["usage"] >= 0.05, "pitch_type"])
    missing = rows["fb_cluster"].iloc[0]
    n_established = len(established)
    if n_established <= 3 and pd.notna(missing):
        peers = league.loc[
            (league["fb_cluster"] == missing)
            & (league["p_throws"] == hand)
            & (league["pitcher"] != pitcher)
            & (league["season"] == season)
        ]
        peer_pitchers = peers.groupby("pitcher")["pitches"].sum()
        peer_pitchers = peer_pitchers.loc[peer_pitchers >= 200]
        peers = peers.loc[peers["pitcher"].isin(peer_pitchers.index)]
        if len(peer_pitchers) >= 12:
            best = None
            for pitch_type, group in peers.groupby("pitch_type"):
                if pitch_type in established:
                    continue
                users = group.loc[group["usage"] >= 0.08, "pitcher"].nunique()
                rate = users / len(peer_pitchers)
                graded = group.loc[(group["usage"] >= 0.08) & (group["pitches"] >= 50)]
                if graded.empty or rate < ADD_PITCH_PEER_RATE:
                    continue
                peer_stuff = float(graded["stuff_plus"].mean())
                if peer_stuff < ADD_PITCH_PEER_STUFF:
                    continue
                candidate = (rate, peer_stuff, pitch_type, graded, users)
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is not None:
                rate, peer_stuff, pitch_type, graded, users = best
                suggestions.append({
                    "kind": "add_pitch",
                    "pitch_type": str(pitch_type),
                    "title": f"Peer arsenals with this fastball often add a {pitch_label(pitch_type).lower()}",
                    "detail": (
                        f"{users} of {len(peer_pitchers)} same-handed pitchers in fastball shape cluster "
                        f"{int(missing)} throw a {pitch_label(pitch_type).lower()} at least 8% of the time "
                        f"({rate * 100:.0f}%). Those pitches average Stuff+ {peer_stuff:.0f}, "
                        f"near {graded['velo'].median():.1f} mph, {graded['ivb'].median():.1f} in IVB, "
                        f"{graded['hb_arm'].median():.1f} in arm-side break. This arsenal does not have one. "
                        f"A bullpen look is justified. A roster move is not, until the pitch exists."
                    ),
                    "evidence": {
                        "pitch_type": str(pitch_type),
                        "peer_pitchers": int(len(peer_pitchers)),
                        "peer_users": int(users),
                        "peer_rate": round(float(rate), 3),
                        "peer_stuff": round(peer_stuff, 1),
                        "target_velo": round(float(graded["velo"].median()), 2),
                        "target_ivb": round(float(graded["ivb"].median()), 2),
                        "target_hb": round(float(graded["hb_arm"].median()), 2),
                        "fb_cluster": int(missing),
                    },
                })

    if not suggestions:
        suggestions.append({
            "kind": "hold",
            "pitch_type": None,
            "title": "No model-backed change",
            "detail": (
                f"Usage-weighted Stuff+ is {arsenal_stuff:.0f} and no single pitch clears the "
                f"bar for a usage cut, a shape change, or a new pitch. Keep the mix."
            ),
            "evidence": {"stuff_plus": round(arsenal_stuff, 1)},
        })
    return suggestions


def suggest_all(arsenal: pd.DataFrame) -> dict[str, list[dict]]:
    """Suggestions keyed by ``season:pitcher`` for every pitcher in the table."""
    output: dict[str, list[dict]] = {}
    for (season, pitcher), rows in arsenal.groupby(["season", "pitcher"], sort=False):
        key = f"{int(season)}:{int(pitcher)}"
        output[key] = suggest_for_pitcher(rows, arsenal.loc[arsenal["season"] == season])
    return output
