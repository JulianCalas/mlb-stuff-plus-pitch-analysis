-- Queries the pipeline actually runs. DuckDB reads the cached parquet
-- and the arsenal table the model just wrote. Names after "-- name:"
-- are the keys the Python loader uses.

-- name: season_inventory
-- How much regular-season Statcast made it into the cache.
SELECT
    game_year AS season,
    COUNT(*) AS pitches,
    COUNT(DISTINCT pitcher) AS pitchers,
    MIN(game_date) AS first_date,
    MAX(game_date) AS last_date
FROM read_parquet('{raw_glob}')
GROUP BY 1
ORDER BY 1;

-- name: whiff_by_pitch_type
-- Raw whiff rate by the official label. Swings only. This is the
-- outcome the whiff model is calibrated against, computed in SQL
-- so the rate does not depend on the pandas feature frame.
SELECT
    pitch_type,
    COUNT(*) AS swings,
    AVG(
        CASE
            WHEN description IN ('swinging_strike', 'swinging_strike_blocked', 'foul_tip') THEN 1.0
            ELSE 0.0
        END
    ) AS whiff_rate
FROM read_parquet('{raw_glob}')
WHERE pitch_type IN ('FF', 'SI', 'FC', 'SL', 'ST', 'SV', 'CU', 'KC', 'CH', 'FS')
  AND description IN (
      'swinging_strike', 'swinging_strike_blocked', 'foul_tip', 'foul',
      'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'
  )
GROUP BY 1
ORDER BY whiff_rate DESC;

-- name: leaderboard
-- Published Stuff+ for the test season. 100 is average among pitcher-seasons
-- with at least 100 of that pitch; the board itself requires 200.
SELECT
    player_name,
    team,
    p_throws,
    pitch_type,
    season,
    pitches,
    usage,
    velo,
    ivb,
    hb_arm,
    vaa,
    whiff_rate,
    stuff_plus,
    actual_rv * 100 AS actual_rv_per_100,
    pred_rv * 100 AS pred_rv_per_100
FROM arsenal
WHERE season = {test_season}
  AND pitches >= {min_pitches}
  AND stuff_plus IS NOT NULL
ORDER BY stuff_plus DESC;

-- name: focus_club
-- Every graded pitch thrown by the focus club in the test season.
SELECT *
FROM arsenal
WHERE team = '{focus_team}'
  AND season = {test_season}
ORDER BY pitches DESC;
