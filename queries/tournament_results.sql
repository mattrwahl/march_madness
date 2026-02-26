-- ============================================================
-- March Madness — Live Tournament Results (2026)
-- Paste individual blocks into Superset SQL Lab
-- DB connection: sqlite:////app/mm_data/march_madness.db
-- ============================================================

-- Change this to your target season
-- (Superset: use a template variable {{ season }} or hardcode)
-- SET @season = 2026;


-- ----------------------------------------------------------------
-- 1. LIVE PICK TRACKER — current status of all 8 picks
-- ----------------------------------------------------------------
SELECT
    p.pick_rank                             AS '#',
    t.name                                  AS team,
    p.seed,
    te.region,
    p.model_score,
    p.status,
    CASE p.round_exit
        WHEN 64 THEN 'Lost R64'
        WHEN 32 THEN 'Lost R32'
        WHEN 16 THEN 'Lost S16'
        WHEN  8 THEN 'Cashed E8'
        WHEN  4 THEN 'Cashed F4'
        WHEN  2 THEN 'Won!'
        ELSE 'Active'
    END                                     AS result,
    ROUND(p.initial_bet_dollars, 2)         AS initial_bet,
    ROUND(COALESCE(p.payout_dollars, p.initial_bet_dollars), 2) AS current_value,
    ROUND(COALESCE(p.units_won, 0), 3)      AS units_won
FROM mm_model_picks p
JOIN mm_teams t ON t.id = p.team_id
LEFT JOIN mm_tournament_entries te ON te.team_id = p.team_id AND te.season = p.season
WHERE p.season = 2026
ORDER BY p.pick_rank;


-- ----------------------------------------------------------------
-- 2. EQUITY CURVE — cumulative units over time (by game date)
-- ----------------------------------------------------------------
WITH pick_games AS (
    SELECT
        p.team_id,
        p.pick_rank,
        t.name          AS team_name,
        g.game_date,
        g.round,
        CASE WHEN g.winner_id = p.team_id THEN 1 ELSE 0 END AS won,
        CASE
            WHEN g.team1_id = p.team_id THEN bl.team1_moneyline
            ELSE bl.team2_moneyline
        END             AS moneyline
    FROM mm_model_picks p
    JOIN mm_teams t ON t.id = p.team_id
    JOIN mm_games g ON g.season = p.season
        AND (g.team1_id = p.team_id OR g.team2_id = p.team_id)
    LEFT JOIN mm_betting_lines bl ON bl.game_id = g.id
    WHERE p.season = 2026
)
SELECT
    game_date,
    round,
    team_name,
    pick_rank,
    won,
    moneyline,
    CASE
        WHEN won = 1 AND moneyline > 0 THEN 25.0 * (moneyline / 100.0)
        WHEN won = 1 AND moneyline < 0 THEN 25.0 * (100.0 / ABS(moneyline))
        WHEN won = 1 THEN 25.0  -- even money fallback
        ELSE -25.0              -- loss (initial bet)
    END AS units_delta_approx
FROM pick_games
ORDER BY game_date, round DESC;


-- ----------------------------------------------------------------
-- 3. GAME LOG — every tournament game involving our picks
-- ----------------------------------------------------------------
SELECT
    g.game_date,
    CASE g.round
        WHEN 64 THEN 'R64' WHEN 32 THEN 'R32'
        WHEN 16 THEN 'S16' WHEN  8 THEN 'E8'
        WHEN  4 THEN 'F4'  WHEN  2 THEN 'Champ'
    END                     AS round,
    t1.name                 AS team1,
    g.team1_score           AS score1,
    g.team2_score           AS score2,
    t2.name                 AS team2,
    winner.name             AS winner,
    -- Flag if either team is one of our picks
    CASE WHEN p1.id IS NOT NULL THEN '#' || p1.pick_rank ELSE '' END AS pick1,
    CASE WHEN p2.id IS NOT NULL THEN '#' || p2.pick_rank ELSE '' END AS pick2,
    bl.team1_moneyline      AS ml_team1,
    bl.team2_moneyline      AS ml_team2
FROM mm_games g
JOIN mm_teams t1 ON t1.id = g.team1_id
JOIN mm_teams t2 ON t2.id = g.team2_id
LEFT JOIN mm_teams winner ON winner.id = g.winner_id
LEFT JOIN mm_betting_lines bl ON bl.game_id = g.id
LEFT JOIN mm_model_picks p1 ON p1.team_id = g.team1_id AND p1.season = g.season
LEFT JOIN mm_model_picks p2 ON p2.team_id = g.team2_id AND p2.season = g.season
WHERE g.season = 2026
  AND g.tournament = 'NCAA'
  AND (p1.id IS NOT NULL OR p2.id IS NOT NULL)
ORDER BY g.game_date, g.round DESC;


-- ----------------------------------------------------------------
-- 4. TOTAL UNITS WON (headline number)
-- ----------------------------------------------------------------
SELECT
    2026                                        AS season,
    COUNT(*)                                    AS picks_made,
    SUM(CASE WHEN status = 'cashed_out' THEN 1 ELSE 0 END)  AS cashed_out,
    SUM(CASE WHEN status = 'active'     THEN 1 ELSE 0 END)  AS still_active,
    SUM(CASE WHEN status = 'eliminated' THEN 1 ELSE 0 END)  AS eliminated,
    ROUND(SUM(COALESCE(units_won, 0)), 3)       AS total_units_won,
    ROUND(SUM(COALESCE(payout_dollars, 0)), 2)  AS total_payout_dollars
FROM mm_model_picks
WHERE season = 2026;
