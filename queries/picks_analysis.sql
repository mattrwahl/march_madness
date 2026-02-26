-- ============================================================
-- March Madness — Historical Picks Analysis
-- Paste individual blocks into Superset SQL Lab
-- DB connection: sqlite:////app/mm_data/march_madness.db
-- ============================================================


-- ----------------------------------------------------------------
-- 1. SEASON SUMMARY: units won per year
-- ----------------------------------------------------------------
SELECT
    p.season,
    COUNT(*)                                                    AS picks_made,
    SUM(CASE WHEN p.status = 'cashed_out' THEN 1 ELSE 0 END)   AS reached_e8,
    SUM(CASE WHEN p.status = 'eliminated' THEN 1 ELSE 0 END)   AS eliminated,
    ROUND(SUM(COALESCE(p.units_won, 0)), 3)                     AS total_units,
    ROUND(AVG(COALESCE(p.units_won, 0)), 3)                     AS avg_units_per_pick,
    ROUND(SUM(COALESCE(p.payout_dollars, 0)), 2)                AS total_payout_dollars
FROM mm_model_picks p
GROUP BY p.season
ORDER BY p.season;


-- ----------------------------------------------------------------
-- 2. PICK-LEVEL DETAIL: all picks with feature scores
-- ----------------------------------------------------------------
SELECT
    p.season,
    p.pick_rank,
    t.name                              AS team,
    p.seed,
    m.net_rank,
    m.seed_rank_gap,
    ROUND(m.net_rating, 2)              AS net_rating,
    ROUND(m.adj_def_rating, 2)          AS adj_def,
    ROUND(m.opp_efg_pct * 100, 1)      AS opp_efg_pct,
    ROUND(m.ft_pct * 100, 1)           AS ft_pct,
    ROUND(p.model_score, 3)             AS model_score,
    p.status,
    p.round_exit,
    ROUND(COALESCE(p.payout_dollars, 0), 2) AS payout,
    ROUND(COALESCE(p.units_won, 0), 3)      AS units_won
FROM mm_model_picks p
JOIN mm_teams t ON t.id = p.team_id
LEFT JOIN mm_team_metrics m ON m.team_id = p.team_id AND m.season = p.season
ORDER BY p.season DESC, p.pick_rank;


-- ----------------------------------------------------------------
-- 3. FEATURE IMPORTANCE: correlation of each feature with units won
-- (Spearman-style ranking comparison — positive = good predictor)
-- ----------------------------------------------------------------
SELECT
    'seed_rank_gap'     AS feature,
    ROUND(AVG(CASE WHEN m.seed_rank_gap < 0 THEN p.units_won END), 3) AS avg_units_negative_gap,
    ROUND(AVG(CASE WHEN m.seed_rank_gap >= 0 THEN p.units_won END), 3) AS avg_units_positive_gap
FROM mm_model_picks p
JOIN mm_team_metrics m ON m.team_id = p.team_id AND m.season = p.season
WHERE p.units_won IS NOT NULL

UNION ALL

SELECT
    'elite_defense (def_rank <= 30)' AS feature,
    ROUND(AVG(CASE WHEN m.def_rank <= 30 THEN p.units_won END), 3) AS elite_defense,
    ROUND(AVG(CASE WHEN m.def_rank > 30 THEN p.units_won END), 3) AS non_elite_defense
FROM mm_model_picks p
JOIN mm_team_metrics m ON m.team_id = p.team_id AND m.season = p.season
WHERE p.units_won IS NOT NULL;


-- ----------------------------------------------------------------
-- 4. ROUND EXIT DISTRIBUTION
-- ----------------------------------------------------------------
SELECT
    CASE g.round
        WHEN 64 THEN 'R64 (First Round)'
        WHEN 32 THEN 'R32 (Second Round)'
        WHEN 16 THEN 'S16 (Sweet Sixteen)'
        WHEN  8 THEN 'E8 (Elite Eight)'
        WHEN  4 THEN 'F4 (Final Four)'
        WHEN  2 THEN 'Champ'
        ELSE 'Unknown'
    END                                 AS round_exited,
    g.round                             AS round_number,
    COUNT(*)                            AS picks,
    ROUND(AVG(p.units_won), 3)          AS avg_units
FROM mm_model_picks p
JOIN mm_teams t ON t.id = p.team_id
JOIN mm_games g ON g.season = p.season
    AND (g.team1_id = p.team_id OR g.team2_id = p.team_id)
    AND g.winner_id != p.team_id        -- last game played (lost here)
WHERE p.round_exit IS NOT NULL
GROUP BY g.round
ORDER BY g.round DESC;


-- ----------------------------------------------------------------
-- 5. STORED MODEL WEIGHTS (most recent training run)
-- ----------------------------------------------------------------
SELECT
    trained_at,
    train_seasons,
    val_seasons,
    ROUND(w_seed_rank_gap, 4)   AS w_seed_rank_gap,
    ROUND(w_def_rank, 4)        AS w_def_rank,
    ROUND(w_opp_efg_pct, 4)    AS w_opp_efg_pct,
    ROUND(w_net_rating, 4)      AS w_net_rating,
    ROUND(w_tov_ratio, 4)       AS w_tov_ratio,
    ROUND(w_ft_pct, 4)          AS w_ft_pct,
    ROUND(w_oreb_pct, 4)        AS w_oreb_pct,
    ROUND(w_pace, 4)            AS w_pace,
    ROUND(train_units_won, 3)   AS train_units,
    ROUND(val_units_won, 3)     AS val_units
FROM mm_model_weights
ORDER BY trained_at DESC
LIMIT 5;
