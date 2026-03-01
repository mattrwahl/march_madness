"""
Feature engineering for the March Madness selection model.
Computes composite selection scores for seeds 5–12.
"""
import sys
import logging
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import ELIGIBLE_SEEDS, MIN_PICKS_PER_YEAR, MAX_PICKS_PER_YEAR

logger = logging.getLogger(__name__)

# Feature definitions: (column_name, negate)
# negate=True for features where lower = better
FEATURES = [
    ("seed_rank_gap",          False),   # negative = under-seeded; lower = better for us, but we negate the sign conceptually
    ("def_rank",               True),    # lower rank number = better defense; negate so higher score = better
    ("opp_efg_pct",            True),    # lower opponent eFG = better defense
    ("net_rating",             False),   # higher = better
    ("tov_ratio",              True),    # lower = fewer turnovers
    ("ft_pct",                 False),   # higher = better clutch shooting
    ("oreb_pct",               False),   # higher = more second chances
    ("pace",                   False),   # mid-pace — use raw value, model learns optimal weight
    # V2: conference tournament momentum features
    ("conf_tourney_wins",      False),   # more conf tourney wins = hotter team
    ("conf_tourney_avg_margin", False),  # higher avg margin = dominant momentum
    # V3: region path difficulty
    ("region_top4_net_avg",    False),   # higher = weaker region top seeds = easier path
]

FEATURE_NAMES = [f[0] for f in FEATURES]
FEATURE_NEGATE = [f[1] for f in FEATURES]

# R64 seed matchup pairs: both teams from same region with these seeds play each other.
# Seeds 5-12 R64 opponents: 5v12, 6v11, 7v10, 8v9.
_R64_OPPONENT_SEED = {5: 12, 12: 5, 6: 11, 11: 6, 7: 10, 10: 7, 8: 9, 9: 8}


def _r64_collision(candidate: dict, existing_picks: list[dict]) -> bool:
    """
    Return True if adding `candidate` would guarantee a R64 matchup against
    an already-selected pick (same region, complementary seed pair).
    """
    c_seed   = candidate.get("seed")
    c_region = candidate.get("region")
    opp_seed = _R64_OPPONENT_SEED.get(c_seed)
    if opp_seed is None or not c_region:
        return False
    return any(
        p.get("seed") == opp_seed and p.get("region") == c_region
        for p in existing_picks
    )


def load_eligible_teams(conn: sqlite3.Connection, season: int) -> list[dict]:
    """
    Load all seeds 5–12 teams for a season with their metrics.
    Returns list of dicts with team info + feature values.
    """
    seed_placeholders = ",".join("?" * len(ELIGIBLE_SEEDS))
    rows = conn.execute(
        f"""
        WITH region_strength AS (
            SELECT
                te2.season,
                te2.region,
                AVG(m2.net_rank) AS region_top4_net_avg
            FROM mm_tournament_entries te2
            JOIN mm_team_metrics m2
              ON m2.team_id = te2.team_id AND m2.season = te2.season
            WHERE te2.seed <= 4
            GROUP BY te2.season, te2.region
        )
        SELECT
            te.team_id,
            te.seed,
            te.region,
            t.name as team_name,
            m.adj_off_rating,
            m.adj_def_rating,
            m.net_rating,
            m.net_rank,
            m.def_rank,
            m.efg_pct,
            m.opp_efg_pct,
            m.tov_ratio,
            m.oreb_pct,
            m.ft_rate,
            m.ft_pct,
            m.pace,
            m.seed_rank_gap,
            m.conf_tourney_wins,
            m.conf_tourney_avg_margin,
            rs.region_top4_net_avg
        FROM mm_tournament_entries te
        JOIN mm_teams t ON t.id = te.team_id
        LEFT JOIN mm_team_metrics m ON m.team_id = te.team_id AND m.season = te.season
        LEFT JOIN region_strength rs ON rs.season = te.season AND rs.region = te.region
        WHERE te.season = ?
          AND te.seed IN ({seed_placeholders})
          AND te.team_id NOT IN (
              SELECT team1_id FROM mm_games g WHERE g.season = te.season AND g.round = 65
              UNION
              SELECT team2_id FROM mm_games g WHERE g.season = te.season AND g.round = 65
          )
        ORDER BY te.seed, t.name
        """,
        [season] + list(ELIGIBLE_SEEDS),
    ).fetchall()

    return [dict(r) for r in rows]


def build_feature_matrix(teams: list[dict]) -> np.ndarray:
    """
    Build and z-score normalize the feature matrix.
    Returns array of shape (n_teams, n_features).
    Negates features where lower = better before normalization.
    Missing values are replaced with 0 (mean after normalization).
    """
    n = len(teams)
    k = len(FEATURES)
    X = np.zeros((n, k))

    for j, (feat, negate) in enumerate(FEATURES):
        vals = np.array([t.get(feat) if t.get(feat) is not None else np.nan
                         for t in teams], dtype=float)
        if negate:
            vals = -vals
        # Z-score normalize (ignore NaN for mean/std)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if std > 0:
            vals = (vals - mean) / std
        else:
            vals = vals - mean
        # Replace NaN with 0 (mean after normalization)
        vals = np.where(np.isnan(vals), 0.0, vals)
        X[:, j] = vals

    return X


def compute_scores(teams: list[dict], weights: np.ndarray) -> list[float]:
    """
    Compute composite selection score for each team given weights.
    weights: array of shape (n_features,)
    Returns list of floats, one per team.
    """
    X = build_feature_matrix(teams)
    scores = X @ weights
    return scores.tolist()


def select_picks(teams: list[dict], weights: np.ndarray, n_picks: int = 8) -> list[dict]:
    """
    Score all eligible teams and select the top n_picks.
    Skips candidates that would face an already-selected pick in R64 (collision guard).
    Returns list of team dicts with 'model_score' and 'pick_rank' added.
    """
    scores = compute_scores(teams, weights)
    indexed = sorted(zip(scores, range(len(teams))), reverse=True)

    picks = []
    for score, idx in indexed:
        if len(picks) >= n_picks:
            break
        team = teams[idx]
        if _r64_collision(team, picks):
            continue
        pick = dict(team)
        pick["model_score"] = round(score, 6)
        pick["pick_rank"]   = len(picks) + 1
        picks.append(pick)

    return picks


def select_picks_threshold(
    teams: list[dict],
    weights: np.ndarray,
    threshold: float,
    min_picks: int = MIN_PICKS_PER_YEAR,
    max_picks: int = MAX_PICKS_PER_YEAR,
) -> list[dict]:
    """
    Variable-N selection: pick all teams whose composite score z-score >= threshold.

    The z-score is computed within the eligible field for that season, so the
    threshold is self-normalizing year-to-year. Teams are always sorted by score;
    the threshold determines where we stop adding picks.

    Enforcements:
      - If fewer than min_picks clear the threshold, take the top min_picks anyway.
      - Never exceed max_picks regardless of how many clear the threshold.

    Returns list of team dicts with 'model_score', 'zscore', and 'pick_rank' added.
    """
    scores = compute_scores(teams, weights)
    scores_arr = np.array(scores, dtype=float)

    mean = np.mean(scores_arr)
    std  = np.std(scores_arr)
    z_scores = (scores_arr - mean) / std if std > 0 else np.zeros_like(scores_arr)

    order = np.argsort(-scores_arr)  # descending by score

    picks = []
    for idx in order:
        if len(picks) >= max_picks:
            break
        z = float(z_scores[idx])
        # Once we have min_picks, only continue if z-score clears the threshold
        if len(picks) >= min_picks and z < threshold:
            break
        team = teams[idx]
        if _r64_collision(team, picks):
            continue
        pick = dict(team)
        pick["model_score"] = round(float(scores_arr[idx]), 6)
        pick["zscore"]      = round(z, 4)
        pick["pick_rank"]   = len(picks) + 1
        picks.append(pick)

    return picks
