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
from config import ELIGIBLE_SEEDS

logger = logging.getLogger(__name__)

# Feature definitions: (column_name, negate)
# negate=True for features where lower = better
FEATURES = [
    ("seed_rank_gap", False),   # negative = under-seeded; lower = better for us, but we negate the sign conceptually
    ("def_rank",      True),    # lower rank number = better defense; negate so higher score = better
    ("opp_efg_pct",   True),    # lower opponent eFG = better defense
    ("net_rating",    False),   # higher = better
    ("tov_ratio",     True),    # lower = fewer turnovers
    ("ft_pct",        False),   # higher = better clutch shooting
    ("oreb_pct",      False),   # higher = more second chances
    ("pace",          False),   # mid-pace — use raw value, model learns optimal weight
]

FEATURE_NAMES = [f[0] for f in FEATURES]
FEATURE_NEGATE = [f[1] for f in FEATURES]


def load_eligible_teams(conn: sqlite3.Connection, season: int) -> list[dict]:
    """
    Load all seeds 5–12 teams for a season with their metrics.
    Returns list of dicts with team info + feature values.
    """
    seed_placeholders = ",".join("?" * len(ELIGIBLE_SEEDS))
    rows = conn.execute(
        f"""
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
            m.seed_rank_gap
        FROM mm_tournament_entries te
        JOIN mm_teams t ON t.id = te.team_id
        LEFT JOIN mm_team_metrics m ON m.team_id = te.team_id AND m.season = te.season
        WHERE te.season = ?
          AND te.seed IN ({seed_placeholders})
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
    Returns list of team dicts with 'model_score' and 'pick_rank' added.
    """
    scores = compute_scores(teams, weights)
    indexed = sorted(zip(scores, range(len(teams))), reverse=True)

    picks = []
    for rank, (score, idx) in enumerate(indexed[:n_picks], start=1):
        pick = dict(teams[idx])
        pick["model_score"] = round(score, 6)
        pick["pick_rank"] = rank
        picks.append(pick)

    return picks
