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
    # V3: region path difficulty (tried, reverted — region-level constant, no within-region discrimination)
    ("region_top4_net_avg",    False),   # higher = weaker region top seeds = easier path
    # V4: R64 opponent quality
    ("opp_seed_rank_gap",      False),   # opponent's (net_rank - seed*10): higher = weaker opp than expected = easier path
]

FEATURE_NAMES = [f[0] for f in FEATURES]
FEATURE_NEGATE = [f[1] for f in FEATURES]

# ---------------------------------------------------------------------------
# V5 feature set — composite indices replace raw V2 sub-components
# ---------------------------------------------------------------------------
FEATURES_V5 = [
    ("seed_rank_gap",     False),  # weight < 0; lower = under-seeded = better
    ("net_rating",        False),  # weight > 0; higher adj net efficiency = better
    ("conf_tourney_wins", False),  # weight > 0; more conf tourney wins = hotter team
    ("cpi",               False),  # weight > 0; higher possession control = better
    ("dfi",               True),   # negate; lower DFI = harder to score on = better
    ("ftli",              False),  # weight > 0; higher FT leverage = better in close games
    ("spmi",              False),  # weight > 0; favorable shot profile mismatch = better
    ("tsi",               True),   # negate; lower TSI = more consistent tempo = better
]

FEATURE_NAMES_V5  = [f[0] for f in FEATURES_V5]
FEATURE_NEGATE_V5 = [f[1] for f in FEATURES_V5]

# Constrained DE bounds for V5: seed_rank_gap weight expected negative,
# all other post-negate features expected positive. Shrinks search space ~500x vs V2.
V5_BOUNDS = [
    (-3.0, 0.0),  # seed_rank_gap
    ( 0.0, 3.0),  # net_rating
    ( 0.0, 3.0),  # conf_tourney_wins
    ( 0.0, 3.0),  # cpi
    ( 0.0, 3.0),  # dfi  (negated before scoring)
    ( 0.0, 3.0),  # ftli
    ( 0.0, 3.0),  # spmi
    ( 0.0, 3.0),  # tsi  (negated before scoring)
]

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
        ),
        r64_opponents AS (
            -- One row per (season, region, seed) via GROUP BY.
            -- For most seed slots there is exactly one team, so AVG = that team's value.
            -- For First Four seed slots (where two teams share the same region/seed),
            -- AVG collapses them to a single pre-tournament quality estimate — the
            -- correct treatment when the actual winner is unknown at selection time.
            SELECT
                te3.season,
                te3.region,
                te3.seed AS opp_seed,
                AVG(m3.seed_rank_gap) AS opp_seed_rank_gap
            FROM mm_tournament_entries te3
            JOIN mm_team_metrics m3
              ON m3.team_id = te3.team_id AND m3.season = te3.season
            GROUP BY te3.season, te3.region, te3.seed
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
            rs.region_top4_net_avg,
            r64.opp_seed_rank_gap,
            -- V5 raw sub-components
            m.opp_tov_pct,
            m.dreb_pct,
            m.opp_3p_pct,
            m.opp_foul_rate,
            m.team_3p_rate,
            m.opp_3p_rate,
            m.team_rim_rate,
            m.tsi
        FROM mm_tournament_entries te
        JOIN mm_teams t ON t.id = te.team_id
        LEFT JOIN mm_team_metrics m ON m.team_id = te.team_id AND m.season = te.season
        LEFT JOIN region_strength rs ON rs.season = te.season AND rs.region = te.region
        LEFT JOIN r64_opponents r64
          ON r64.season = te.season
         AND r64.region = te.region
         AND r64.opp_seed = CASE te.seed
               WHEN 5  THEN 12 WHEN 12 THEN 5
               WHEN 6  THEN 11 WHEN 11 THEN 6
               WHEN 7  THEN 10 WHEN 10 THEN 7
               WHEN 8  THEN 9  WHEN 9  THEN 8
               ELSE NULL END
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


def compute_composite_features(teams: list[dict]) -> list[dict]:
    """
    Compute V5 composite features from raw sub-components stored in DB.
    Adds cpi, dfi, ftli, spmi to each team dict (tsi is pre-computed in DB).

    Formulas:
      CPI  = (opp_tov_pct - tov_ratio) + oreb_pct + dreb_pct
      DFI  = opp_efg_pct + opp_3p_pct
      FTLI = z(ft_rate) + z(ft_pct) + z(opp_foul_rate)   [z within eligible pool]
      SPMI = (team_rim_rate - team_3p_rate) + opp_3p_rate

    Modifies and returns the list in place.
    """
    def _f(team, key):
        v = team.get(key)
        return float(v) if v is not None else float("nan")

    # CPI and DFI are simple arithmetic — no cross-team normalization needed
    for t in teams:
        opp_tov  = _f(t, "opp_tov_pct")
        tov      = _f(t, "tov_ratio")
        oreb     = _f(t, "oreb_pct")
        dreb     = _f(t, "dreb_pct")
        opp_efg  = _f(t, "opp_efg_pct")
        opp_3p   = _f(t, "opp_3p_pct")
        rim_rate = _f(t, "team_rim_rate")
        t3p_rate = _f(t, "team_3p_rate")
        opp_3pr  = _f(t, "opp_3p_rate")

        # tov_ratio and opp_tov_pct are decimal (~0.17-0.25); oreb/dreb are percentage
        # form (~25-75). Multiply tov net by 100 to put all terms in percentage scale.
        t["cpi"]  = ((opp_tov - tov) * 100.0) + oreb + dreb
        t["dfi"]  = opp_efg + opp_3p
        t["spmi"] = (rim_rate - t3p_rate) + opp_3pr
        # tsi is already in the dict from DB; ftli computed below after z-scoring

    # FTLI: z-score each sub-component within this season's eligible pool, then sum
    for key in ("ft_rate", "ft_pct", "opp_foul_rate"):
        vals = np.array([_f(t, key) for t in teams], dtype=float)
        mean = np.nanmean(vals)
        std  = np.nanstd(vals)
        z = (vals - mean) / std if std > 0 else vals - mean
        for i, t in enumerate(teams):
            t.setdefault("_ftli_z_" + key, 0.0)
            t["_ftli_z_" + key] = 0.0 if np.isnan(z[i]) else float(z[i])

    for t in teams:
        t["ftli"] = (
            t.get("_ftli_z_ft_rate",      0.0)
            + t.get("_ftli_z_ft_pct",     0.0)
            + t.get("_ftli_z_opp_foul_rate", 0.0)
        )
        # Clean up temp keys
        for key in ("_ftli_z_ft_rate", "_ftli_z_ft_pct", "_ftli_z_opp_foul_rate"):
            t.pop(key, None)

    return teams


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


# ---------------------------------------------------------------------------
# V5 feature matrix and scoring
# ---------------------------------------------------------------------------

def build_feature_matrix_v5(teams: list[dict]) -> np.ndarray:
    """
    Build and z-score normalize the V5 feature matrix.
    Assumes composite features (cpi, dfi, ftli, spmi) are already computed
    and present in each team dict; tsi is loaded from DB.
    Returns array of shape (n_teams, n_features_v5).
    """
    n = len(teams)
    k = len(FEATURES_V5)
    X = np.zeros((n, k))

    for j, (feat, negate) in enumerate(FEATURES_V5):
        vals = np.array([t.get(feat) if t.get(feat) is not None else np.nan
                         for t in teams], dtype=float)
        if negate:
            vals = -vals
        mean = np.nanmean(vals)
        std  = np.nanstd(vals)
        if std > 0:
            vals = (vals - mean) / std
        else:
            vals = vals - mean
        vals = np.where(np.isnan(vals), 0.0, vals)
        X[:, j] = vals

    return X


def compute_scores_v5(teams: list[dict], weights: np.ndarray) -> list[float]:
    """
    Compute V5 composite scores. Calls compute_composite_features first to
    populate cpi, dfi, ftli, spmi from raw sub-components in each team dict.
    weights: array of shape (n_features_v5,)
    """
    compute_composite_features(teams)
    X = build_feature_matrix_v5(teams)
    return (X @ weights).tolist()


def select_picks_v5(teams: list[dict], weights: np.ndarray, n_picks: int = 8) -> list[dict]:
    """
    Score all eligible teams with V5 model and select top n_picks.
    Calls compute_composite_features first.
    Applies R64 collision guard (same logic as V2).
    """
    scores = compute_scores_v5(teams, weights)
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


def select_picks_threshold_v5(
    teams: list[dict],
    weights: np.ndarray,
    threshold: float,
    min_picks: int = MIN_PICKS_PER_YEAR,
    max_picks: int = MAX_PICKS_PER_YEAR,
) -> list[dict]:
    """
    Variable-N threshold selection with V5 model.
    Calls compute_composite_features first.
    """
    scores = compute_scores_v5(teams, weights)
    scores_arr = np.array(scores, dtype=float)

    mean = np.mean(scores_arr)
    std  = np.std(scores_arr)
    z_scores = (scores_arr - mean) / std if std > 0 else np.zeros_like(scores_arr)

    order = np.argsort(-scores_arr)

    picks = []
    for idx in order:
        if len(picks) >= max_picks:
            break
        z = float(z_scores[idx])
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


# ---------------------------------------------------------------------------
# V6 feature sets — hybrid 4-feature models
# ---------------------------------------------------------------------------
FEATURES_V6A = [
    ("seed_rank_gap",     False),  # weight < 0; lower = under-seeded = better
    ("conf_tourney_wins", False),  # weight > 0; more wins = hotter team
    ("opp_efg_pct",       True),   # negate; lower opp eFG% = better defense
    ("tsi",               True),   # negate; lower TSI = more consistent tempo
]
FEATURES_V6B = [
    ("seed_rank_gap",     False),  # weight < 0; lower = under-seeded = better
    ("conf_tourney_wins", False),  # weight > 0; more wins = hotter team
    ("dfi",               True),   # negate; lower DFI = harder to score on
    ("tsi",               True),   # negate; lower TSI = more consistent tempo
]

FEATURE_NAMES_V6A  = [f[0] for f in FEATURES_V6A]
FEATURE_NAMES_V6B  = [f[0] for f in FEATURES_V6B]

V6A_BOUNDS = [
    (-3.0, 0.0),  # seed_rank_gap
    ( 0.0, 3.0),  # conf_tourney_wins
    ( 0.0, 3.0),  # opp_efg_pct (negated before scoring)
    ( 0.0, 3.0),  # tsi (negated before scoring)
]
V6B_BOUNDS = [
    (-3.0, 0.0),  # seed_rank_gap
    ( 0.0, 3.0),  # conf_tourney_wins
    ( 0.0, 3.0),  # dfi (negated before scoring)
    ( 0.0, 3.0),  # tsi (negated before scoring)
]

# Pre-specified geomean weights for v6-fixed-8-geomean (coreB feature set).
# Derived from single-feature model results: w_i = geomean(SFM_val_i, SFM_test_i),
# then normalized so weights sum to 1.
#   seed_rank_gap:     val=+6.52u  test=+5.83u  geomean=6.165 -> w=0.2795
#   conf_tourney_wins: val=+9.55u  test=+5.27u  geomean=7.094 -> w=0.3217
#   dfi:               val=+6.95u  test=+2.87u  geomean=4.466 -> w=0.2025
#   tsi:               val=+5.21u  test=+3.60u  geomean=4.331 -> w=0.1963
_V6_GM = np.array([
    np.sqrt(6.52 * 5.83),   # seed_rank_gap
    np.sqrt(9.55 * 5.27),   # conf_tourney_wins
    np.sqrt(6.95 * 2.87),   # dfi
    np.sqrt(5.21 * 3.60),   # tsi
])
V6_GEOMEAN_W = _V6_GM / _V6_GM.sum()


def _build_feature_matrix_v6(teams: list[dict], features: list) -> np.ndarray:
    """Z-score-normalized feature matrix for a V6 variant."""
    n = len(teams)
    X = np.zeros((n, len(features)))
    for j, (feat, negate) in enumerate(features):
        vals = np.array([t.get(feat) if t.get(feat) is not None else np.nan
                         for t in teams], dtype=float)
        if negate:
            vals = -vals
        mean = np.nanmean(vals)
        std  = np.nanstd(vals)
        if std > 0:
            vals = (vals - mean) / std
        else:
            vals = vals - mean
        X[:, j] = np.where(np.isnan(vals), 0.0, vals)
    return X


def select_picks_v6(
    teams: list[dict],
    weights: np.ndarray,
    variant: str,
    n_picks: int = 8,
) -> list[dict]:
    """
    Score all eligible teams with a V6 hybrid model and select top n_picks.
    variant: 'coreA' (opp_efg_pct) or 'coreB' (dfi; calls compute_composite_features).
    Applies R64 collision guard.
    """
    if variant == "coreB":
        compute_composite_features(teams)
    features = FEATURES_V6A if variant == "coreA" else FEATURES_V6B
    scores = (_build_feature_matrix_v6(teams, features) @ weights).tolist()
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
