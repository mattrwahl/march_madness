"""
Feature diagnostics for V5 model development.

Three analyses on combined V2 raw + V5 composite features:

  1. Feature-to-feature Pearson correlation matrix (train seasons only)
     Full 15-feature matrix (10 V2 raw + 5 V5 composites) plus a focused
     cross-block showing how composites correlate with the V2 features they
     are intended to replace.

  2. Feature-to-target Pearson correlations
     Target A: units won per eligible team (payout-based, collision ignored)
     Target B: R64 win probability residual = actual_win - novig_implied_prob
     Sorted by |r| to show strongest individual predictors.

  3. Single-feature V5 models
     For each of the 8 V5 features: retrain with only that feature active,
     all others locked to 0. Uses constrained V5_BOUNDS for the active feature.
     Reports train / val / test units and compares to V2 and full V5 baselines.

Usage:
  python feature_diagnostics.py            # all three analyses
  python feature_diagnostics.py --corr     # correlation analyses only (no training)
  python feature_diagnostics.py --sfm      # single-feature models only
  python feature_diagnostics.py --fast     # faster DE for SFM (maxiter=100, popsize=10)
"""
import sys
import argparse
import sqlite3
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.optimize import differential_evolution

from processors.features import (
    FEATURE_NAMES_V5, V5_BOUNDS,
    load_eligible_teams, compute_composite_features, build_feature_matrix_v5,
)
from processors.model import simulate_pick_payout, INITIAL_BET_DOLLARS
from config import (
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    DEFAULT_CASH_OUT_ROUND, DEFAULT_BET_STYLE,
    NUM_PICKS, L2_LAMBDA, DB_PATH,
)

# ---------------------------------------------------------------------------
# Combined feature list for correlation analysis
# V2 raw (10) + V5 new composites (5)
# ---------------------------------------------------------------------------
V2_RAW_FEATURES = [
    "seed_rank_gap", "def_rank", "opp_efg_pct", "net_rating",
    "tov_ratio", "ft_pct", "oreb_pct", "pace",
    "conf_tourney_wins", "conf_tourney_avg_margin",
]
V5_NEW_COMPOSITES = ["cpi", "dfi", "ftli", "spmi", "tsi"]
ALL_CORR_FEATURES = V2_RAW_FEATURES + V5_NEW_COMPOSITES

# Abbreviated display names (max 8 chars) for compact correlation matrix
_ABBREV = {
    "seed_rank_gap":          "srg",
    "def_rank":               "def_r",
    "opp_efg_pct":            "oefg",
    "net_rating":             "net",
    "tov_ratio":              "tov",
    "ft_pct":                 "ft%",
    "oreb_pct":               "oreb",
    "pace":                   "pace",
    "conf_tourney_wins":      "ctw",
    "conf_tourney_avg_margin":"ctm",
    "cpi":                    "CPI",
    "dfi":                    "DFI",
    "ftli":                   "FTLI",
    "spmi":                   "SPMI",
    "tsi":                    "TSI",
}

# Which V2 features each composite is designed to replace/improve
_COMPOSITE_ANCESTRY = {
    "cpi":  ["tov_ratio", "oreb_pct"],
    "dfi":  ["def_rank", "opp_efg_pct"],
    "ftli": ["ft_pct"],
    "spmi": ["pace"],
    "tsi":  ["pace", "conf_tourney_avg_margin"],
}

# R64 collision guard
_R64_OPPONENT_SEED = {5: 12, 12: 5, 6: 11, 11: 6, 7: 10, 10: 7, 8: 9, 9: 8}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_season_full(conn: sqlite3.Connection, season: int) -> list[dict]:
    """
    Load all eligible teams for a season with:
      - All V2 raw feature values (from load_eligible_teams)
      - V5 composite features computed (cpi, dfi, ftli, spmi, tsi)
      - units_won: actual payout units (other_pick_ids ignored)
      - r64_actual: 1.0 win / 0.0 loss in R64 (None if no R64 game)
      - r64_residual: r64_actual - novig_implied_prob (None if no line)
    """
    teams = load_eligible_teams(conn, season)
    if not teams:
        return []

    compute_composite_features(teams)
    team_id_set = {t["team_id"] for t in teams}

    # Units won per team
    for t in teams:
        payout, _ = simulate_pick_payout(
            conn, t["team_id"], season,
            initial_bet=INITIAL_BET_DOLLARS,
            cash_out_round=DEFAULT_CASH_OUT_ROUND,
            other_pick_ids=None,
            bet_style=DEFAULT_BET_STYLE,
        )
        t["units_won"] = (payout - INITIAL_BET_DOLLARS) / 100.0

    # R64 residuals
    r64_rows = conn.execute(
        """
        SELECT g.team1_id, g.team2_id, g.winner_id,
               bl.team1_novig_prob, bl.team2_novig_prob
        FROM mm_games g
        LEFT JOIN mm_betting_lines bl ON bl.game_id = g.id
        WHERE g.season = ? AND g.round = 64
        """,
        (season,),
    ).fetchall()

    r64_lookup = {}  # team_id -> (actual, implied_prob | None)
    for row in r64_rows:
        t1, t2, winner = row["team1_id"], row["team2_id"], row["winner_id"]
        for tid, novig in ((t1, row["team1_novig_prob"]), (t2, row["team2_novig_prob"])):
            if tid in team_id_set:
                actual  = 1.0 if winner == tid else 0.0
                implied = float(novig) if novig is not None else None
                r64_lookup[tid] = (actual, implied)

    for t in teams:
        entry = r64_lookup.get(t["team_id"])
        if entry is not None:
            actual, implied = entry
            t["r64_actual"]   = actual
            t["r64_residual"] = (actual - implied) if implied is not None else None
        else:
            t["r64_actual"]   = None
            t["r64_residual"] = None

    return teams


def collect_train_teams(conn: sqlite3.Connection) -> list[dict]:
    all_teams = []
    for s in TRAIN_SEASONS:
        all_teams.extend(load_season_full(conn, s))
    return all_teams


# ---------------------------------------------------------------------------
# Correlation utilities
# ---------------------------------------------------------------------------

def nanpearsonr(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    n = mask.sum()
    if n < 5:
        return np.nan
    xm, ym = x[mask], y[mask]
    return float(np.corrcoef(xm, ym)[0, 1])


def build_feature_array(teams: list[dict], feature: str) -> np.ndarray:
    return np.array([
        float(t[feature]) if t.get(feature) is not None else np.nan
        for t in teams
    ], dtype=float)


# ---------------------------------------------------------------------------
# Analysis 1: Feature-to-feature correlation matrix
# ---------------------------------------------------------------------------

def run_correlation_matrix(teams: list[dict]) -> None:
    n_feat = len(ALL_CORR_FEATURES)
    vals = {f: build_feature_array(teams, f) for f in ALL_CORR_FEATURES}

    # Compute full correlation matrix
    C = np.full((n_feat, n_feat), np.nan)
    for i, fi in enumerate(ALL_CORR_FEATURES):
        for j, fj in enumerate(ALL_CORR_FEATURES):
            C[i, j] = nanpearsonr(vals[fi], vals[fj])

    abbrevs = [_ABBREV[f] for f in ALL_CORR_FEATURES]
    col_w = 6  # column width

    def _fmt(v):
        if np.isnan(v):
            return " " * col_w
        s = f"{v:+.2f}"
        return s.rjust(col_w)

    def _marker(v):
        if np.isnan(v):  return " "
        a = abs(v)
        if a >= 0.7:     return "##"
        if a >= 0.5:     return "=="
        if a >= 0.35:    return "--"
        return "  "

    print("\n" + "=" * 80)
    print("FEATURE-TO-FEATURE CORRELATION MATRIX  (train seasons, Pearson r)")
    print(f"  N = {len(teams)} team-seasons  ({len(TRAIN_SEASONS)} seasons)")
    print("  Markers: ## |r|>=0.7  == |r|>=0.5  -- |r|>=0.35")
    print("  V2 raw: rows 1-10  |  V5 composites: rows 11-15")
    print("=" * 80)

    # Header rows
    header1 = "  " + " " * 7
    header2 = "  " + f"{'Feature':<7}"
    for j, ab in enumerate(abbrevs):
        label = f"{j+1}:{ab}"
        header1 += label.rjust(col_w)
    print(header1)

    # Divider between V2 and V5 blocks
    print("  " + "-" * (7 + n_feat * col_w))

    for i, fi in enumerate(ALL_CORR_FEATURES):
        ab = _ABBREV[fi]
        row_lbl = f"{i+1}:{ab}"
        line = f"  {row_lbl:<7}"
        for j in range(n_feat):
            v = C[i, j]
            if i == j:
                line += "  1.00"
            else:
                line += _fmt(v)
        # Add marker for high correlations in this row (excluding self)
        row_cors = [(abs(C[i, j]), ALL_CORR_FEATURES[j]) for j in range(n_feat) if j != i and not np.isnan(C[i, j])]
        top = sorted(row_cors, reverse=True)[:3]
        markers = "  " + ", ".join(f"{_ABBREV[n]}:{v:+.2f}" for v, n in top if v >= 0.25)
        print(line + markers)
        # Divider after V2 block
        if i == len(V2_RAW_FEATURES) - 1:
            print("  " + "-" * (7 + n_feat * col_w))

    # Cross-block summary: composites vs their intended ancestors
    print()
    print("  COMPOSITE -> ANCESTOR correlations (key for hybrid design):")
    print("  " + "-" * 60)
    for comp, ancestors in _COMPOSITE_ANCESTRY.items():
        for anc in ancestors:
            if anc in vals and comp in vals:
                r = nanpearsonr(vals[comp], vals[anc])
                bar = "#" * int(abs(r) * 20)
                print(f"  {_ABBREV[comp]:>5} <-> {_ABBREV[anc]:<5}  r={r:+.3f}  {bar}")

    # High-correlation pairs (|r| >= 0.4, excluding self, deduplicated)
    print()
    print("  High-correlation pairs (|r| >= 0.40):")
    print("  " + "-" * 60)
    pairs_seen = set()
    high_pairs = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            v = C[i, j]
            if not np.isnan(v) and abs(v) >= 0.40:
                high_pairs.append((abs(v), v, ALL_CORR_FEATURES[i], ALL_CORR_FEATURES[j]))
    high_pairs.sort(reverse=True)
    if high_pairs:
        for abs_v, v, fi, fj in high_pairs:
            print(f"  {_ABBREV[fi]:>5} <-> {_ABBREV[fj]:<5}  r={v:+.3f}")
    else:
        print("  (none)")
    print()


# ---------------------------------------------------------------------------
# Analysis 2: Feature-to-target correlations
# ---------------------------------------------------------------------------

def run_target_correlations(teams: list[dict]) -> None:
    units  = build_feature_array(teams, "units_won")
    resid  = build_feature_array(teams, "r64_residual")

    n_units = int((~np.isnan(units)).sum())
    n_resid = int((~np.isnan(resid)).sum())

    print("=" * 70)
    print("FEATURE-TO-TARGET CORRELATIONS  (train seasons, Pearson r)")
    print(f"  Target A - units_won:     N={n_units} (all eligible teams)")
    print(f"  Target B - r64_residual:  N={n_resid} (teams with R64 betting line)")
    print("  Note: lower def_rank/opp_efg_pct/tov_ratio = better (raw values used)")
    print("=" * 70)
    print(f"  {'Feature':<25}  {'r(units)':>9}  {'r(r64_res)':>10}  {'|r| avg':>8}  {'Signal?'}")
    print("  " + "-" * 65)

    rows = []
    for f in ALL_CORR_FEATURES:
        vals = build_feature_array(teams, f)
        r_u = nanpearsonr(vals, units)
        r_r = nanpearsonr(vals, resid)
        avg = np.nanmean([abs(r_u), abs(r_r)])
        rows.append((f, r_u, r_r, avg))

    # Sort by average |r|
    rows.sort(key=lambda x: x[3], reverse=True)

    for f, r_u, r_r, avg in rows:
        ru_s = f"{r_u:+.3f}" if not np.isnan(r_u) else "  n/a"
        rr_s = f"{r_r:+.3f}" if not np.isnan(r_r) else "   n/a"
        avg_s = f"{avg:.3f}" if not np.isnan(avg) else "  n/a"
        signal = "**" if avg >= 0.12 else ("*" if avg >= 0.07 else "")
        comp_marker = " [V5]" if f in V5_NEW_COMPOSITES else "     "
        print(f"  {_ABBREV[f]:<5}{comp_marker}  {f:<20}  {ru_s:>9}  {rr_s:>10}  {avg_s:>8}  {signal}")

    print()


# ---------------------------------------------------------------------------
# Single-feature model fast infrastructure (reuses V5 precomputed matrices)
# ---------------------------------------------------------------------------

def precompute_v5_data(conn: sqlite3.Connection, seasons: list[int]) -> dict:
    """Precompute V5 feature matrices and payouts for fast in-memory objective."""
    data = {}
    for season in seasons:
        teams = load_eligible_teams(conn, season)
        if not teams:
            continue
        compute_composite_features(teams)
        X = build_feature_matrix_v5(teams)

        team_ids = [t["team_id"] for t in teams]
        seeds    = [t["seed"]    for t in teams]
        regions  = [t.get("region") for t in teams]

        payouts = {}
        for tid in team_ids:
            payout, _ = simulate_pick_payout(
                conn, tid, season,
                initial_bet=INITIAL_BET_DOLLARS,
                cash_out_round=DEFAULT_CASH_OUT_ROUND,
                other_pick_ids=None,
                bet_style=DEFAULT_BET_STYLE,
            )
            payouts[tid] = payout

        data[season] = {
            "feature_matrix": X,
            "team_ids": team_ids,
            "seeds": seeds,
            "regions": regions,
            "payouts": payouts,
        }
    return data


def _select_fast(team_ids, seeds, regions, scores, n_picks=NUM_PICKS):
    order = np.argsort(-np.array(scores, dtype=float))
    selected, meta = [], []
    for idx in order:
        if len(selected) >= n_picks:
            break
        seed, region = seeds[idx], regions[idx]
        opp = _R64_OPPONENT_SEED.get(seed)
        if opp is not None and region is not None:
            if any(m[0] == opp and m[1] == region for m in meta):
                continue
        selected.append(idx)
        meta.append((seed, region))
    return selected


def make_objective(season_data: dict, seasons: list[int], lambda_l2: float = L2_LAMBDA):
    def _obj(weights):
        total = 0.0
        for s in seasons:
            if s not in season_data:
                continue
            d = season_data[s]
            scores = d["feature_matrix"] @ weights
            for idx in _select_fast(d["team_ids"], d["seeds"], d["regions"], scores):
                payout = d["payouts"].get(d["team_ids"][idx], INITIAL_BET_DOLLARS)
                total += (payout - INITIAL_BET_DOLLARS) / 100.0
        l2 = lambda_l2 * float(np.sum(weights ** 2))
        return -total + l2
    return _obj


def eval_units(season_data: dict, weights: np.ndarray, seasons: list[int]) -> float:
    obj = make_objective(season_data, seasons, lambda_l2=0.0)
    return -obj(weights)


# ---------------------------------------------------------------------------
# Analysis 3: Single-feature models
# ---------------------------------------------------------------------------

def run_single_feature_models(season_data: dict, maxiter: int, popsize: int) -> None:
    n_feat = len(FEATURE_NAMES_V5)

    # V5 baseline (id=17, lambda=0.1 - most recent)
    v5_base_train = eval_units(season_data, np.zeros(n_feat), TRAIN_SEASONS)  # placeholder
    # We'll compute baselines fresh from the precomputed data using saved weights later;
    # for now compare relative to single-feature results.

    print("=" * 80)
    print(f"SINGLE-FEATURE V5 MODELS  (maxiter={maxiter}, popsize={popsize})")
    print(f"  Each feature trained alone; all others locked to 0.")
    print(f"  Bounds: V5_BOUNDS for active feature (sign-constrained).")
    print(f"  Reference baselines:")
    print(f"    V2        train=+29.33u  val=+2.08u  test=+3.29u")
    print(f"    V5 L=0.01 train=+33.49u  val=-1.64u  test=+3.59u  (id=16)")
    print(f"    V5 L=0.10 train=+33.14u  val=+3.79u  test=+3.59u  (id=17)")
    print("=" * 80)
    print(f"  {'#':<3} {'Feature':<25} {'Val':>8}  {'Train':>8}  {'Test':>8}  {'w*':>8}  vs V2 val")
    print("  " + "-" * 75)

    results = []
    for i, name in enumerate(FEATURE_NAMES_V5):
        t0 = time.time()
        print(f"  {i+1:<3} {name:<25}  running...", end="", flush=True)

        # Build bounds: all (0,0) except active feature
        bounds = [(0.0, 0.0)] * n_feat
        bounds[i] = V5_BOUNDS[i]

        _obj = make_objective(season_data, TRAIN_SEASONS)
        result = differential_evolution(
            _obj, bounds=bounds,
            seed=42, maxiter=maxiter, tol=1e-5,
            workers=1, popsize=popsize, init="sobol",
        )
        w = result.x
        train_u = eval_units(season_data, w, TRAIN_SEASONS)
        val_u   = eval_units(season_data, w, VAL_SEASONS)
        test_u  = eval_units(season_data, w, TEST_SEASONS)
        w_star  = w[i]

        elapsed = time.time() - t0
        delta = val_u - 2.08  # vs V2 val
        marker = " ^" if delta > 0 else " v"
        print(f"\r  {i+1:<3} {name:<25} {val_u:>+8.3f}  {train_u:>+8.3f}  {test_u:>+8.3f}  {w_star:>+8.4f}  {delta:>+6.3f}{marker}  ({elapsed:.0f}s)")
        results.append((name, val_u, train_u, test_u, w_star, delta))

    print()
    print("=" * 80)
    print("Summary - ranked by val performance:")
    print("=" * 80)
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    for rank, (name, val_u, train_u, test_u, w_star, delta) in enumerate(results_sorted, 1):
        bar = "#" * max(0, int((val_u + 3) * 3))
        print(f"  {rank:2}. {name:<25}  val={val_u:>+7.3f}  train={train_u:>+7.3f}  test={test_u:>+7.3f}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V5 feature diagnostics")
    parser.add_argument("--corr", action="store_true", help="Correlation analyses only (no training)")
    parser.add_argument("--sfm",  action="store_true", help="Single-feature models only")
    parser.add_argument("--fast", action="store_true", help="Faster DE for SFM (maxiter=100, popsize=10)")
    args = parser.parse_args()

    run_corr = not args.sfm   # default: run corr unless --sfm only
    run_sfm  = not args.corr  # default: run SFM unless --corr only
    if args.corr and args.sfm:
        run_corr = run_sfm = True

    maxiter = 100 if args.fast else 200
    popsize = 10  if args.fast else 12

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if run_corr:
        print(f"\nLoading train data for correlation analysis ({len(TRAIN_SEASONS)} seasons)...", flush=True)
        t0 = time.time()
        train_teams = collect_train_teams(conn)
        print(f"  {len(train_teams)} team-seasons loaded in {time.time()-t0:.1f}s\n")

        run_correlation_matrix(train_teams)
        run_target_correlations(train_teams)

    if run_sfm:
        all_seasons = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS
        print(f"Precomputing V5 feature matrices for {len(all_seasons)} seasons...", flush=True)
        t0 = time.time()
        season_data = precompute_v5_data(conn, all_seasons)
        print(f"  Done in {time.time()-t0:.1f}s\n")

        run_single_feature_models(season_data, maxiter=maxiter, popsize=popsize)

    conn.close()


if __name__ == "__main__":
    main()
