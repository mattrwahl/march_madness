"""
Feature analysis for V2 model.

Steps:
  1. Print V2 weight table with sign-check vs. priors
  2. Run leave-one-feature-out (LOO) analysis:
     - Precompute all feature matrices and payouts so the optimizer runs in memory
     - For each of the 10 V2 features, retrain with that feature locked to 0
     - Compare val to V2 baseline
  3. Summary: rank features by val impact of removal

Usage:
  python analyze_features.py            # full LOO (maxiter=300, ~5-10 min total)
  python analyze_features.py --fast     # quick LOO (maxiter=100, ~2-3 min, noisier)
  python analyze_features.py --weights  # weight table only, no retraining
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
    FEATURES, FEATURE_NAMES, load_eligible_teams, build_feature_matrix,
)
from processors.model import (
    simulate_pick_payout, get_team_tournament_results,
    INITIAL_BET_DOLLARS,
)
from config import (
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    DEFAULT_CASH_OUT_ROUND, DEFAULT_BET_STYLE,
    NUM_PICKS, L2_LAMBDA, DB_PATH,
)

# V2 feature set (first 10 features in FEATURES list)
V2_N_FEATURES = 10

# Expected sign of weight for each V2 feature
# (+) = want positive weight after negate-transform, (-) = want negative, (?) = no prior
EXPECTED_SIGNS = {
    "seed_rank_gap":           "-",  # more negative = more under-seeded = want
    "def_rank":                "+",  # negated; + = lower actual def_rank = better
    "opp_efg_pct":             "+",  # negated; + = lower actual opp_efg = better
    "net_rating":              "+",  # higher = better
    "tov_ratio":               "+",  # negated; + = lower actual TOs = better
    "ft_pct":                  "+",  # higher = better
    "oreb_pct":                "+",  # higher = better
    "pace":                    "?",  # no directional prior
    "conf_tourney_wins":       "+",  # higher = better
    "conf_tourney_avg_margin": "+",  # higher = better
}

# R64 seed opponent map for collision guard
_R64_OPPONENT_SEED = {5: 12, 12: 5, 6: 11, 11: 6, 7: 10, 10: 7, 8: 9, 9: 8}


# ---------------------------------------------------------------------------
# Precomputed data structures for fast in-memory objective
# ---------------------------------------------------------------------------

def precompute_season_data(conn: sqlite3.Connection, seasons: list[int]) -> dict:
    """
    Precompute everything needed for fast in-memory objective evaluation.

    Returns dict keyed by season, each value is:
      {
        "feature_matrix": np.ndarray shape (n_teams, n_all_features),
        "team_ids":        list[int],
        "seeds":           list[int],
        "regions":         list[str|None],
        "payouts":         dict {team_id: (payout_dollars, round_exit)},
        "other_ids":       dict {team_id: set of all team_ids in this season},
      }
    """
    data = {}
    all_team_ids_by_season = {}

    # First pass: load teams and payouts per season
    for season in seasons:
        teams = load_eligible_teams(conn, season)
        if not teams:
            continue

        # Build full feature matrix (all features, z-score normalized)
        X = build_feature_matrix(teams)

        team_ids = [t["team_id"] for t in teams]
        seeds    = [t["seed"]    for t in teams]
        regions  = [t.get("region") for t in teams]

        # Precompute payout for each eligible team
        # We pass other_pick_ids=set() for now; the collision-among-picks case
        # is handled at pick-selection time, not payout time — the payout uses
        # the full pick_ids set. We'll handle this by computing payouts lazily
        # or we precompute with the full eligible set as other_ids.
        #
        # Correct approach: simulate_pick_payout needs other_pick_ids = other
        # SELECTED picks, not all eligible teams. But we don't know selections
        # until after scoring. So we precompute payout with other_pick_ids=None
        # (ignoring the collision-between-picks cancellation) and apply
        # collision logic at selection time.
        #
        # This matches how simulate_season works — it builds other_ids_map
        # AFTER selection. For LOO the key insight is: same teams will rarely
        # collide differently across LOO runs, so small payout error from
        # ignoring between-pick collision is acceptable for relative comparison.
        payouts = {}
        for tid in team_ids:
            payout, rnd = simulate_pick_payout(
                conn, tid, season,
                initial_bet=INITIAL_BET_DOLLARS,
                cash_out_round=DEFAULT_CASH_OUT_ROUND,
                other_pick_ids=None,
                bet_style=DEFAULT_BET_STYLE,
            )
            payouts[tid] = payout

        all_team_ids_by_season[season] = set(team_ids)
        data[season] = {
            "feature_matrix": X,
            "team_ids": team_ids,
            "seeds": seeds,
            "regions": regions,
            "payouts": payouts,
        }

    return data


def _select_picks_fast(team_ids, seeds, regions, scores, n_picks=NUM_PICKS):
    """
    Select top n_picks from scored teams with R64 collision guard.
    Returns list of team_id indices in the original arrays.
    """
    order = np.argsort(-np.array(scores, dtype=float))
    selected = []
    selected_meta = []  # (seed, region) of picked teams

    for idx in order:
        if len(selected) >= n_picks:
            break
        seed   = seeds[idx]
        region = regions[idx]
        opp_seed = _R64_OPPONENT_SEED.get(seed)

        # Collision guard: skip if R64 opponent already selected
        if opp_seed is not None and region is not None:
            if any(sm[0] == opp_seed and sm[1] == region for sm in selected_meta):
                continue

        selected.append(idx)
        selected_meta.append((seed, region))

    return selected


def make_fast_objective(season_data: dict, seasons: list[int], lambda_l2: float = L2_LAMBDA):
    """
    Return a fast in-memory objective function for the given seasons.
    The objective takes a weight vector of length V2_N_FEATURES.
    """
    def _objective(weights):
        total_units = 0.0
        for season in seasons:
            if season not in season_data:
                continue
            d = season_data[season]
            X = d["feature_matrix"]  # shape (n_teams, n_all_features)
            # Use only the first V2_N_FEATURES columns; rest zeroed
            w_padded = np.zeros(X.shape[1])
            w_padded[:len(weights)] = weights
            scores = X @ w_padded

            picked_indices = _select_picks_fast(
                d["team_ids"], d["seeds"], d["regions"], scores
            )
            for idx in picked_indices:
                tid = d["team_ids"][idx]
                payout = d["payouts"].get(tid, INITIAL_BET_DOLLARS)
                total_units += (payout - INITIAL_BET_DOLLARS) / 100.0

        l2 = lambda_l2 * float(np.sum(weights[:V2_N_FEATURES] ** 2))
        return -total_units + l2

    return _objective


def eval_units_fast(season_data: dict, weights: np.ndarray, seasons: list[int]) -> float:
    """Evaluate total units on given seasons using precomputed data."""
    obj = make_fast_objective(season_data, seasons, lambda_l2=0.0)
    return -obj(weights)


# ---------------------------------------------------------------------------
# Weight table display
# ---------------------------------------------------------------------------

def load_v2_weights(conn: sqlite3.Connection) -> np.ndarray:
    """Load V2 fixed-8 weights (id=11) from DB — first 10 features."""
    row = conn.execute(
        """
        SELECT w_seed_rank_gap, w_def_rank, w_opp_efg_pct, w_net_rating,
               w_tov_ratio, w_ft_pct, w_oreb_pct, w_pace,
               w_conf_tourney_wins, w_conf_tourney_avg_margin
        FROM mm_model_weights WHERE id = 11
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("V2 weights (id=11) not found in DB")
    return np.array(list(row), dtype=float)


def print_weight_table(weights: np.ndarray, train_u: float, val_u: float) -> None:
    abs_w = np.abs(weights)
    max_abs = abs_w.max() if abs_w.max() > 0 else 1.0

    print("=" * 88)
    print(f"V2 Fixed-8 Weights (id=11)  --  Train: {train_u:+.4f}u  Val: {val_u:+.4f}u")
    print("=" * 88)
    print(f"  {'#':<3} {'Feature':<28} {'Weight':>10}  {'|Weight|':>9}  {'Rel%':>6}  {'Sign':>5}  {'Exp':>4}  {'OK?':>6}")
    print("  " + "-" * 76)

    feature_names_v2 = FEATURE_NAMES[:V2_N_FEATURES]
    for i, name in enumerate(feature_names_v2):
        w = weights[i]
        rel = abs_w[i] / max_abs * 100
        sign = "+" if w > 0 else "-"
        exp = EXPECTED_SIGNS.get(name, "?")
        ok = "YES" if exp == "?" or sign == exp else "*** NO"
        print(f"  {i+1:<3} {name:<28} {w:>10.6f}  {abs_w[i]:>9.6f}  {rel:>5.1f}%  {sign:>5}  {exp:>4}  {ok:>6}")

    print()
    ranked = sorted(range(V2_N_FEATURES), key=lambda i: abs_w[i], reverse=True)
    print("  Ranked by |weight|:")
    for rank, i in enumerate(ranked, 1):
        name = feature_names_v2[i]
        w = weights[i]
        exp = EXPECTED_SIGNS.get(name, "?")
        sign = "+" if w > 0 else "-"
        ok = "" if exp == "?" or sign == exp else "  << WRONG SIGN"
        print(f"    {rank:2}. {name:<28} {w:+.6f}{ok}")
    print()


# ---------------------------------------------------------------------------
# LOO analysis
# ---------------------------------------------------------------------------

def loo_train_fast(train_data: dict, drop_idx: int, maxiter: int, popsize: int, seed: int = 42) -> dict:
    """
    Retrain with feature `drop_idx` locked to 0 using fast in-memory objective.
    Returns dict with train_units, weights.
    """
    bounds = [(-3.0, 3.0)] * V2_N_FEATURES
    bounds[drop_idx] = (0.0, 0.0)

    _obj = make_fast_objective(train_data, TRAIN_SEASONS)

    result = differential_evolution(
        _obj,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        tol=1e-5,
        workers=1,
        popsize=popsize,
        init="sobol",
    )

    w = result.x
    train_u = eval_units_fast(train_data, w, TRAIN_SEASONS)
    return {"train_units": train_u, "weights": w}


def run_loo_analysis(
    conn: sqlite3.Connection,
    train_data: dict,
    val_data: dict,
    test_data: dict,
    all_data: dict,
    v2_weights: np.ndarray,
    v2_val: float,
    maxiter: int,
    popsize: int,
) -> None:
    feature_names_v2 = FEATURE_NAMES[:V2_N_FEATURES]

    # V2 baseline on val via fast evaluator (should match DB val_units_won)
    v2_val_fast = eval_units_fast(all_data, v2_weights, VAL_SEASONS)
    v2_train_fast = eval_units_fast(all_data, v2_weights, TRAIN_SEASONS)
    v2_test_fast = eval_units_fast(all_data, v2_weights, TEST_SEASONS)

    print("=" * 84)
    print(f"LOO Analysis (maxiter={maxiter}, popsize={popsize})")
    print(f"V2 baseline  --  Train: {v2_train_fast:+.4f}u  Val: {v2_val_fast:+.4f}u  "
          f"Test: {v2_test_fast:+.4f}u  (DB val: {v2_val:+.4f}u)")
    print("  Note: fast evaluator ignores between-pick collision payout adjustment;")
    print("        relative dVal comparisons remain valid.")
    print("=" * 84)
    print(f"  {'#':<3} {'Feature':<28} {'Val (LOO)':>10}  {'dVal':>7}  {'Train':>8}  {'Impact'}")
    print("  " + "-" * 72)

    results = []
    for i, name in enumerate(feature_names_v2):
        t0 = time.time()
        print(f"  {i+1:<3} {name:<28}  running...", end="", flush=True)

        r = loo_train_fast(train_data, i, maxiter=maxiter, popsize=popsize)
        val_u  = eval_units_fast(all_data, r["weights"], VAL_SEASONS)
        test_u = eval_units_fast(all_data, r["weights"], TEST_SEASONS)

        elapsed = time.time() - t0
        delta = val_u - v2_val_fast
        impact = "DROP (helps)" if delta > 0.1 else ("KEEP (hurts to remove)" if delta < -0.1 else "neutral")
        print(f"\r  {i+1:<3} {name:<28} {val_u:>10.4f}  {delta:>+7.3f}  {r['train_units']:>8.4f}  {impact}  ({elapsed:.0f}s)")
        results.append((name, i, val_u, delta, r["train_units"], test_u))

    print()
    print("=" * 84)
    print("Summary -- most harmful to remove (top) to most beneficial to drop (bottom):")
    print("=" * 84)
    results_sorted = sorted(results, key=lambda x: x[3])
    for name, i, val_u, delta, train_u, test_u in results_sorted:
        sign = EXPECTED_SIGNS.get(name, "?")
        w_actual = v2_weights[i]
        w_sign = "+" if w_actual > 0 else "-"
        sign_ok = "OK " if sign == "?" or w_sign == sign else "BAD"
        print(f"  {name:<28}  dVal={delta:>+6.3f}  Val={val_u:>+6.3f}  Train={train_u:>+6.3f}  "
              f"Test={test_u:>+6.3f}  sign={sign_ok}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V2 feature weight analysis and LOO")
    parser.add_argument("--fast",    action="store_true", help="Fast LOO: maxiter=100, popsize=10 (~2-3 min)")
    parser.add_argument("--weights", action="store_true", help="Weight table only -- skip LOO retraining")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        "SELECT train_units_won, val_units_won FROM mm_model_weights WHERE id=11"
    ).fetchone()
    v2_train = row["train_units_won"]
    v2_val   = row["val_units_won"]

    weights = load_v2_weights(conn)
    print_weight_table(weights, v2_train, v2_val)

    if args.weights:
        conn.close()
        return

    if args.fast:
        maxiter, popsize = 100, 10
        print("Fast mode: approximate results (good for relative ranking)\n")
    else:
        maxiter, popsize = 300, 15

    # Precompute all data
    all_seasons = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS
    print(f"Precomputing feature matrices and payouts for {len(all_seasons)} seasons...", flush=True)
    t0 = time.time()
    all_data   = precompute_season_data(conn, all_seasons)
    train_data = {s: all_data[s] for s in TRAIN_SEASONS if s in all_data}
    val_data   = {s: all_data[s] for s in VAL_SEASONS   if s in all_data}
    test_data  = {s: all_data[s] for s in TEST_SEASONS  if s in all_data}
    print(f"Precompute done in {time.time()-t0:.1f}s\n")

    conn.close()

    run_loo_analysis(
        conn=None,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        all_data=all_data,
        v2_weights=weights,
        v2_val=v2_val,
        maxiter=maxiter,
        popsize=popsize,
    )


if __name__ == "__main__":
    main()
