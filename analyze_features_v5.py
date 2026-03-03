"""
Feature analysis for V5 model.

Steps:
  1. Print V5 weight table (id=16) with sign-check vs. priors
  2. Run leave-one-feature-out (LOO) analysis:
     - Precompute V5 composite feature matrices and payouts in memory
     - For each of the 8 V5 features, retrain with that feature locked to 0
       (constrained V5_BOUNDS maintained for remaining features)
     - Compare val to V5 baseline
  3. Summary: rank features by val impact of removal

Usage:
  python analyze_features_v5.py            # full LOO (maxiter=300, ~5-10 min)
  python analyze_features_v5.py --fast     # quick LOO (maxiter=100, ~2-3 min)
  python analyze_features_v5.py --weights  # weight table only, no retraining
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
from processors.model import (
    simulate_pick_payout, INITIAL_BET_DOLLARS,
)
from config import (
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    DEFAULT_CASH_OUT_ROUND, DEFAULT_BET_STYLE,
    NUM_PICKS, L2_LAMBDA, DB_PATH,
)

# Expected sign after negate-transform (all features should have + weight;
# seed_rank_gap is the exception — it has negate=False and lower is better,
# so its weight should be negative).
EXPECTED_SIGNS = {
    "seed_rank_gap":     "-",  # lower (more negative) = more under-seeded = want
    "net_rating":        "+",  # higher adj net = better
    "conf_tourney_wins": "+",  # more conf wins = hotter
    "cpi":               "+",  # higher possession control index = better
    "dfi":               "+",  # negated in matrix; + = lower actual DFI = better defense
    "ftli":              "+",  # higher free-throw leverage = better in close games
    "spmi":              "+",  # higher shot profile mismatch = better
    "tsi":               "+",  # negated in matrix; + = lower actual TSI = more consistent
}

# R64 seed opponent map for collision guard
_R64_OPPONENT_SEED = {5: 12, 12: 5, 6: 11, 11: 6, 7: 10, 10: 7, 8: 9, 9: 8}


# ---------------------------------------------------------------------------
# Precomputed data structures for fast in-memory objective
# ---------------------------------------------------------------------------

def precompute_season_data_v5(conn: sqlite3.Connection, seasons: list[int]) -> dict:
    """
    Precompute V5 feature matrices and payouts for fast in-memory objective.

    Calls compute_composite_features() first (adds cpi, dfi, ftli, spmi to each
    team dict), then builds the V5 feature matrix. Payouts are precomputed with
    other_pick_ids=None (ignores between-pick collision adjustment — acceptable
    for relative LOO comparisons).
    """
    data = {}
    for season in seasons:
        teams = load_eligible_teams(conn, season)
        if not teams:
            continue

        # Compute composites in place, then build V5 matrix
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
            "feature_matrix": X,   # shape (n_teams, 8)
            "team_ids": team_ids,
            "seeds": seeds,
            "regions": regions,
            "payouts": payouts,
        }

    return data


def _select_picks_fast(team_ids, seeds, regions, scores, n_picks=NUM_PICKS):
    """Select top n_picks with R64 collision guard. Returns list of indices."""
    order = np.argsort(-np.array(scores, dtype=float))
    selected = []
    selected_meta = []

    for idx in order:
        if len(selected) >= n_picks:
            break
        seed   = seeds[idx]
        region = regions[idx]
        opp_seed = _R64_OPPONENT_SEED.get(seed)
        if opp_seed is not None and region is not None:
            if any(sm[0] == opp_seed and sm[1] == region for sm in selected_meta):
                continue
        selected.append(idx)
        selected_meta.append((seed, region))

    return selected


def make_fast_objective_v5(season_data: dict, seasons: list[int], lambda_l2: float = L2_LAMBDA):
    """
    Return a fast in-memory objective for the given seasons.
    Takes an 8-element weight vector (V5 features).
    """
    def _objective(weights):
        total_units = 0.0
        for season in seasons:
            if season not in season_data:
                continue
            d = season_data[season]
            X = d["feature_matrix"]  # (n_teams, 8)
            scores = X @ weights

            picked_indices = _select_picks_fast(
                d["team_ids"], d["seeds"], d["regions"], scores
            )
            for idx in picked_indices:
                tid = d["team_ids"][idx]
                payout = d["payouts"].get(tid, INITIAL_BET_DOLLARS)
                total_units += (payout - INITIAL_BET_DOLLARS) / 100.0

        l2 = lambda_l2 * float(np.sum(weights ** 2))
        return -total_units + l2

    return _objective


def eval_units_fast_v5(season_data: dict, weights: np.ndarray, seasons: list[int]) -> float:
    """Evaluate total units for given seasons using precomputed V5 data."""
    obj = make_fast_objective_v5(season_data, seasons, lambda_l2=0.0)
    return -obj(weights)


# ---------------------------------------------------------------------------
# Weight table display
# ---------------------------------------------------------------------------

def load_v5_weights(conn: sqlite3.Connection) -> tuple[np.ndarray, int]:
    """Load the latest V5 fixed-8 weights. Returns (weights, id)."""
    row = conn.execute(
        """
        SELECT id, w_seed_rank_gap, w_net_rating, w_conf_tourney_wins,
               w_cpi, w_dfi, w_ftli, w_spmi, w_tsi,
               train_units_won, val_units_won
        FROM mm_model_weights
        WHERE notes LIKE 'model=v5 %'
        ORDER BY trained_at DESC LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("No V5 weights found. Run: python main.py train --v5")
    weights = np.array([
        row["w_seed_rank_gap"]     or 0.0,
        row["w_net_rating"]        or 0.0,
        row["w_conf_tourney_wins"] or 0.0,
        row["w_cpi"]               or 0.0,
        row["w_dfi"]               or 0.0,
        row["w_ftli"]              or 0.0,
        row["w_spmi"]              or 0.0,
        row["w_tsi"]               or 0.0,
    ], dtype=float)
    return weights, row["id"], row["train_units_won"], row["val_units_won"]


def print_weight_table_v5(weights: np.ndarray, wid: int, train_u: float, val_u: float) -> None:
    abs_w = np.abs(weights)
    max_abs = abs_w.max() if abs_w.max() > 0 else 1.0

    print("=" * 90)
    print(f"V5 Fixed-8 Weights (id={wid})  --  Train: {train_u:+.4f}u  Val: {val_u:+.4f}u")
    print("=" * 90)
    print(f"  {'#':<3} {'Feature':<28} {'Weight':>10}  {'|Weight|':>9}  {'Rel%':>6}  {'Sign':>5}  {'Exp':>4}  {'OK?':>6}")
    print("  " + "-" * 76)

    for i, name in enumerate(FEATURE_NAMES_V5):
        w = weights[i]
        rel = abs_w[i] / max_abs * 100
        sign = "+" if w > 0 else ("-" if w < 0 else "0")
        exp = EXPECTED_SIGNS.get(name, "?")
        ok = "YES" if exp == "?" or sign == exp else "*** NO"
        print(f"  {i+1:<3} {name:<28} {w:>10.6f}  {abs_w[i]:>9.6f}  {rel:>5.1f}%  {sign:>5}  {exp:>4}  {ok:>6}")

    print()
    ranked = sorted(range(len(FEATURE_NAMES_V5)), key=lambda i: abs_w[i], reverse=True)
    print("  Ranked by |weight|:")
    for rank, i in enumerate(ranked, 1):
        name = FEATURE_NAMES_V5[i]
        w = weights[i]
        exp = EXPECTED_SIGNS.get(name, "?")
        sign = "+" if w > 0 else ("-" if w < 0 else "0")
        ok = "" if exp == "?" or sign == exp else "  << WRONG SIGN"
        print(f"    {rank:2}. {name:<28} {w:+.6f}{ok}")
    print()


# ---------------------------------------------------------------------------
# LOO analysis
# ---------------------------------------------------------------------------

def loo_train_fast_v5(season_data: dict, drop_idx: int, maxiter: int, popsize: int, seed: int = 42) -> dict:
    """
    Retrain V5 with feature `drop_idx` locked to 0.
    Maintains V5 sign constraints for all other features.
    """
    bounds = list(V5_BOUNDS)       # copy the constrained bounds
    bounds[drop_idx] = (0.0, 0.0) # lock this feature out

    _obj = make_fast_objective_v5(season_data, TRAIN_SEASONS)

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
    train_u = eval_units_fast_v5(season_data, w, TRAIN_SEASONS)
    return {"train_units": train_u, "weights": w}


def run_loo_analysis_v5(
    season_data: dict,
    v5_weights: np.ndarray,
    v5_val_db: float,
    maxiter: int,
    popsize: int,
) -> None:
    v5_train_fast = eval_units_fast_v5(season_data, v5_weights, TRAIN_SEASONS)
    v5_val_fast   = eval_units_fast_v5(season_data, v5_weights, VAL_SEASONS)
    v5_test_fast  = eval_units_fast_v5(season_data, v5_weights, TEST_SEASONS)

    print("=" * 86)
    print(f"V5 LOO Analysis (maxiter={maxiter}, popsize={popsize})")
    print(f"V5 baseline  --  Train: {v5_train_fast:+.4f}u  Val: {v5_val_fast:+.4f}u  "
          f"Test: {v5_test_fast:+.4f}u  (DB val: {v5_val_db:+.4f}u)")
    print("  Note: fast evaluator ignores between-pick collision payout adjustment;")
    print("        relative dVal comparisons remain valid.")
    print("=" * 86)
    print(f"  {'#':<3} {'Feature':<28} {'Val (LOO)':>10}  {'dVal':>7}  {'Train':>8}  {'Impact'}")
    print("  " + "-" * 74)

    results = []
    for i, name in enumerate(FEATURE_NAMES_V5):
        t0 = time.time()
        print(f"  {i+1:<3} {name:<28}  running...", end="", flush=True)

        r = loo_train_fast_v5(season_data, i, maxiter=maxiter, popsize=popsize)
        val_u  = eval_units_fast_v5(season_data, r["weights"], VAL_SEASONS)
        test_u = eval_units_fast_v5(season_data, r["weights"], TEST_SEASONS)

        elapsed = time.time() - t0
        delta = val_u - v5_val_fast
        impact = "DROP (helps)" if delta > 0.1 else ("KEEP (hurts to remove)" if delta < -0.1 else "neutral")
        print(f"\r  {i+1:<3} {name:<28} {val_u:>10.4f}  {delta:>+7.3f}  {r['train_units']:>8.4f}  {impact}  ({elapsed:.0f}s)")
        results.append((name, i, val_u, delta, r["train_units"], test_u))

    print()
    print("=" * 86)
    print("Summary -- most harmful to remove (top) to most beneficial to drop (bottom):")
    print("=" * 86)
    results_sorted = sorted(results, key=lambda x: x[3])
    for name, i, val_u, delta, train_u, test_u in results_sorted:
        exp = EXPECTED_SIGNS.get(name, "?")
        w_actual = v5_weights[i]
        w_sign = "+" if w_actual > 0 else ("-" if w_actual < 0 else "0")
        sign_ok = "OK " if exp == "?" or w_sign == exp else "BAD"
        print(f"  {name:<28}  dVal={delta:>+6.3f}  Val={val_u:>+6.3f}  Train={train_u:>+6.3f}  "
              f"Test={test_u:>+6.3f}  sign={sign_ok}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V5 feature weight analysis and LOO")
    parser.add_argument("--fast",    action="store_true", help="Fast LOO: maxiter=100, popsize=10")
    parser.add_argument("--weights", action="store_true", help="Weight table only -- skip LOO retraining")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    weights, wid, train_u, val_u = load_v5_weights(conn)
    print_weight_table_v5(weights, wid, train_u, val_u)

    if args.weights:
        conn.close()
        return

    if args.fast:
        maxiter, popsize = 100, 10
        print("Fast mode: approximate results (good for relative ranking)\n")
    else:
        maxiter, popsize = 300, 15

    all_seasons = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS
    print(f"Precomputing V5 composite feature matrices and payouts for {len(all_seasons)} seasons...", flush=True)
    t0 = time.time()
    all_data = precompute_season_data_v5(conn, all_seasons)
    print(f"Precompute done in {time.time()-t0:.1f}s\n")

    conn.close()

    run_loo_analysis_v5(
        season_data=all_data,
        v5_weights=weights,
        v5_val_db=val_u,
        maxiter=maxiter,
        popsize=popsize,
    )


if __name__ == "__main__":
    main()
