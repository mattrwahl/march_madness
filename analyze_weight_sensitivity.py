"""
Weight perturbation sweep for v6-fixed-8-geomean.

For each of the 4 features, multiply its weight by {0.8, 0.9, 1.0, 1.1, 1.2},
renormalize (so abs values still sum to 1), then simulate all 11 seasons.
Reports train / val / test units for both flat and tiered strategies.
"""
import sys, os, sqlite3
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from processors.features import (
    load_eligible_teams, compute_composite_features,
    FEATURES_V6B, _build_feature_matrix_v6, select_picks_v6, V6_GEOMEAN_W,
)
from processors.model import simulate_pick_payout, ROUND_LABELS
from config import (
    ALL_SEASONS, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    DEFAULT_CASH_OUT_ROUND,
)

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "march_madness.db")
FEATURE_NAMES = ["srg", "ctw", "dfi", "tsi"]
MULTIPLIERS   = [0.8, 0.9, 1.0, 1.1, 1.2]
CASH_OUT      = DEFAULT_CASH_OUT_ROUND   # E8 = 8
FLAT_BET      = 25.0
TIER1, TIER2  = 37.50, 12.50
NUM_PICKS     = 8


def run_seasons(conn, weights):
    """
    Simulate all 11 seasons with the given weights.
    Returns dict: season -> (flat_units, tiered_units).
    """
    results = {}
    for season in ALL_SEASONS:
        teams = load_eligible_teams(conn, season)
        if not teams:
            continue
        compute_composite_features(teams)
        picks = select_picks_v6(teams, weights, "coreB", NUM_PICKS)
        other_ids = {p["team_id"] for p in picks}

        flat_total = 0.0
        tiered_total = 0.0
        for pick in picks:
            others = other_ids - {pick["team_id"]}
            bet = TIER1 if pick["pick_rank"] <= 4 else TIER2

            fp, _ = simulate_pick_payout(
                conn, pick["team_id"], season,
                initial_bet=FLAT_BET, cash_out_round=CASH_OUT,
                other_pick_ids=others, bet_style="flat",
            )
            tp, _ = simulate_pick_payout(
                conn, pick["team_id"], season,
                initial_bet=bet, cash_out_round=CASH_OUT,
                other_pick_ids=others, bet_style="flat",
            )
            flat_total   += (fp - FLAT_BET) / 100.0
            tiered_total += (tp - bet) / 100.0

        results[season] = (flat_total, tiered_total)
    return results


def sum_splits(results, seasons):
    flat = sum(results[s][0] for s in seasons if s in results)
    tier = sum(results[s][1] for s in seasons if s in results)
    return flat, tier


def perturb(base_weights, feat_idx, factor):
    """Multiply weight[feat_idx] by factor, renormalize so |w| sums to 1."""
    w = base_weights.copy()
    w[feat_idx] *= factor
    w /= np.abs(w).sum()
    return w


def fmt_w(w):
    return "  ".join(f"{n}={v:+.4f}" for n, v in zip(FEATURE_NAMES, w))


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    base = V6_GEOMEAN_W.copy()
    print("=" * 90)
    print("V6-GEOMEAN WEIGHT PERTURBATION SWEEP  (E8 cash-out, all 11 seasons)")
    print(f"Base weights: {fmt_w(base)}")
    print("=" * 90)

    # Run baseline once
    base_results = run_seasons(conn, base)
    b_train_f, b_train_t = sum_splits(base_results, TRAIN_SEASONS)
    b_val_f,   b_val_t   = sum_splits(base_results, VAL_SEASONS)
    b_v23_f,   b_v23_t   = sum_splits(base_results, [2023])
    b_v24_f,   b_v24_t   = sum_splits(base_results, [2024])
    b_test_f,  b_test_t  = sum_splits(base_results, TEST_SEASONS)
    b_tot_f  = b_train_f + b_val_f + b_test_f
    b_tot_t  = b_train_t + b_val_t + b_test_t

    # Header
    print()
    print(f"{'Feature':<6} {'Mult':>5}  {'Weights (renorm)':42}  "
          f"{'--- FLAT ----------------------------':36}  "
          f"{'--- TIERED --------------------------':36}")
    print(f"{'':6} {'':5}  {'':42}  "
          f"{'Train':>7} {'Val23':>6} {'Val24':>6} {'Val':>6} {'Test':>6} {'Total':>7}  "
          f"{'Train':>7} {'Val23':>6} {'Val24':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    print("-" * 150)

    for fi, fname in enumerate(FEATURE_NAMES):
        first_row = True
        for mult in MULTIPLIERS:
            w = perturb(base, fi, mult)
            if np.allclose(w, base):
                res = base_results
            else:
                res = run_seasons(conn, w)

            tr_f, tr_t = sum_splits(res, TRAIN_SEASONS)
            v23_f, v23_t = sum_splits(res, [2023])
            v24_f, v24_t = sum_splits(res, [2024])
            val_f, val_t = v23_f + v24_f, v23_t + v24_t
            te_f, te_t = sum_splits(res, TEST_SEASONS)
            tot_f = tr_f + val_f + te_f
            tot_t = tr_t + val_t + te_t

            # Delta from baseline for total
            d_f = tot_f - b_tot_f
            d_t = tot_t - b_tot_t

            baseline_marker = " <--BASE" if mult == 1.0 else ""
            label = fname if first_row else ""
            first_row = False

            print(
                f"{label:<6} {mult:5.1f}  {fmt_w(w):42}  "
                f"{tr_f:+7.2f} {v23_f:+6.2f} {v24_f:+6.2f} {val_f:+6.2f} {te_f:+6.2f} {tot_f:+7.2f}  "
                f"{tr_t:+7.2f} {v23_t:+6.2f} {v24_t:+6.2f} {val_t:+6.2f} {te_t:+6.2f} {tot_t:+7.2f}"
                f"  (d_total flat={d_f:+.2f} tiered={d_t:+.2f}){baseline_marker}"
            )
        print()

    print("-" * 150)
    print(
        f"{'BASE':6} {'1.0':>5}  {fmt_w(base):42}  "
        f"{b_train_f:+7.2f} {b_v23_f:+6.2f} {b_v24_f:+6.2f} {b_val_f:+6.2f} {b_test_f:+6.2f} {b_tot_f:+7.2f}  "
        f"{b_train_t:+7.2f} {b_v23_t:+6.2f} {b_v24_t:+6.2f} {b_val_t:+6.2f} {b_test_t:+6.2f} {b_tot_t:+7.2f}"
    )
    print("=" * 150)
    print()
    print("STABILITY SUMMARY")
    print("-" * 60)
    print(f"{'':6}  {'Flat total range':25}  {'Tiered total range':25}")

    all_runs = {}
    for fi, fname in enumerate(FEATURE_NAMES):
        flat_tots, tier_tots = [], []
        for mult in MULTIPLIERS:
            w = perturb(base, fi, mult)
            res = base_results if np.allclose(w, base) else run_seasons(conn, w)
            tr_f, tr_t = sum_splits(res, TRAIN_SEASONS)
            val_f, val_t = sum_splits(res, VAL_SEASONS)
            te_f, te_t = sum_splits(res, TEST_SEASONS)
            flat_tots.append(tr_f + val_f + te_f)
            tier_tots.append(tr_t + val_t + te_t)
        lo_f, hi_f = min(flat_tots), max(flat_tots)
        lo_t, hi_t = min(tier_tots), max(tier_tots)
        print(f"  {fname:<5}  [{lo_f:+.2f}, {hi_f:+.2f}] spread={hi_f-lo_f:.2f}u    "
              f"[{lo_t:+.2f}, {hi_t:+.2f}] spread={hi_t-lo_t:.2f}u")

    conn.close()
    print()
    print("Note: spread < 2u across ±20% weight range indicates low razor-tuning risk.")


if __name__ == "__main__":
    main()
