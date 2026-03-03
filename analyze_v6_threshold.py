"""
v6-fixed-8-geomean threshold analysis.

The geomean weights are fixed. This script sweeps a single parameter —
the composite-score cutoff — to find whether concentrating on the
highest-conviction picks improves ROI per pick vs the flat 8-pick baseline.

Composite score = X @ V6_GEOMEAN_W, where X is the per-feature z-scored matrix.
The score itself is already on a z-score-like scale; we threshold directly on it.

Output:
  1. Threshold sweep table  — for each cutoff, avg picks/yr, units, ROI/pick
  2. Per-season breakdown   — pick-by-pick scores and payouts for the full 8-pick run
  3. Score distribution     — histogram of scores so we can see where the mass lies

Run:
  python march_madness/analyze_v6_threshold.py
"""
import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from processors.features import (
    FEATURES_V6B, FEATURE_NAMES_V6B,
    V6_GEOMEAN_W,
    load_eligible_teams, compute_composite_features,
    _build_feature_matrix_v6,
    _r64_collision,
)
from processors.model import simulate_pick_payout, INITIAL_BET_DOLLARS
from config import (
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    DEFAULT_CASH_OUT_ROUND, DEFAULT_BET_STYLE,
    MIN_PICKS_PER_YEAR, MAX_PICKS_PER_YEAR,
    DB_PATH,
)

ALL_SEASONS  = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS
ROUND_LABELS = {65: "FF", 64: "R64", 32: "R32", 16: "S16", 8: "E8", 4: "F4", 2: "Champ"}


# ---------------------------------------------------------------------------
# Score + threshold selection
# ---------------------------------------------------------------------------

def _season_scores(conn, season):
    """Return (teams, scores) — teams with their geomean composite scores."""
    teams = load_eligible_teams(conn, season)
    if not teams:
        return [], []
    compute_composite_features(teams)
    X = _build_feature_matrix_v6(teams, FEATURES_V6B)
    scores = (X @ V6_GEOMEAN_W).tolist()
    return teams, scores


def _threshold_picks(teams, scores, threshold, min_n=MIN_PICKS_PER_YEAR, max_n=MAX_PICKS_PER_YEAR):
    """
    Pick all teams with score >= threshold, respecting min/max and R64 guard.
    min_n is a floor (take at least this many even if below threshold).
    """
    indexed = sorted(zip(scores, range(len(teams))), reverse=True)
    picks = []
    for score, idx in indexed:
        if len(picks) >= max_n:
            break
        if len(picks) >= min_n and score < threshold:
            break
        team = teams[idx]
        if _r64_collision(team, picks):
            continue
        pick = dict(team)
        pick["score"]      = round(float(score), 6)
        pick["pick_rank"]  = len(picks) + 1
        picks.append(pick)
    return picks


def _season_payout(conn, picks, season, cash_out_round=DEFAULT_CASH_OUT_ROUND):
    """Compute flat $25 payouts for a pick list. Returns (total_units, pick_rows)."""
    pick_ids = frozenset(p["team_id"] for p in picks)
    total = 0.0
    rows  = []
    for pick in picks:
        other = pick_ids - {pick["team_id"]}
        po, rnd = simulate_pick_payout(
            conn, pick["team_id"], season,
            cash_out_round=cash_out_round,
            other_pick_ids=other,
            bet_style=DEFAULT_BET_STYLE,
        )
        u = (po - INITIAL_BET_DOLLARS) / 100.0
        total += u
        rows.append((pick, u, rnd))
    return total, rows


# ---------------------------------------------------------------------------
# 1. Threshold sweep
# ---------------------------------------------------------------------------

def print_threshold_sweep(conn):
    """
    Sweep threshold from -2.0 to +1.5 in 0.25 steps.
    For each threshold, show avg picks/yr and units/ROI per split.
    """
    # Precompute all season scores
    season_data = {}
    for s in ALL_SEASONS:
        teams, scores = _season_scores(conn, s)
        if teams:
            season_data[s] = (teams, scores)

    thresholds = [round(t * 0.25, 2) for t in range(-8, 7)]  # -2.0 to +1.5 step 0.25

    print("=" * 90)
    print("v6-fixed-8-geomean THRESHOLD SWEEP")
    print("Fixed weights: " + "  ".join(f"{n}={w:.4f}" for n, w in zip(FEATURE_NAMES_V6B, V6_GEOMEAN_W)))
    print(f"Threshold applied to composite score (weighted sum of per-feature z-scores).")
    print(f"min_picks={MIN_PICKS_PER_YEAR}  max_picks={MAX_PICKS_PER_YEAR}  cash_out=E8  flat $25/pick")
    print("=" * 90)
    print(f"  {'Thresh':>7}  {'AvgN':>5}  {'Train':>9}  {'ROI/pk':>8}  {'Val':>9}  {'ROI/pk':>8}  "
          f"{'Test':>9}  {'ROI/pk':>8}  {'Total':>9}")
    print("  " + "-" * 86)

    # Baseline = 8-pick fixed (threshold = -inf effectively, max_n=8)
    baseline_row = None

    for thr in thresholds:
        train_u = train_n = 0.0
        val_u   = val_n   = 0.0
        test_u  = test_n  = 0.0

        for s in ALL_SEASONS:
            if s not in season_data:
                continue
            teams, scores = season_data[s]
            picks = _threshold_picks(teams, scores, thr)
            if not picks:
                continue
            units, _ = _season_payout(conn, picks, s)
            n = len(picks)

            if s in TRAIN_SEASONS:
                train_u += units
                train_n += n
            elif s in VAL_SEASONS:
                val_u   += units
                val_n   += n
            else:
                test_u  += units
                test_n  += n

        tr_roi = train_u / train_n if train_n else 0.0
        vl_roi = val_u   / val_n   if val_n   else 0.0
        te_roi = test_u  / test_n  if test_n  else 0.0
        total  = train_u + val_u + test_u
        avg_n  = (train_n + val_n + test_n) / len(season_data) if season_data else 0

        marker = ""
        if abs(thr) < 0.001:              # thr == 0.0
            marker = " <-- zero cutoff"
        elif abs(thr + 1.0) < 0.001:     # thr == -1.0 (broad selection)
            marker = " <-- broad"

        print(f"  {thr:>+7.2f}  {avg_n:>5.1f}  {train_u:>+9.3f}  {tr_roi:>+8.4f}  "
              f"{val_u:>+9.3f}  {vl_roi:>+8.4f}  {test_u:>+9.3f}  {te_roi:>+8.4f}  "
              f"{total:>+9.3f}{marker}")

        if abs(thr - (-99)) < 0.001 or (baseline_row is None and train_n > 0):
            baseline_row = (train_u, val_u, test_u, avg_n)

    print()
    # Print fixed-8 (no threshold) as explicit baseline
    fix_train = fix_val = fix_test = fix_n = 0.0
    for s in ALL_SEASONS:
        if s not in season_data:
            continue
        teams, scores = season_data[s]
        picks = _threshold_picks(teams, scores, threshold=-999.0, min_n=8, max_n=8)
        if not picks:
            continue
        units, _ = _season_payout(conn, picks, s)
        if s in TRAIN_SEASONS:
            fix_train += units; fix_n += len(picks)
        elif s in VAL_SEASONS:
            fix_val += units
        else:
            fix_test += units
    fix_total = fix_train + fix_val + fix_test
    fix_roi   = fix_total / fix_n if fix_n else 0.0
    print(f"  FIXED-8 BASELINE (threshold=-inf, always 8 picks):")
    print(f"    train={fix_train:+.3f}u  val={fix_val:+.3f}u  test={fix_test:+.3f}u  "
          f"total={fix_total:+.3f}u  ROI/pick={fix_roi:+.4f}")
    print()


# ---------------------------------------------------------------------------
# 2. Per-season score distribution
# ---------------------------------------------------------------------------

def print_score_distribution(conn):
    """Show per-season pick scores so we can see where the mass lies."""
    print("=" * 90)
    print("PER-SEASON SCORE DISTRIBUTION (all 8 picks, sorted by score desc)")
    print("Shows composite score and payout for each pick.")
    print("=" * 90)

    for s in ALL_SEASONS:
        teams, scores = _season_scores(conn, s)
        if not teams:
            continue
        picks = _threshold_picks(teams, scores, threshold=-999.0, min_n=8, max_n=8)
        if not picks:
            continue
        _, rows = _season_payout(conn, picks, s)

        tag = ""
        if s in VAL_SEASONS:  tag = " [VAL]"
        if s in TEST_SEASONS: tag = " [TEST]"

        # Total units for season
        total_u = sum(u for _, u, _ in rows)
        print(f"\n  {s}{tag}  total={total_u:+.3f}u")
        print(f"  {'#':>2}  {'Team':<28}  {'Sd':>3}  {'Score':>8}  {'Exit':<5}  {'Units':>7}")
        print("  " + "-" * 60)

        for pick, u, rnd in rows:
            exit_str = ROUND_LABELS.get(rnd, "?") if rnd else "?"
            print(f"  {pick['pick_rank']:>2}  {pick['team_name']:<28}  "
                  f"{pick.get('seed','?'):>3}  {pick['score']:>8.4f}  "
                  f"{exit_str:<5}  {u:>+7.3f}u")
    print()


# ---------------------------------------------------------------------------
# 3. Specific threshold comparisons
# ---------------------------------------------------------------------------

def print_threshold_detail(conn, thresholds_to_show):
    """Show which picks are kept/dropped at specific thresholds vs full 8."""
    print("=" * 90)
    print("THRESHOLD DETAIL -- picks kept vs dropped at selected cutoffs")
    print("=" * 90)

    season_data = {}
    for s in ALL_SEASONS:
        teams, scores = _season_scores(conn, s)
        if teams:
            season_data[s] = (teams, scores)

    for thr in thresholds_to_show:
        train_u = val_u = test_u = 0.0
        train_n = val_n = test_n = 0
        print(f"\n  --- threshold >= {thr:+.2f} ---")
        print(f"  {'Season':<12}  {'N':>3}  {'Units':>8}  {'ROI':>8}  Picks kept")

        for s in ALL_SEASONS:
            if s not in season_data:
                continue
            teams, scores = season_data[s]
            picks_full   = _threshold_picks(teams, scores, -999.0, min_n=8, max_n=8)
            picks_thresh = _threshold_picks(teams, scores, thr)
            units, _ = _season_payout(conn, picks_thresh, s)
            n = len(picks_thresh)
            roi = units / n if n else 0.0

            kept_ids   = {p["team_id"] for p in picks_thresh}
            drop_names = [p["team_name"].split()[0] for p in picks_full
                          if p["team_id"] not in kept_ids]
            tag = ""
            if s in VAL_SEASONS:  tag = "[V]"
            if s in TEST_SEASONS: tag = "[T]"

            drop_str = "  dropped: " + ", ".join(drop_names) if drop_names else ""
            print(f"  {s} {tag:<4}  {n:>3}  {units:>+8.3f}u  {roi:>+8.4f}{drop_str}")

            if s in TRAIN_SEASONS:
                train_u += units; train_n += n
            elif s in VAL_SEASONS:
                val_u += units; val_n += n
            else:
                test_u += units; test_n += n

        tr_roi = train_u / train_n if train_n else 0.0
        vl_roi = val_u   / val_n   if val_n   else 0.0
        te_roi = test_u  / test_n  if test_n  else 0.0
        print(f"  {'TOTAL':<12}  {'':>3}  {'':>8}  {'':>8}  "
              f"train={train_u:+.3f}u/{train_n}pk ({tr_roi:+.4f}/pk)  "
              f"val={val_u:+.3f}u/{val_n}pk ({vl_roi:+.4f}/pk)  "
              f"test={test_u:+.3f}u/{test_n}pk ({te_roi:+.4f}/pk)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print_threshold_sweep(conn)
    print_score_distribution(conn)
    print_threshold_detail(conn, thresholds_to_show=[0.25, 0.00, -0.25, -0.50])

    conn.close()


if __name__ == "__main__":
    main()
