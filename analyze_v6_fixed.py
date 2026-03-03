"""
V6 fixed-weight model comparison: equal weights vs geomean vs Borda count.
Features: seed_rank_gap, conf_tourney_wins, dfi, tsi  (coreB set)
No optimization -- weights are pre-specified from SFM single-feature results.

Methods:
  equal   -- uniform 0.25 per feature (z-score normalized composite)
  geomean -- weights proportional to geomean(SFM val, SFM test) per feature
  borda   -- rank each feature independently, sum ranks (Borda count)

Geomean weight derivation (from single-feature model analysis on V5 features):
  seed_rank_gap:     val=+6.52u  test=+5.83u  geomean=6.165 -> w=0.2795
  conf_tourney_wins: val=+9.55u  test=+5.27u  geomean=7.094 -> w=0.3217
  dfi:               val=+6.95u  test=+2.87u  geomean=4.466 -> w=0.2025
  tsi:               val=+5.21u  test=+3.60u  geomean=4.331 -> w=0.1963

Run:
  python march_madness/analyze_v6_fixed.py
"""
import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.stats import rankdata

from processors.features import (
    FEATURES_V6B, FEATURE_NAMES_V6B,
    load_eligible_teams, compute_composite_features,
    _build_feature_matrix_v6,
)
from processors.model import simulate_pick_payout, INITIAL_BET_DOLLARS
from config import (
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    DEFAULT_CASH_OUT_ROUND, DEFAULT_BET_STYLE,
    NUM_PICKS, DB_PATH,
)

# ---------------------------------------------------------------------------
# Weight vectors  (feature order: srg, ctw, dfi, tsi)
# ---------------------------------------------------------------------------

EQUAL_W = np.array([0.25, 0.25, 0.25, 0.25])

_gm = np.array([
    np.sqrt(6.52 * 5.83),   # srg
    np.sqrt(9.55 * 5.27),   # ctw
    np.sqrt(6.95 * 2.87),   # dfi
    np.sqrt(5.21 * 3.60),   # tsi
])
GEOMEAN_W = _gm / _gm.sum()

METHODS = ["equal", "geomean", "borda"]
ROUND_LABELS = {65: "FF", 64: "R64", 32: "R32", 16: "S16", 8: "E8", 4: "F4", 2: "Champ"}

# ---------------------------------------------------------------------------
# R64 collision guard (local)
# ---------------------------------------------------------------------------

_OPP_SEED = {5: 12, 12: 5, 6: 11, 11: 6, 7: 10, 10: 7, 8: 9, 9: 8}


def _r64_collision(candidate, existing):
    opp = _OPP_SEED.get(candidate.get("seed"))
    reg = candidate.get("region")
    if opp is None or not reg:
        return False
    return any(p.get("seed") == opp and p.get("region") == reg for p in existing)


def _select_picks(teams, scores):
    indexed = sorted(zip(scores, range(len(teams))), reverse=True)
    picks = []
    for score, idx in indexed:
        if len(picks) >= NUM_PICKS:
            break
        team = teams[idx]
        if _r64_collision(team, picks):
            continue
        pick = dict(team)
        pick["score"] = round(float(score), 6)
        pick["pick_rank"] = len(picks) + 1
        picks.append(pick)
    return picks


# ---------------------------------------------------------------------------
# Scoring functions  (all take the z-scored feature matrix X)
# ---------------------------------------------------------------------------

def _score_equal(X):
    return (X @ EQUAL_W).tolist()


def _score_geomean(X):
    return (X @ GEOMEAN_W).tolist()


def _score_borda(X):
    """Borda count: rank each feature col (higher z = rank 1), sum ranks, negate."""
    n, k = X.shape
    rank_sum = np.zeros(n)
    for j in range(k):
        rank_sum += rankdata(-X[:, j], method="average")
    return (-rank_sum).tolist()   # negate: higher = better (lower rank sum)


_SCORER = {"equal": _score_equal, "geomean": _score_geomean, "borda": _score_borda}


# ---------------------------------------------------------------------------
# Season evaluation
# ---------------------------------------------------------------------------

def evaluate_season(conn, season):
    teams = load_eligible_teams(conn, season)
    if not teams:
        return None
    compute_composite_features(teams)
    X = _build_feature_matrix_v6(teams, FEATURES_V6B)

    out = {}
    for m in METHODS:
        scores = _SCORER[m](X)
        picks  = _select_picks(teams, scores)
        pids   = frozenset(p["team_id"] for p in picks)
        omap   = {p["team_id"]: pids - {p["team_id"]} for p in picks}

        total = 0.0
        for pick in picks:
            po, rnd = simulate_pick_payout(
                conn, pick["team_id"], season,
                cash_out_round=DEFAULT_CASH_OUT_ROUND,
                other_pick_ids=omap[pick["team_id"]],
                bet_style=DEFAULT_BET_STYLE,
            )
            u = (po - INITIAL_BET_DOLLARS) / 100.0
            pick["units"]      = u
            pick["round_exit"] = rnd
            total += u

        out[m] = {"picks": picks, "ids": pids, "units": total}

    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _tag(season):
    if season in VAL_SEASONS:
        return " [VAL]"
    if season in TEST_SEASONS:
        return " [TEST]"
    return ""


def _split(season):
    if season in VAL_SEASONS:
        return "val"
    if season in TEST_SEASONS:
        return "test"
    return "train"


def print_header():
    print("=" * 90)
    print("V6 FIXED-WEIGHT COMPARISON  (seed_rank_gap, conf_tourney_wins, dfi, tsi)")
    print("=" * 90)
    feat_str = "  ".join(f"{n}" for n in FEATURE_NAMES_V6B)
    print(f"  Features (order): {feat_str}")
    print()
    print(f"  {'Method':<10}  Weights (srg, ctw, dfi, tsi)")
    eq = "  ".join(f"{w:.3f}" for w in EQUAL_W)
    gm = "  ".join(f"{w:.3f}" for w in GEOMEAN_W)
    print(f"  {'equal':<10}  [{eq}]  (uniform)")
    print(f"  {'geomean':<10}  [{gm}]  (geomean of SFM val+test)")
    print(f"  {'borda':<10}  rank each feature, sum ranks (lower sum = higher score)")
    print()


def print_summary_table(season_results):
    print("=" * 90)
    print("  SEASON SUMMARY")
    print("=" * 90)
    print(f"  {'Season':<12}  {'Equal':>8}  {'Geomean':>9}  {'Borda':>8}"
          f"  | E=G  E=B  G=B  All3")
    print("  " + "-" * 72)

    split_totals = {
        "train": {m: 0.0 for m in METHODS},
        "val":   {m: 0.0 for m in METHODS},
        "test":  {m: 0.0 for m in METHODS},
    }
    grand = {m: 0.0 for m in METHODS}

    for season in TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS:
        if season not in season_results:
            continue
        r = season_results[season]
        eu = r["equal"]["units"]
        gu = r["geomean"]["units"]
        bu = r["borda"]["units"]
        eg = len(r["equal"]["ids"] & r["geomean"]["ids"])
        eb = len(r["equal"]["ids"] & r["borda"]["ids"])
        gb = len(r["geomean"]["ids"] & r["borda"]["ids"])
        a3 = len(r["equal"]["ids"] & r["geomean"]["ids"] & r["borda"]["ids"])
        tag = _tag(season)
        sp  = _split(season)

        print(f"  {season}{tag:<10}  {eu:>+8.3f}  {gu:>+9.3f}  {bu:>+8.3f}"
              f"  | {eg:>3}  {eb:>3}  {gb:>3}  {a3:>3}")

        for m in METHODS:
            split_totals[sp][m] += r[m]["units"]
            grand[m] += r[m]["units"]

    print("  " + "-" * 72)
    print(f"  {'TOTAL (11 seas)':<14}  {grand['equal']:>+8.3f}  {grand['geomean']:>+9.3f}  {grand['borda']:>+8.3f}")
    print()
    for label, key in [("Train (8 seasons)", "train"), ("Val   (2 seasons)", "val"), ("Test  (1 season) ", "test")]:
        st = split_totals[key]
        print(f"  {label}:  Equal={st['equal']:>+7.3f}u  "
              f"Geomean={st['geomean']:>+7.3f}u  Borda={st['borda']:>+7.3f}u")

    # Compare vs V2 val gate
    v2_val = 2.08
    print()
    print(f"  V2 val gate: +{v2_val:.2f}u  "
          f"| dVal: Equal={split_totals['val']['equal']-v2_val:>+.3f}  "
          f"Geomean={split_totals['val']['geomean']-v2_val:>+.3f}  "
          f"Borda={split_totals['val']['borda']-v2_val:>+.3f}")
    print()


def print_season_detail(season_results):
    print("=" * 90)
    print("  SEASON DETAIL  (X = method picked this team | * = unique to that method)")
    print("  Note: 'unique' means NOT picked by either of the other two methods.")
    print("=" * 90)

    for season in TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS:
        if season not in season_results:
            continue
        r = season_results[season]
        tag = _tag(season)

        # Units line
        u_str = "  ".join(f"{m}:{r[m]['units']:>+7.3f}u" for m in METHODS)
        print(f"\n  {season}{tag}  --  {u_str}")

        # Union of all picked team_ids
        all_ids = r["equal"]["ids"] | r["geomean"]["ids"] | r["borda"]["ids"]

        # Build team info lookup from any method's picks
        tid_info = {}
        for m in METHODS:
            for p in r[m]["picks"]:
                tid_info[p["team_id"]] = p

        # Sort union by equal method rank (or seed as fallback)
        def _sort_key(tid):
            for p in r["equal"]["picks"]:
                if p["team_id"] == tid:
                    return p["pick_rank"]
            return 99

        sorted_ids = sorted(all_ids, key=_sort_key)

        # Header
        print(f"  {'Team':<28}  {'Sd':>3}  {'Exit':<5}  {'units':>7}  EQ  GM  BO")
        print("  " + "-" * 60)

        for tid in sorted_ids:
            info      = tid_info[tid]
            name      = info.get("team_name", str(tid))[:27]
            seed      = info.get("seed", "?")
            rnd       = info.get("round_exit")
            exit_str  = ROUND_LABELS.get(rnd, "?") if rnd else "?"
            units     = info.get("units", 0.0)

            in_eq  = tid in r["equal"]["ids"]
            in_gm  = tid in r["geomean"]["ids"]
            in_bo  = tid in r["borda"]["ids"]

            eq_flag = "X" if in_eq else "."
            gm_flag = "X" if in_gm else "."
            bo_flag = "X" if in_bo else "."

            # Unique to one method only
            n_methods = sum([in_eq, in_gm, in_bo])
            if n_methods == 1:
                if in_eq:   eq_flag = "*"
                elif in_gm: gm_flag = "*"
                else:        bo_flag = "*"

            print(f"  {name:<28}  {seed:>3}  {exit_str:<5}  {units:>+7.3f}  {eq_flag}   {gm_flag}   {bo_flag}")

    print()


def print_divergence_summary(season_results):
    """Show which seasons the methods most disagreed, and direction."""
    print("=" * 90)
    print("  DIVERGENCE SUMMARY  (seasons where methods differed most in picks)")
    print("=" * 90)

    rows = []
    for season in TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS:
        if season not in season_results:
            continue
        r = season_results[season]
        all3 = len(r["equal"]["ids"] & r["geomean"]["ids"] & r["borda"]["ids"])
        min_pair = min(
            len(r["equal"]["ids"] & r["geomean"]["ids"]),
            len(r["equal"]["ids"] & r["borda"]["ids"]),
            len(r["geomean"]["ids"] & r["borda"]["ids"]),
        )
        # disagreement = how few are shared by all 3
        rows.append((all3, min_pair, season, r))

    rows.sort(key=lambda x: x[0])   # least consensus first

    print(f"  {'Season':<12}  {'All3':>5}  {'MinPair':>8}  {'Equal':>8}  {'Geomean':>9}  {'Borda':>8}")
    print("  " + "-" * 65)
    for all3, min_pair, season, r in rows:
        tag = _tag(season)
        eu = r["equal"]["units"]
        gu = r["geomean"]["units"]
        bu = r["borda"]["units"]
        print(f"  {season}{tag:<10}  {all3:>5}  {min_pair:>8}  {eu:>+8.3f}  {gu:>+9.3f}  {bu:>+8.3f}")

    print()


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    all_seasons = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS
    print(f"Evaluating {len(all_seasons)} seasons...", flush=True)

    season_results = {}
    for season in all_seasons:
        r = evaluate_season(conn, season)
        if r:
            season_results[season] = r
        print(f"  {season} done", flush=True)

    conn.close()
    print()

    print_header()
    print_summary_table(season_results)
    print_season_detail(season_results)
    print_divergence_summary(season_results)


if __name__ == "__main__":
    main()
