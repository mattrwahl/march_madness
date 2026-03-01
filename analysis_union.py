"""
Union strategy analysis [1 unit = $100]:

  Both models flag team  -> 1/2 unit = $50/round
  Either model flags team -> 1/4 unit = $25/round

Pool = fixed-8 V2 UNION variable-N V2
  Fixed-8 always picks 8; variable-N always picks 4 (min floor).
  Union size = 8 + 4 - n_overlap, ranging 8-12.

Budget: n*$50 + (8-n)*$25 + (4-n)*$25
      = 50n + (12-2n)*25
      = 50n + 300 - 50n
      = $300/yr  (constant regardless of overlap count)

Compare against flat ($200/yr) and tiered ($200/yr) on ROI/$100 invested.
"""
import sys, sqlite3
import numpy as np

sys.path.insert(0, ".")

from processors.features import (
    FEATURE_NAMES, load_eligible_teams, select_picks, select_picks_threshold,
)
from processors.model import simulate_pick_payout
from config import (
    DB_PATH, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    INITIAL_BET_DOLLARS, NUM_PICKS, DEFAULT_CASH_OUT_ROUND,
    BET_STYLE_FLAT, TIER1_BET, TIER2_BET, MIN_PICKS_PER_YEAR, MAX_PICKS_PER_YEAR,
)

BOTH_BET   = 50.0   # 1/2 unit — both models agree
EITHER_BET = 25.0   # 1/4 unit — only one model flags

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row


def load_weights(wid):
    c = conn.cursor()
    c.execute("SELECT * FROM mm_model_weights WHERE id=?", (wid,))
    row = c.fetchone()
    weights = np.array([row["w_" + f] for f in FEATURE_NAMES], dtype=float)
    return weights, row["w_threshold"]


w_fixed, _    = load_weights(11)
w_var, thresh = load_weights(12)

ALL_EVAL = sorted(TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS)


def season_row(season):
    teams = load_eligible_teams(conn, season)
    if not teams:
        return None

    picks_f = select_picks(teams, w_fixed, NUM_PICKS)
    picks_v = select_picks_threshold(
        teams, w_var, thresh, MIN_PICKS_PER_YEAR, MAX_PICKS_PER_YEAR
    )

    fids    = {p["team_id"] for p in picks_f}
    var_ids = {p["team_id"] for p in picks_v}
    both    = fids & var_ids
    union   = fids | var_ids

    # Build a lookup: team_id -> team dict for all union members
    team_lookup = {t["team_id"]: t for t in teams}

    # other_pick_ids for collision detection = full union minus self
    fu = tu = uu = 0.0
    flat_stake = INITIAL_BET_DOLLARS * NUM_PICKS          # $200
    tier_stake = (NUM_PICKS // 2) * TIER1_BET + (NUM_PICKS // 2) * TIER2_BET  # $200
    union_stake = len(both) * BOTH_BET + (len(union) - len(both)) * EITHER_BET  # always $300

    # --- Flat & Tiered: over fixed-8 picks only ---
    fixed_other = {p["team_id"]: fids - {p["team_id"]} for p in picks_f}
    for i, pick in enumerate(picks_f):
        tid = pick["team_id"]
        o   = fixed_other[tid]

        pf, _ = simulate_pick_payout(
            conn, tid, season, initial_bet=INITIAL_BET_DOLLARS,
            cash_out_round=DEFAULT_CASH_OUT_ROUND,
            other_pick_ids=o, bet_style=BET_STYLE_FLAT,
        )
        fu += (pf - INITIAL_BET_DOLLARS) / 100.0

        tb = TIER1_BET if i < NUM_PICKS // 2 else TIER2_BET
        pt, _ = simulate_pick_payout(
            conn, tid, season, initial_bet=tb,
            cash_out_round=DEFAULT_CASH_OUT_ROUND,
            other_pick_ids=o, bet_style=BET_STYLE_FLAT,
        )
        tu += (pt - tb) / 100.0

    # --- Union strategy: over all union picks ---
    for tid in union:
        ob    = BOTH_BET if tid in both else EITHER_BET
        other = union - {tid}
        po, _ = simulate_pick_payout(
            conn, tid, season, initial_bet=ob,
            cash_out_round=DEFAULT_CASH_OUT_ROUND,
            other_pick_ids=other, bet_style=BET_STYLE_FLAT,
        )
        uu += (po - ob) / 100.0

    return dict(
        fu=fu, tu=tu, uu=uu,
        flat_stake=flat_stake, tier_stake=tier_stake, union_stake=union_stake,
        n_union=len(union), n_both=len(both), n_var_only=len(var_ids - fids),
    )


# ── Report ────────────────────────────────────────────────────────────────────
print("=" * 86)
print("UNION STRATEGY vs FLAT vs TIERED  [1 unit = $100]")
print(f"  Flat:   $200/yr  ($25 x 8 fixed-8 picks)")
print(f"  Tiered: $200/yr  ($37.50 top-4 / $12.50 bottom-4 by score)")
print(f"  Union:  $300/yr  ($50 both-model picks + $25 either-model picks)")
print(f"  Note: union budget is always $300/yr (algebraically constant)")
print("=" * 86)
print(f"  {'Season':<10} {'Flat':>8} {'Tiered':>8} {'Union':>8}  "
      f"{'FlatROI':>8} {'TierROI':>8} {'UnionROI':>9}  Picks  #Both  +VarOnly  Marker")
print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}  "
      f"{'-'*8} {'-'*8} {'-'*9}  -----  -----  --------  ------")

totals = dict(fu=0.0, tu=0.0, uu=0.0, fs=0.0, ts=0.0, us=0.0)

for season in ALL_EVAL:
    r = season_row(season)
    if r is None:
        print(f"  {season:<10}  no data")
        continue
    totals["fu"] += r["fu"];  totals["fs"] += r["flat_stake"]
    totals["tu"] += r["tu"];  totals["ts"] += r["tier_stake"]
    totals["uu"] += r["uu"];  totals["us"] += r["union_stake"]

    f_roi = r["fu"] / (r["flat_stake"]  / 100.0)
    t_roi = r["tu"] / (r["tier_stake"]  / 100.0)
    u_roi = r["uu"] / (r["union_stake"] / 100.0)

    marker = ""
    if season in VAL_SEASONS:    marker = "[VAL]"
    elif season in TEST_SEASONS: marker = "[TEST]"

    print(f"  {season:<10} {r['fu']:>+8.3f}u {r['tu']:>+8.3f}u {r['uu']:>+8.3f}u  "
          f"{f_roi:>+8.4f} {t_roi:>+8.4f} {u_roi:>+9.4f}  "
          f"{r['n_union']:>5}  {r['n_both']:>5}  {r['n_var_only']:>8}  {marker}")

n = len(ALL_EVAL)
print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}  {'-'*8} {'-'*8} {'-'*9}")
print(f"  {'TOTAL':<10} {totals['fu']:>+8.3f}u {totals['tu']:>+8.3f}u {totals['uu']:>+8.3f}u")
f_roi = totals["fu"] / (totals["fs"] / 100.0)
t_roi = totals["tu"] / (totals["ts"] / 100.0)
u_roi = totals["uu"] / (totals["us"] / 100.0)
print(f"  {'ROI/$100':<10} {f_roi:>+8.4f}  {t_roi:>+8.4f}  {u_roi:>+9.4f}")
print(f"  Avg/yr:    flat=${totals['fs']/n:.0f}  tiered=${totals['ts']/n:.0f}  union=${totals['us']/n:.0f}")
print("=" * 86)
print()

print("Subtotals by split:")
for label, seasons in [
    ("TRAIN (8yr)", TRAIN_SEASONS),
    ("VAL (2yr)",   VAL_SEASONS),
    ("TEST (2025)", TEST_SEASONS),
]:
    fu = tu = uu = fs = ts = us = 0.0
    for s in seasons:
        r = season_row(s)
        if r is None:
            continue
        fu += r["fu"]; tu += r["tu"]; uu += r["uu"]
        fs += r["flat_stake"]; ts += r["tier_stake"]; us += r["union_stake"]
    print(f"  {label:<12}"
          f"  flat={fu:>+8.3f}u (ROI {fu/(fs/100):>+.4f})"
          f"   tiered={tu:>+8.3f}u (ROI {tu/(ts/100):>+.4f})"
          f"   union={uu:>+8.3f}u (ROI {uu/(us/100):>+.4f})"
          f"   stake/yr=${us/len(seasons):.0f}")
