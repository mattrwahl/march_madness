"""
Overlap-tiered analysis (budget-neutral, 1 unit = $100).

Three strategies, all $200/yr total stake:
  A) Flat:          $25 all 8 picks
  B) Tiered:        $37.50 top-4 by score / $12.50 bottom-4 by score
  C) Overlap:       3:1 ratio between picks in BOTH models vs fixed-8 only,
                    scaled so total = $200 each year regardless of overlap count.
                    With n overlap picks:
                      solo_bet    = 200 / (2n + 8)
                      overlap_bet = 3 * solo_bet = 600 / (2n + 8)
                    Special case n=4: overlap=$37.50, solo=$12.50 (same amounts
                    as tiered but assigned by cross-model agreement, not score rank).

Fixed-8 V2  = weight ID 11 (always 8 picks)
Variable-N V2 = weight ID 12 (always 4 picks at min-floor, threshold=1.2516)
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

BUDGET       = INITIAL_BET_DOLLARS * NUM_PICKS   # $200
OVERLAP_RATIO = 3.0   # overlap picks get 3x the stake of solo picks

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row


def load_weights(wid):
    c = conn.cursor()
    c.execute("SELECT * FROM mm_model_weights WHERE id=?", (wid,))
    row = c.fetchone()
    weights = np.array([row["w_" + f] for f in FEATURE_NAMES], dtype=float)
    return weights, row["w_threshold"]


w_fixed, _    = load_weights(11)   # fixed-8 flat/E8 V2
w_var, thresh = load_weights(12)   # variable-N flat/E8 V2

ALL_EVAL = sorted(TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS)


def overlap_bets(n_overlap):
    """
    Return (overlap_bet, solo_bet) maintaining 3:1 ratio and total=$200.
    Derivation: n*3Y + (8-n)*Y = 200  =>  Y = 200 / (2n + 8)
    """
    solo = BUDGET / (OVERLAP_RATIO * n_overlap + (NUM_PICKS - n_overlap))
    return round(OVERLAP_RATIO * solo, 6), round(solo, 6)


def season_row(season):
    teams = load_eligible_teams(conn, season)
    if not teams:
        return None

    picks_f = select_picks(teams, w_fixed, NUM_PICKS)
    picks_v = select_picks_threshold(
        teams, w_var, thresh, MIN_PICKS_PER_YEAR, MAX_PICKS_PER_YEAR
    )
    var_ids  = {p["team_id"] for p in picks_v}
    fids     = {p["team_id"] for p in picks_f}
    n_ol     = len(fids & var_ids)
    omap     = {p["team_id"]: fids - {p["team_id"]} for p in picks_f}

    ol_bet, solo_bet = overlap_bets(n_ol) if n_ol > 0 else (INITIAL_BET_DOLLARS, INITIAL_BET_DOLLARS)

    fu = tu = ou = 0.0

    for i, pick in enumerate(picks_f):
        tid = pick["team_id"]
        o   = omap[tid]

        # A) Flat $25
        pf, _ = simulate_pick_payout(
            conn, tid, season, initial_bet=INITIAL_BET_DOLLARS,
            cash_out_round=DEFAULT_CASH_OUT_ROUND,
            other_pick_ids=o, bet_style=BET_STYLE_FLAT,
        )
        fu += (pf - INITIAL_BET_DOLLARS) / 100.0

        # B) Tiered: top-4 $37.50, bottom-4 $12.50 (by fixed-8 score rank)
        tb = TIER1_BET if i < NUM_PICKS // 2 else TIER2_BET
        pt, _ = simulate_pick_payout(
            conn, tid, season, initial_bet=tb,
            cash_out_round=DEFAULT_CASH_OUT_ROUND,
            other_pick_ids=o, bet_style=BET_STYLE_FLAT,
        )
        tu += (pt - tb) / 100.0

        # C) Overlap-tiered: higher bet if also picked by variable-N
        ob = ol_bet if tid in var_ids else solo_bet
        po, _ = simulate_pick_payout(
            conn, tid, season, initial_bet=ob,
            cash_out_round=DEFAULT_CASH_OUT_ROUND,
            other_pick_ids=o, bet_style=BET_STYLE_FLAT,
        )
        ou += (po - ob) / 100.0

    return dict(fu=fu, tu=tu, ou=ou, n_ol=n_ol, ol_bet=ol_bet, solo_bet=solo_bet)


# ── Report ────────────────────────────────────────────────────────────────────
print("=" * 82)
print("OVERLAP-TIERED (budget-neutral) vs FLAT vs TIERED  [1 unit = $100]")
print(f"  All strategies: ${BUDGET:.0f}/yr total stake")
print(f"  Flat:    ${INITIAL_BET_DOLLARS:.2f} all 8 picks")
print(f"  Tiered:  ${TIER1_BET:.2f} top-4 by score / ${TIER2_BET:.2f} bottom-4 by score")
print(f"  Overlap: 3:1 ratio (overlap vs solo), scaled to ${BUDGET:.0f}/yr")
print(f"           e.g. 4OL -> ${overlap_bets(4)[0]:.2f}/${overlap_bets(4)[1]:.2f}  "
      f"3OL -> ${overlap_bets(3)[0]:.2f}/${overlap_bets(3)[1]:.2f}  "
      f"2OL -> ${overlap_bets(2)[0]:.2f}/${overlap_bets(2)[1]:.2f}  "
      f"1OL -> ${overlap_bets(1)[0]:.2f}/${overlap_bets(1)[1]:.2f}")
print("=" * 82)
print(f"  {'Season':<10} {'Flat':>8} {'Tiered':>8} {'Overlap':>8}  "
      f"{'OL$':>6} {'Solo$':>6}  #OL  Marker")
print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}  {'-'*6} {'-'*6}  ---  ------")

totals = dict(fu=0.0, tu=0.0, ou=0.0)

for season in ALL_EVAL:
    r = season_row(season)
    if r is None:
        print(f"  {season:<10}  no data")
        continue
    for k in totals:
        totals[k] += r[k]
    marker = ""
    if season in VAL_SEASONS:    marker = "[VAL]"
    elif season in TEST_SEASONS: marker = "[TEST]"
    print(f"  {season:<10} {r['fu']:>+8.3f}u {r['tu']:>+8.3f}u {r['ou']:>+8.3f}u  "
          f"${r['ol_bet']:>5.2f} ${r['solo_bet']:>5.2f}  {r['n_ol']:>3}  {marker}")

n = len(ALL_EVAL)
print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'TOTAL':<10} {totals['fu']:>+8.3f}u {totals['tu']:>+8.3f}u {totals['ou']:>+8.3f}u")
print(f"  {'Avg/yr':<10} {totals['fu']/n:>+8.3f}u {totals['tu']/n:>+8.3f}u {totals['ou']/n:>+8.3f}u")
print("=" * 82)
print()

# ── Subtotals ─────────────────────────────────────────────────────────────────
print("Subtotals by split:")
for label, seasons in [
    ("TRAIN (8yr)", TRAIN_SEASONS),
    ("VAL (2yr)",   VAL_SEASONS),
    ("TEST (2025)", TEST_SEASONS),
]:
    fu = tu = ou = 0.0
    for s in seasons:
        r = season_row(s)
        if r is None:
            continue
        fu += r["fu"]; tu += r["tu"]; ou += r["ou"]
    roi_f = fu / (len(seasons) * NUM_PICKS * INITIAL_BET_DOLLARS / 100.0)
    roi_t = tu / (len(seasons) * NUM_PICKS * INITIAL_BET_DOLLARS / 100.0)
    roi_o = ou / (len(seasons) * NUM_PICKS * INITIAL_BET_DOLLARS / 100.0)
    print(f"  {label:<12}  flat={fu:>+8.3f}u (ROI {roi_f:>+.4f})"
          f"   tiered={tu:>+8.3f}u (ROI {roi_t:>+.4f})"
          f"   overlap={ou:>+8.3f}u (ROI {roi_o:>+.4f})")

conn.close()
