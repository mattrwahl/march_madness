"""
Bootstrap risk analysis for v6-fixed-8-geomean (score-tiered, E8 cash-out).

Uses the 11 actual season-by-season tiered unit returns to estimate:
  - P(losing season)
  - Expected units/year distribution
  - Multi-year cumulative return scenarios (3, 5, 10 yr)
  - Worst-case drawdown distribution
"""
import sys, os, sqlite3
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from processors.features import (
    load_eligible_teams, compute_composite_features,
    select_picks_v6, V6_GEOMEAN_W,
)
from processors.model import simulate_pick_payout
from config import ALL_SEASONS, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "march_madness.db")
CASH_OUT  = 8        # E8
TIER1     = 37.50
TIER2     = 12.50
NUM_PICKS = 8
N_BOOT    = 200_000
np.random.seed(42)

# Only the 11 model seasons (modern analytics era, excl 2020)
MODEL_SEASONS = sorted(
    s for s in ALL_SEASONS
    if s in TRAIN_SEASONS or s in VAL_SEASONS or s in TEST_SEASONS
)


# ---------------------------------------------------------------------------
# Step 1: collect actual season-by-season tiered units
# ---------------------------------------------------------------------------

def get_season_units(conn):
    results = {}
    for season in MODEL_SEASONS:
        teams = load_eligible_teams(conn, season)
        if not teams:
            continue
        compute_composite_features(teams)
        picks = select_picks_v6(teams, V6_GEOMEAN_W, "coreB", NUM_PICKS)
        other_ids = {p["team_id"] for p in picks}
        season_total = 0.0
        for pick in picks:
            others = other_ids - {pick["team_id"]}
            bet = TIER1 if pick["pick_rank"] <= 4 else TIER2
            payout, _ = simulate_pick_payout(
                conn, pick["team_id"], season,
                initial_bet=bet, cash_out_round=CASH_OUT,
                other_pick_ids=others, bet_style="flat",
            )
            season_total += (payout - bet) / 100.0
        results[season] = round(season_total, 4)
    return results


# ---------------------------------------------------------------------------
# Step 2: helper — max drawdown of a cumulative return series
# ---------------------------------------------------------------------------

def max_drawdown(seq):
    """Max peak-to-trough drop in cumulative unit returns."""
    cum = np.cumsum(seq)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum          # drawdown at each point (>=0)
    return float(dd.max())


def consecutive_losses(seq):
    """Length of longest consecutive losing-season streak."""
    best, cur = 0, 0
    for v in seq:
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


# ---------------------------------------------------------------------------
# Step 3: bootstrap
# ---------------------------------------------------------------------------

def bootstrap(season_arr, n_years_list, n_boot=N_BOOT):
    n = len(season_arr)

    # --- per-season stats ---
    single = np.random.choice(season_arr, size=(n_boot,), replace=True)
    p_losing_single = (single < 0).mean()

    # --- multi-year scenario stats ---
    multi = {}
    for ny in n_years_list:
        samp = np.random.choice(season_arr, size=(n_boot, ny), replace=True)
        cum  = samp.sum(axis=1)
        mean = samp.mean(axis=1)

        # drawdown and losing streak for each bootstrap run
        dd_arr  = np.array([max_drawdown(row) for row in samp])
        cons_arr = np.array([consecutive_losses(row) for row in samp])
        frac_losing = (samp < 0).sum(axis=1) / ny  # fraction of losing seasons

        multi[ny] = dict(
            cum=cum, mean=mean, dd=dd_arr, cons=cons_arr, frac_losing=frac_losing,
            p_net_neg=(cum < 0).mean(),
            p_any_losing=(frac_losing > 0).mean(),
            p_half_losing=(frac_losing >= 0.5).mean(),
        )
    return p_losing_single, multi


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def pct(arr, p):
    return float(np.percentile(arr, p))

def split_label(season):
    if season in TRAIN_SEASONS: return "Train"
    if season in VAL_SEASONS:   return "Val"
    if season in TEST_SEASONS:  return "Test"
    return "?"

def bar(v, lo, hi, width=30):
    """ASCII bar showing where v sits in [lo, hi]."""
    if hi == lo:
        return "|" + " " * width + "|"
    pos = int((v - lo) / (hi - lo) * width)
    pos = max(0, min(width - 1, pos))
    s = list(" " * width)
    s[pos] = "#"
    return "|" + "".join(s) + "|"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("Loading season results...")
    season_units = get_season_units(conn)
    conn.close()

    arr = np.array(list(season_units.values()))
    n   = len(arr)

    # -----------------------------------------------------------------------
    # SECTION 1: empirical season-by-season
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("V6-GEOMEAN BOOTSTRAP RISK ANALYSIS  (score-tiered, E8 cash-out)")
    print("=" * 70)
    print()
    print("EMPIRICAL SEASON-BY-SEASON RESULTS (tiered units)")
    print("-" * 50)
    for s, u in season_units.items():
        bar_str = bar(u, -2, 12, 25)
        marker  = f"  [{split_label(s)}]"
        print(f"  {s}  {u:+7.3f}u  {bar_str}{marker}")
    print(f"  {'':4}  {'-------'}")
    print(f"  Mean   {arr.mean():+7.3f}u/yr")
    print(f"  Std    {arr.std(ddof=1):7.3f}u/yr")
    print(f"  Min    {arr.min():+7.3f}u  (worst single season)")
    print(f"  Max    {arr.max():+7.3f}u  (best single season)")
    print(f"  Losing seasons: {(arr < 0).sum()}/{n}  ({100*(arr<0).mean():.1f}% empirical rate)")
    print()

    # -----------------------------------------------------------------------
    # SECTION 2: bootstrap
    # -----------------------------------------------------------------------
    n_years_list = [3, 5, 10]
    print(f"Running bootstrap (n={N_BOOT:,} iterations per scenario)...")
    p_losing_single, multi = bootstrap(arr, n_years_list)
    print()

    # -----------------------------------------------------------------------
    # SECTION 3: single-season risk
    # -----------------------------------------------------------------------
    print("SINGLE-SEASON RISK PROFILE  (bootstrap)")
    print("-" * 50)
    print(f"  P(losing season)         {p_losing_single*100:5.1f}%")
    print(f"  Expected units/yr        {arr.mean():+.3f}u   (bootstrapped mean = {np.mean(np.random.choice(arr, N_BOOT)):+.3f}u)")
    print()
    print("  Annual return percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        v = pct(np.random.choice(arr, (N_BOOT,), replace=True), p)  # re-draw to get percentiles of bootstrap dist
        print(f"    p{p:2d}:  {v:+.2f}u")
    print()

    # -----------------------------------------------------------------------
    # SECTION 4: multi-year scenarios
    # -----------------------------------------------------------------------
    for ny in n_years_list:
        d = multi[ny]
        print(f"{ny}-YEAR SCENARIO  (bootstrap over {ny} seasons sampled with replacement)")
        print("-" * 50)
        print(f"  Expected cumulative return      {d['cum'].mean():+.2f}u  ({d['mean'].mean():+.2f}u/yr)")
        print(f"  P(net negative over {ny} yrs)      {d['p_net_neg']*100:5.1f}%")
        print(f"  P(at least 1 losing season)     {d['p_any_losing']*100:5.1f}%")
        print(f"  P(majority of seasons losing)   {d['p_half_losing']*100:5.1f}%")
        print()
        print(f"  Cumulative return percentiles:")
        for p in [5, 10, 25, 50, 75, 90, 95]:
            print(f"    p{p:2d}:  {pct(d['cum'], p):+.2f}u  ({pct(d['cum'], p)/ny:+.2f}u/yr)")
        print()
        print(f"  Worst drawdown within {ny}-yr run:")
        print(f"    Mean worst drawdown           {d['dd'].mean():.2f}u")
        print(f"    p50 worst drawdown            {pct(d['dd'], 50):.2f}u")
        print(f"    p75 worst drawdown            {pct(d['dd'], 75):.2f}u")
        print(f"    p90 worst drawdown            {pct(d['dd'], 90):.2f}u")
        print(f"    p95 worst drawdown            {pct(d['dd'], 95):.2f}u")
        print()
        print(f"  Longest consecutive losing streak within {ny}-yr run:")
        for k in range(ny + 1):
            p_k = (d['cons'] >= k).mean()
            if p_k < 0.01:
                break
            note = "  <-- most likely worst case" if k == int(np.median(d['cons'][d['cons']>0])) else ""
            print(f"    P(>= {k} in a row losing):  {p_k*100:5.1f}%{note}")
        print()

    # -----------------------------------------------------------------------
    # SECTION 5: contextual note
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  Empirical: {(arr<0).sum()} losing season(s) in {n} ({100*(arr<0).mean():.0f}%),")
    print(f"             mean {arr.mean():+.2f}u/yr, std {arr.std(ddof=1):.2f}u/yr.")
    print()
    d5 = multi[5]
    print(f"  Over a 5-year horizon:")
    print(f"    90% of bootstrap runs finish between"
          f" {pct(d5['cum'],5):+.1f}u and {pct(d5['cum'],95):+.1f}u cumulative.")
    print(f"    Median expected: {pct(d5['cum'],50):+.1f}u over 5 years.")
    print(f"    P(net loss over 5 yrs): {d5['p_net_neg']*100:.1f}%")
    print(f"    Worst drawdown 90th pct: {pct(d5['dd'],90):.1f}u  "
          f"(${pct(d5['dd'],90)*100:.0f} at $100/unit)")
    print()
    print("  Note: bootstrap treats each season as i.i.d. and resamples from the")
    print("  observed 11-season distribution. True out-of-sample variance is likely")
    print("  higher due to model/regime risk not captured by resampling historical returns.")
    print("=" * 70)


if __name__ == "__main__":
    main()
