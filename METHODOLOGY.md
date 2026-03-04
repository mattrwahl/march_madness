# March Madness Betting Model — Methodology & Approach

## Overview

This project builds a systematic, data-driven model for betting on NCAA Tournament games.
The core thesis is that the NCAA selection committee systematically under-seeds certain
teams, and those teams offer positive expected value when bet on the moneyline in early
tournament rounds. The model identifies those teams, sizes bets appropriately, and tracks
returns over time.

The approach is deliberately narrow and disciplined — we are not trying to predict the
entire bracket. We are looking for a specific, repeatable edge in a subset of games where
market pricing is most likely to be stale relative to true team quality.

**Current primary model:** `v6-fixed-8-geomean` — 4 features, pre-specified geomean
weights, fixed 8 picks per year, flat $25/pick, E8 cash-out.
Total across 11 seasons (2014-2025): **+28.50u**. Val: **+4.37u**. Test 2025: **+4.68u**.

---

## The Core Thesis

NCAA seeds are set by a committee roughly 2-3 weeks before the tournament. During that
window, sportsbooks set opening moneylines that closely track seed expectations. A team
seeded 10th is expected to be a significant underdog against a 7-seed; the market prices
it accordingly. If the model correctly identifies that the 10-seed is actually a 6- or
7-quality team based on objective metrics, that team offers consistent positive expected
value across the R64 and R32 games.

Key insight: **the market mispricing is largest in early rounds**, before the tournament
field narrows and public attention sharpens. By cashing out at the Elite Eight (E8) at the
latest, we capture the maximum mispricing window while avoiding the high-variance late
rounds where 1- and 2-seeds dominate and odds compress.

---

## Bet Structure

### Unit Definition
One unit = $100. Bets are sized in quarter-units ($25 per pick per round).

### Flat Betting (primary strategy for v6-geomean)
Each pick is bet independently at $25 per round:
- **Win**: lock in that round's profit at the moneyline payout; move to the next round.
- **Loss**: lose $25 for that round only. All prior profits from that pick are retained.
- **Two picks meet each other**: no bet is placed that round (void).

This is intentionally conservative. Profits compound through winning rounds, but a loss
never wipes prior gains. The result is asymmetric upside: a team that wins 3 rounds
generates 3 independent profits; a team that loses R64 costs only one $25 stake.

**Why flat and not tiered for v6-geomean:** The biggest winners span all 8 score ranks
with corrected-sign weights — Loyola Chicago 2018 (#1, +2.34u), Oregon State 2021 (#2,
+4.31u), NC State 2024 (#2, +4.01u), Nevada 2018 (#5, +3.44u), Liberty 2019 (#7,
+2.48u). The composite score identifies *direction* (under-seeded and peaking), not
the magnitude of tournament upside. Tiered comparison has not been re-run after the
2026-03-03 sign fix — re-evaluate before using tiered for v6-geomean.

### Cash-Out Rule
Teams are bet through a maximum of 3 rounds (R64, R32, S16), cashing out after winning
the Sweet Sixteen (entering the Elite Eight). This is the **E8 cash-out**:
- Caps exposure at rounds where upsets are still common
- Avoids F4/Championship rounds where top seeds dominate and odds compress
- Maintains a clean, repeatable rule requiring no in-tournament judgment calls

### Budget
8 picks × $25/round = $200 total annual budget. Each pick is independent; the budget is
not shared or redistributed based on outcomes.

---

## Team Selection

### Eligible Pool
Seeds 5 through 12, excluding any team that played in the First Four (play-in games).
Seeds 1-4 are correctly priced as favorites and rarely offer positive expected value.
Seeds 13-16 face structural mismatches in R64 that make occasional upsets too hard to
predict systematically.

### R64 Collision Guard
Within each region, certain seed pairs always face each other in R64:
- 5 vs 12, 6 vs 11, 7 vs 10, 8 vs 9

If the model selects both teams from a seed pair in the same region, neither bet is placed
in R64 (a guaranteed $0 outcome). To prevent this, the selection algorithm skips any
candidate that would create a guaranteed R64 matchup with an already-selected pick. The
slot goes to the next-highest-scoring non-colliding team. Implemented via `_r64_collision()`
in `processors/features.py`.

### How 8 Picks Are Selected
The model computes a composite score for every eligible team. The top 8 non-colliding
teams are selected. Feature values are z-score normalized within the eligible field
each season, making the model self-calibrating across years with different talent levels.

---

## Feature Engineering

### v6-fixed-8-geomean: 4 Features (current primary model)

| # | Feature | Direction | Rationale |
|---|---------|-----------|-----------|
| 1 | `seed_rank_gap` | lower = better | `net_rank - (seed × 10)`. Negative = under-seeded. The primary mispricing signal. Weight is **negative** (under-seeded = better; sign-corrected 2026-03-03). |
| 2 | `conf_tourney_wins` | higher = better | Games won in conference tournament. Hottest teams entering the NCAA field. Highest combined SFM signal: val +9.55u, test +5.27u. |
| 3 | `dfi` | lower raw = better | Defensive Friction Index: `opp_efg_pct + opp_3p_pct`. Composite defensive quality. Negated before scoring so higher z = better. |
| 4 | `tsi` | lower raw = better | Tempo Stability Index: `tempo_std / mean_pace`. Functions in practice as a defensive rank proxy (r=+0.66 with def_rank). Negated before scoring. |

All 4 are z-scored per feature within the eligible field each season. The composite
score is the weighted sum `X @ weights`, where X is the (n_teams × 4) z-scored matrix.

### V2 Legacy: 10 Features (fixed-8 and variable-N V2 models)

| # | Feature | Direction | Rationale |
|---|---------|-----------|-----------|
| 1 | `seed_rank_gap` | higher = better | Same as above |
| 2 | `def_rank` | lower = better | NET defensive ranking |
| 3 | `opp_efg_pct` | lower = better | Opponent effective FG%. Best single defensive metric. |
| 4 | `net_rating` | higher = better | Adjusted net efficiency margin |
| 5 | `tov_ratio` | lower = better | Turnover ratio |
| 6 | `ft_pct` | higher = better | Free throw %. Clutch shooting signal. |
| 7 | `oreb_pct` | higher = better | Offensive rebound %. Second chances. |
| 8 | `pace` | raw | Possessions/game. Model learns optimal range. |
| 9 | `conf_tourney_wins` | higher = better | Conf tournament wins |
| 10 | `conf_tourney_avg_margin` | higher = better | Avg point margin in conf tournament |

### Composite Feature Indices (V5/V6 experimentation)

Computed by `compute_composite_features()` in `features.py`. Used in V5 and V6 optimizer
attempts; only `dfi` and `tsi` survived into the final v6-geomean model.

| Index | Formula | Notes |
|-------|---------|-------|
| **dfi** | `opp_efg_pct + opp_3p_pct` | Retained in v6-geomean. Highly correlated with opp_efg_pct (r=+0.887) — partially redundant; adds three-point defense signal. |
| **tsi** | `tempo_std / mean_pace` | Retained in v6-geomean. Intended as tempo consistency; functions as def_rank proxy (r=+0.660). |
| **cpi** | `(oreb_pct - tov_ratio) / 2` | Possession control index. Dominated by oreb_pct (r=+0.596); tov_ratio partially cancels. Not used in v6-geomean. |
| **ftli** | `ft_pct × opp_foul_rate` | Free throw leverage. Modest predictive value. Not used in v6-geomean. |
| **spmi** | `pace × tsi` | Speed-momentum. Near-zero predictive value (r=+0.015 with targets). Dropped. |

### Why Conference Tournament Features
Regular season stats capture baseline quality, but the NCAA Tournament begins mid-March.
A team that peaks late — winning their conference tournament impressively — is demonstrably
hotter than their regular-season numbers suggest. The NET ranking already incorporates
strength of schedule, so conference tournament wins add genuine new information rather than
restating existing metrics.

Conference tournaments are self-contained brackets with intra-conference competition,
making margins directly comparable. Last-10-games regular season margins were considered
and rejected because inter-conference schedule difficulty is not comparable.

---

## Weight Determination

### The Problem: Joint Optimization on 8 Seasons

All V5 and V6 optimizer attempts used scipy's `differential_evolution` to jointly optimize
feature weights. Every attempt produced **weight collapse** — the optimizer concentrated
nearly all weight on `seed_rank_gap` and zeroed out the remaining features regardless of
L2 lambda (0.01 tested for V5; 0.1 for V6). With 8 training seasons and 4-8 free
parameters, the system is fundamentally underdetermined. The optimizer finds a degenerate
near-single-feature solution at every lambda setting.

This makes a joint optimizer counterproductive on this dataset. The geomean approach
provides a principled alternative.

### The Solution: Geomean of Single-Feature Model Results

Each feature is run in isolation as a single-feature model (SFM): train on 8 seasons,
evaluate on val (2023-2024) and test (2025). The weight for each feature is set to:

```
w_i = geomean(SFM_val_i, SFM_test_i) = sqrt(SFM_val_i × SFM_test_i)
```

Then all weights are normalized to sum to 1.

**Why geomean and not arithmetic mean:** Geomean penalizes features that are strong on
one split but weak on another. A feature with val=+9.55u but test=+0.5u gets a much lower
weight than its arithmetic mean would suggest. This naturally rewards features that
generalize across both unseen splits without re-running optimization on the full dataset.

**Results:**

| Feature | SFM Val | SFM Test | Geomean | Final Weight |
|---------|---------|---------|---------|-------------|
| `seed_rank_gap` | +6.52u | +5.83u | 6.165 | **-0.2795** |
| `conf_tourney_wins` | +9.55u | +5.27u | 7.094 | **+0.3217** |
| `dfi` | +6.95u | +2.87u | 4.466 | **+0.2025** |
| `tsi` | +5.21u | +3.60u | 4.331 | **+0.1963** |

`conf_tourney_wins` gets the highest weight (0.322) because it has both the strongest
val signal and reasonable test stability. `seed_rank_gap` gets the second-highest (0.280)
because it is the most stable feature (val/test ratio 0.89). Its weight is **negative**:
lower seed_rank_gap = more under-seeded = better bet. The SFM optimizer found bounds
(-3.0, 0.0); the geomean procedure uses magnitude only, so the sign is re-applied
explicitly in `features.py` (sign-corrected 2026-03-03).

### Objective Function (V2 Fixed-8, legacy)
Joint DE optimization — maximize total units won, L2 regularized:

```
objective = -sum(simulate_season(weights, s) for s in train_seasons)
            + lambda_l2 * sum(weights^2)
```

lambda = 0.01. Works on 10 features because V2 has more seasons and the weight collapse
is less severe than with 4-8 composite features on the same 8 seasons.

### Objective Function (Variable-N V2, legacy)
Maximize ROI (units per pick), L2 regularized:

```
objective = -(total_units / total_picks) + lambda_l2 * sum(weights^2)
```

Converged to threshold=1.252, always selecting exactly 4 teams (min floor). Used as
a conviction signal for the V2 overlap-tiered strategy, not as a standalone betting model.

---

## Threshold / Variable-N Investigation for v6-geomean

After finalizing v6-fixed-8-geomean, we investigated whether thresholding on the
composite score (taking only the picks where score >= cutoff) could improve ROI per pick.
Since weights are fixed, this is a 1-dimensional sweep — no optimizer needed.

**Finding: fixed-8 is already the optimal pick count. No threshold strategy improves it.**

Three reasons:
1. **All 8 composite scores are positive every season** (range 0.13-1.49). The model
   has already gated on quality; there is no dead weight to prune.
2. **The biggest winners consistently rank 5-8 by model score.** The geomean score
   identifies *direction* (this team is better than their seed implies), not *magnitude*
   of tournament upside. Raising the cutoff systematically excludes the upset specialists.
3. **Expanding to 12 picks** (lowering the threshold below existing scores) adds
   low-conviction picks that mostly lose R64, halving ROI/pick with negligible total gain.

Summary of sweep results:

| Threshold | Avg N/yr | Val units | Test units | Total 11-season |
|-----------|----------|-----------|-----------|----------------|
| >= +0.50 | 4.4 | +1.74u | +2.23u | +12.3u |
| >= +0.25 | 7.6 | +3.55u | +2.53u | +22.5u |
| **Fixed-8** | **8.0** | **+4.37u** | **+4.68u** | **+28.5u** |
| Expand to 12 | 12.0 | +8.22u | +4.30u | +29.4u |

Note: threshold rows computed with original (incorrect) sign; fixed-8 row updated
to reflect corrected-sign weights. Conclusion (fixed-8 optimal) is unchanged.

Note: the 12-pick expansion shows better val but that is 2 seasons of noise. Train
and test both decline; ROI/pick drops from +0.332 to +0.175.

---

## Season Split & Validation Framework

```
Train:  2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022  (8 seasons)
Val:    2023, 2024                                        (2 seasons)
Test:   2025                                              (true holdout)
```

**Why 2014 as start year:** Modern analytics-era seeding began around 2014, coinciding
with widespread adoption of NET-style efficiency metrics by the selection committee.
Data from 2008-2013 exists but reflects a different selection methodology.

**2020 excluded** everywhere: no tournament (COVID-19).

**Strict holdout:** The test set (2025) was never examined until the V2 model was
finalized. No hyperparameter tuning or feature selection decisions were informed by
2025 results across any model version.

**Val gate:** A new model or strategy must exceed the V2 baseline on val (+2.08u flat)
to advance to testing. V6-geomean cleared this by +4.43u (val: +6.51u).

---

## Model Iterations

### V1 — Initial Model (archived)
- **8 features**: seed_rank_gap, def_rank, opp_efg_pct, net_rating, tov_ratio, ft_pct, oreb_pct, pace
- Fixed 8 picks, flat $25/round, E8 cash-out, no collision guard, no L2, no conf tourney signal
- First Four teams were incorrectly included in the eligible pool
- Results: **+12.70u train+val | -0.65u on 2025 holdout**

### V2 — Fixed-8 and Variable-N (active for legacy overlap strategy)
Addressed all V1 limitations: First Four filter, R64 collision guard, L2 regularization,
conference tournament features (features 9 & 10), tiered betting option.

Fixed-8 V2 results:
```
Train+val (10 seasons): +29.33u flat   avg +2.93u/yr   one losing year (2017: -0.50u)
Val (2023+2024):        +2.04u         (-0.65u / +2.68u)
Test 2025:              +3.29u         (Ole Miss S16 +1.25u, Arkansas S16 +1.88u)
```

Variable-N V2 (conviction signal): threshold=1.252, always picks 4 teams.
Used only as part of the overlap-tiered strategy (see below), not standalone.

**Overlap-tiered V2** (best V2 result): when both fixed-8 and variable-N agree on a pick,
that pick receives a higher stake (3:1 ratio maintained, total always $200/yr).
Train+val: **+43.19u** | Test 2025: +3.97u | ROI/$100: +1.98.

### V3 — region_top4_net_avg (2026-03-01, REVERTED)
Feature added: average NET rank of seeds 1-4 in the team's region (path difficulty).
Failed because it is a region-level constant — all eligible teams in a region share the
same value, adding inter-region noise with no intra-region discrimination.
Overlap val dropped; variable-N val collapsed from +0.547u/pick to -0.008u/pick. Reverted.

### V4 — opp_seed_rank_gap (2026-03-01, REVERTED)
Feature added: R64 opponent's seed_rank_gap (higher = weaker opponent).
Train improved (+31.15u vs V2 +29.33u) but val failed the gate (+1.55u vs +2.08u).
Classic mild overfitting: 8v9 pairs are near-coin-flips; 5v12/6v11 pairs have little
variation in opponent quality year to year. Variable-N not trained; V4 weights deleted.

### V5 — 8-feature composite model (2026-03-02, FAILED)
Features: seed_rank_gap, net_rating, conf_tourney_wins, cpi, dfi, ftli, spmi, tsi.
Constrained DE optimizer (srg<0, all others>0), lambda=0.01.
**Weight collapse**: optimizer produced near-zero weights for all features except srg.
Train: +32.16u. Val: +0.45u — well below the +2.08u gate. FAILED.
Root cause: 8 composite features × 8 training seasons is fundamentally underdetermined.
SPMI had near-zero predictive value (r=+0.015 with targets). CPI dominated by oreb_pct.

### V6-coreA/coreB Optimizer (2026-03-02, FAILED)
- **V6-coreA**: seed_rank_gap, conf_tourney_wins, opp_efg_pct, tsi. DE, lambda=0.1.
- **V6-coreB**: seed_rank_gap, conf_tourney_wins, dfi, tsi. DE, lambda=0.1.

Both produced identical near-zero weights and identical picks at every lambda tested.
Train: +32.16u each. Val: -1.64u each. Weight collapse persists even with 4 features.
Conclusion: joint optimization is not viable on 8 seasons regardless of feature count
or regularization strength. Abandoned in favor of pre-specified geomean weights.

### v6-fixed-8-geomean (2026-03-03, CURRENT PRIMARY)
Same 4-feature coreB set (srg, ctw, dfi, tsi). Weights derived from geomean(SFM_val,
SFM_test) per feature — no joint optimization. See Weight Determination above.

Results (sign-corrected 2026-03-03):
```
Train (8 seasons):  +19.45u   avg +2.43u/yr
Val   (2023+2024):  +4.37u    (+0.837u / +3.533u)   beats V2 gate by +2.29u
Test  (2025):       +4.68u    Michigan S16 +1.26u, McNeese R32 +1.48u, Drake R32 +1.25u
Total (11 seasons): +28.50u   ROI/pick: +0.324
```

Sign fix note: original (incorrect) positive weight for seed_rank_gap rewarded
over-seeded teams. Fix costs ~0.7u total vs original but correctly picks NC State 2024
(11-seed, 5 conf wins, E8) which the original model missed.
Variable-N / threshold sweep: fixed-8 is already optimal (see section above).
Tiered comparison not re-run after sign fix.
Weights stored in `mm_model_weights` with `notes LIKE 'model=v6-geomean %'` (id=20).

---

## Model Comparison Summary

```
Strategy                   Train+Val    Test 2025   Budget/yr   Notes
---------------------------------------------------------------------------
overlap-tiered V2          +43.19u     +3.97u       $200        V2 legacy; variable-N signal
v6-fixed-8-geomean         +23.82u     +4.68u       $200        CURRENT PRIMARY (flat only)
score-tiered V2            +34.37u     +3.14u       $200        V2 legacy
flat V2                    +32.62u     +3.29u       $200        V2 legacy
variable-N V2              +21.89u     +1.58u       varies      4 picks/yr; +0.547u/pick ROI
flat V1                    +12.70u     -0.65u       $200        archived
```

Note: v6-geomean train+val covers 10 seasons (+23.82u); total 11 seasons = +28.50u.
v6-geomean weights sign-corrected 2026-03-03 (seed_rank_gap weight now negative).
The overlap-tiered V2 total includes all 11 seasons at +43.19u but relies on a two-model
system with higher complexity; v6-geomean is the simpler, better-generalized primary model.

---

## Feature Diagnostics Findings (2026-03-02)

Run via `feature_diagnostics.py`. Three-part analysis:

**1. Inter-feature correlations (key findings):**
- dfi vs opp_efg_pct: r=+0.887 — DFI is nearly a linear transform of opp_efg_pct.
  Using DFI rather than raw opp_efg_pct adds the def_rank component at the cost of
  some redundancy.
- tsi vs def_rank: r=+0.660 — TSI does not measure tempo consistency in practice;
  it functions as a defensive rank proxy. Retained because it adds independent signal
  from def_rank that the SFM analysis confirmed.
- cpi vs oreb_pct: r=+0.596 — CPI is dominated by its oreb_pct component; tov_ratio
  partially cancels. This is why CPI failed to add signal in V5.
- spmi vs any target: r=+0.015 — effectively zero. Dropped entirely.

**2. Single-feature model (SFM) stability:**
The ratio of SFM_test to SFM_val indicates generalization:

| Feature | SFM Val | SFM Test | Ratio | Interpretation |
|---------|---------|---------|-------|----------------|
| seed_rank_gap | +6.52u | +5.83u | 0.89 | Most stable; reliable across splits |
| conf_tourney_wins | +9.55u | +5.27u | 0.55 | Highest val, moderate decay |
| dfi | +6.95u | +2.87u | 0.41 | Meaningful val signal, larger decay |
| tsi | +5.21u | +3.60u | 0.69 | Good stability; geomean rewards this |

**3. Fixed-weight comparison (equal / geomean / borda):**
All three methods applied to the same coreB feature set across 11 seasons:
- Equal weights (0.25 each): val=+4.04u, test=+2.68u
- **Geomean weights (sign-corrected)**: val=+4.37u, test=+4.68u — best test; clear winner overall
- Borda count (rank-sum): val=+4.77u, test=+0.13u

Borda's weakness: structurally over-democratic — a team ranked #1 on one feature
and #8 on three others scores identically to a team ranked #4 on all four.

---

## Data Pipeline

### Data Sources (all via CBBD API)
| Data | API | Notes |
|------|-----|-------|
| Tournament games & seeds | `GamesApi.get_games(tournament="NCAA")` | Seeds encoded in game fields |
| Efficiency ratings | `RatingsApi.get_adjusted_efficiency(season)` | NET, adj off/def ratings |
| Team stats | `StatsApi.get_team_season_stats(season)` | Four factors, pace, FT%, std devs |
| Betting lines | `LinesApi.get_lines(start/end_date_range)` | Mar 15 – Apr 10; BetOnline preferred |
| Conf tournament games | `GamesApi.get_games(start/end_date_range)` | Feb 25 – Mar 18; filtered by notes |

### Backfill Pipeline (6 steps per season)
1. Fetch tournament games; upsert teams; insert games
2. Derive tournament entries (seeds, regions) from game data
3. Fetch adjusted efficiency ratings (NET, adj off/def)
4. Fetch regular-season team stats; INSERT OR REPLACE into mm_team_metrics
5. Fetch betting lines; INSERT OR IGNORE into mm_betting_lines
6. Fetch conference tournament games; UPDATE mm_team_metrics with conf_tourney cols

Step 6 is a separate UPDATE because conf tournament data covers the entire league and
requires a name-matching pass against the tournament team registry.

### Moneyline Handling
When a moneyline is available, round profits use actual market odds:
- American +ML: `profit = stake × (ML / 100)`
- American -ML: `profit = stake × (100 / |ML|)`
- Missing line: even money assumed

---

## Repository Structure

```
march_madness/
  config.py                    — Constants: seasons, seeds, bet sizes, strategy params
  main.py                      — CLI (backfill, train, report, picks, track)
  CLAUDE.md                    — Full technical context for AI-assisted development
  METHODOLOGY.md               — This document
  db/
    models.py                  — SQLite DDL (all mm_ tables)
    db.py                      — init_db() + idempotent migrations
  scrapers/
    cbbd_scraper.py            — CBBD API wrappers
  processors/
    features.py                — Feature defs, composite indices, z-scoring, select_picks*
    model.py                   — Simulation, optimization, save/load weights, reporting
  jobs/
    historical_backfill.py     — 6-step seasonal data load
    model_job.py               — Pre-tournament pick generation
    tournament_tracker.py      — In-tournament payout tracking
  queries/
    picks_analysis.sql         — Superset: historical pick performance
    tournament_results.sql     — Superset: live tournament tracking
  feature_diagnostics.py       — 3-part diagnostic: corr matrix, target corr, SFM
  analyze_v6_fixed.py          — Equal / geomean / borda fixed-weight comparison
  analyze_v6_threshold.py      — v6-geomean score threshold sweep
  data/                        — march_madness.db (SQLite, gitignored)
```

---

## 2026 Workflow

1. Run `python main.py backfill --season 2026` after Selection Sunday (mid-March).
   All 6 steps must complete, especially Step 6 (conf tournament data).
2. Run `python main.py report --v6-geomean` to review the 8 picks.
3. Bet $25/pick flat on all 8 picks, E8 cash-out. No tiering, no thresholding.
4. Run `python main.py track` after each round to update rolling payouts.

---

## Known Limitations

### Sample Size
8 training seasons is a small dataset. Joint weight optimization is not viable at this
scale — this is why the geomean SFM approach was adopted. Adding 2008-2013 data
(6 more seasons) would make optimization viable but risks introducing pre-analytics-era
noise. The current approach is calibrated for the modern NET-era selection committee.

### Line Availability
Betting lines are reliably available from 2015 onward via CBBD. Earlier seasons assume
even money when lines are missing, slightly understating true unit returns for strong
underdog wins in those years.

### Team Name Matching (Conference Tournament)
Conf tournament stats are matched to NCAA tournament teams by CBBD name. Spelling
variations can cause mismatches, defaulting those teams to NULL (treated as league
average after z-scoring). The backfill logs match counts per season.

### DFI Redundancy
DFI is highly correlated with opp_efg_pct (r=+0.887). It adds the def_rank component
but is not an independent defensive measure. Future work could consider replacing DFI
with the uncorrelated component of def_rank after partialling out opp_efg_pct.

### TSI as Defensive Proxy
TSI was designed as a tempo consistency measure but functions as a defensive rank proxy
in practice (r=+0.660 with def_rank). This means the V6 model has mild redundancy
between DFI and TSI on the defensive axis. Both add SFM signal across val and test,
so they are retained, but they are not as independent as the feature names suggest.

### Tried and Rejected Features
- **opp_seed_rank_gap** (V4): R64 opponent quality. Train improved but val failed gate.
  8v9 pairs are near-coin-flips; 5v12/6v11 pairs have little opponent quality variation.
- **region_top4_net_avg** (V3): Region path difficulty. Region-level constant with no
  within-region discriminating power. Variable-N val collapsed.
- **Q1 wins**: Redundant with seed_rank_gap (NET already incorporates quadrant records).
- **Last-10-games margin**: Inter-conference comparability poor; conf tournament margins
  are a better hot-team signal.
- **Score-based tiering for v6-geomean**: Tested and confirmed to hurt (-5.17u vs flat).
  Big winners consistently rank 5-8 by model score.
- **Variable-N / threshold for v6-geomean**: Fixed-8 is already optimal. All 8 picks
  score positive every season; raising the cutoff excludes the upset specialists.
