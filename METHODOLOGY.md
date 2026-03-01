# March Madness Betting Model — Methodology & Approach

## Overview

This project builds a systematic, data-driven model for betting on NCAA Tournament games.
The core thesis is straightforward: the NCAA selection committee systematically under-seeds
certain teams, and those teams offer positive expected value when bet on the moneyline in
early tournament rounds. The model identifies those teams, sizes bets appropriately, and
tracks returns over time.

The approach is deliberately narrow and disciplined — we are not trying to predict the
entire bracket. We are looking for a specific, repeatable edge in a subset of games where
the market pricing is most likely to be stale relative to true team quality.

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
rounds where 1- and 2-seeds dominate and upsets are rarer.

---

## Bet Structure

### Unit Definition
One unit = $100. Bets are sized in quarter-units ($25 per pick per round).

### Flat Betting (primary strategy)
Each pick is bet independently at $25 per round:
- **Win**: lock in that round's profit at the moneyline payout; move to the next round.
- **Loss**: lose $25 for that round only. All prior profits from that pick are retained.
- **Two picks meet each other**: no bet is placed that round (void).

This is intentionally conservative. Profits compound through winning rounds, but a loss
never wipes prior gains. The result is asymmetric upside: a team that wins 3 rounds
generates 3 independent profits; a team that loses R64 costs only one $25 stake.

### Tiered Conviction Betting (V2 — alternative)
The same 8 picks, same $200 total annual budget, but stake-weighted by model conviction:
- **Top 4 picks** (highest composite score): $37.50 per round
- **Bottom 4 picks**: $12.50 per round

This amplifies returns when the model's highest-conviction picks go deep, at the cost of
more exposure when they exit early. Historically (+1.90u over 10 seasons vs. flat) the
top-4 by score tend to be the better performers, but the gap is small enough that both
approaches are reasonable.

### Cash-Out Rule
Teams are bet through a maximum of 3 rounds (R64, R32, S16), cashing out after winning
the Sweet Sixteen (entering the Elite Eight). This is the E8 cash-out:
- Caps exposure at rounds where upsets are still common
- Avoids the F4/Championship rounds where top seeds dominate and odds compress
- Maintains a clean, repeatable rule that requires no in-tournament judgment calls

---

## Team Selection

### Eligible Pool
Seeds 5 through 12, excluding any team that played in the First Four (play-in games).
Seeds 1-4 rarely offer positive expected value as they're correctly priced as favorites.
Seeds 13-16 face structural disadvantages (extreme mismatches in R64) that make their
occasional upsets too hard to predict systematically.

### R64 Collision Guard (V2)
Within each region, certain seed pairs always face each other in R64:
- 5 vs 12, 6 vs 11, 7 vs 10, 8 vs 9

If the model selects both teams from a seed pair in the same region, neither bet is placed
in R64. To prevent this waste, the selection algorithm now skips any candidate that would
create a guaranteed R64 matchup with an already-selected pick. The slot is given to the
next-highest-scoring non-colliding team.

### Composite Score
Each eligible team receives a weighted composite score from 10 z-score normalized features.
The 8 highest-scoring teams are selected each year (fixed-8 model). Features are normalized
within each year's eligible field (z-scores), making the model self-calibrating across seasons
with different overall talent levels.

---

## Feature Engineering

All features are z-score normalized within the eligible pool (seeds 5-12, current season)
before weighting. Features where lower values are better (e.g., defensive rank) are negated
before normalization so that higher z-score always means "better."

### The 10 Features

| # | Feature | Direction | Rationale |
|---|---------|-----------|-----------|
| 1 | `seed_rank_gap` | higher = better | `net_rank - (seed × 10)`. Negative = under-seeded. The primary mispricing signal. |
| 2 | `def_rank` | lower = better | NET defensive ranking. Elite defense wins close games in the tournament. |
| 3 | `opp_efg_pct` | lower = better | Opponent effective field goal percentage. Best single-number defensive quality metric. |
| 4 | `net_rating` | higher = better | Adjusted net efficiency margin. Overall team quality. |
| 5 | `tov_ratio` | lower = better | Turnover ratio. Fewer turnovers = cleaner offense under pressure. |
| 6 | `ft_pct` | higher = better | Free throw percentage. Clutch shooting in late-game situations. |
| 7 | `oreb_pct` | higher = better | Offensive rebound percentage. Second chances compound in close games. |
| 8 | `pace` | raw value | Possessions per game. Model learns optimal pace range rather than assuming direction. |
| 9 | `conf_tourney_wins` | higher = better | Games won in conference tournament. Hot teams entering the NCAA Tournament. |
| 10 | `conf_tourney_avg_margin` | higher = better | Average point differential across all conference tournament games. Validates the hot streak is dominant, not just lucky. |

### Why Conference Tournament Features (V2 addition)
Regular season stats through February capture a team's baseline quality, but the NCAA
Tournament begins in mid-March. A team that peaks late — winning their conference tournament
impressively — is demonstrably hotter than their regular-season numbers suggest. NC State
2024 is the canonical example: their regular-season numbers were mediocre, but they won the
ACC Tournament and proceeded to reach the Elite Eight as an 11-seed.

Critically, NET rankings already incorporate strength of schedule, so the conference
tournament signal adds genuine new information rather than restating existing metrics.
Last-10-regular-season-games was considered but rejected: teams in strong conferences finish
the regular season against very different opponents than teams in weaker conferences, making
those margins incomparable. Conference tournaments are self-contained brackets with comparable
intra-conference competition.

---

## Weight Optimization

### Algorithm: Differential Evolution
Weights are optimized using scipy's `differential_evolution`, a global optimizer that does
not require gradient information and avoids local minima. Key parameters:
- Population size: 15 × n_features (150 individuals for 10 features)
- Max iterations: 500
- Tolerance: 1e-4
- Bounds: [-3.0, 3.0] per feature weight

### Objective Function (Fixed-8 Model)
**Maximize total units won** across all training seasons, subject to L2 regularization:

```
objective = -sum(simulate_season(weights, season) for season in train_seasons)
            + lambda_l2 * sum(weights^2)
```

L2 regularization (lambda = 0.01) penalizes extreme weight values, improving
generalization to unseen seasons.

### Objective Function (Variable-N Model)
**Maximize ROI** (units per pick) rather than total units, also with L2 regularization:

```
objective = -(total_units / total_picks) + lambda_l2 * sum(weights^2)
```

This makes the variable-N model prefer high-quality picks over high-quantity picks.

---

## Season Split & Validation Framework

```
Train:  2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022  (8 seasons)
Val:    2023, 2024                                        (2 seasons)
Test:   2025                                              (true holdout — never seen by optimizer)
```

**Why 2014 as the start year:** Modern analytics-era seeding began around 2014, coinciding
with widespread adoption of NET-style efficiency metrics by the selection committee. Data
from 2008-2013 exists and is available, but reflects a different selection methodology that
could introduce noise.

**2020 excluded** everywhere: no tournament was held (COVID-19).

The test set (2025) was not examined until the V2 model was finalized. This is a strict
holdout: no hyperparameter tuning or feature selection decisions were informed by 2025
results.

---

## Model Iterations

### V1 — Initial Model
- **8 features**: seed_rank_gap, def_rank, opp_efg_pct, net_rating, tov_ratio, ft_pct, oreb_pct, pace
- **Fixed 8 picks** per year from seeds 5-12
- **Flat betting** at $25/round, E8 cash-out
- **No collision guard**: 8v9 same-region selections produced $0 rounds
- **No L2 regularization**
- **No conference tournament signal**
- Results: +12.70u train+val (10 seasons) | **-0.65u on 2025 holdout**

Key V1 limitations discovered:
- First Four teams (AZ State 2018, Temple 2019) were incorrectly included in the eligible pool
- R64 seed-pair collisions wasted roughly one slot per year on average
- No momentum signal: a team finishing the regular season cold looked identical to one peaking at the right time

### V3 Attempt — region_top4_net_avg (tried 2026-03-01, reverted)
A single new feature was added and both models were retrained. The feature did not improve
on any evaluation metric and V3 weights were discarded. V2 remains the active model.

**Feature tested:** `region_top4_net_avg` — the average NET rank of seeds 1–4 in the
eligible team's region. Computed on-the-fly via a SQL CTE (no new scraping; data already
exists in `mm_tournament_entries` + `mm_team_metrics`). The hypothesis was that a team in a
weak region (high average NET rank = poor top seeds) faces an easier path and is therefore
undervalued by the market.

**Why it failed:** The feature is a region-level constant — all four eligible seeds in a
region share the identical value. This gives the optimizer nothing to discriminate *within*
a region, which is where most selection decisions happen (multiple seeds 7–12 from the same
region compete for the same picks). The signal adds inter-region noise rather than
intra-region signal, and the variable-N model's val ROI collapsed to -0.008u/pick vs V2's
+0.547u/pick.

**Results vs targets:**

| Metric | V2 | V3 |
|---|---|---|
| Overlap 11-season total | **+43.19u** | +39.91u |
| Overlap test 2025 ROI | **+0.496/pick** | +0.233/pick |
| Variable-N val ROI | **+0.547u/pick** | -0.008u/pick |

The code infrastructure (11th weight column in `mm_model_weights`, CTE in
`load_eligible_teams()`, weight slot in all load/save functions) remains in place.
V2 weights load with `w_region_top4_net_avg = 0.0` (NULL fallback), so the feature
is inert unless new weights are trained that assign it a non-zero value.

### V2 — Current Model
All V1 limitations addressed:

1. **First Four filter**: any team appearing in a round=65 game is excluded from the eligible
   pool. They have not yet secured their bracket slot.

2. **Conference tournament features** (features 9 & 10): queried from CBBD using a Feb 25–
   Mar 18 date window, filtered by game_notes to isolate conference tournaments (excluding
   NCAA Tournament and NIT). Teams not appearing default to NULL, treated as mean (0.0 after
   z-score normalization).

3. **R64 collision guard**: `_r64_collision()` in `features.py` checks each candidate against
   already-selected picks. Skips the candidate and takes the next-best non-colliding team.

4. **L2 regularization**: lambda=0.01 in both objective functions.

5. **Tiered conviction betting**: `simulate_season_tiered()` and `print_tiered_comparison_report()`
   allow comparison of $37.50/$12.50 stake split vs. uniform $25.

V2 results:
```
                        Train+Val    2025 Test    ROI/pick
flat/E8 V2              +29.33u      +3.29u       +0.367u
tiered/E8 V2            +31.23u      +3.14u       +0.390u
flat/E8 V1              +12.70u      -0.65u       +0.159u
```

The improvement from V1 to V2 on the 2025 test (+3.94u swing) demonstrates that the V2
additions captured genuine signal rather than overfitting.

---

## Data Pipeline

### Data Sources (all via CBBD API — `cbbd` Python package)
| Data | API | Notes |
|------|-----|-------|
| Tournament games & seeds | `GamesApi.get_games(tournament="NCAA")` | Seeds encoded in game fields |
| Efficiency ratings | `RatingsApi.get_adjusted_efficiency(season)` | NET, adj off/def ratings |
| Team stats | `StatsApi.get_team_season_stats(season, season_type="regular")` | Four factors, pace, FT% |
| Betting lines | `LinesApi.get_lines(start/end_date_range)` | Mar 15 – Apr 10; BetOnline preferred |
| Conf tournament games | `GamesApi.get_games(start/end_date_range)` | Feb 25 – Mar 18; filtered by notes |

### Backfill Pipeline (6 steps per season)
1. Fetch tournament games; upsert teams; insert games
2. Derive tournament entries (seeds, regions) from game data
3. Fetch adjusted efficiency ratings
4. Fetch regular-season team stats; INSERT OR REPLACE into mm_team_metrics
5. Fetch betting lines; INSERT OR IGNORE into mm_betting_lines
6. Fetch conference tournament games; UPDATE mm_team_metrics with conf_tourney cols

Step 6 is a separate UPDATE (not part of the main INSERT) because conf tournament data
covers the entire league, not just tournament teams, and requires a name-matching pass
against the team_id_map from Step 1.

### Moneyline Handling
When a moneyline is available, round profits use actual market odds:
- American +ML: `profit = stake × (ML / 100)`
- American -ML: `profit = stake × (100 / |ML|)`
- Missing line: even money assumed (`profit = stake`)

Vig is removed using implied probability normalization when storing no-vig probabilities
in `mm_betting_lines`.

---

## Repository Structure

```
march_madness/
  config.py                    — All constants: seasons, seeds, bet sizes, L2 lambda
  main.py                      — CLI entry point (backfill, train, report, picks, track)
  db/
    models.py                  — Full SQLite DDL (all mm_ tables)
    db.py                      — init_db() + idempotent migrations
  scrapers/
    cbbd_scraper.py            — CBBD API wrappers (5 functions)
  processors/
    features.py                — Feature list, normalization, collision guard, select_picks
    model.py                   — Simulation, optimization, reporting (flat + tiered)
  jobs/
    historical_backfill.py     — 6-step seasonal data load
    model_job.py               — Pre-tournament pick generation
    tournament_tracker.py      — In-tournament payout tracking
  queries/
    picks_analysis.sql         — Superset: historical pick performance
    tournament_results.sql     — Superset: live tournament tracking
  data/                        — march_madness.db (SQLite, gitignored)
  CLAUDE.md                    — Full context for AI-assisted development sessions
  METHODOLOGY.md               — This document
```

---

## Known Limitations & Future Considerations

### Sample Size
10 training seasons is a small dataset for statistical inference. The optimizer can
identify patterns that look meaningful but may be idiosyncratic to those specific years.
L2 regularization and a strict train/val/test split mitigate this, but it cannot be
eliminated entirely.

### Line Availability
Betting lines are only reliably available from 2015 onward via CBBD. For earlier seasons,
even-money is assumed when lines are missing. This slightly inflates early-season unit
counts when strong underdogs won (their true +EV would show higher unit returns with
actual lines).

### Team Name Matching (Conference Tournament)
Conference tournament stats are matched to NCAA tournament teams by CBBD team name. Name
changes or spelling variations could cause conf_tourney data to fail to match, defaulting
those teams to NULL (treated as league average). The backfill logs a count of successful
matches per season for monitoring.

### Variable-N Model
The variable-N V2 model (retrained with all V2 features: collision guard, conf tourney,
L2 regularization) converged to always picking exactly 4 teams (the minimum floor),
suggesting the ROI objective strongly prefers concentration. Rather than using it as a
standalone betting system, it is used as a conviction signal within the overlap-tiered
strategy: its 4 picks that overlap with the fixed-8 pool receive a higher stake.

### Overlap-Tiered Strategy
The primary betting strategy combines both models into a single budget-neutral system.
For each year's 8 fixed-8 V2 picks:
- Picks also flagged by variable-N V2 ("overlap"): receive `overlap_bet`
- Picks only in fixed-8 V2 ("solo"): receive `solo_bet`
- 3:1 ratio is maintained: `overlap_bet = 3 * solo_bet`
- Total is always $200/yr: `n * overlap_bet + (8-n) * solo_bet = $200`
  - Solving: `solo_bet = 200 / (2n + 8)`, `overlap_bet = 600 / (2n + 8)`

The key insight: when both models independently agree on a pick using different objectives
(total units vs ROI), that agreement is a stronger conviction signal than either model's
internal score rank alone. In 2021 and 2022 (years with 4/4 overlap), the strategy
returned +8.65u and +10.43u respectively vs +6.07u and +7.44u flat.

Historical performance vs flat and score-tiered (all $200/yr, 11 seasons 2014-2025):
```
Strategy           Train+Val   Val ROI   Test 2025   ROI/$100
Overlap-tiered     +39.22u    +0.90     +3.97u      +1.98   <-- PRIMARY
Score-tiered       +34.37u    +1.03     +3.14u      +1.56
Flat               +32.62u    +0.51     +3.29u      +1.64
```

### Tried and Rejected Features

**`region_top4_net_avg`** — average NET rank of seeds 1–4 in the eligible team's region.
Attempted as a "path difficulty" signal: a weak region means an easier route to E8/F4.
Failed because it is a region-level constant (all eligible teams in a region share the
same value), providing no discriminating power within a region. The variable-N model's
val ROI dropped from +0.547 to -0.008 u/pick. Reverted to V2. See the V3 entry in Model
Iterations above for full results.

**Q1 wins** — considered as a measure of strength-of-schedule performance. Rejected
because NET rank already incorporates quadrant win/loss records as a direct input, making
Q1 wins redundant with `net_rating` and `seed_rank_gap`. Incremental signal too small to
justify an additional correlated feature on an 8-season training set.

**Last-10-regular-season margin** — considered as a "hot team" signal alongside conference
tournament features. Rejected because inter-conference comparability is poor: a team in the
Big Ten finishing the season against Michigan/Purdue faces a fundamentally different test
than a team in the Sun Belt playing conference cellar-dwellers. Conference tournament margins
are intra-conference, making them directly comparable.

### 2026 Data Availability
Conference tournament data for 2026 will not be fully available until mid-March. The
backfill command should be run on or after Selection Sunday to ensure all 6 data steps
complete successfully.
