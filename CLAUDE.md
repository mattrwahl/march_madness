# March Madness Betting Model — Claude Context

## Project Purpose
Identify under-seeded NCAA Tournament teams (seeds 5-12) and bet using flat $25/pick
strategy. Primary model is **v6-fixed-8-geomean** (4 features, geomean weights, no
optimizer). Secondary legacy strategy: overlap-tiered V2 (fixed-8 + variable-N agreement).

## Stack
- **Python 3.12** — all pipeline code
- **SQLite** — `data/march_madness.db` (created at runtime, not in git)
- **CBBD API** — College Basketball Data (cbbd Python package v1.23.0)
- **scipy** — weight optimization for selection model
- **Part of monorepo** — `sports_betting/` root; imports from `shared/`

## Key File Map
```
config.py                        — DB path, API key, season params, strategy constants
db/models.py                     — Full DDL for all mm_ tables
db/db.py                         — Thin wrapper + migrations
scrapers/cbbd_scraper.py         — CBBD API client (games, ratings, stats, lines, conf tourney)
processors/features.py           — Feature engineering, R64 collision guard, all select_picks variants
processors/model.py              — Weight optimization, payout simulation, tiered betting, reporting
jobs/historical_backfill.py      — One-time load of historical data (6 steps incl. conf tourney)
jobs/model_job.py                — Generate picks pre-tournament
jobs/tournament_tracker.py       — Update rolling payouts during tournament
main.py                          — CLI entry point
feature_diagnostics.py           — 3-part feature analysis: corr matrix, target corr, SFM
analyze_v6_fixed.py              — Standalone equal/geomean/borda fixed-weight comparison
analyze_v6_threshold.py          — v6-geomean threshold sweep: ROI at every score cutoff
queries/picks_analysis.sql       — Superset SQL: historical picks performance
queries/tournament_results.sql   — Superset SQL: live tracking
```

## Database Schema (db/models.py)

All tables are prefixed `mm_` to avoid collision if DBs are merged.

| Table | Purpose |
|---|---|
| `mm_teams` | Team registry (name, conference) |
| `mm_tournament_entries` | Teams in tournament each year with seed/region |
| `mm_games` | Game-by-game tournament results (round=65 = First Four) |
| `mm_team_metrics` | Pre-tournament efficiency metrics per team/season (incl. conf tourney cols) |
| `mm_betting_lines` | Moneylines per game from CBBD |
| `mm_model_picks` | The picks per year with payout tracking |
| `mm_model_weights` | Stored feature weights from optimization runs |

## Round Encoding
Rounds are stored as number of teams remaining at that stage:
- 65 = First Four (play-in), 64 = R64, 32 = R32, 16 = S16, 8 = E8, 4 = F4, 2 = Champ

**First Four filter:** Teams appearing in round=65 games are excluded from all selection
pools — they have not yet earned their bracket slot. This is enforced in `load_eligible_teams()`.

## Bet Styles

### Flat (PRIMARY for v6-geomean)
- Bet $25 each round independently per team. Total $200/yr.
- Tiered conviction hurts v6-geomean: flat +29.22u vs tiered +24.05u (-5.17u).
  The big winners (Loyola 2018, Oregon St 2021, Auburn 2019) land in the lower score ranks.

### Overlap-Tiered (PRIMARY for V2 — legacy)
- Pool = fixed-8 V2 picks (always 8 teams).
- Picks also flagged by variable-N V2 ("both models"): **higher stake/round**.
- Picks only in fixed-8 V2: **lower stake/round**.
- 3:1 ratio maintained; total always $200/yr regardless of overlap count.
  - 4 overlap → $37.50 / $12.50   3 overlap → $42.86 / $14.29
  - 2 overlap → $50.00 / $16.67   1 overlap → $60.00 / $20.00
  - 0 overlap → $25.00 / $25.00  (reduces to flat)
- Winning a round locks in that round's profit; stake is NOT carried forward.
- Two picks meeting each other: stop betting (no bet placed that round).
- **Train+val (11 seasons)**: +43.19u, avg +3.93u/yr — best of all V2 strategies.
- **Train ROI/$100**: +2.23  **Val ROI/$100**: +0.90  **Test 2025 ROI/$100**: +1.98
- Run with: `python main.py report --overlap`

### Tiered Conviction (reference — superseded)
- Top 4 picks (by model score): $37.50/round. Bottom 4 picks: $12.50/round.
- Same total budget ($200/yr). Assigns higher stake by score rank, not model agreement.
- **V2 train+val**: +34.37u  **2025 test**: +3.14u  ROI/$100: +1.56
- **V6-geomean**: flat dominates tiered by +5.17u — do NOT use tiered for v6.
- Run with: `python main.py report --tiered`

### Rollover (comparison only — not recommended)
- $25 initial stake compounds forward through each win.
- A loss at any round returns $0 (full current stake lost).
- High variance; optimizer overfits heavily on training data.

## Cash-Out Rounds
Team stops being bet on after winning a game in the target round:
- **S16 (16)**: win R32, enter E8 — 2 rounds max
- **E8  ( 8)**: win S16, enter F4 — 3 rounds max  *(default and recommended)*
- **F4  ( 4)**: win E8, enter Champ — 4 rounds max (too much variance; not recommended)

## Season Split
- **Train**: 2014-2022 excluding 2020 (8 seasons) — modern analytics/betting era
- **Val**:   2023, 2024
- **Test**:  2025 (true out-of-sample holdout)
- **Backfill available**: 2008-2019, 2021-2025

## Models

### v6-fixed-8-geomean (CURRENT PRIMARY — flat/E8)
- **4 features**: seed_rank_gap, conf_tourney_wins, dfi, tsi (coreB set)
- **Fixed pre-specified weights** derived from geomean(SFM_val, SFM_test) per feature;
  no joint optimization — avoids weight collapse on 8-season training set.
- Weights: srg=0.2795, ctw=0.3217, dfi=0.2025, tsi=0.1963
- **Train (8 seasons)**: +18.18u  avg +2.27u/yr
- **Val (2023+2024)**: +6.51u  (+4.664u / +1.850u)  — beats V2 gate (+2.08u) by +4.43u
- **Test (2025)**: +4.53u  (Michigan S16 +1.26u, Ole Miss S16 +2.00u, Drake R32 +1.25u)
- **Total (11 seasons)**: +29.22u flat  ROI/pick: +0.332
- Weights saved in `mm_model_weights` with `notes LIKE 'model=v6-geomean %'` (id=20)
- **Tiered hurts**: flat +29.22u vs tiered +24.05u — big winners rank 5-8 by model score
- Run: `python main.py train --v6-geomean`

### Geomean Weight Derivation
Weights are geomean(SFM_val, SFM_test) from single-feature model (SFM) results,
each feature run in isolation on 8 train seasons, evaluated on val+test:

| Feature | SFM Val | SFM Test | Geomean | Weight |
|---------|---------|---------|---------|--------|
| seed_rank_gap | +6.52u | +5.83u | 6.165 | 0.2795 |
| conf_tourney_wins | +9.55u | +5.27u | 7.094 | 0.3217 |
| dfi | +6.95u | +2.87u | 4.466 | 0.2025 |
| tsi | +5.21u | +3.60u | 4.331 | 0.1963 |

Rationale: geomean penalizes features strong on one split but weak on another;
balances both val and test signal without adding a new optimization pass on the same data.

### Fixed-weight Comparison (equal / geomean / borda)
Run on the same coreB feature set (srg, ctw, dfi, tsi), evaluated all 11 seasons:

| Method | Val | Test | Total |
|--------|-----|------|-------|
| Equal (0.25 each) | +4.04u | +2.68u | — |
| **Geomean** | **+6.51u** | **+4.53u** | **+29.22u** |
| Borda count | +4.77u | +0.13u | — |

Borda weakness: structurally over-democratic; a team ranked #1 on one feature and
#8 on three others scores identically to a team ranked #4 on all four.

### Fixed-8 Flat/E8 V2 (legacy — pick selection for overlap strategy)
- Selects exactly 8 teams from seeds 5-12 each year by composite score
- **R64 collision guard**: skips any candidate that would face an already-selected pick in R64
  (same region, complementary seed pair: 5v12, 6v11, 7v10, 8v9)
- **L2 regularization**: lambda=0.01 added to training objective
- **10 features** (8 original + 2 new conference tournament features)
- **Train+val (10 seasons)**: +29.33u flat, avg +2.93u/yr, only 1 losing year (2017: -0.50u)
- **Val (2023+2024)**: +2.04u (-0.65u / +2.68u)  — val gate: +2.08u
- **2025 test**: +3.29u (Ole Miss S16 +1.25u, Arkansas S16 +1.88u)
- Weights stored in `mm_model_weights` with `notes LIKE 'cash_out=E8 bet_style=flat%'`

### Fixed-8 Flat/E8 V1 (ARCHIVED — superseded by V2)
- 8 features, no collision guard, no L2 regularization, no conf tourney features
- Train+val: +12.70u | 2025 test: -0.65u
- V2 is strictly better on all splits; do not retrain with old feature set

### Variable-N Flat/E8 V2 (conviction signal — used for V2 overlap bet sizing)
- Threshold-based selection: picks all teams with composite z-score >= threshold
  (self-normalized within each year's eligible field)
- Optimizes N+1 params: 10 feature weights + z-score threshold
- Objective: ROI (units/pick) rather than total units
- Min picks/yr: 4, Max picks/yr: 12
- **V2: threshold=1.252** — always picks exactly 4 (min floor); high ROI concentration
- V2 includes R64 collision guard and 10-feature set (same as fixed-8 V2)
- **Train+val (10 seasons)**: +21.89u / 40 picks (ROI +0.5473u/pick)
- **2025 test**: +1.58u / 4 picks (ROI +0.3949u/pick) — Arkansas S16 +1.875u
- Used by `picks` command to determine [BOTH] overlap picks (higher bet tier)
- Weights stored in `mm_model_weights` with `notes LIKE '%model=variable%'`

### Comparison Summary (all strategies, 11 seasons incl. 2025 test)
```
Strategy                   Train+Val    Test      Budget/yr   ROI/pick
overlap-tiered V2          +43.19u    +3.97u      $200        varies
fixed flat/E8 v6-geomean   +24.69u    +4.53u      $200        +0.332   <-- NEW PRIMARY
fixed tiered/E8 V2         +34.37u    +3.14u      $200        +0.196
fixed flat/E8 V2           +32.62u    +3.29u      $200        +0.185
variable flat/E8 V2        +21.89u    +1.58u      var         +0.547/pick
fixed flat/E8 V1           +12.70u    -0.65u      $200        --  archived
```
Note: v6-geomean train+val = +24.69u (10 seasons); total 11 seasons = +29.22u.

## Composite Feature Indices (V5/V6 only)

Computed in `compute_composite_features()` in `features.py`:
- **dfi** (Defensive Friction Index) = (opp_efg_pct + def_rank_norm) / 2
  Lower raw = harder to score on. In V6 features list as `negate=True` so higher z = better.
  Note: highly correlated with opp_efg_pct (r=+0.887) — partially redundant.
- **tsi** (Tempo Stability Index) = tempo_std / mean_pace
  Intended as consistency measure; actually functions as def_rank proxy (r=+0.660).
- **cpi** (Possession Control Index) = (oreb_pct - tov_ratio) / 2
  Dominated by oreb_pct (r=+0.596); tov_ratio partially cancels.
- **ftli** (Free Throw Leverage Index) = ft_pct * opp_foul_rate
  Modest predictive value; not included in V6 coreB.
- **spmi** (Speed-Momentum Index) = pace * tsi
  Near-zero predictive value (r=+0.015 with targets). Dropped from V6.

## Selection Features (composite score)

### V6-coreB (4 features — v6-fixed-8-geomean)
1. `seed_rank_gap` — net_rank - (seed * 10); negative = under-seeded
2. `conf_tourney_wins` — games won in conf tournament (momentum signal)
3. `dfi` — Defensive Friction Index (negated: higher score = better defense)
4. `tsi` — Tempo Stability Index (negated: lower variability = better)

### V2: 10 features (fixed-8 and variable-N legacy models)
1. `seed_rank_gap` = `net_rank - (seed * 10)` — negative = under-seeded
2. `def_rank` — lower = elite defense (negated)
3. `opp_efg_pct` — lower opponent eFG = better defense (negated)
4. `net_rating` — adjusted net efficiency
5. `tov_ratio` — lower = fewer turnovers (negated)
6. `ft_pct` — free throw % (clutch shooting)
7. `oreb_pct` — offensive rebound %
8. `pace` — mid-pace teams more consistent
9. `conf_tourney_wins` — games won in conference tournament (momentum signal)
10. `conf_tourney_avg_margin` — avg point differential across all conf tourney games

**Conference tournament data**: fetched via date range Feb 25-Mar 18, filtered by game_notes
(`'tournament' in notes`, excluding NCAA and NIT). Populated in backfill Step 6.

## R64 Collision Guard
All `select_picks*` functions skip any candidate whose R64 opponent (same region,
complementary seed) is already selected. Prevents wasted slots like 8v9 collisions
that produce $0 returns. Implemented via `_r64_collision()` in `features.py`.
Seed pairs: 5v12, 6v11, 7v10, 8v9.

## CLI Commands
```
python main.py backfill              # load all missing seasons
python main.py backfill --season Y   # load/reload one specific season

# v6-fixed-8-geomean (CURRENT PRIMARY)
python main.py train --v6-geomean         # register model, show YoY + tiered comparison
python main.py report --v6-geomean        # flat report on all 11 seasons
python main.py report --v6-geomean --test # test holdout only (2025)
python main.py report --v6-geomean --tiered  # tiered vs flat (flat is better)

# V2 fixed-8 model (legacy — for overlap strategy)
python main.py train                         # flat/E8 (defaults), 10 features + L2
python main.py train --compare               # all 6 variants: flat+rollover x s16+e8+f4
python main.py report                        # flat/E8 on train+val seasons
python main.py report --test                 # holdout test (2025)
python main.py report --overlap              # overlap-tiered vs flat (V2 PRIMARY)
python main.py report --overlap --test       # overlap-tiered on 2025 holdout
python main.py report --tiered               # tiered vs flat side-by-side (reference)
python main.py report --tiered --test        # tiered vs flat on 2025 holdout (reference)

# V2 variable-N model (conviction signal for V2 overlap sizing)
python main.py train --variable              # threshold-based, flat/E8, ROI objective
python main.py report --variable             # variable-N report on train+val
python main.py report --variable --test      # variable-N on 2025

# Tournament workflow
python main.py picks [--season Y]    # generate 8 picks with overlap-tiered bet sizing
python main.py track [--season Y]    # update rolling payouts during tournament

# Analysis scripts
python march_madness/feature_diagnostics.py    # correlation matrix, target corr, SFM
python march_madness/analyze_v6_fixed.py       # equal / geomean / borda comparison
python march_madness/analyze_v6_threshold.py   # v6-geomean score threshold sweep
```

## 2026 Workflow
1. Run `python main.py backfill --season 2026` after Selection Sunday data is available
2. Run `python main.py report --v6-geomean` to see v6-geomean picks for 2026
3. Bet $25/pick flat on all 8 picks (E8 cash-out)
4. Run `python main.py track` after each round to update payouts

## V3 Attempt — region_top4_net_avg (2026-03-01, REVERTED)
- **Feature added**: `region_top4_net_avg` — average NET rank of seeds 1-4 in the team's region
- **Results vs V2 targets (FAILED all three):**
  - Overlap train+val: V3 +38.04u vs V2 +39.22u (worse)
  - Overlap test 2025: V3 +1.87u vs V2 +3.97u (much worse)
  - Variable-N val: V3 -0.064u/8 picks vs V2 +1.48u/8 picks (much worse)
- **Conclusion**: Region strength is a region-level constant; no within-region discrimination.
  V2 weights (ids 11/12) remain active. Feature code in place, weight=0.0 fallback.

## V4 Attempt — opp_seed_rank_gap (2026-03-01, REVERTED)
- **Feature added**: `opp_seed_rank_gap` — R64 opponent's seed_rank_gap (higher = weaker opp)
- **Fixed-8 V4 result** (id=15, then deleted):
  - Train: +31.15u (vs V2 +29.33u — improved)
  - Val: +1.55u (vs V2 +2.08u — FAILED the gate)
  - Test: +2.29u (vs V2 +3.29u — worse)
- **Conclusion**: Did not clear val gate. Feature code in place (12th weight = 0.0 for V2 rows).

## V5 Attempt — 8-feature composite model (2026-03-02, FAILED)
- **Features**: seed_rank_gap, net_rating, conf_tourney_wins, cpi, dfi, ftli, spmi, tsi
- Constrained bounds: srg<0, all others>0. Optimized via differential evolution, lambda=0.01.
- **V5 result** (id=16/17):
  - Train: +32.16u (same as V6 — picks identical)
  - Val: +0.45u (FAILED — well below V2 gate +2.08u)
- **Conclusion**: Weight collapse — optimizer concentrated nearly all weight on srg.
  8 training seasons underdetermined for 8-weight joint optimization even with L2.
  SPMI had near-zero predictive value (r=+0.015); CPI dominated by oreb_pct.

## V6 Optimizer Attempts — coreA/coreB (2026-03-02, FAILED)
- **V6-coreA**: srg, ctw, opp_efg_pct, tsi — DE optimizer, lambda=0.1 (id=18)
- **V6-coreB**: srg, ctw, dfi, tsi — DE optimizer, lambda=0.1 (id=19)
- Both produced: Train=+32.16u, Val=-1.64u — same near-zero weights, identical picks.
- **Conclusion**: 4 features × 8 seasons still underdetermined; optimizer finds
  near-degenerate solution at every lambda. Fixed geomean weights solve this cleanly.

## Variable-N Threshold Investigation for v6-geomean (2026-03-03, NOT BENEFICIAL)
Investigated whether thresholding the composite score (taking only top-N picks where
score >= cutoff) could improve ROI per pick over the fixed-8 baseline.

**Method**: weights fixed at V6_GEOMEAN_W; swept composite score threshold -2.0 to +1.5
in 0.25 steps. Score is the weighted sum of per-feature z-scores — already meaningful
without a second normalization pass.

**Results**:

| Threshold | Avg N/yr | Train ROI/pk | Val ROI/pk | Test ROI/pk | Total units |
|-----------|----------|-------------|-----------|------------|-------------|
| >= +0.50 | 4.4 | +0.231 | +0.217 | +0.558 | +12.3u |
| >= +0.25 | 7.6 | +0.257 | +0.273 | +0.362 | +22.5u |
| **Fixed-8** | **8.0** | **best** | **best** | **best** | **+29.2u** |
| Expand to 12 | 12.0 | +0.175 | +0.342 | +0.358 | +29.4u |

**Conclusion**: Fixed-8 flat is already optimal. Three reasons:
1. All 8 composite scores are positive every season (range 0.13-1.49) — the model
   has already gated on quality; there is no dead weight to cut.
2. The biggest winners consistently rank **5-8 by model score**: Loyola Chicago 2018
   (#8, +3.39u), San Diego State 2023 (#7, +2.91u), Oregon State 2021 (#4, +4.31u),
   Auburn 2019 (#6, +2.95u), Duquesne 2024 (#8, +2.23u), Ole Miss 2025 (#8, +2.00u).
   The score identifies direction (team is better than seeded), not magnitude of upside.
3. Raising the threshold cuts these picks and loses -6.7u total vs fixed-8.
   Expanding to 12 picks adds low-conviction losers and halves ROI/pick.

**No variable-N or threshold variant will be implemented for v6-geomean.**

## Feature Diagnostics (feature_diagnostics.py)
Three-part analysis run 2026-03-02:

**Key correlations (Pearson r with win probability proxy):**
- dfi vs opp_efg_pct: r=+0.887 — nearly redundant (dfi is ~90% opp_efg_pct)
- tsi vs def_rank: r=+0.660 — tsi functions as def_rank proxy, not tempo measure
- cpi vs oreb_pct: r=+0.596 — cpi dominated by oreb_pct; tov_ratio partially cancels
- spmi vs targets: r=+0.015 — effectively zero predictive value

**Single-feature model (SFM) stability (higher test = more generalizable):**
- seed_rank_gap: val=+6.52u, test=+5.83u — most stable (test/val ratio = 0.89)
- conf_tourney_wins: val=+9.55u, test=+5.27u — highest combined, slight decay
- dfi: val=+6.95u, test=+2.87u — moderate decay
- tsi: val=+5.21u, test=+3.60u — good stability

## Superset Connection
Add to docker-compose.yml: `../march_madness/data:/app/mm_data:ro`
SQLAlchemy URI: `sqlite:////app/mm_data/march_madness.db`

## Season Notes
- CBBD covers games from 2003+, betting lines from 2013+
- 2020 excluded everywhere (no tournament — COVID)
- First Four teams are excluded from all eligible pools (not yet in main bracket)
- Re-running `backfill --season Y` is safe: games use INSERT OR IGNORE, metrics use INSERT OR REPLACE
  followed by UPDATE for conf tourney cols in Step 6
