# March Madness Betting Model — Claude Context

## Project Purpose
Identify under-seeded NCAA Tournament teams (seeds 5-12) and bet using an
overlap-tiered flat strategy: picks flagged by both fixed-8 V2 and variable-N V2
receive a higher stake; fixed-8-only picks receive a lower stake. Total always
$200/yr (3:1 ratio, budget-neutral). Profits locked in each round independently.

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
processors/features.py           — Feature engineering, R64 collision guard, select_picks, select_picks_threshold
processors/model.py              — Weight optimization, payout simulation, tiered betting, reporting
jobs/historical_backfill.py      — One-time load of historical data (6 steps incl. conf tourney)
jobs/model_job.py                — Generate picks pre-tournament
jobs/tournament_tracker.py       — Update rolling payouts during tournament
main.py                          — CLI entry point
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

### Overlap-Tiered (PRIMARY — use for 2026)
- Pool = fixed-8 V2 picks (always 8 teams).
- Picks also flagged by variable-N V2 ("both models"): **higher stake/round**.
- Picks only in fixed-8 V2: **lower stake/round**.
- 3:1 ratio maintained; total always $200/yr regardless of overlap count.
  - 4 overlap → $37.50 / $12.50   3 overlap → $42.86 / $14.29
  - 2 overlap → $50.00 / $16.67   1 overlap → $60.00 / $20.00
  - 0 overlap → $25.00 / $25.00  (reduces to flat)
- Winning a round locks in that round's profit; stake is NOT carried forward.
- Two picks meeting each other: stop betting (no bet placed that round).
- **Train+val (11 seasons)**: +43.19u, avg +3.93u/yr — best of all strategies.
- **Train ROI/$100**: +2.23  **Val ROI/$100**: +0.90  **Test 2025 ROI/$100**: +1.98
- Run with: `python main.py report --overlap`
- Pick sheet (`python main.py picks`) shows `[BOTH]` marker and per-pick bet size.

### Flat (reference — superseded by overlap-tiered)
- Bet $25 each round independently per team. Total $200/yr.
- **Train+val**: +29.33u  **2025 test**: +3.29u  ROI/$100: +1.48
- Still useful as a baseline comparison.

### Tiered Conviction (reference — superseded by overlap-tiered)
- Top 4 picks (by model score): $37.50/round. Bottom 4 picks: $12.50/round.
- Same total budget ($200/yr). Assigns higher stake by score rank, not model agreement.
- **Train+val**: +34.37u  **2025 test**: +3.14u  ROI/$100: +1.56
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

### Fixed-8 Flat/E8 V2 (pick selection — combined with variable-N for bet sizing)
- Selects exactly 8 teams from seeds 5-12 each year by composite score
- **R64 collision guard**: skips any candidate that would face an already-selected pick in R64
  (same region, complementary seed pair: 5v12, 6v11, 7v10, 8v9)
- **L2 regularization**: lambda=0.01 added to training objective
- **10 features** (8 original + 2 new conference tournament features)
- **Train+val (10 seasons)**: +29.33u flat, avg +2.93u/yr, only 1 losing year (2017: -0.50u)
- **Val (2023+2024)**: +2.04u (-0.65u / +2.68u)
- **2025 test**: +3.29u (Ole Miss S16 +1.25u, Arkansas S16 +1.88u)
- Weights stored in `mm_model_weights` with `notes LIKE 'cash_out=E8 bet_style=flat%'`

### Fixed-8 Flat/E8 V1 (ARCHIVED — superseded by V2)
- 8 features, no collision guard, no L2 regularization, no conf tourney features
- Train+val: +12.70u | 2025 test: -0.65u
- V2 is strictly better on all splits; do not retrain with old feature set

### Variable-N Flat/E8 V2 (conviction signal — used for overlap bet sizing)
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

### Comparison Summary (train+val, 11 seasons incl. 2025 test)
```
Strategy                  Total Units   Budget/yr   ROI/$100    2025 test ROI/$100
overlap-tiered V2         +43.19u       $200        +1.72       +1.98   <-- PRIMARY
fixed tiered/E8 V2        +34.37u       $200        +1.56       +1.57
fixed flat/E8 V2          +32.62u       $200        +1.48       +1.64
variable flat/E8 V2       +21.89u       var         +0.55/pick  +1.58/4picks
fixed flat/E8 V1          +12.70u       $200        --          -0.65   <-- archived
```

## Selection Features (composite score) — V2: 10 features
1. `seed_rank_gap` = `net_rank - (seed * 10)` — negative = under-seeded
2. `def_rank` — lower = elite defense (negated)
3. `opp_efg_pct` — lower = better defense (negated)
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
`select_picks()` and `select_picks_threshold()` skip any candidate whose R64 opponent
(same region, complementary seed) is already selected. Prevents wasted slots like 8v9
collisions that produce $0 returns. Implemented via `_r64_collision()` in `features.py`.

## CLI Commands
```
python main.py backfill              # load all missing seasons
python main.py backfill --season Y   # load/reload one specific season

# Fixed-8 model (pick selection)
python main.py train                         # flat/E8 (defaults), 10 features + L2
python main.py train --compare               # all 6 variants: flat+rollover x s16+e8+f4
python main.py report                        # flat/E8 on train+val seasons
python main.py report --test                 # holdout test (2025)
python main.py report --overlap              # overlap-tiered vs flat (PRIMARY)
python main.py report --overlap --test       # overlap-tiered on 2025 holdout
python main.py report --tiered               # tiered vs flat side-by-side (reference)
python main.py report --tiered --test        # tiered vs flat on 2025 holdout (reference)

# Variable-N model (conviction signal for overlap sizing)
python main.py train --variable              # threshold-based, flat/E8, ROI objective
python main.py report --variable             # variable-N report on train+val
python main.py report --variable --test      # variable-N on 2025

# Tournament workflow
python main.py picks [--season Y]    # generate 8 picks with overlap-tiered bet sizing
python main.py track [--season Y]    # update rolling payouts during tournament
```

## 2026 Workflow
1. Run `python main.py backfill --season 2026` after Selection Sunday data is available
2. Run `python main.py picks` — outputs 8 picks with per-pick bet size already computed
   - [BOTH] picks = flagged by both models → higher stake (overlap tier)
   - Fixed-8 only picks → lower stake; total always $200/yr
3. Run `python main.py report --overlap` to review historical overlap-tiered performance
4. Run `python main.py track` after each round to update payouts

## V3 Attempt — region_top4_net_avg (implemented 2026-03-01, REVERTED)
- **Feature added**: `region_top4_net_avg` — average NET rank of seeds 1-4 in the team's region
  - Code changes fully implemented in features.py, db/db.py, and processors/model.py
  - Computed via CTE in load_eligible_teams(); no new DB column in mm_team_metrics
  - mm_model_weights has w_region_top4_net_avg column (added via migration)
- **Retrained**: fixed-8 V3 (id=13) and variable-N V3 (id=14) — then deleted after comparison
- **Results vs V2 targets (FAILED all three):**
  - Overlap train+val (10 seasons): V3 +38.04u vs V2 +39.22u (worse)
  - Overlap test 2025: V3 +1.87u (ROI +0.233) vs V2 +3.97u (ROI +0.496) (much worse)
  - Variable-N val: V3 -0.064u/8 picks vs V2 +1.48u/8 picks (much worse)
- **Conclusion**: V3 did not improve; V2 weights (ids 11/12) remain active.
  The region feature hurt the variable-N model significantly; root cause likely that
  region strength is a region-level constant (all 4 eligible seeds in a region share
  the same value), providing no within-region discrimination — it adds noise to the
  optimizer rather than signal. Feature code remains in place (load_eligible_teams
  returns the column); it will simply be ignored if V2 weights are loaded (11th
  weight = 0.0 fallback).
- **Q1 wins: considered and rejected** — already partially captured by seed_rank_gap since
  NET rank uses quadrant records as a direct input; incremental signal too small to justify
  an additional correlated feature on an 8-season training set

## Superset Connection
Add to docker-compose.yml: `../march_madness/data:/app/mm_data:ro`
SQLAlchemy URI: `sqlite:////app/mm_data/march_madness.db`

## Season Notes
- CBBD covers games from 2003+, betting lines from 2013+
- 2020 excluded everywhere (no tournament — COVID)
- First Four teams are excluded from all eligible pools (not yet in main bracket)
- Re-running `backfill --season Y` is safe: games use INSERT OR IGNORE, metrics use INSERT OR REPLACE
  followed by UPDATE for conf tourney cols in Step 6
