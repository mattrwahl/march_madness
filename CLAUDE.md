# March Madness Betting Model — Claude Context

## Project Purpose
Identify under-seeded NCAA Tournament teams (seeds 5-12) and bet $25/quarter-unit
per team per round using a flat betting strategy, locking in profits each round
independently. Optimize for total units won (fixed model) and ROI per pick (variable model).

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

### Flat (default — recommended)
- Bet $25 each round independently per team.
- Winning a round locks in that round's profit; stake is NOT carried forward.
- Losing a round costs only that round's $25; prior profits are kept.
- Two picks meeting each other: stop betting (no bet placed that round).
- `units = (sum_of_round_profits - sum_of_round_losses) / 100`

### Tiered Conviction (V2 — comparison available)
- Top 4 picks (by model score): $37.50/round. Bottom 4 picks: $12.50/round.
- Same total budget as flat ($200/yr). Amplifies wins/losses on high-conviction picks.
- **Train+val**: +31.23u vs +29.33u flat (+1.90u); ROI +0.3903 vs +0.3666 u/pick.
- **2025 test**: +3.14u vs +3.29u flat (-0.14u — essentially equal).
- Run with: `python main.py report --tiered`

### Rollover (comparison only — not recommended for 2026)
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

### Fixed-8 Flat/E8 V2 (PRIMARY — use for 2026)
- Selects exactly 8 teams from seeds 5-12 each year by composite score
- **R64 collision guard**: skips any candidate that would face an already-selected pick in R64
  (same region, complementary seed pair: 5v12, 6v11, 7v10, 8v9)
- **L2 regularization**: lambda=0.01 added to training objective
- **10 features** (8 original + 2 new conference tournament features)
- **Train+val (10 seasons)**: +29.33u, avg +2.93u/yr, only 1 losing year (2017: -0.50u)
- **Val (2023+2024)**: +2.04u (-0.65u / +2.68u)
- **2025 test**: +3.29u (Ole Miss S16 +1.25u, Arkansas S16 +1.88u)
- Weights stored in `mm_model_weights` with `notes LIKE 'cash_out=E8 bet_style=flat%'`

### Fixed-8 Flat/E8 V1 (ARCHIVED — superseded by V2)
- 8 features, no collision guard, no L2 regularization, no conf tourney features
- Train+val: +12.70u | 2025 test: -0.65u
- V2 is strictly better on all splits; do not retrain with old feature set

### Variable-N Flat/E8 (REFERENCE — informative but not primary)
- Threshold-based selection: picks all teams with composite z-score >= threshold
  (self-normalized within each year's eligible field)
- Optimizes N+1 params: 10 feature weights + z-score threshold
- Objective: ROI (units/pick) rather than total units
- Min picks/yr: 4, Max picks/yr: 12
- **Optimizer converged to threshold=1.815** — effectively always picks exactly 4 (the floor)
- Key weakness: with only 4 picks, an 8v9 collision wastes 50% of the year; needs retrain with V2 collision guard
- Run: `python main.py train --variable` then `python main.py report --variable`

### Comparison Summary (train+val, 10 seasons, V2 weights)
```
Strategy               Total Units   Picks/yr   ROI/pick    2025 test
fixed flat/E8 V2       +29.33u       8          +0.367u     +3.29u   <-- PRIMARY
fixed tiered/E8 V2     +31.23u       8          +0.390u     +3.14u   <-- alternative
fixed flat/E8 V1       +12.70u       8          +0.159u     -0.65u   <-- archived
variable flat/E8 V1    +9.34u        4          +0.234u     -0.28u   <-- reference (needs V2 retrain)
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

# Fixed-8 model (primary)
python main.py train                         # flat/E8 (defaults), 10 features + L2
python main.py train --compare               # all 6 variants: flat+rollover x s16+e8+f4
python main.py report                        # flat/E8 on train+val seasons
python main.py report --test                 # holdout test (2025)
python main.py report --tiered               # tiered vs flat side-by-side (train+val)
python main.py report --tiered --test        # tiered vs flat on 2025 holdout

# Variable-N model
python main.py train --variable              # threshold-based, flat/E8, ROI objective
python main.py report --variable             # variable-N report on train+val
python main.py report --variable --test      # variable-N on 2025

# Tournament workflow
python main.py picks [--season Y]    # generate picks (defaults to 2026)
python main.py track [--season Y]    # update rolling payouts during tournament
```

## 2026 Workflow
1. Run `python main.py backfill --season 2026` after Selection Sunday data is available
2. Run `python main.py picks` to generate 8 picks using fixed flat/E8 V2 weights
3. Run `python main.py report --tiered` to see flat vs tiered split for conviction sizing
4. Decide: flat $25/pick or tiered $37.50/$12.50 based on confidence in pick ordering
5. Run `python main.py track` after each round to update payouts

## Superset Connection
Add to docker-compose.yml: `../march_madness/data:/app/mm_data:ro`
SQLAlchemy URI: `sqlite:////app/mm_data/march_madness.db`

## Season Notes
- CBBD covers games from 2003+, betting lines from 2013+
- 2020 excluded everywhere (no tournament — COVID)
- First Four teams are excluded from all eligible pools (not yet in main bracket)
- Re-running `backfill --season Y` is safe: games use INSERT OR IGNORE, metrics use INSERT OR REPLACE
  followed by UPDATE for conf tourney cols in Step 6
