# March Madness Betting Model — Claude Context

## Project Purpose
Identify 8 "under-seeded" teams (seeds 5–12) before the NCAA Tournament.
Bet $25 (¼ unit) per team, rolling winnings forward game by game until
the team reaches the Elite Eight or two picks face each other.
Max loss: $200 (8 × $25). Optimize for total units won.

## Stack
- **Python 3.12** — all pipeline code
- **SQLite** — `data/march_madness.db` (created at runtime, not in git)
- **CBBD API** — College Basketball Data (cbbd Python package v1.23.0)
- **scipy** — weight optimization for selection model
- **Part of monorepo** — `sports_betting/` root; imports from `shared/`

## Key File Map
```
config.py                        — DB path, API key, season params
db/models.py                     — Full DDL for all mm_ tables
db/db.py                         — Thin wrapper using shared.sqlite_helpers
scrapers/cbbd_scraper.py         — CBBD API client (games, ratings, stats, lines)
processors/features.py           — Feature engineering + composite score
processors/model.py              — Weight optimization, pick selection
jobs/historical_backfill.py      — One-time load 2021–2025 data
jobs/model_job.py                — Generate 8 picks pre-tournament
jobs/tournament_tracker.py       — Update rolling payouts during tournament
main.py                          — CLI entry point (backfill / picks / track / report)
queries/picks_analysis.sql       — Superset SQL: historical picks performance
queries/tournament_results.sql   — Superset SQL: live tracking
```

## Database Schema (db/models.py)

All tables are prefixed `mm_` to avoid collision if DBs are merged.

| Table | Purpose |
|---|---|
| `mm_teams` | Team registry (name, conference) |
| `mm_tournament_entries` | Teams in tournament each year with seed/region |
| `mm_games` | Game-by-game tournament results |
| `mm_team_metrics` | Pre-tournament efficiency metrics per team/season |
| `mm_betting_lines` | Moneylines per game from CBBD |
| `mm_model_picks` | The 8 picks per year with rolling payout tracking |
| `mm_model_weights` | Stored feature weights from optimization runs |

## Round Encoding
Rounds are stored as number of teams remaining at that stage:
- 64 = First Round, 32 = Second Round, 16 = Sweet Sixteen, 8 = Elite Eight
- 4 = Final Four, 2 = Championship, 1 = Champion

## Rolling Bet Logic
Starting with $25, each win rolls the full payout into the next round.
The bet cashes out at Elite Eight (8 teams remaining) or if two of our
8 picks face each other. `units_won = (payout_dollars - 25) / 100`

## Selection Features (used in composite score)
1. `seed_rank_gap` = `net_rank - (seed * 10)` (negative = under-seeded)
2. `def_rank` — lower = elite defense (negated in score)
3. `opp_efg_pct` — lower = better defense (negated in score)
4. `net_rating` — adjusted net efficiency
5. `tov_ratio` — lower = fewer turnovers (negated in score)
6. `ft_pct` — free throw % (clutch shooting)
7. `oreb_pct` — offensive rebound %
8. `pace` — mid-pace teams more consistent

## CLI Commands
```
python main.py backfill              # one-time historical load 2021–2025
python main.py picks [--season Y]    # generate 8 picks (defaults to current year)
python main.py track [--season Y]    # update rolling payouts for active picks
python main.py report                # print units won summary across all years
```

## Superset Connection
Add to docker-compose.yml: `../march_madness/data:/app/mm_data:ro`
SQLAlchemy URI: `sqlite:////app/mm_data/march_madness.db`

## Season Notes
- CBBD covers games from 2003+, betting lines from 2013+
- Historical training: 2021–2025 (5 seasons)
- 2026 workflow: run `picks` on Selection Sunday (mid-March 2026)
