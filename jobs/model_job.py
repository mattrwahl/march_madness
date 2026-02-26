"""
Generate the 8 pre-tournament picks for a given season.
Run on Selection Sunday after CBBD has the bracket data.
"""
import sys
import logging
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.db import init_db, upsert_team
from scrapers.cbbd_scraper import (
    get_tournament_games,
    get_adjusted_efficiency_ratings,
    get_team_season_stats,
    get_betting_lines,
)
from processors.features import load_eligible_teams, select_picks
from processors.model import load_latest_weights, simulate_season
from config import DB_PATH, CURRENT_SEASON, NUM_PICKS, INITIAL_BET_DOLLARS

from shared.sqlite_helpers import get_db

logger = logging.getLogger(__name__)


def fetch_current_season_data(conn, season: int):
    """
    Fetch and store the current season's tournament data from CBBD.
    Called on Selection Sunday after bracket is announced.
    """
    from jobs.historical_backfill import load_season
    load_season(conn, season)


def generate_picks(conn, season: int, weights=None) -> list[dict]:
    """
    Generate the 8 picks for the given season.
    If weights is None, loads the most recent trained weights from DB.
    """
    import numpy as np

    if weights is None:
        weights = load_latest_weights(conn)

    if weights is None:
        # Fallback: equal weights if no trained weights exist
        from processors.features import FEATURE_NAMES
        logger.warning("No trained weights found — using equal weights")
        weights = np.ones(len(FEATURE_NAMES))

    teams = load_eligible_teams(conn, season)
    if not teams:
        raise ValueError(f"No eligible teams found for season {season}. "
                         f"Has the bracket been loaded? Run backfill first.")

    picks = select_picks(teams, weights, NUM_PICKS)
    return picks


def save_picks(conn, season: int, picks: list[dict]):
    """Insert picks into mm_model_picks."""
    # Clear existing picks for this season first
    conn.execute("DELETE FROM mm_model_picks WHERE season = ?", (season,))

    for pick in picks:
        conn.execute(
            """
            INSERT INTO mm_model_picks
                (season, team_id, seed, pick_rank, model_score,
                 initial_bet_dollars, status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                season,
                pick["team_id"],
                pick["seed"],
                pick["pick_rank"],
                pick["model_score"],
                INITIAL_BET_DOLLARS,
            ),
        )

    logger.info(f"Saved {len(picks)} picks for season {season}")


def get_opening_lines(conn, team_id: int, season: int) -> dict | None:
    """Get the R64 opening line for a team."""
    row = conn.execute(
        """
        SELECT
            g.round,
            bl.team1_moneyline, bl.team2_moneyline,
            CASE WHEN g.team1_id = ? THEN bl.team1_moneyline ELSE bl.team2_moneyline END as team_ml
        FROM mm_games g
        JOIN mm_betting_lines bl ON bl.game_id = g.id
        WHERE g.season = ?
          AND (g.team1_id = ? OR g.team2_id = ?)
          AND g.round = 64
        LIMIT 1
        """,
        (team_id, season, team_id, team_id),
    ).fetchone()

    return dict(row) if row else None


def print_pick_sheet(conn, season: int, picks: list[dict]):
    """Print a formatted pick sheet to stdout."""
    print(f"\n{'=' * 70}")
    print(f"MARCH MADNESS MODEL PICKS — {season}")
    print(f"{'=' * 70}")
    print(f"{'#':>3}  {'Team':<28} {'Seed':>4}  {'Score':>7}  {'R64 Line':>9}  {'Notes'}")
    print(f"{'-' * 70}")

    for pick in picks:
        line_info = get_opening_lines(conn, pick["team_id"], season)
        ml_str = f"{line_info['team_ml']:+d}" if line_info and line_info.get("team_ml") else "  N/A"
        region = pick.get("region", "")
        print(
            f"{pick['pick_rank']:>3}. {pick['team_name']:<28} "
            f"#{pick['seed']:>2}  "
            f"{pick['model_score']:>+7.3f}  "
            f"{ml_str:>9}  "
            f"{region}"
        )

    print(f"\n  Strategy: $25 per pick, roll winnings through R64 → R32 → S16 → E8")
    print(f"  Cash out at Elite Eight. Max loss: ${NUM_PICKS * INITIAL_BET_DOLLARS:.0f}")
    print(f"{'=' * 70}\n")


def run(season: int | None = None, fetch_data: bool = True):
    """
    Generate and display picks for a given season.
    If fetch_data=True, pulls fresh data from CBBD first.
    """
    if season is None:
        season = CURRENT_SEASON

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    init_db()

    with get_db(DB_PATH) as conn:
        if fetch_data:
            logger.info(f"Fetching {season} tournament data from CBBD...")
            fetch_current_season_data(conn, season)

        picks = generate_picks(conn, season)
        save_picks(conn, season, picks)
        print_pick_sheet(conn, season, picks)

    return picks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate March Madness picks")
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip CBBD data fetch (use existing DB data)")
    args = parser.parse_args()

    run(season=args.season, fetch_data=not args.no_fetch)
