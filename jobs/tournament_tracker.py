"""
Daily tournament tracker — updates rolling payouts for active picks.
Run each day during the NCAA Tournament (mid-March through early April).
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.db import init_db
from scrapers.cbbd_scraper import get_tournament_games, get_betting_lines
from processors.model import simulate_pick_payout
from config import DB_PATH, CURRENT_SEASON, INITIAL_BET_DOLLARS

from shared.sqlite_helpers import get_db

logger = logging.getLogger(__name__)


def refresh_tournament_games(conn, season: int):
    """
    Refresh game results from CBBD for the current season.
    Updates scores and winners in mm_games.
    """
    games = get_tournament_games(season, tournament="NCAA")

    for g in games:
        cbbd_id = g["cbbd_game_id"]
        home_score = g.get("home_score")
        away_score = g.get("away_score")

        if home_score is None or away_score is None:
            continue  # Game not yet played

        # Look up local game record
        row = conn.execute(
            "SELECT id, team1_id, team2_id FROM mm_games WHERE cbbd_game_id = ?",
            (cbbd_id,)
        ).fetchone()

        if not row:
            continue

        game_id = row["id"]
        winner_id = row["team1_id"] if home_score > away_score else row["team2_id"]

        conn.execute(
            """
            UPDATE mm_games
            SET team1_score = ?, team2_score = ?, winner_id = ?
            WHERE id = ?
            """,
            (home_score, away_score, winner_id, game_id),
        )

    # Also refresh betting lines for new games
    cbbd_game_ids = [g["cbbd_game_id"] for g in games]
    if cbbd_game_ids:
        lines = get_betting_lines(season, game_ids=cbbd_game_ids)
        for line in lines:
            row = conn.execute(
                "SELECT id FROM mm_games WHERE cbbd_game_id = ?",
                (line["cbbd_game_id"],)
            ).fetchone()
            if not row:
                continue
            game_id = row["id"]
            # Insert if not already present
            conn.execute(
                """
                INSERT OR IGNORE INTO mm_betting_lines
                    (game_id, provider, team1_moneyline, team2_moneyline,
                     team1_novig_prob, team2_novig_prob)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    line.get("provider"),
                    line.get("home_moneyline"),
                    line.get("away_moneyline"),
                    line.get("home_novig_prob"),
                    line.get("away_novig_prob"),
                ),
            )

    logger.info(f"Refreshed {len(games)} game records for {season}")


def update_pick_payouts(conn, season: int):
    """
    Recompute rolling payouts for all picks in a season.
    Updates mm_model_picks with current payout/status.
    """
    picks = conn.execute(
        """
        SELECT p.id, p.team_id, p.seed, p.initial_bet_dollars,
               t.name as team_name
        FROM mm_model_picks p
        JOIN mm_teams t ON t.id = p.team_id
        WHERE p.season = ?
        ORDER BY p.pick_rank
        """,
        (season,),
    ).fetchall()

    if not picks:
        logger.warning(f"No picks found for season {season}")
        return

    pick_ids = {p["team_id"] for p in picks}

    updates = []
    for pick in picks:
        other_ids = pick_ids - {pick["team_id"]}
        payout, round_exit = simulate_pick_payout(
            conn,
            pick["team_id"],
            season,
            initial_bet=pick["initial_bet_dollars"],
            other_pick_ids=other_ids,
        )

        # Determine status
        if round_exit is None:
            status = "pending"
        elif round_exit <= 8:
            # Reached Elite Eight or cashed out
            status = "cashed_out"
        else:
            # Check if still active (no exit yet means still in tournament)
            # A pick is 'active' if they've won at least one game but not exited
            games_won = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM mm_games
                WHERE season = ? AND winner_id = ?
                """,
                (season, pick["team_id"]),
            ).fetchone()["cnt"]

            if games_won > 0 and payout > pick["initial_bet_dollars"]:
                status = "active"
            elif round_exit is not None:
                status = "eliminated"
            else:
                status = "pending"

        units = (payout - pick["initial_bet_dollars"]) / 100.0 if payout else None

        updates.append((payout, round_exit, units, status, pick["id"]))

    for payout, round_exit, units, status, pick_id in updates:
        conn.execute(
            """
            UPDATE mm_model_picks
            SET payout_dollars = ?, round_exit = ?, units_won = ?, status = ?
            WHERE id = ?
            """,
            (payout, round_exit, units, status, pick_id),
        )

    logger.info(f"Updated payouts for {len(picks)} picks ({season})")


def print_tracking_report(conn, season: int):
    """Print a formatted tracking report for the current season."""
    picks = conn.execute(
        """
        SELECT p.pick_rank, t.name as team_name, p.seed,
               p.status, p.round_exit, p.payout_dollars, p.units_won
        FROM mm_model_picks p
        JOIN mm_teams t ON t.id = p.team_id
        WHERE p.season = ?
        ORDER BY p.pick_rank
        """,
        (season,),
    ).fetchall()

    if not picks:
        print(f"No picks found for {season}.")
        return

    print(f"\n{'=' * 65}")
    print(f"TOURNAMENT TRACKER — {season}")
    print(f"{'=' * 65}")
    print(f"{'#':>3}  {'Team':<28} {'Seed':>4}  {'Status':<12}  {'Payout':>8}  {'Units':>7}")
    print(f"{'-' * 65}")

    total_units = 0.0
    for p in picks:
        units_str = f"{p['units_won']:+.3f}" if p["units_won"] is not None else "  pend"
        payout_str = f"${p['payout_dollars']:.2f}" if p["payout_dollars"] else "  ?"
        round_str = f"(R{p['round_exit']})" if p["round_exit"] else ""
        status = p["status"] or "pending"
        print(
            f"{p['pick_rank']:>3}. {p['team_name']:<28} "
            f"#{p['seed']:>2}  "
            f"{status + ' ' + round_str:<12}  "
            f"{payout_str:>8}  "
            f"{units_str:>7}"
        )
        if p["units_won"] is not None:
            total_units += p["units_won"]

    print(f"{'-' * 65}")
    print(f"{'Total units:':<50}  {total_units:>+7.3f}")
    print(f"{'=' * 65}\n")


def run(season: int | None = None):
    """Update tournament results and print tracking report."""
    if season is None:
        season = CURRENT_SEASON

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    init_db()

    with get_db(DB_PATH) as conn:
        logger.info(f"Refreshing tournament results for {season}...")
        refresh_tournament_games(conn, season)
        update_pick_payouts(conn, season)
        print_tracking_report(conn, season)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update March Madness tournament tracker")
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    args = parser.parse_args()

    run(season=args.season)
