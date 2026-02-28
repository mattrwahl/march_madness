"""
One-time historical data load for 2021–2025.
Populates mm_games, mm_tournament_entries, mm_team_metrics, mm_betting_lines.
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.db import init_db, upsert_team, get_team_id
from scrapers.cbbd_scraper import (
    get_tournament_games,
    get_adjusted_efficiency_ratings,
    get_team_season_stats,
    get_betting_lines,
    get_conf_tournament_stats_by_team,
)
from config import HISTORICAL_SEASONS, DB_PATH

from shared.sqlite_helpers import get_db

logger = logging.getLogger(__name__)


def _extract_seeds_from_games(games: list[dict]) -> dict[str, int]:
    """
    Extract seed information from game notes or game records.
    Returns dict: team_name -> seed
    CBBD encodes seeds in the notes field for NCAA tournament games.
    """
    seed_map = {}
    for g in games:
        notes = g.get("notes") or ""
        # Notes may contain seed info like "#5 seed" or similar patterns
        # We'll also infer from round 64 matchups: seeds 1v16, 2v15, 3v14, etc.
        # For now, seeds are loaded when available from CBBD game attributes
        ht_seed = g.get("home_seed")
        at_seed = g.get("away_seed")
        if ht_seed and g.get("home_team"):
            seed_map[g["home_team"]] = ht_seed
        if at_seed and g.get("away_team"):
            seed_map[g["away_team"]] = at_seed

    return seed_map


def _infer_seeds_from_r64(games: list[dict]) -> dict[str, int]:
    """
    Infer seeds from Round of 64 matchups.
    Standard NCAA bracket pairs: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
    We can use the relative ordering + pre-game ELO rankings as a proxy.
    This is a best-effort fallback when CBBD doesn't provide explicit seeds.
    """
    # We need a different approach — use the tournament entry/bracket data
    # For now return empty; seeds will be populated from dedicated endpoint
    return {}


def load_season(conn, season: int) -> dict:
    """
    Load all data for a single season.
    Returns summary dict with counts.
    """
    logger.info(f"Loading season {season}...")
    counts = {"games": 0, "teams": 0, "metrics": 0, "lines": 0, "conf_tourney": 0}

    # --- Step 1: Tournament games ---
    games = get_tournament_games(season, tournament="NCAA")
    if not games:
        logger.warning(f"No tournament games found for {season}")
        return counts

    # Collect all unique team names and upsert them
    team_names = set()
    for g in games:
        if g.get("home_team"):
            team_names.add(g["home_team"])
        if g.get("away_team"):
            team_names.add(g["away_team"])

    team_id_map = {}
    for name in sorted(team_names):
        tid = upsert_team(conn, name)
        team_id_map[name] = tid

    counts["teams"] = len(team_id_map)

    # Insert games
    game_id_map = {}  # cbbd_game_id -> local db id
    for g in games:
        home_id = team_id_map.get(g.get("home_team"))
        away_id = team_id_map.get(g.get("away_team"))
        if not home_id or not away_id:
            continue

        home_score = g.get("home_score")
        away_score = g.get("away_score")
        winner_id = None
        if home_score is not None and away_score is not None:
            winner_id = home_id if home_score > away_score else away_id

        # Use the round already parsed by the scraper from game_notes
        game_round = g.get("round")

        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO mm_games
                    (cbbd_game_id, season, game_date, round, tournament,
                     team1_id, team2_id, team1_seed, team2_seed,
                     team1_score, team2_score, winner_id, neutral_site)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    g["cbbd_game_id"], season, g.get("game_date"),
                    game_round, g.get("tournament", "NCAA"),
                    home_id, away_id,
                    g.get("home_seed"), g.get("away_seed"),
                    home_score, away_score,
                    winner_id,
                    1,  # tournament games always neutral site
                ),
            )
            # Get local id for this game
            row = conn.execute(
                "SELECT id FROM mm_games WHERE cbbd_game_id = ?", (g["cbbd_game_id"],)
            ).fetchone()
            if row:
                game_id_map[g["cbbd_game_id"]] = row["id"]
                counts["games"] += 1
        except Exception as e:
            logger.warning(f"Error inserting game {g['cbbd_game_id']}: {e}")

    # --- Step 2: Tournament entries (seeds/regions) ---
    # We derive this from the games data where seeds are available
    # and from R64 games specifically
    entry_map = {}  # team_id -> {seed, region}
    for g in games:
        if g.get("home_seed") and g.get("home_team") in team_id_map:
            tid = team_id_map[g["home_team"]]
            entry_map[tid] = {"seed": g["home_seed"], "region": g.get("region"),
                               "cbbd_team": g["home_team"]}
        if g.get("away_seed") and g.get("away_team") in team_id_map:
            tid = team_id_map[g["away_team"]]
            entry_map[tid] = {"seed": g["away_seed"], "region": g.get("region"),
                               "cbbd_team": g["away_team"]}

    for team_id, info in entry_map.items():
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO mm_tournament_entries
                    (season, team_id, seed, region, cbbd_team)
                VALUES (?, ?, ?, ?, ?)
                """,
                (season, team_id, info["seed"], info.get("region"), info.get("cbbd_team")),
            )
        except Exception as e:
            logger.warning(f"Error inserting tournament entry: {e}")

    # --- Step 3: Adjusted efficiency ratings ---
    ratings = get_adjusted_efficiency_ratings(season)
    ratings_map = {r["team"]: r for r in ratings}

    # --- Step 4: Team season stats ---
    stats = get_team_season_stats(season, season_type="regular")
    stats_map = {s["team"]: s for s in stats}

    for team_name, team_id in team_id_map.items():
        r = ratings_map.get(team_name, {})
        s = stats_map.get(team_name, {})

        # Get seed from tournament entry if available
        entry_info = entry_map.get(team_id, {})
        seed = entry_info.get("seed")
        net_rank = r.get("net_rank")
        seed_rank_gap = None
        if seed and net_rank:
            seed_rank_gap = net_rank - (seed * 10)

        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO mm_team_metrics
                    (season, team_id,
                     adj_off_rating, adj_def_rating, net_rating, net_rank, def_rank,
                     efg_pct, opp_efg_pct, tov_ratio, oreb_pct, ft_rate,
                     ft_pct, pace, seed_rank_gap,
                     conf_tourney_wins, conf_tourney_avg_margin)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    season, team_id,
                    r.get("adj_off_rating"), r.get("adj_def_rating"),
                    r.get("net_rating"), net_rank, r.get("def_rank"),
                    s.get("efg_pct"), s.get("opp_efg_pct"),
                    s.get("tov_ratio"), s.get("oreb_pct"), s.get("ft_rate"),
                    s.get("ft_pct"), s.get("pace"),
                    seed_rank_gap,
                    None, None,  # conf_tourney_wins/avg_margin populated in Step 6
                ),
            )
            counts["metrics"] += 1
        except Exception as e:
            logger.warning(f"Error inserting metrics for {team_name}: {e}")

    # --- Step 5: Betting lines ---
    cbbd_game_ids = list(game_id_map.keys())
    if cbbd_game_ids:
        lines = get_betting_lines(season, game_ids=cbbd_game_ids)
        for line in lines:
            local_game_id = game_id_map.get(line["cbbd_game_id"])
            if not local_game_id:
                continue
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO mm_betting_lines
                        (game_id, provider, team1_moneyline, team2_moneyline,
                         team1_novig_prob, team2_novig_prob)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        local_game_id,
                        line.get("provider"),
                        line.get("home_moneyline"),
                        line.get("away_moneyline"),
                        line.get("home_novig_prob"),
                        line.get("away_novig_prob"),
                    ),
                )
                counts["lines"] += 1
            except Exception as e:
                logger.warning(f"Error inserting line for game {line['cbbd_game_id']}: {e}")

    # --- Step 6: Conference tournament stats ---
    # Query games in the 3 weeks before the NCAA Tournament (Feb 25 - Mar 18)
    # and aggregate per-team wins and avg point differential in conf tournaments.
    conf_stats = get_conf_tournament_stats_by_team(season)
    for team_name, stats in conf_stats.items():
        team_id = team_id_map.get(team_name)
        if not team_id:
            continue
        try:
            conn.execute(
                """
                UPDATE mm_team_metrics
                SET conf_tourney_wins = ?, conf_tourney_avg_margin = ?
                WHERE season = ? AND team_id = ?
                """,
                (
                    stats["conf_tourney_wins"],
                    stats["conf_tourney_avg_margin"],
                    season,
                    team_id,
                ),
            )
            counts["conf_tourney"] += 1
        except Exception as e:
            logger.warning(f"Error updating conf tourney stats for {team_name}: {e}")

    logger.info(
        f"Season {season}: {counts['games']} games, {counts['teams']} teams, "
        f"{counts['metrics']} metrics rows, {counts['lines']} betting lines, "
        f"{counts['conf_tourney']} conf tourney updates"
    )
    return counts


def run(seasons: list[int] | None = None):
    """Run historical backfill for specified seasons (default: all HISTORICAL_SEASONS)."""
    if seasons is None:
        seasons = HISTORICAL_SEASONS

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(__file__).parent.parent / "logs" / "backfill.log",
                encoding="utf-8"
            ),
        ],
    )

    init_db()
    logger.info(f"Starting historical backfill for seasons: {seasons}")

    with get_db(DB_PATH) as conn:
        for season in seasons:
            try:
                counts = load_season(conn, season)
            except Exception as e:
                logger.error(f"Failed to load season {season}: {e}", exc_info=True)

    logger.info("Backfill complete.")


if __name__ == "__main__":
    run()
