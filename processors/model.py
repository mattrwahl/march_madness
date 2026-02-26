"""
March Madness selection model.
Optimizes feature weights to maximize total units won on historical data.
"""
import sys
import logging
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.optimize import minimize, differential_evolution

from processors.features import (
    FEATURE_NAMES, load_eligible_teams, compute_scores, select_picks
)
from config import INITIAL_BET_DOLLARS, NUM_PICKS, HISTORICAL_SEASONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rolling bet payout calculation
# ---------------------------------------------------------------------------

def compute_rolling_payout(
    initial_bet: float,
    moneylines: list[int | None],
    won_rounds: list[bool],
) -> float:
    """
    Compute the final payout for a pick that made it through some rounds.

    Args:
        initial_bet: Starting bet amount in dollars (e.g. 25.0)
        moneylines: American ML for each round the team played, in order
        won_rounds: Whether the team won each round (True/False)

    Returns:
        Final payout in dollars (initial_bet if lost R64, for example)
    """
    current_stake = initial_bet

    for ml, won in zip(moneylines, won_rounds):
        if not won:
            # Lost — return what was staked going into this round
            return current_stake if current_stake != initial_bet else 0.0

        if ml is None:
            # No line available — assume even money as fallback
            ml = 100

        if ml > 0:
            payout = current_stake * (ml / 100.0)
        else:
            payout = current_stake * (100.0 / abs(ml))

        current_stake = current_stake + payout  # roll winnings forward

    return current_stake


def get_team_tournament_results(
    conn: sqlite3.Connection, team_id: int, season: int
) -> list[dict]:
    """
    Get the ordered list of tournament games for a team in a given season.
    Returns games from R64 onward, ordered by round (64 → 1).
    """
    rows = conn.execute(
        """
        SELECT
            g.id as game_id,
            g.cbbd_game_id,
            g.round,
            g.team1_id, g.team2_id,
            g.team1_score, g.team2_score,
            g.winner_id,
            bl.team1_moneyline, bl.team2_moneyline,
            bl.team1_novig_prob, bl.team2_novig_prob,
            -- Determine which side this team is on
            CASE WHEN g.team1_id = ? THEN 'team1' ELSE 'team2' END as team_side
        FROM mm_games g
        LEFT JOIN mm_betting_lines bl ON bl.game_id = g.id
        WHERE g.season = ?
          AND (g.team1_id = ? OR g.team2_id = ?)
          AND g.round <= 64    -- only main bracket rounds
        ORDER BY g.round DESC  -- 64, 32, 16, 8 (ascending rounds = descending "round" number)
        """,
        (team_id, season, team_id, team_id),
    ).fetchall()

    results = []
    for r in rows:
        r = dict(r)
        side = r["team_side"]
        if side == "team1":
            r["team_moneyline"] = r["team1_moneyline"]
            r["won"] = r["winner_id"] == r["team1_id"]
        else:
            r["team_moneyline"] = r["team2_moneyline"]
            r["won"] = r["winner_id"] == r["team2_id"]
        results.append(r)

    return results


def simulate_pick_payout(
    conn: sqlite3.Connection,
    team_id: int,
    season: int,
    initial_bet: float = INITIAL_BET_DOLLARS,
    cash_out_round: int = 8,  # Elite Eight
    other_pick_ids: set | None = None,
) -> tuple[float, int | None]:
    """
    Simulate the rolling payout for a single pick.

    Args:
        conn: DB connection
        team_id: The picked team
        season: Tournament season
        initial_bet: Starting dollars
        cash_out_round: Cash out when this round is reached (8 = Elite Eight)
        other_pick_ids: Set of other team_ids — cash out if we face them

    Returns:
        (payout_dollars, round_exit)
        payout_dollars: Final payout (0.0 if lost in R64)
        round_exit: Round where team was eliminated or cashed out
    """
    games = get_team_tournament_results(conn, team_id, season)

    if not games:
        return 0.0, None

    stake = initial_bet
    round_exit = None

    for game in games:
        current_round = game["round"]

        # Cash out condition: reached Elite Eight
        if current_round <= cash_out_round and game["won"]:
            # Team reached E8 (won the Sweet 16 game to get here)
            # Actually: if round == 8, they ARE in E8 and we cash out win/loss
            pass

        # Check if facing another pick
        opponent_id = (game["team2_id"] if game["team_side"] == "team1"
                       else game["team1_id"])
        if other_pick_ids and opponent_id in other_pick_ids:
            # Two picks facing each other — cash out at current stake
            round_exit = current_round
            return stake, round_exit

        ml = game.get("team_moneyline")
        won = game.get("won", False)

        if not won:
            round_exit = current_round
            return 0.0, round_exit

        # Won this round — compute payout and roll forward
        if ml is not None:
            if ml > 0:
                payout = stake * (ml / 100.0)
            else:
                payout = stake * (100.0 / abs(ml))
        else:
            # No line — even money
            payout = stake

        stake = stake + payout
        round_exit = current_round

        # Cash out if we've reached or passed the Elite Eight round
        if current_round <= cash_out_round:
            return stake, round_exit

    return stake, round_exit


def simulate_season(
    conn: sqlite3.Connection,
    season: int,
    weights: np.ndarray,
) -> float:
    """
    Simulate one tournament season with given weights.
    Returns total units won across all 8 picks.
    """
    teams = load_eligible_teams(conn, season)
    if not teams:
        logger.warning(f"No eligible teams found for season {season}")
        return 0.0

    picks = select_picks(teams, weights, NUM_PICKS)
    pick_ids = {p["team_id"] for p in picks}
    other_ids_map = {p["team_id"]: pick_ids - {p["team_id"]} for p in picks}

    total_units = 0.0
    for pick in picks:
        payout, _ = simulate_pick_payout(
            conn,
            pick["team_id"],
            season,
            other_pick_ids=other_ids_map[pick["team_id"]],
        )
        units = (payout - INITIAL_BET_DOLLARS) / 100.0
        total_units += units

    return total_units


# ---------------------------------------------------------------------------
# Weight optimization
# ---------------------------------------------------------------------------

def objective(weights: np.ndarray, conn: sqlite3.Connection, seasons: list[int]) -> float:
    """
    Objective function: negative total units won across all training seasons.
    (Negative because scipy minimizes.)
    """
    total_units = sum(simulate_season(conn, s, weights) for s in seasons)
    return -total_units


def train_weights(
    conn: sqlite3.Connection,
    train_seasons: list[int],
    val_seasons: list[int] | None = None,
    method: str = "differential_evolution",
    seed: int = 42,
) -> dict:
    """
    Optimize feature weights to maximize units won on training seasons.

    Args:
        conn: DB connection
        train_seasons: Seasons to train on
        val_seasons: Seasons to validate on (not used in training)
        method: 'differential_evolution' (global) or 'nelder-mead' (local)
        seed: Random seed for reproducibility

    Returns:
        dict with weights, train_units, val_units
    """
    n_features = len(FEATURE_NAMES)
    bounds = [(-3.0, 3.0)] * n_features  # weight bounds for z-score normalized features

    logger.info(f"Training weights on seasons {train_seasons} using {method}")

    if method == "differential_evolution":
        result = differential_evolution(
            objective,
            bounds=bounds,
            args=(conn, train_seasons),
            seed=seed,
            maxiter=500,
            tol=1e-4,
            workers=1,
            popsize=15,
        )
        best_weights = result.x
    else:
        # Local optimization with multiple random starts
        best_val = float("inf")
        best_weights = np.zeros(n_features)
        rng = np.random.default_rng(seed)

        for _ in range(50):
            w0 = rng.uniform(-1.0, 1.0, n_features)
            res = minimize(
                objective,
                w0,
                args=(conn, train_seasons),
                method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_weights = res.x

    train_units = sum(simulate_season(conn, s, best_weights) for s in train_seasons)
    val_units = None
    if val_seasons:
        val_units = sum(simulate_season(conn, s, best_weights) for s in val_seasons)

    logger.info(f"Training complete. Train units: {train_units:.3f}, Val units: {val_units}")
    logger.info(f"Weights: {dict(zip(FEATURE_NAMES, best_weights))}")

    return {
        "weights": best_weights,
        "feature_names": FEATURE_NAMES,
        "train_units": train_units,
        "val_units": val_units,
        "train_seasons": train_seasons,
        "val_seasons": val_seasons,
    }


def save_weights(conn: sqlite3.Connection, result: dict) -> int:
    """Save trained weights to mm_model_weights. Returns row id."""
    w = result["weights"]
    conn.execute(
        """
        INSERT INTO mm_model_weights (
            train_seasons, val_seasons,
            w_seed_rank_gap, w_def_rank, w_opp_efg_pct, w_net_rating,
            w_tov_ratio, w_ft_pct, w_oreb_pct, w_pace,
            train_units_won, val_units_won
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ",".join(str(s) for s in result["train_seasons"]),
            ",".join(str(s) for s in (result["val_seasons"] or [])),
            float(w[0]), float(w[1]), float(w[2]), float(w[3]),
            float(w[4]), float(w[5]), float(w[6]), float(w[7]),
            result["train_units"],
            result["val_units"],
        ),
    )
    row = conn.execute(
        "SELECT id FROM mm_model_weights ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["id"]


def load_latest_weights(conn: sqlite3.Connection) -> np.ndarray | None:
    """Load the most recently trained weights from DB."""
    row = conn.execute(
        """
        SELECT w_seed_rank_gap, w_def_rank, w_opp_efg_pct, w_net_rating,
               w_tov_ratio, w_ft_pct, w_oreb_pct, w_pace
        FROM mm_model_weights
        ORDER BY trained_at DESC
        LIMIT 1
        """
    ).fetchone()

    if row is None:
        return None

    return np.array([
        row["w_seed_rank_gap"], row["w_def_rank"], row["w_opp_efg_pct"],
        row["w_net_rating"], row["w_tov_ratio"], row["w_ft_pct"],
        row["w_oreb_pct"], row["w_pace"],
    ])


def print_validation_report(conn: sqlite3.Connection, weights: np.ndarray):
    """Print a formatted validation report across all historical seasons."""
    print("\n" + "=" * 60)
    print("MODEL VALIDATION REPORT")
    print("=" * 60)

    total_units = 0.0
    for season in HISTORICAL_SEASONS:
        teams = load_eligible_teams(conn, season)
        if not teams:
            print(f"  {season}: no data")
            continue

        picks = select_picks(teams, weights, NUM_PICKS)
        pick_ids = {p["team_id"] for p in picks}
        other_ids_map = {p["team_id"]: pick_ids - {p["team_id"]} for p in picks}

        season_units = 0.0
        n_reached_e8 = 0

        for pick in picks:
            payout, round_exit = simulate_pick_payout(
                conn, pick["team_id"], season,
                other_pick_ids=other_ids_map[pick["team_id"]],
            )
            units = (payout - INITIAL_BET_DOLLARS) / 100.0
            season_units += units
            if round_exit is not None and round_exit <= 8:
                n_reached_e8 += 1

        total_units += season_units
        print(f"  {season}: {season_units:+.3f} units  ({n_reached_e8}/{NUM_PICKS} reached E8)")

        for pick in picks:
            payout, round_exit = simulate_pick_payout(
                conn, pick["team_id"], season,
                other_pick_ids=other_ids_map[pick["team_id"]],
            )
            units = (payout - INITIAL_BET_DOLLARS) / 100.0
            print(f"    #{pick['pick_rank']} {pick['team_name']:30s} "
                  f"seed={pick['seed']:2d}  "
                  f"score={pick['model_score']:+.3f}  "
                  f"units={units:+.3f}")

    print("-" * 60)
    print(f"  TOTAL: {total_units:+.3f} units across {len(HISTORICAL_SEASONS)} seasons")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point for training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from db.db import init_db, get_db as _get_db
    from config import DB_PATH, HISTORICAL_SEASONS

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train/evaluate the March Madness model")
    parser.add_argument("--train", action="store_true", help="Run weight optimization")
    parser.add_argument("--report", action="store_true", help="Print validation report using latest weights")
    parser.add_argument("--method", default="differential_evolution",
                        choices=["differential_evolution", "nelder-mead"])
    args = parser.parse_args()

    from shared.sqlite_helpers import get_db as _shared_get_db

    if args.train:
        train_seasons = HISTORICAL_SEASONS[:-1]   # 2021–2024
        val_seasons   = HISTORICAL_SEASONS[-1:]   # 2025

        with _shared_get_db(DB_PATH) as conn:
            result = train_weights(conn, train_seasons, val_seasons, method=args.method)
            wid = save_weights(conn, result)
            print(f"\nWeights saved (id={wid})")
            print_validation_report(conn, result["weights"])

    elif args.report:
        with _shared_get_db(DB_PATH) as conn:
            weights = load_latest_weights(conn)
            if weights is None:
                print("No trained weights found. Run --train first.")
            else:
                print_validation_report(conn, weights)
