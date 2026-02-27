"""
March Madness selection model.
Optimizes feature weights to maximize total units won on historical data.
Supports configurable cash-out round (E8=8 or S16=16).
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
from config import (
    INITIAL_BET_DOLLARS, NUM_PICKS, ALL_SEASONS,
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    DEFAULT_CASH_OUT_ROUND,
)

logger = logging.getLogger(__name__)

ROUND_LABELS = {
    65: "First Four", 64: "R64", 32: "R32", 16: "S16",
    8: "E8", 4: "F4", 2: "Champ",
}


# ---------------------------------------------------------------------------
# Rolling bet payout calculation
# ---------------------------------------------------------------------------

def get_team_tournament_results(
    conn: sqlite3.Connection, team_id: int, season: int
) -> list[dict]:
    """
    Get the ordered list of tournament games for a team in a given season.
    Returns games from R64 onward, ordered earliest round first (64 → 2).
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
            CASE WHEN g.team1_id = ? THEN 'team1' ELSE 'team2' END as team_side
        FROM mm_games g
        LEFT JOIN mm_betting_lines bl ON bl.game_id = g.id
        WHERE g.season = ?
          AND (g.team1_id = ? OR g.team2_id = ?)
          AND g.round <= 64
        ORDER BY g.round DESC
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
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    other_pick_ids: set | None = None,
) -> tuple[float, int | None]:
    """
    Simulate the rolling payout for a single pick.

    Cash-out rules:
      - After winning a game in `cash_out_round` (e.g. 8=E8, 16=S16), collect stake.
      - If two picks face each other at any point, both cash out at current stake.
      - A loss at any round returns 0.0 (total risk was the initial $25).

    Returns:
        (payout_dollars, round_exit)
    """
    games = get_team_tournament_results(conn, team_id, season)

    if not games:
        return 0.0, None

    stake = initial_bet
    round_exit = None

    for game in games:
        current_round = game["round"]

        # Two picks facing each other — void both bets, return current stake
        opponent_id = (game["team2_id"] if game["team_side"] == "team1"
                       else game["team1_id"])
        if other_pick_ids and opponent_id in other_pick_ids:
            round_exit = current_round
            return stake, round_exit

        won = game.get("won", False)

        if not won:
            round_exit = current_round
            return 0.0, round_exit

        # Won — compute payout and roll forward
        ml = game.get("team_moneyline")
        if ml is not None:
            if ml > 0:
                profit = stake * (ml / 100.0)
            else:
                profit = stake * (100.0 / abs(ml))
        else:
            profit = stake  # even money fallback for seasons without lines

        stake = stake + profit
        round_exit = current_round

        # Cash out after winning the target round
        if current_round <= cash_out_round:
            return stake, round_exit

    return stake, round_exit


def simulate_season(
    conn: sqlite3.Connection,
    season: int,
    weights: np.ndarray,
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
) -> float:
    """
    Simulate one tournament season with given weights.
    Returns total units won across all picks.
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
            cash_out_round=cash_out_round,
            other_pick_ids=other_ids_map[pick["team_id"]],
        )
        units = (payout - INITIAL_BET_DOLLARS) / 100.0
        total_units += units

    return total_units


# ---------------------------------------------------------------------------
# Weight optimization
# ---------------------------------------------------------------------------

def objective(
    weights: np.ndarray,
    conn: sqlite3.Connection,
    seasons: list[int],
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
) -> float:
    """Negative total units won (scipy minimizes)."""
    total = sum(simulate_season(conn, s, weights, cash_out_round) for s in seasons)
    return -total


def train_weights(
    conn: sqlite3.Connection,
    train_seasons: list[int],
    val_seasons: list[int] | None = None,
    test_seasons: list[int] | None = None,
    method: str = "differential_evolution",
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    seed: int = 42,
) -> dict:
    """
    Optimize feature weights to maximize units won on training seasons.

    Returns dict with weights, train_units, val_units, test_units, cash_out_round.
    """
    n_features = len(FEATURE_NAMES)
    bounds = [(-3.0, 3.0)] * n_features

    co_label = f"E8" if cash_out_round == 8 else f"S16" if cash_out_round == 16 else f"R{cash_out_round}"
    logger.info(
        f"Training ({co_label} cash-out) on {len(train_seasons)} seasons "
        f"[{min(train_seasons)}–{max(train_seasons)}] using {method}"
    )

    _obj = lambda w: objective(w, conn, train_seasons, cash_out_round)

    if method == "differential_evolution":
        result = differential_evolution(
            _obj,
            bounds=bounds,
            seed=seed,
            maxiter=500,
            tol=1e-4,
            workers=1,
            popsize=15,
        )
        best_weights = result.x
    else:
        best_val = float("inf")
        best_weights = np.zeros(n_features)
        rng = np.random.default_rng(seed)
        for _ in range(50):
            w0 = rng.uniform(-1.0, 1.0, n_features)
            res = minimize(
                _obj, w0, method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_weights = res.x

    train_units = sum(simulate_season(conn, s, best_weights, cash_out_round) for s in train_seasons)
    val_units = (sum(simulate_season(conn, s, best_weights, cash_out_round) for s in val_seasons)
                 if val_seasons else None)
    test_units = (sum(simulate_season(conn, s, best_weights, cash_out_round) for s in test_seasons)
                  if test_seasons else None)

    logger.info(
        f"Done. Train: {train_units:+.2f}u  Val: {val_units:+.2f}u  "
        f"Test: {test_units:+.2f}u  ({co_label} cash-out)"
    )
    return {
        "weights": best_weights,
        "feature_names": FEATURE_NAMES,
        "train_units": train_units,
        "val_units": val_units,
        "test_units": test_units,
        "train_seasons": train_seasons,
        "val_seasons": val_seasons,
        "test_seasons": test_seasons,
        "cash_out_round": cash_out_round,
    }


def save_weights(conn: sqlite3.Connection, result: dict) -> int:
    """Save trained weights to mm_model_weights. Returns row id."""
    w = result["weights"]
    cash_out_round = result.get("cash_out_round", DEFAULT_CASH_OUT_ROUND)
    co_label = "E8" if cash_out_round == 8 else "S16" if cash_out_round == 16 else str(cash_out_round)

    conn.execute(
        """
        INSERT INTO mm_model_weights (
            train_seasons, val_seasons,
            w_seed_rank_gap, w_def_rank, w_opp_efg_pct, w_net_rating,
            w_tov_ratio, w_ft_pct, w_oreb_pct, w_pace,
            train_units_won, val_units_won,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ",".join(str(s) for s in result["train_seasons"]),
            ",".join(str(s) for s in (result["val_seasons"] or [])),
            float(w[0]), float(w[1]), float(w[2]), float(w[3]),
            float(w[4]), float(w[5]), float(w[6]), float(w[7]),
            result["train_units"],
            result["val_units"],
            f"cash_out={co_label} test={result.get('test_units')}",
        ),
    )
    row = conn.execute(
        "SELECT id FROM mm_model_weights ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["id"]


def load_latest_weights(
    conn: sqlite3.Connection,
    cash_out_round: int | None = None,
) -> np.ndarray | None:
    """
    Load the most recently trained weights.
    If cash_out_round is specified, loads the latest weights for that cash-out style.
    """
    if cash_out_round is not None:
        co_label = "E8" if cash_out_round == 8 else "S16" if cash_out_round == 16 else str(cash_out_round)
        row = conn.execute(
            """
            SELECT w_seed_rank_gap, w_def_rank, w_opp_efg_pct, w_net_rating,
                   w_tov_ratio, w_ft_pct, w_oreb_pct, w_pace
            FROM mm_model_weights
            WHERE notes LIKE ?
            ORDER BY trained_at DESC LIMIT 1
            """,
            (f"cash_out={co_label}%",),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT w_seed_rank_gap, w_def_rank, w_opp_efg_pct, w_net_rating,
                   w_tov_ratio, w_ft_pct, w_oreb_pct, w_pace
            FROM mm_model_weights
            ORDER BY trained_at DESC LIMIT 1
            """
        ).fetchone()

    if row is None:
        return None

    return np.array([
        row["w_seed_rank_gap"], row["w_def_rank"], row["w_opp_efg_pct"],
        row["w_net_rating"], row["w_tov_ratio"], row["w_ft_pct"],
        row["w_oreb_pct"], row["w_pace"],
    ])


def print_validation_report(
    conn: sqlite3.Connection,
    weights: np.ndarray,
    seasons: list[int] | None = None,
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    label: str | None = None,
):
    """Print a formatted per-season validation report."""
    if seasons is None:
        seasons = ALL_SEASONS

    co_label = "E8" if cash_out_round == 8 else "S16" if cash_out_round == 16 else f"R{cash_out_round}"
    header = label or f"MODEL REPORT ({co_label} cash-out)"

    print("\n" + "=" * 65)
    print(header)
    print("=" * 65)

    total_units = 0.0
    seasons_with_data = 0

    for season in sorted(seasons):
        teams = load_eligible_teams(conn, season)
        if not teams:
            print(f"  {season}: no data")
            continue

        picks = select_picks(teams, weights, NUM_PICKS)
        pick_ids = {p["team_id"] for p in picks}
        other_ids_map = {p["team_id"]: pick_ids - {p["team_id"]} for p in picks}

        season_units = 0.0
        n_cashed = 0

        pick_rows = []
        for pick in picks:
            payout, round_exit = simulate_pick_payout(
                conn, pick["team_id"], season,
                cash_out_round=cash_out_round,
                other_pick_ids=other_ids_map[pick["team_id"]],
            )
            units = (payout - INITIAL_BET_DOLLARS) / 100.0
            season_units += units
            if round_exit is not None and round_exit <= cash_out_round and payout > INITIAL_BET_DOLLARS:
                n_cashed += 1
            pick_rows.append((pick, units, round_exit))

        total_units += season_units
        seasons_with_data += 1

        # Marker for special seasons
        marker = ""
        if season in VAL_SEASONS:
            marker = " [VAL]"
        elif season in TEST_SEASONS:
            marker = " [TEST]"

        print(f"  {season}{marker}: {season_units:+.3f} units  ({n_cashed}/{NUM_PICKS} cashed)")
        for pick, units, round_exit in pick_rows:
            exit_str = ROUND_LABELS.get(round_exit, f"R{round_exit}") if round_exit else "?"
            print(f"    #{pick['pick_rank']} {pick['team_name']:30s} "
                  f"seed={pick['seed']:2d}  {exit_str:<8}  {units:+.3f}u")

    print("-" * 65)
    print(f"  TOTAL: {total_units:+.3f} units  ({seasons_with_data} seasons)")
    print("=" * 65)
