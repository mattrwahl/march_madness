"""
March Madness selection model.
Optimizes feature weights to maximize total units won on historical data.

Bet styles:
  flat     — fixed $25 bet each round; profits locked in independently per round
  rollover — $25 initial stake compounds forward through each win (parlay-style)

Cash-out rounds (teams-remaining encoding):
  S16 (16) — cash out after winning R32  → enter E8   (2 rounds max)
  E8  ( 8) — cash out after winning S16  → enter F4   (3 rounds max)
  F4  ( 4) — cash out after winning E8   → enter Champ (4 rounds max)
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
    FEATURE_NAMES, load_eligible_teams, compute_scores,
    select_picks, select_picks_threshold,
)
from config import (
    INITIAL_BET_DOLLARS, NUM_PICKS, ALL_SEASONS,
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    CASH_OUT_ROUND_S16, CASH_OUT_ROUND_E8, CASH_OUT_ROUND_F4,
    DEFAULT_CASH_OUT_ROUND,
    BET_STYLE_FLAT, BET_STYLE_ROLLOVER, DEFAULT_BET_STYLE,
    MIN_PICKS_PER_YEAR, MAX_PICKS_PER_YEAR, DEFAULT_THRESHOLD,
    TIER1_BET, TIER2_BET, L2_LAMBDA,
)

logger = logging.getLogger(__name__)

ROUND_LABELS = {
    65: "First Four", 64: "R64", 32: "R32", 16: "S16",
    8: "E8", 4: "F4", 2: "Champ",
}

_CO_LABEL = {
    CASH_OUT_ROUND_S16: "S16",
    CASH_OUT_ROUND_E8:  "E8",
    CASH_OUT_ROUND_F4:  "F4",
}


def _co_label(cash_out_round: int) -> str:
    return _CO_LABEL.get(cash_out_round, f"R{cash_out_round}")


# ---------------------------------------------------------------------------
# Tournament results helper
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


def _ml_profit(stake: float, ml: float | None) -> float:
    """Net profit on `stake` dollars given an American moneyline (None → even money)."""
    if ml is None:
        return stake
    if ml > 0:
        return stake * (ml / 100.0)
    return stake * (100.0 / abs(ml))


# ---------------------------------------------------------------------------
# Payout simulation — flat and rollover
# ---------------------------------------------------------------------------

def simulate_pick_payout(
    conn: sqlite3.Connection,
    team_id: int,
    season: int,
    initial_bet: float = INITIAL_BET_DOLLARS,
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    other_pick_ids: set | None = None,
    bet_style: str = DEFAULT_BET_STYLE,
) -> tuple[float, int | None]:
    """
    Simulate the payout for a single pick.

    Flat style:
      - Bet `initial_bet` each round independently.
      - Winning a round locks in that round's profit; stake is NOT rolled over.
      - Losing a round costs `initial_bet` for that round; prior profits kept.
      - If two picks face each other, stop betting (no bet placed that round).
      - Returns (initial_bet + total_net_profit, round_exit).

    Rollover style:
      - `initial_bet` is the starting stake; winnings compound each round.
      - A loss at any round returns 0.0 (full stake lost).
      - If two picks face each other, return current stake (void the bet).
      - Returns (final_stake, round_exit).

    In both cases: units_won = (payout - initial_bet) / 100.
    """
    games = get_team_tournament_results(conn, team_id, season)

    if not games:
        return 0.0, None

    round_exit = None

    # ---- FLAT ---------------------------------------------------------------
    if bet_style == BET_STYLE_FLAT:
        net_profit = 0.0

        for game in games:
            current_round = game["round"]

            # Two picks meet — stop before placing this round's bet
            opponent_id = (game["team2_id"] if game["team_side"] == "team1"
                           else game["team1_id"])
            if other_pick_ids and opponent_id in other_pick_ids:
                round_exit = current_round
                break

            won = game.get("won", False)

            if not won:
                net_profit -= initial_bet          # lose this round's bet
                round_exit = current_round
                break

            # Won — lock in this round's profit
            profit = _ml_profit(initial_bet, game.get("team_moneyline"))
            net_profit += profit
            round_exit = current_round

            # Cash out after winning the target round
            if current_round <= cash_out_round:
                break

        return initial_bet + net_profit, round_exit

    # ---- ROLLOVER -----------------------------------------------------------
    stake = initial_bet

    for game in games:
        current_round = game["round"]

        # Two picks meet — void both bets, return current stake
        opponent_id = (game["team2_id"] if game["team_side"] == "team1"
                       else game["team1_id"])
        if other_pick_ids and opponent_id in other_pick_ids:
            round_exit = current_round
            return stake, round_exit

        won = game.get("won", False)

        if not won:
            round_exit = current_round
            return 0.0, round_exit

        # Won — compound the stake
        profit = _ml_profit(stake, game.get("team_moneyline"))
        stake += profit
        round_exit = current_round

        # Cash out after winning the target round
        if current_round <= cash_out_round:
            return stake, round_exit

    return stake, round_exit


# ---------------------------------------------------------------------------
# Season simulation
# ---------------------------------------------------------------------------

def simulate_season(
    conn: sqlite3.Connection,
    season: int,
    weights: np.ndarray,
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    bet_style: str = DEFAULT_BET_STYLE,
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
            cash_out_round=cash_out_round,
            other_pick_ids=other_ids_map[pick["team_id"]],
            bet_style=bet_style,
        )
        total_units += (payout - INITIAL_BET_DOLLARS) / 100.0

    return total_units


# ---------------------------------------------------------------------------
# Weight optimization
# ---------------------------------------------------------------------------

def objective(
    weights: np.ndarray,
    conn: sqlite3.Connection,
    seasons: list[int],
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    bet_style: str = DEFAULT_BET_STYLE,
    lambda_l2: float = L2_LAMBDA,
) -> float:
    """Negative total units won + L2 regularization on weights (scipy minimizes)."""
    total = sum(
        simulate_season(conn, s, weights, cash_out_round, bet_style)
        for s in seasons
    )
    l2 = lambda_l2 * float(np.sum(weights ** 2))
    return -total + l2


def train_weights(
    conn: sqlite3.Connection,
    train_seasons: list[int],
    val_seasons: list[int] | None = None,
    test_seasons: list[int] | None = None,
    method: str = "differential_evolution",
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    bet_style: str = DEFAULT_BET_STYLE,
    seed: int = 42,
) -> dict:
    """
    Optimize feature weights to maximize units won on training seasons.

    Returns dict with weights, train_units, val_units, test_units,
    cash_out_round, and bet_style.
    """
    n_features = len(FEATURE_NAMES)
    bounds = [(-3.0, 3.0)] * n_features

    co = _co_label(cash_out_round)
    logger.info(
        f"Training ({bet_style}/{co} cash-out) on {len(train_seasons)} seasons "
        f"[{min(train_seasons)}–{max(train_seasons)}] using {method}"
    )

    _obj = lambda w: objective(w, conn, train_seasons, cash_out_round, bet_style)

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

    def _eval(seasons_list):
        if not seasons_list:
            return None
        return sum(
            simulate_season(conn, s, best_weights, cash_out_round, bet_style)
            for s in seasons_list
        )

    train_units = _eval(train_seasons)
    val_units   = _eval(val_seasons)
    test_units  = _eval(test_seasons)

    logger.info(
        f"Done ({bet_style}/{co}). "
        f"Train: {train_units:+.2f}u  Val: {val_units:+.2f}u  Test: {test_units:+.2f}u"
    )
    return {
        "weights":       best_weights,
        "feature_names": FEATURE_NAMES,
        "train_units":   train_units,
        "val_units":     val_units,
        "test_units":    test_units,
        "train_seasons": train_seasons,
        "val_seasons":   val_seasons,
        "test_seasons":  test_seasons,
        "cash_out_round": cash_out_round,
        "bet_style":     bet_style,
    }


def save_weights(conn: sqlite3.Connection, result: dict) -> int:
    """Save trained weights to mm_model_weights. Returns row id."""
    w = result["weights"]
    co = _co_label(result.get("cash_out_round", DEFAULT_CASH_OUT_ROUND))
    bs = result.get("bet_style", DEFAULT_BET_STYLE)
    # Pad to 11 features if older weights are passed
    w_vals = [float(w[i]) if i < len(w) else 0.0 for i in range(11)]

    conn.execute(
        """
        INSERT INTO mm_model_weights (
            train_seasons, val_seasons,
            w_seed_rank_gap, w_def_rank, w_opp_efg_pct, w_net_rating,
            w_tov_ratio, w_ft_pct, w_oreb_pct, w_pace,
            w_conf_tourney_wins, w_conf_tourney_avg_margin,
            w_region_top4_net_avg,
            train_units_won, val_units_won,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ",".join(str(s) for s in result["train_seasons"]),
            ",".join(str(s) for s in (result["val_seasons"] or [])),
            w_vals[0], w_vals[1], w_vals[2], w_vals[3],
            w_vals[4], w_vals[5], w_vals[6], w_vals[7],
            w_vals[8], w_vals[9],
            w_vals[10],
            result["train_units"],
            result["val_units"],
            f"cash_out={co} bet_style={bs} test={result.get('test_units')}",
        ),
    )
    row = conn.execute(
        "SELECT id FROM mm_model_weights ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["id"]


def load_latest_weights(
    conn: sqlite3.Connection,
    cash_out_round: int | None = None,
    bet_style: str | None = None,
) -> np.ndarray | None:
    """
    Load the most recently trained weights, optionally filtered by
    cash_out_round and/or bet_style.
    """
    conditions = []
    params = []

    if cash_out_round is not None:
        co = _co_label(cash_out_round)
        conditions.append("notes LIKE ?")
        params.append(f"cash_out={co}%")

    if bet_style is not None:
        conditions.append("notes LIKE ?")
        params.append(f"%bet_style={bet_style}%")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    row = conn.execute(
        f"""
        SELECT w_seed_rank_gap, w_def_rank, w_opp_efg_pct, w_net_rating,
               w_tov_ratio, w_ft_pct, w_oreb_pct, w_pace,
               w_conf_tourney_wins, w_conf_tourney_avg_margin,
               w_region_top4_net_avg
        FROM mm_model_weights
        {where}
        ORDER BY trained_at DESC LIMIT 1
        """,
        params,
    ).fetchone()

    if row is None:
        return None

    # Fall back to 0.0 for columns not present in older weight rows
    return np.array([
        row["w_seed_rank_gap"], row["w_def_rank"], row["w_opp_efg_pct"],
        row["w_net_rating"], row["w_tov_ratio"], row["w_ft_pct"],
        row["w_oreb_pct"], row["w_pace"],
        row["w_conf_tourney_wins"]        or 0.0,
        row["w_conf_tourney_avg_margin"]  or 0.0,
        row["w_region_top4_net_avg"]      or 0.0,
    ])


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def print_validation_report(
    conn: sqlite3.Connection,
    weights: np.ndarray,
    seasons: list[int] | None = None,
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    bet_style: str = DEFAULT_BET_STYLE,
    label: str | None = None,
):
    """Print a formatted per-season validation report."""
    if seasons is None:
        seasons = ALL_SEASONS

    co = _co_label(cash_out_round)
    header = label or f"MODEL REPORT ({bet_style.upper()} / {co} cash-out)"

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
                bet_style=bet_style,
            )
            units = (payout - INITIAL_BET_DOLLARS) / 100.0
            season_units += units
            if round_exit is not None and round_exit <= cash_out_round and payout > INITIAL_BET_DOLLARS:
                n_cashed += 1
            pick_rows.append((pick, units, round_exit))

        total_units += season_units
        seasons_with_data += 1

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


# ---------------------------------------------------------------------------
# Variable-N model — threshold-based selection + ROI objective
# ---------------------------------------------------------------------------

def simulate_season_variable(
    conn: sqlite3.Connection,
    season: int,
    weights: np.ndarray,
    threshold: float,
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    bet_style: str = DEFAULT_BET_STYLE,
    min_picks: int = MIN_PICKS_PER_YEAR,
    max_picks: int = MAX_PICKS_PER_YEAR,
) -> tuple[float, int]:
    """
    Simulate one season using variable-N threshold selection.
    Returns (total_units, n_picks).
    """
    teams = load_eligible_teams(conn, season)
    if not teams:
        logger.warning(f"No eligible teams found for season {season}")
        return 0.0, 0

    picks = select_picks_threshold(teams, weights, threshold, min_picks, max_picks)
    if not picks:
        return 0.0, 0

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
            bet_style=bet_style,
        )
        total_units += (payout - INITIAL_BET_DOLLARS) / 100.0

    return total_units, len(picks)


def objective_roi(
    params: np.ndarray,
    conn: sqlite3.Connection,
    seasons: list[int],
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    bet_style: str = DEFAULT_BET_STYLE,
    min_picks: int = MIN_PICKS_PER_YEAR,
    max_picks: int = MAX_PICKS_PER_YEAR,
    lambda_l2: float = L2_LAMBDA,
) -> float:
    """
    Negative ROI (units/pick) + L2 regularization across all training seasons.
    params = [w0..wN, threshold]  (N = len(FEATURE_NAMES))
    """
    weights   = params[:-1]
    threshold = params[-1]

    total_units = 0.0
    total_picks = 0

    for season in seasons:
        units, n = simulate_season_variable(
            conn, season, weights, threshold,
            cash_out_round, bet_style, min_picks, max_picks,
        )
        total_units += units
        total_picks += n

    if total_picks == 0:
        return 0.0  # degenerate; min_picks floor should prevent this

    l2 = lambda_l2 * float(np.sum(weights ** 2))
    return -(total_units / total_picks) + l2


def train_weights_variable(
    conn: sqlite3.Connection,
    train_seasons: list[int],
    val_seasons: list[int] | None = None,
    test_seasons: list[int] | None = None,
    method: str = "differential_evolution",
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    bet_style: str = DEFAULT_BET_STYLE,
    min_picks: int = MIN_PICKS_PER_YEAR,
    max_picks: int = MAX_PICKS_PER_YEAR,
    seed: int = 42,
) -> dict:
    """
    Optimize feature weights + z-score threshold to maximize ROI (units/pick).
    Optimizes n_features+1 parameters: [w0..wN, threshold].

    Returns dict with weights, threshold, roi metrics, and pick counts.
    """
    n_features = len(FEATURE_NAMES)
    # N feature weights + 1 threshold (z-score range: -1.0 = inclusive, 2.0 = selective)
    bounds = [(-3.0, 3.0)] * n_features + [(-1.0, 2.0)]

    co = _co_label(cash_out_round)
    logger.info(
        f"Training VARIABLE-N ({bet_style}/{co}) on {len(train_seasons)} seasons "
        f"[{min(train_seasons)}-{max(train_seasons)}] "
        f"min_picks={min_picks} max_picks={max_picks} using {method}"
    )

    _obj = lambda p: objective_roi(
        p, conn, train_seasons, cash_out_round, bet_style, min_picks, max_picks
    )

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
        best_params = result.x
    else:
        best_val = float("inf")
        best_params = np.zeros(n_features + 1)
        rng = np.random.default_rng(seed)
        for _ in range(50):
            p0 = np.concatenate([
                rng.uniform(-1.0, 1.0, n_features),
                rng.uniform(-0.5, 1.5, 1),
            ])
            res = minimize(
                _obj, p0, method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x

    best_weights   = best_params[:-1]
    best_threshold = float(best_params[-1])

    def _eval(seasons_list):
        if not seasons_list:
            return None, None, None
        total_u, total_n = 0.0, 0
        for s in seasons_list:
            u, n = simulate_season_variable(
                conn, s, best_weights, best_threshold,
                cash_out_round, bet_style, min_picks, max_picks,
            )
            total_u += u
            total_n += n
        roi = (total_u / total_n) if total_n > 0 else None
        return total_u, total_n, roi

    train_u, train_n, train_roi = _eval(train_seasons)
    val_u,   val_n,   val_roi   = _eval(val_seasons)
    test_u,  test_n,  test_roi  = _eval(test_seasons)

    logger.info(
        f"Done VARIABLE-N ({bet_style}/{co}). threshold={best_threshold:.3f}  "
        f"Train: {train_u:+.2f}u / {train_n} picks (ROI {train_roi:+.4f}u/pick)  "
        f"Val: {val_u:+.2f}u / {val_n} picks (ROI {val_roi:+.4f}u/pick)"
    )

    return {
        "weights":       best_weights,
        "threshold":     best_threshold,
        "feature_names": FEATURE_NAMES,
        "train_units":   train_u,  "train_picks": train_n,  "train_roi": train_roi,
        "val_units":     val_u,    "val_picks":   val_n,    "val_roi":   val_roi,
        "test_units":    test_u,   "test_picks":  test_n,   "test_roi":  test_roi,
        "train_seasons": train_seasons,
        "val_seasons":   val_seasons,
        "test_seasons":  test_seasons,
        "cash_out_round": cash_out_round,
        "bet_style":     bet_style,
        "min_picks":     min_picks,
        "max_picks":     max_picks,
        "model_type":    "variable",
    }


def save_weights_variable(conn: sqlite3.Connection, result: dict) -> int:
    """Save variable-N trained weights + threshold to mm_model_weights. Returns row id."""
    w  = result["weights"]
    co = _co_label(result.get("cash_out_round", DEFAULT_CASH_OUT_ROUND))
    bs = result.get("bet_style", DEFAULT_BET_STYLE)
    th = result.get("threshold", None)
    # Pad to 11 features if older weights are passed
    w_vals = [float(w[i]) if i < len(w) else 0.0 for i in range(11)]

    conn.execute(
        """
        INSERT INTO mm_model_weights (
            train_seasons, val_seasons,
            w_seed_rank_gap, w_def_rank, w_opp_efg_pct, w_net_rating,
            w_tov_ratio, w_ft_pct, w_oreb_pct, w_pace,
            w_conf_tourney_wins, w_conf_tourney_avg_margin,
            w_region_top4_net_avg,
            w_threshold,
            train_units_won, val_units_won,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ",".join(str(s) for s in result["train_seasons"]),
            ",".join(str(s) for s in (result["val_seasons"] or [])),
            w_vals[0], w_vals[1], w_vals[2], w_vals[3],
            w_vals[4], w_vals[5], w_vals[6], w_vals[7],
            w_vals[8], w_vals[9],
            w_vals[10],
            float(th) if th is not None else None,
            result["train_units"],
            result["val_units"],
            (f"model=variable cash_out={co} bet_style={bs} "
             f"threshold={th:.4f} test={result.get('test_units')}"),
        ),
    )
    row = conn.execute(
        "SELECT id FROM mm_model_weights ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["id"]


def load_latest_weights_variable(
    conn: sqlite3.Connection,
    cash_out_round: int | None = None,
    bet_style: str | None = None,
) -> tuple[np.ndarray, float] | None:
    """
    Load the most recently trained variable-N weights and threshold.
    Returns (weights_array, threshold) or None if not found.
    """
    conditions = ["notes LIKE '%model=variable%'"]
    params = []

    if cash_out_round is not None:
        co = _co_label(cash_out_round)
        conditions.append("notes LIKE ?")
        params.append(f"%cash_out={co}%")

    if bet_style is not None:
        conditions.append("notes LIKE ?")
        params.append(f"%bet_style={bet_style}%")

    where = "WHERE " + " AND ".join(conditions)

    row = conn.execute(
        f"""
        SELECT w_seed_rank_gap, w_def_rank, w_opp_efg_pct, w_net_rating,
               w_tov_ratio, w_ft_pct, w_oreb_pct, w_pace,
               w_conf_tourney_wins, w_conf_tourney_avg_margin,
               w_region_top4_net_avg,
               w_threshold
        FROM mm_model_weights
        {where}
        ORDER BY trained_at DESC LIMIT 1
        """,
        params,
    ).fetchone()

    if row is None:
        return None

    weights = np.array([
        row["w_seed_rank_gap"], row["w_def_rank"], row["w_opp_efg_pct"],
        row["w_net_rating"], row["w_tov_ratio"], row["w_ft_pct"],
        row["w_oreb_pct"], row["w_pace"],
        row["w_conf_tourney_wins"]        or 0.0,
        row["w_conf_tourney_avg_margin"]  or 0.0,
        row["w_region_top4_net_avg"]      or 0.0,
    ])
    threshold = row["w_threshold"]
    return weights, threshold


def print_validation_report_variable(
    conn: sqlite3.Connection,
    weights: np.ndarray,
    threshold: float,
    seasons: list[int] | None = None,
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    bet_style: str = DEFAULT_BET_STYLE,
    min_picks: int = MIN_PICKS_PER_YEAR,
    max_picks: int = MAX_PICKS_PER_YEAR,
    label: str | None = None,
):
    """Print a formatted per-season report for the variable-N model."""
    if seasons is None:
        seasons = ALL_SEASONS

    co = _co_label(cash_out_round)
    header = label or (
        f"VARIABLE-N MODEL REPORT ({bet_style.upper()} / {co} cash-out)  "
        f"threshold={threshold:.3f}  min={min_picks} max={max_picks}"
    )

    print("\n" + "=" * 70)
    print(header)
    print("=" * 70)

    total_units = 0.0
    total_picks = 0
    seasons_with_data = 0

    for season in sorted(seasons):
        teams = load_eligible_teams(conn, season)
        if not teams:
            print(f"  {season}: no data")
            continue

        picks = select_picks_threshold(teams, weights, threshold, min_picks, max_picks)
        if not picks:
            print(f"  {season}: no picks cleared threshold")
            continue

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
                bet_style=bet_style,
            )
            units = (payout - INITIAL_BET_DOLLARS) / 100.0
            season_units += units
            if round_exit is not None and round_exit <= cash_out_round and payout > INITIAL_BET_DOLLARS:
                n_cashed += 1
            pick_rows.append((pick, units, round_exit))

        n_picks = len(picks)
        season_roi = season_units / n_picks if n_picks > 0 else 0.0
        total_units += season_units
        total_picks += n_picks
        seasons_with_data += 1

        marker = ""
        if season in VAL_SEASONS:
            marker = " [VAL]"
        elif season in TEST_SEASONS:
            marker = " [TEST]"

        print(f"  {season}{marker} [{n_picks} picks]: "
              f"{season_units:+.3f}u  ROI {season_roi:+.3f}u/pick  "
              f"({n_cashed}/{n_picks} cashed)")
        for pick, units, round_exit in pick_rows:
            exit_str = ROUND_LABELS.get(round_exit, f"R{round_exit}") if round_exit else "?"
            z_str = f"z={pick.get('zscore', 0.0):+.2f}"
            print(f"    #{pick['pick_rank']} {pick['team_name']:30s} "
                  f"seed={pick['seed']:2d}  {z_str}  {exit_str:<8}  {units:+.3f}u")

    overall_roi = (total_units / total_picks) if total_picks > 0 else 0.0
    print("-" * 70)
    print(f"  TOTAL:  {total_units:+.3f}u  |  {total_picks} picks  "
          f"|  {seasons_with_data} seasons  |  avg {total_picks/max(seasons_with_data,1):.1f} picks/yr")
    print(f"  ROI:    {overall_roi:+.4f}u per pick")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Tiered conviction betting — top-half at TIER1_BET, bottom-half at TIER2_BET
# ---------------------------------------------------------------------------

def simulate_season_tiered(
    conn: sqlite3.Connection,
    season: int,
    weights: np.ndarray,
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
) -> float:
    """
    Simulate one season with tiered flat betting.

    Picks are sorted by score (descending). Top NUM_PICKS//2 get TIER1_BET per round;
    the remaining picks get TIER2_BET per round. Total budget equals standard flat
    (TIER1_BET * 4 + TIER2_BET * 4 == INITIAL_BET_DOLLARS * 8 == $200).

    Returns total units won (using each pick's own stake as denominator).
    """
    teams = load_eligible_teams(conn, season)
    if not teams:
        return 0.0

    picks = select_picks(teams, weights, NUM_PICKS)
    if not picks:
        return 0.0

    pick_ids = {p["team_id"] for p in picks}
    other_ids_map = {p["team_id"]: pick_ids - {p["team_id"]} for p in picks}

    n_tier1 = NUM_PICKS // 2  # top 4

    total_units = 0.0
    for i, pick in enumerate(picks):  # picks already ordered by score desc (pick_rank)
        tier_bet = TIER1_BET if i < n_tier1 else TIER2_BET
        payout, _ = simulate_pick_payout(
            conn,
            pick["team_id"],
            season,
            initial_bet=tier_bet,
            cash_out_round=cash_out_round,
            other_pick_ids=other_ids_map[pick["team_id"]],
            bet_style=BET_STYLE_FLAT,
        )
        total_units += (payout - tier_bet) / 100.0

    return total_units


def print_tiered_comparison_report(
    conn: sqlite3.Connection,
    weights: np.ndarray,
    seasons: list[int] | None = None,
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    label: str | None = None,
):
    """
    Side-by-side comparison of standard flat ($25/pick) vs tiered conviction
    ($37.50 top-4, $12.50 bottom-4) for every season in `seasons`.

    Both strategies use the same total budget ($200/yr), so net units and ROI
    per pick are directly comparable.
    """
    if seasons is None:
        seasons = ALL_SEASONS

    co = _co_label(cash_out_round)
    header = label or f"TIERED vs FLAT COMPARISON ({co} cash-out)"

    print("\n" + "=" * 72)
    print(header)
    print(f"  Standard flat: ${INITIAL_BET_DOLLARS:.2f}/pick x {NUM_PICKS} = "
          f"${INITIAL_BET_DOLLARS * NUM_PICKS:.2f}/yr")
    print(f"  Tiered:        ${TIER1_BET:.2f} top-{NUM_PICKS//2} + "
          f"${TIER2_BET:.2f} bottom-{NUM_PICKS//2} = "
          f"${TIER1_BET*(NUM_PICKS//2) + TIER2_BET*(NUM_PICKS//2):.2f}/yr")
    print("=" * 72)
    print(f"  {'Season':<10} {'Flat':>10} {'Tiered':>10} {'Delta':>10}  Marker")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}  ------")

    flat_total   = 0.0
    tiered_total = 0.0
    seasons_with_data = 0

    for season in sorted(seasons):
        teams = load_eligible_teams(conn, season)
        if not teams:
            print(f"  {season:<10} {'no data':>10}")
            continue

        picks = select_picks(teams, weights, NUM_PICKS)
        if not picks:
            print(f"  {season:<10} {'no picks':>10}")
            continue

        pick_ids = {p["team_id"] for p in picks}
        other_ids_map = {p["team_id"]: pick_ids - {p["team_id"]} for p in picks}

        n_tier1 = NUM_PICKS // 2
        flat_units   = 0.0
        tiered_units = 0.0

        for i, pick in enumerate(picks):
            # Standard flat
            payout_flat, _ = simulate_pick_payout(
                conn, pick["team_id"], season,
                initial_bet=INITIAL_BET_DOLLARS,
                cash_out_round=cash_out_round,
                other_pick_ids=other_ids_map[pick["team_id"]],
                bet_style=BET_STYLE_FLAT,
            )
            flat_units += (payout_flat - INITIAL_BET_DOLLARS) / 100.0

            # Tiered
            tier_bet = TIER1_BET if i < n_tier1 else TIER2_BET
            payout_tiered, _ = simulate_pick_payout(
                conn, pick["team_id"], season,
                initial_bet=tier_bet,
                cash_out_round=cash_out_round,
                other_pick_ids=other_ids_map[pick["team_id"]],
                bet_style=BET_STYLE_FLAT,
            )
            tiered_units += (payout_tiered - tier_bet) / 100.0

        flat_total   += flat_units
        tiered_total += tiered_units
        seasons_with_data += 1
        delta = tiered_units - flat_units

        marker = ""
        if season in VAL_SEASONS:
            marker = "[VAL]"
        elif season in TEST_SEASONS:
            marker = "[TEST]"

        print(f"  {season:<10} {flat_units:>+10.3f}u {tiered_units:>+10.3f}u "
              f"{delta:>+10.3f}u  {marker}")

    n = max(NUM_PICKS, 1)
    flat_roi   = flat_total   / (seasons_with_data * n) if seasons_with_data else 0.0
    tiered_roi = tiered_total / (seasons_with_data * n) if seasons_with_data else 0.0
    delta_total = tiered_total - flat_total

    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':<10} {flat_total:>+10.3f}u {tiered_total:>+10.3f}u "
          f"{delta_total:>+10.3f}u")
    print(f"  {'ROI/pick':<10} {flat_roi:>+10.4f}  {tiered_roi:>+10.4f}")
    print("=" * 72)

# ---------------------------------------------------------------------------
# Overlap-tiered strategy
# ---------------------------------------------------------------------------

OVERLAP_RATIO = 3.0   # overlap picks bet 3x the stake of solo picks


def overlap_bet_sizes(
    n_overlap: int,
    budget: float = INITIAL_BET_DOLLARS * NUM_PICKS,
    ratio: float = OVERLAP_RATIO,
) -> tuple[float, float]:
    """
    Return (overlap_bet, solo_bet) for the budget-neutral overlap strategy.

    Maintains `ratio`:1 weighting between picks flagged by both models vs
    fixed-8 only, while keeping total stake = budget regardless of n_overlap.

    Derivation: n*ratio*Y + (NUM_PICKS-n)*Y = budget
                Y = budget / (n*(ratio-1) + NUM_PICKS)

    With ratio=3, budget=$200:
      n=4 -> $37.50 / $12.50   (same amounts as tiered, different assignment)
      n=3 -> $42.86 / $14.29
      n=2 -> $50.00 / $16.67
      n=1 -> $60.00 / $20.00
      n=0 -> $25.00 / $25.00   (reduces to flat)
    """
    if n_overlap == 0:
        flat = budget / NUM_PICKS
        return flat, flat
    solo = budget / (n_overlap * (ratio - 1) + NUM_PICKS)
    return ratio * solo, solo


def simulate_season_overlap(
    conn: sqlite3.Connection,
    season: int,
    w_fixed: np.ndarray,
    w_var: np.ndarray,
    threshold: float,
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    min_picks: int = MIN_PICKS_PER_YEAR,
    max_picks: int = MAX_PICKS_PER_YEAR,
) -> float:
    """
    Simulate one season using the overlap-tiered strategy.

    Takes the fixed-8 pool and weights by cross-model agreement:
      - Picks also flagged by variable-N V2: overlap_bet per round
      - Picks only in fixed-8 V2:            solo_bet per round
    Total stake always = $200 (budget-neutral, 3:1 ratio).
    """
    teams = load_eligible_teams(conn, season)
    if not teams:
        return 0.0

    picks_f = select_picks(teams, w_fixed, NUM_PICKS)
    picks_v = select_picks_threshold(teams, w_var, threshold, min_picks, max_picks)

    var_ids   = {p["team_id"] for p in picks_v}
    fids      = {p["team_id"] for p in picks_f}
    n_ol      = len(fids & var_ids)
    ol_bet, solo_bet = overlap_bet_sizes(n_ol)
    other_map = {p["team_id"]: fids - {p["team_id"]} for p in picks_f}

    total_units = 0.0
    for pick in picks_f:
        tid = pick["team_id"]
        bet = ol_bet if tid in var_ids else solo_bet
        payout, _ = simulate_pick_payout(
            conn, tid, season,
            initial_bet=bet,
            cash_out_round=cash_out_round,
            other_pick_ids=other_map[tid],
            bet_style=BET_STYLE_FLAT,
        )
        total_units += (payout - bet) / 100.0

    return total_units


def print_overlap_report(
    conn: sqlite3.Connection,
    w_fixed: np.ndarray,
    w_var: np.ndarray,
    threshold: float,
    seasons: list[int] | None = None,
    cash_out_round: int = DEFAULT_CASH_OUT_ROUND,
    label: str | None = None,
    min_picks: int = MIN_PICKS_PER_YEAR,
    max_picks: int = MAX_PICKS_PER_YEAR,
):
    """
    Side-by-side comparison of flat ($25 all 8) vs overlap-tiered (3:1, $200 total).

    Overlap rule: picks flagged by both fixed-8 V2 and variable-N V2 get 3x the
    stake of fixed-8-only picks. Total always $200/yr (budget-neutral).
    """
    if seasons is None:
        seasons = ALL_SEASONS

    co     = _co_label(cash_out_round)
    header = label or f"OVERLAP-TIERED vs FLAT  (fixed-8 V2 x variable-N V2, {co} cash-out)"
    budget = INITIAL_BET_DOLLARS * NUM_PICKS

    print("\n" + "=" * 72)
    print(header)
    print(f"  Standard flat:   ${INITIAL_BET_DOLLARS:.2f}/pick x {NUM_PICKS} = ${budget:.2f}/yr")
    print(f"  Overlap-tiered:  3:1 ratio (both models vs fixed-8 only), ${budget:.2f}/yr")
    print(
        f"  e.g. 4OL->${overlap_bet_sizes(4)[0]:.2f}/${overlap_bet_sizes(4)[1]:.2f}  "
        f"3OL->${overlap_bet_sizes(3)[0]:.2f}/${overlap_bet_sizes(3)[1]:.2f}  "
        f"2OL->${overlap_bet_sizes(2)[0]:.2f}/${overlap_bet_sizes(2)[1]:.2f}  "
        f"1OL->${overlap_bet_sizes(1)[0]:.2f}/${overlap_bet_sizes(1)[1]:.2f}"
    )
    print("=" * 72)
    print(f"  {'Season':<10} {'Flat':>10} {'Overlap':>10} {'Delta':>10}  #OL  Marker")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}  ---  ------")

    flat_total    = 0.0
    overlap_total = 0.0
    seasons_with_data = 0

    for season in sorted(seasons):
        teams = load_eligible_teams(conn, season)
        if not teams:
            print(f"  {season:<10} {'no data':>10}")
            continue

        picks_f = select_picks(teams, w_fixed, NUM_PICKS)
        picks_v = select_picks_threshold(teams, w_var, threshold, min_picks, max_picks)

        var_ids   = {p["team_id"] for p in picks_v}
        fids      = {p["team_id"] for p in picks_f}
        n_ol      = len(fids & var_ids)
        ol_bet, solo_bet = overlap_bet_sizes(n_ol)
        other_map = {p["team_id"]: fids - {p["team_id"]} for p in picks_f}

        flat_units    = 0.0
        overlap_units = 0.0

        for pick in picks_f:
            tid = pick["team_id"]
            o   = other_map[tid]

            pf, _ = simulate_pick_payout(
                conn, tid, season,
                initial_bet=INITIAL_BET_DOLLARS,
                cash_out_round=cash_out_round,
                other_pick_ids=o,
                bet_style=BET_STYLE_FLAT,
            )
            flat_units += (pf - INITIAL_BET_DOLLARS) / 100.0

            bet = ol_bet if tid in var_ids else solo_bet
            po, _ = simulate_pick_payout(
                conn, tid, season,
                initial_bet=bet,
                cash_out_round=cash_out_round,
                other_pick_ids=o,
                bet_style=BET_STYLE_FLAT,
            )
            overlap_units += (po - bet) / 100.0

        flat_total    += flat_units
        overlap_total += overlap_units
        seasons_with_data += 1
        delta = overlap_units - flat_units

        marker = ""
        if season in VAL_SEASONS:    marker = "[VAL]"
        elif season in TEST_SEASONS: marker = "[TEST]"

        print(
            f"  {season:<10} {flat_units:>+10.3f}u {overlap_units:>+10.3f}u "
            f"{delta:>+10.3f}u  {n_ol:>3}  {marker}"
        )

    n = max(NUM_PICKS, 1)
    flat_roi    = flat_total    / (seasons_with_data * n) if seasons_with_data else 0.0
    overlap_roi = overlap_total / (seasons_with_data * n) if seasons_with_data else 0.0
    delta_total = overlap_total - flat_total

    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':<10} {flat_total:>+10.3f}u {overlap_total:>+10.3f}u {delta_total:>+10.3f}u")
    print(f"  {'ROI/pick':<10} {flat_roi:>+10.4f}  {overlap_roi:>+10.4f}")
    print("=" * 72)
