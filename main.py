"""
March Madness betting model CLI.

Usage:
    python main.py backfill [--season Y]              # load historical data (default: all missing)
    python main.py picks [--season Y]                 # generate 8 picks (defaults to 2026)
    python main.py track [--season Y]                 # update rolling payouts during tournament
    python main.py train [--bet-style flat|rollover]  # optimize weights (default: flat/E8)
                         [--cash-out s16|e8|f4]
                         [--method M]
    python main.py train --compare                    # train all 6 variants (2 styles x 3 cash-outs)
    python main.py report [--bet-style flat|rollover] [--cash-out e8|s16|f4]
    python main.py report --test                      # run on holdout test season (2025)
    python main.py report --tiered                    # tiered vs flat conviction comparison
    python main.py report --tiered --test             # tiered comparison on holdout test (2025)
    python main.py report --overlap                   # overlap-tiered vs flat (PRIMARY strategy)
    python main.py report --overlap --test            # overlap-tiered on holdout test (2025)

    # Variable-N model (threshold-based selection, ROI objective):
    python main.py train --variable
    python main.py report --variable
    python main.py report --variable --test

Bet styles:
    flat     — fixed $25 per round; profits locked in each round independently (default)
    rollover — $25 initial stake compounds forward through wins (parlay-style)

Cash-out rounds:
    s16 — cash out after winning R32, enter E8    (2 rounds max)
    e8  — cash out after winning S16, enter F4    (3 rounds max)  [default]
    f4  — cash out after winning E8,  enter Champ (4 rounds max)
"""
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CURRENT_SEASON, ALL_SEASONS, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    DB_PATH,
    CASH_OUT_ROUND_S16, CASH_OUT_ROUND_E8, CASH_OUT_ROUND_F4, DEFAULT_CASH_OUT_ROUND,
    BET_STYLE_FLAT, BET_STYLE_ROLLOVER, DEFAULT_BET_STYLE,
    MIN_PICKS_PER_YEAR, MAX_PICKS_PER_YEAR, DEFAULT_THRESHOLD,
)


def _parse_cash_out(val: str) -> int:
    v = val.lower()
    if v in ("e8", "elite8", "elite-eight", "8"):
        return CASH_OUT_ROUND_E8
    if v in ("s16", "sweet16", "sweet-sixteen", "16"):
        return CASH_OUT_ROUND_S16
    if v in ("f4", "final4", "final-four", "4"):
        return CASH_OUT_ROUND_F4
    try:
        return int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Unknown cash-out value: {val!r}. Use 's16', 'e8', or 'f4'.")


def _parse_bet_style(val: str) -> str:
    v = val.lower()
    if v in (BET_STYLE_FLAT, "f"):
        return BET_STYLE_FLAT
    if v in (BET_STYLE_ROLLOVER, "roll", "r"):
        return BET_STYLE_ROLLOVER
    raise argparse.ArgumentTypeError(f"Unknown bet style: {val!r}. Use 'flat' or 'rollover'.")


def cmd_backfill(args):
    from jobs.historical_backfill import run
    if args.season:
        run(seasons=[args.season])
    else:
        from shared.sqlite_helpers import get_db
        from db.db import init_db
        init_db()
        with get_db(DB_PATH) as conn:
            existing = {r["season"] for r in
                        conn.execute("SELECT DISTINCT season FROM mm_games").fetchall()}
        missing = [s for s in ALL_SEASONS if s not in existing]
        if not missing:
            print("All seasons already loaded.")
            return
        print(f"Loading {len(missing)} seasons: {missing}")
        run(seasons=sorted(missing))


def cmd_picks(args):
    from jobs.model_job import run
    run(season=args.season or CURRENT_SEASON, fetch_data=not args.no_fetch)


def cmd_track(args):
    from jobs.tournament_tracker import run
    run(season=args.season or CURRENT_SEASON)


def cmd_train(args):
    from shared.sqlite_helpers import get_db
    from db.db import init_db

    init_db()

    _co_label = {CASH_OUT_ROUND_S16: "S16", CASH_OUT_ROUND_E8: "E8", CASH_OUT_ROUND_F4: "F4"}

    # ---- V6-geomean fixed-weight model (no optimization) --------------------
    if getattr(args, 'v6_geomean', False):
        from processors.model import (
            simulate_season_v6, save_weights_v6_geomean,
            print_validation_report_v6, print_tiered_comparison_report_v6,
        )
        from processors.features import V6_GEOMEAN_W, FEATURE_NAMES_V6B
        co = _co_label.get(args.cash_out, f"R{args.cash_out}")
        print(f"\n{'='*65}")
        print(f"REGISTERING v6-fixed-8-geomean -- FLAT / {co} cash-out")
        print(f"Features: seed_rank_gap, conf_tourney_wins, dfi, tsi")
        print(f"Weights (geomean of SFM val+test, no optimization):")
        for name, w in zip(FEATURE_NAMES_V6B, V6_GEOMEAN_W):
            print(f"  {name:<25s} {w:+.4f}")
        print(f"{'='*65}")

        all_seasons = TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS
        with get_db(DB_PATH) as conn:
            train_units = sum(
                simulate_season_v6(conn, s, V6_GEOMEAN_W, "coreB",
                                   cash_out_round=args.cash_out, bet_style=args.bet_style)
                for s in TRAIN_SEASONS
            )
            val_units = sum(
                simulate_season_v6(conn, s, V6_GEOMEAN_W, "coreB",
                                   cash_out_round=args.cash_out, bet_style=args.bet_style)
                for s in VAL_SEASONS
            )
            test_units = sum(
                simulate_season_v6(conn, s, V6_GEOMEAN_W, "coreB",
                                   cash_out_round=args.cash_out, bet_style=args.bet_style)
                for s in TEST_SEASONS
            )
            wid = save_weights_v6_geomean(
                conn, train_units, val_units, test_units,
                cash_out_round=args.cash_out, bet_style=args.bet_style,
            )
            print(f"Weights saved (id={wid})  "
                  f"train={train_units:+.2f}u  val={val_units:+.2f}u  test={test_units:+.2f}u")

            print_validation_report_v6(
                conn, V6_GEOMEAN_W, "coreB",
                seasons=all_seasons,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
                label="v6-fixed-8-geomean YEAR-OVER-YEAR (all 11 seasons)",
            )
            print_tiered_comparison_report_v6(
                conn, V6_GEOMEAN_W, "coreB",
                seasons=all_seasons,
                cash_out_round=args.cash_out,
                label="v6-fixed-8-geomean TIERED vs FLAT (all 11 seasons)",
            )
        return

    # ---- V6 hybrid models (coreA / coreB) -----------------------------------
    if args.v6a or args.v6b:
        from processors.model import (
            train_weights_v6, save_weights_v6, print_validation_report_v6,
        )
        variant = "coreA" if args.v6a else "coreB"
        lambda_l2 = args.l2_lambda if args.l2_lambda is not None else 0.1
        features_desc = (
            "seed_rank_gap, conf_tourney_wins, opp_efg_pct, tsi" if args.v6a
            else "seed_rank_gap, conf_tourney_wins, dfi, tsi"
        )
        co = _co_label.get(args.cash_out, f"R{args.cash_out}")
        print(f"\n{'='*65}")
        print(f"TRAINING V6-{variant} FIXED-8 -- {args.bet_style.upper()} / {co} cash-out")
        print(f"Train: {min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)} ({len(TRAIN_SEASONS)} seasons)"
              f"   Val: {VAL_SEASONS}")
        print(f"Features: {features_desc}")
        print(f"Bounds:   constrained (seed_rank_gap<0, all others>0)  lambda={lambda_l2}")
        print(f"{'='*65}")

        with get_db(DB_PATH) as conn:
            result = train_weights_v6(
                conn, TRAIN_SEASONS,
                variant=variant,
                val_seasons=VAL_SEASONS,
                test_seasons=TEST_SEASONS,
                method=args.method,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
                lambda_l2=lambda_l2,
            )
            wid = save_weights_v6(conn, result)

            val_str  = f"{result['val_units']:+.2f}u"  if result["val_units"]  is not None else "n/a"
            test_str = f"{result['test_units']:+.2f}u" if result["test_units"] is not None else "n/a"
            print(f"Weights saved (id={wid})  "
                  f"train={result['train_units']:+.2f}u  val={val_str}  test={test_str}")

            for name, w in zip(result["feature_names"], result["weights"]):
                print(f"  {name:<25s} {w:+.4f}")

            print_validation_report_v6(
                conn, result["weights"], variant,
                seasons=TRAIN_SEASONS + VAL_SEASONS,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
            )
        return

    # ---- V5 fixed-8 model ---------------------------------------------------
    if args.v5:
        from processors.model import (
            train_weights_v5, save_weights_v5, print_validation_report_v5,
        )
        from config import L2_LAMBDA
        lambda_l2 = args.l2_lambda if args.l2_lambda is not None else L2_LAMBDA
        co = _co_label.get(args.cash_out, f"R{args.cash_out}")
        print(f"\n{'='*65}")
        print(f"TRAINING V5 FIXED-8 — {args.bet_style.upper()} / {co} cash-out")
        print(f"Train: {min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)} ({len(TRAIN_SEASONS)} seasons)"
              f"   Val: {VAL_SEASONS}")
        print(f"Features: seed_rank_gap, net_rating, conf_tourney_wins, cpi, dfi, ftli, spmi, tsi")
        print(f"Bounds:   constrained (seed_rank_gap<0, all others>0)  lambda={lambda_l2}")
        print(f"{'='*65}")

        with get_db(DB_PATH) as conn:
            result = train_weights_v5(
                conn, TRAIN_SEASONS,
                val_seasons=VAL_SEASONS,
                test_seasons=TEST_SEASONS,
                method=args.method,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
                lambda_l2=lambda_l2,
            )
            wid = save_weights_v5(conn, result)

            val_str  = f"{result['val_units']:+.2f}u"  if result["val_units"]  is not None else "n/a"
            test_str = f"{result['test_units']:+.2f}u" if result["test_units"] is not None else "n/a"
            print(f"Weights saved (id={wid})  "
                  f"train={result['train_units']:+.2f}u  val={val_str}  test={test_str}")

            from processors.features import FEATURE_NAMES_V5
            for name, w in zip(FEATURE_NAMES_V5, result["weights"]):
                print(f"  {name:<25s} {w:+.4f}")

            print_validation_report_v5(
                conn, result["weights"],
                seasons=TRAIN_SEASONS + VAL_SEASONS,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
            )
        return

    # ---- Variable-N model ---------------------------------------------------
    if args.variable:
        from processors.model import (
            train_weights_variable, save_weights_variable,
            print_validation_report_variable,
        )
        co = _co_label.get(args.cash_out, f"R{args.cash_out}")
        print(f"\n{'='*70}")
        print(f"TRAINING VARIABLE-N — {args.bet_style.upper()} / {co} cash-out")
        print(f"Train: {min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)} ({len(TRAIN_SEASONS)} seasons)"
              f"   Val: {VAL_SEASONS}")
        print(f"Min picks/yr: {MIN_PICKS_PER_YEAR}   Max picks/yr: {MAX_PICKS_PER_YEAR}")
        print(f"{'='*70}")

        with get_db(DB_PATH) as conn:
            result = train_weights_variable(
                conn, TRAIN_SEASONS,
                val_seasons=VAL_SEASONS,
                test_seasons=TEST_SEASONS,
                method=args.method,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
            )
            wid = save_weights_variable(conn, result)

            def _fmt(u, n, roi):
                if u is None:
                    return "n/a"
                return f"{u:+.2f}u/{n}picks (ROI {roi:+.4f})"

            print(f"Weights saved (id={wid})")
            print(f"  threshold = {result['threshold']:+.4f}")
            print(f"  train = {_fmt(result['train_units'], result['train_picks'], result['train_roi'])}")
            print(f"  val   = {_fmt(result['val_units'],   result['val_picks'],   result['val_roi'])}")
            print(f"  test  = {_fmt(result['test_units'],  result['test_picks'],  result['test_roi'])}")

            print_validation_report_variable(
                conn, result["weights"], result["threshold"],
                seasons=TRAIN_SEASONS + VAL_SEASONS,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
            )
        return

    # ---- Fixed-8 model ------------------------------------------------------
    from processors.model import train_weights, save_weights, print_validation_report

    # Build list of (bet_style, cash_out_round) combos to train
    if args.compare:
        combos = [
            (bs, co)
            for bs in (BET_STYLE_FLAT, BET_STYLE_ROLLOVER)
            for co in (CASH_OUT_ROUND_S16, CASH_OUT_ROUND_E8, CASH_OUT_ROUND_F4)
        ]
    else:
        combos = [(args.bet_style, args.cash_out)]

    results = []

    with get_db(DB_PATH) as conn:
        for bet_style, co_round in combos:
            co = _co_label.get(co_round, f"R{co_round}")
            print(f"\n{'='*65}")
            print(f"TRAINING — {bet_style.upper()} / {co} cash-out")
            print(f"Train: {min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)} "
                  f"({len(TRAIN_SEASONS)} seasons)   Val: {VAL_SEASONS}")
            print(f"{'='*65}")

            result = train_weights(
                conn,
                TRAIN_SEASONS,
                val_seasons=VAL_SEASONS,
                test_seasons=TEST_SEASONS,
                method=args.method,
                cash_out_round=co_round,
                bet_style=bet_style,
            )
            wid = save_weights(conn, result)
            val_str  = f"{result['val_units']:+.2f}u"  if result["val_units"]  is not None else "n/a"
            test_str = f"{result['test_units']:+.2f}u" if result["test_units"] is not None else "n/a"
            print(f"Weights saved (id={wid})  "
                  f"train={result['train_units']:+.2f}u  "
                  f"val={val_str}  test={test_str}")

            if not args.compare:
                print_validation_report(
                    conn, result["weights"],
                    seasons=TRAIN_SEASONS + VAL_SEASONS,
                    cash_out_round=co_round,
                    bet_style=bet_style,
                )

            results.append((bet_style, co, result))

    # Side-by-side summary for --compare
    if args.compare:
        print(f"\n{'='*65}")
        print("COMPARISON SUMMARY")
        print(f"Train: {min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)}  "
              f"Val: {VAL_SEASONS}  Test: {TEST_SEASONS}")
        print(f"{'='*65}")
        print(f"{'Strategy':<22} {'Train':>8} {'Val':>8} {'Test':>8}")
        print(f"{'-'*22} {'-'*8} {'-'*8} {'-'*8}")
        for bet_style, co, result in results:
            label = f"{bet_style}/{co}"
            val_str  = f"{result['val_units']:>+8.2f}u"  if result["val_units"]  is not None else "     n/a"
            test_str = f"{result['test_units']:>+8.2f}u" if result["test_units"] is not None else "     n/a"
            print(f"{label:<22} {result['train_units']:>+8.2f}u {val_str} {test_str}")
        print("=" * 65)


def cmd_report(args):
    from shared.sqlite_helpers import get_db
    from db.db import init_db

    init_db()

    # ---- Variable-N model ---------------------------------------------------
    if args.variable:
        from processors.model import (
            load_latest_weights_variable, print_validation_report_variable,
        )
        with get_db(DB_PATH) as conn:
            result = load_latest_weights_variable(
                conn,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
            )
            if result is None:
                print(f"No variable-N weights found for bet_style={args.bet_style}, "
                      f"cash_out={args.cash_out}. Run: python main.py train --variable")
                sys.exit(1)

            weights, threshold = result
            seasons = TEST_SEASONS if args.test else TRAIN_SEASONS + VAL_SEASONS
            lbl = f"VARIABLE-N HOLDOUT TEST ({TEST_SEASONS})" if args.test else None

            print_validation_report_variable(
                conn, weights, threshold,
                seasons=seasons,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
                label=lbl,
            )
        return

    # ---- V6-geomean fixed-weight model --------------------------------------
    if getattr(args, 'v6_geomean', False):
        from processors.model import (
            load_latest_weights_v6_geomean, print_validation_report_v6,
            print_tiered_comparison_report_v6,
        )
        with get_db(DB_PATH) as conn:
            weights = load_latest_weights_v6_geomean(
                conn, cash_out_round=args.cash_out, bet_style=args.bet_style,
            )
            if weights is None:
                print("No v6-geomean weights found. Run: python main.py train --v6-geomean")
                sys.exit(1)
            seasons = TEST_SEASONS if args.test else TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS
            lbl = f"v6-fixed-8-geomean HOLDOUT TEST ({TEST_SEASONS})" if args.test else None
            print_validation_report_v6(
                conn, weights, "coreB",
                seasons=seasons,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
                label=lbl,
            )
            if args.tiered:
                print_tiered_comparison_report_v6(
                    conn, weights, "coreB",
                    seasons=seasons,
                    cash_out_round=args.cash_out,
                )
        return

    # ---- V6 hybrid models ---------------------------------------------------
    if args.v6a or args.v6b:
        from processors.model import load_latest_weights_v6, print_validation_report_v6
        variant = "coreA" if args.v6a else "coreB"
        flag = "--v6a" if args.v6a else "--v6b"
        with get_db(DB_PATH) as conn:
            weights = load_latest_weights_v6(
                conn, variant, cash_out_round=args.cash_out, bet_style=args.bet_style,
            )
            if weights is None:
                print(f"No V6-{variant} weights found. Run: python main.py train {flag}")
                sys.exit(1)
            seasons = TEST_SEASONS if args.test else TRAIN_SEASONS + VAL_SEASONS
            lbl = f"V6-{variant} HOLDOUT TEST ({TEST_SEASONS})" if args.test else None
            print_validation_report_v6(
                conn, weights, variant,
                seasons=seasons,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
                label=lbl,
            )
        return

    # ---- V5 fixed-8 model ---------------------------------------------------
    if args.v5:
        from processors.model import load_latest_weights_v5, print_validation_report_v5
        with get_db(DB_PATH) as conn:
            weights = load_latest_weights_v5(
                conn, cash_out_round=args.cash_out, bet_style=args.bet_style,
            )
            if weights is None:
                print(f"No V5 weights found for bet_style={args.bet_style}, "
                      f"cash_out={args.cash_out}. Run: python main.py train --v5")
                sys.exit(1)
            seasons = TEST_SEASONS if args.test else TRAIN_SEASONS + VAL_SEASONS
            lbl = f"V5 HOLDOUT TEST ({TEST_SEASONS})" if args.test else None
            print_validation_report_v5(
                conn, weights,
                seasons=seasons,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
                label=lbl,
            )
        return

    # ---- Fixed-8 model ------------------------------------------------------
    from processors.model import (
        load_latest_weights, print_validation_report, print_tiered_comparison_report,
    )

    with get_db(DB_PATH) as conn:
        weights = load_latest_weights(
            conn,
            cash_out_round=args.cash_out,
            bet_style=args.bet_style,
        )
        if weights is None:
            print(f"No trained weights found for bet_style={args.bet_style}, "
                  f"cash_out={args.cash_out}. Run: python main.py train")
            sys.exit(1)

        if args.overlap:
            from processors.model import load_latest_weights_variable, print_overlap_report
            var_result = load_latest_weights_variable(
                conn, cash_out_round=args.cash_out, bet_style=args.bet_style,
            )
            if var_result is None:
                print("No variable-N weights found. Run: python main.py train --variable")
                sys.exit(1)
            w_var, threshold = var_result
            seasons = TEST_SEASONS if args.test else TRAIN_SEASONS + VAL_SEASONS
            lbl = f"OVERLAP-TIERED vs FLAT — HOLDOUT TEST ({TEST_SEASONS})" if args.test else None
            print_overlap_report(
                conn, weights, w_var, threshold,
                seasons=seasons,
                cash_out_round=args.cash_out,
                label=lbl,
            )
            return

        if args.tiered:
            # Tiered vs flat comparison
            seasons = TEST_SEASONS if args.test else TRAIN_SEASONS + VAL_SEASONS
            lbl = f"TIERED vs FLAT — HOLDOUT TEST ({TEST_SEASONS})" if args.test else None
            print_tiered_comparison_report(
                conn, weights,
                seasons=seasons,
                cash_out_round=args.cash_out,
                label=lbl,
            )
            return

        if args.test:
            print_validation_report(
                conn, weights,
                seasons=TEST_SEASONS,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
                label=f"HOLDOUT TEST REPORT ({TEST_SEASONS})",
            )
        else:
            print_validation_report(
                conn, weights,
                seasons=TRAIN_SEASONS + VAL_SEASONS,
                cash_out_round=args.cash_out,
                bet_style=args.bet_style,
            )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(
        description="March Madness Betting Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # backfill
    p_backfill = sub.add_parser("backfill", help="Load historical data (default: all missing seasons)")
    p_backfill.add_argument("--season", type=int, help="Load only this specific season")
    p_backfill.set_defaults(func=cmd_backfill)

    # picks
    p_picks = sub.add_parser("picks", help="Generate 8 pre-tournament picks")
    p_picks.add_argument("--season", type=int, help=f"Season year (default: {CURRENT_SEASON})")
    p_picks.add_argument("--no-fetch", action="store_true",
                         help="Skip CBBD data fetch, use existing DB data")
    p_picks.set_defaults(func=cmd_picks)

    # track
    p_track = sub.add_parser("track", help="Update rolling payouts during tournament")
    p_track.add_argument("--season", type=int, help=f"Season year (default: {CURRENT_SEASON})")
    p_track.set_defaults(func=cmd_track)

    # train
    p_train = sub.add_parser("train", help="Optimize feature weights on historical data")
    p_train.add_argument("--method", default="differential_evolution",
                         choices=["differential_evolution", "nelder-mead"])
    p_train.add_argument("--bet-style", type=_parse_bet_style, default=DEFAULT_BET_STYLE,
                         dest="bet_style", metavar="flat|rollover",
                         help=f"Bet style: 'flat' (default) or 'rollover'")
    p_train.add_argument("--cash-out", type=_parse_cash_out, default=DEFAULT_CASH_OUT_ROUND,
                         metavar="s16|e8|f4",
                         help="Cash-out round: 's16', 'e8' (default), or 'f4'")
    p_train.add_argument("--compare", action="store_true",
                         help="Train all 6 fixed-8 variants (flat+rollover x s16+e8+f4) and compare")
    p_train.add_argument("--variable", action="store_true",
                         help="Use variable-N threshold model (ROI objective, z-score selection)")
    p_train.add_argument("--v5", action="store_true",
                         help="Train V5 composite model (cpi, dfi, ftli, spmi, tsi) with constrained bounds")
    p_train.add_argument("--v6a", action="store_true",
                         help="Train V6-coreA hybrid: seed_rank_gap, conf_tourney_wins, opp_efg_pct, tsi")
    p_train.add_argument("--v6b", action="store_true",
                         help="Train V6-coreB hybrid: seed_rank_gap, conf_tourney_wins, dfi, tsi")
    p_train.add_argument("--v6-geomean", action="store_true", dest="v6_geomean",
                         help="Register v6-fixed-8-geomean model (fixed geomean weights, no optimization)")
    p_train.add_argument("--l2-lambda", type=float, default=None, dest="l2_lambda",
                         metavar="LAMBDA",
                         help="L2 regularization lambda override (V5 default: 0.01; V6 default: 0.1)")
    p_train.set_defaults(func=cmd_train)

    # report
    p_report = sub.add_parser("report", help="Print historical units won summary")
    p_report.add_argument("--bet-style", type=_parse_bet_style, default=DEFAULT_BET_STYLE,
                          dest="bet_style", metavar="flat|rollover")
    p_report.add_argument("--cash-out", type=_parse_cash_out, default=DEFAULT_CASH_OUT_ROUND,
                          metavar="s16|e8|f4")
    p_report.add_argument("--test", action="store_true",
                          help="Run on holdout test season (2025) instead of training seasons")
    p_report.add_argument("--variable", action="store_true",
                          help="Use variable-N threshold model")
    p_report.add_argument("--tiered", action="store_true",
                          help="Show tiered conviction vs standard flat comparison")
    p_report.add_argument("--overlap", action="store_true",
                          help="Show overlap-tiered vs flat (PRIMARY strategy)")
    p_report.add_argument("--v5", action="store_true",
                          help="Show V5 composite model report")
    p_report.add_argument("--v6a", action="store_true",
                          help="Show V6-coreA hybrid model report")
    p_report.add_argument("--v6b", action="store_true",
                          help="Show V6-coreB hybrid model report")
    p_report.add_argument("--v6-geomean", action="store_true", dest="v6_geomean",
                          help="Show v6-fixed-8-geomean model report")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
