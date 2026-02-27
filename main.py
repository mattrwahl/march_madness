"""
March Madness betting model CLI.

Usage:
    python main.py backfill [--season Y]         # load historical data (default: all seasons)
    python main.py picks [--season Y]            # generate 8 picks (defaults to 2026)
    python main.py track [--season Y]            # update rolling payouts during tournament
    python main.py train [--method M]            # optimize weights, E8 cash-out (default)
    python main.py train --cash-out s16          # optimize weights, S16 cash-out
    python main.py train --compare               # train both E8 and S16, print side-by-side
    python main.py report [--cash-out e8|s16]    # print units won summary
    python main.py report --test                 # run on holdout test seasons (2005–2007)
"""
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CURRENT_SEASON, ALL_SEASONS, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS,
    DB_PATH, CASH_OUT_ROUND_E8, CASH_OUT_ROUND_S16, DEFAULT_CASH_OUT_ROUND,
)


def _parse_cash_out(val: str) -> int:
    v = val.lower()
    if v in ("e8", "elite8", "elite-eight", "8"):
        return CASH_OUT_ROUND_E8
    if v in ("s16", "sweet16", "sweet-sixteen", "16"):
        return CASH_OUT_ROUND_S16
    try:
        return int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Unknown cash-out value: {val!r}. Use 'e8' or 's16'.")


def cmd_backfill(args):
    from jobs.historical_backfill import run
    if args.season:
        run(seasons=[args.season])
    else:
        # Default: backfill all seasons not yet in DB
        from shared.sqlite_helpers import get_db
        from db.db import init_db
        init_db()
        with get_db(DB_PATH) as conn:
            existing = {r["season"] for r in
                        conn.execute("SELECT DISTINCT season FROM mm_games").fetchall()}
        missing = [s for s in ALL_SEASONS + TEST_SEASONS if s not in existing]
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
    from processors.model import train_weights, save_weights, print_validation_report
    from shared.sqlite_helpers import get_db
    from db.db import init_db

    init_db()

    cash_out_rounds = (
        [CASH_OUT_ROUND_E8, CASH_OUT_ROUND_S16] if args.compare
        else [args.cash_out]
    )

    with get_db(DB_PATH) as conn:
        for co_round in cash_out_rounds:
            co_label = "E8" if co_round == CASH_OUT_ROUND_E8 else "S16"
            print(f"\n{'='*65}")
            print(f"TRAINING — {co_label} cash-out")
            print(f"Train: {min(TRAIN_SEASONS)}–{max(TRAIN_SEASONS)} "
                  f"({len(TRAIN_SEASONS)} seasons)   Val: {VAL_SEASONS}")
            print(f"{'='*65}")

            result = train_weights(
                conn,
                TRAIN_SEASONS,
                val_seasons=VAL_SEASONS,
                test_seasons=TEST_SEASONS,
                method=args.method,
                cash_out_round=co_round,
            )
            wid = save_weights(conn, result)
            print(f"Weights saved (id={wid})  "
                  f"train={result['train_units']:+.2f}u  "
                  f"val={result['val_units']:+.2f}u  "
                  f"test={result['test_units']:+.2f}u")
            print_validation_report(
                conn, result["weights"],
                seasons=TRAIN_SEASONS + VAL_SEASONS,
                cash_out_round=co_round,
            )


def cmd_report(args):
    from processors.model import load_latest_weights, print_validation_report
    from shared.sqlite_helpers import get_db
    from db.db import init_db

    init_db()

    with get_db(DB_PATH) as conn:
        weights = load_latest_weights(conn, cash_out_round=args.cash_out)
        if weights is None:
            print(f"No trained weights found for cash-out={args.cash_out}. Run: python main.py train")
            sys.exit(1)

        if args.test:
            print_validation_report(
                conn, weights,
                seasons=TEST_SEASONS,
                cash_out_round=args.cash_out,
                label=f"HOLDOUT TEST REPORT (2005–2007)",
            )
        else:
            print_validation_report(
                conn, weights,
                seasons=TRAIN_SEASONS + VAL_SEASONS,
                cash_out_round=args.cash_out,
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
    p_train.add_argument("--cash-out", type=_parse_cash_out, default=DEFAULT_CASH_OUT_ROUND,
                         metavar="e8|s16",
                         help="Cash-out round: 'e8' (Elite Eight, default) or 's16' (Sweet Sixteen)")
    p_train.add_argument("--compare", action="store_true",
                         help="Train both E8 and S16 cash-out variants and compare")
    p_train.set_defaults(func=cmd_train)

    # report
    p_report = sub.add_parser("report", help="Print historical units won summary")
    p_report.add_argument("--cash-out", type=_parse_cash_out, default=DEFAULT_CASH_OUT_ROUND,
                          metavar="e8|s16")
    p_report.add_argument("--test", action="store_true",
                          help="Run on holdout test seasons (2005–2007) instead of training seasons")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
