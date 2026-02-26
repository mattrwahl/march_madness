"""
March Madness betting model CLI.

Usage:
    python main.py backfill              # one-time historical load 2021–2025
    python main.py picks [--season Y]    # generate 8 picks (defaults to 2026)
    python main.py track [--season Y]    # update rolling payouts for active picks
    python main.py train [--method M]    # optimize feature weights
    python main.py report                # print units won summary across all years
"""
import sys
import argparse
import logging
from pathlib import Path

# Ensure monorepo root is on the path so shared/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from config import CURRENT_SEASON, HISTORICAL_SEASONS, DB_PATH


def cmd_backfill(args):
    from jobs.historical_backfill import run
    seasons = [args.season] if args.season else None
    run(seasons=seasons)


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
    import numpy as np

    init_db()

    train_seasons = HISTORICAL_SEASONS[:-1]  # 2021–2024
    val_seasons   = HISTORICAL_SEASONS[-1:]  # 2025

    with get_db(DB_PATH) as conn:
        result = train_weights(
            conn,
            train_seasons,
            val_seasons,
            method=args.method,
        )
        wid = save_weights(conn, result)
        print(f"\nWeights saved to mm_model_weights (id={wid})")
        print_validation_report(conn, result["weights"])


def cmd_report(args):
    from processors.model import load_latest_weights, print_validation_report
    from shared.sqlite_helpers import get_db
    from db.db import init_db

    init_db()

    with get_db(DB_PATH) as conn:
        weights = load_latest_weights(conn)
        if weights is None:
            print("No trained weights found. Run: python main.py train")
            sys.exit(1)
        print_validation_report(conn, weights)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    parser = argparse.ArgumentParser(
        description="March Madness Betting Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # backfill
    p_backfill = sub.add_parser("backfill", help="One-time historical data load 2021–2025")
    p_backfill.add_argument("--season", type=int, help="Load only this season (default: all)")
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
    p_train.add_argument(
        "--method",
        default="differential_evolution",
        choices=["differential_evolution", "nelder-mead"],
        help="Optimization algorithm (default: differential_evolution)",
    )
    p_train.set_defaults(func=cmd_train)

    # report
    p_report = sub.add_parser("report", help="Print historical units won summary")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
