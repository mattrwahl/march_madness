"""
Backfill synthetic betting lines for 2014-2017 (which have no CBBD lines).

Two-pass operation:
  Pass 1: Fix round encoding for 2014-2015 (CBBD stored these shifted by one).
           round=64 (First Four) -> 65
           round=32 (actual R64) -> 64
           round=NULL (actual R32) -> 32

  Pass 2: Build a lookup table of average moneylines by (team1_seed, team2_seed, round)
           from 2018-2025 consensus lines, then insert synthetic 'historical_avg'
           records for all 2014-2017 games that still have no lines.

Safe to re-run: deletes existing 'historical_avg' records before re-inserting.
"""
import sys, os, sqlite3
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "march_madness.db")
BACKFILL_SEASONS = [2014, 2015, 2016, 2017]
LOOKUP_SEASONS   = list(range(2018, 2026))   # 2018-2025 (excl. 2020 already excluded by data)


# ---------------------------------------------------------------------------
# Pass 1: Fix round encoding for 2014-2015
# ---------------------------------------------------------------------------

def fix_round_encoding(conn):
    """
    In CBBD data for 2014-2015, rounds are shifted:
      round=64  (4 games)  = First Four play-in     -> should be round=65
      round=32  (32 games) = actual R64              -> should be round=64
      round=NULL (16 games) = actual R32             -> should be round=32
      round=16, 8, 4, 2    = already correct

    2016+ have correct encoding (round=65 for First Four, round=64 for R64, etc.).
    """
    print("Pass 1: Fixing round encoding for 2014-2015...")

    # Must run in order: 65 first (so round=64 First Four games become 65,
    # freeing the 64 slot for actual R64 games that are currently at 32).
    c1 = conn.execute(
        "UPDATE mm_games SET round = 65 WHERE season IN (2014, 2015) AND round = 64"
    ).rowcount
    c2 = conn.execute(
        "UPDATE mm_games SET round = 64 WHERE season IN (2014, 2015) AND round = 32"
    ).rowcount
    c3 = conn.execute(
        "UPDATE mm_games SET round = 32 WHERE season IN (2014, 2015) AND round IS NULL"
    ).rowcount

    conn.commit()
    print(f"  First Four fixed (64->65): {c1} games")
    print(f"  R64 fixed (32->64):        {c2} games")
    print(f"  R32 fixed (NULL->32):      {c3} games")
    print()

    # Verify
    for season in (2014, 2015):
        print(f"  {season} round counts after fix:")
        for r in conn.execute(
            "SELECT round, COUNT(*) as cnt FROM mm_games WHERE season=? GROUP BY round ORDER BY round DESC",
            (season,)
        ):
            print(f"    round={r['round']}  cnt={r['cnt']}")
    print()


# ---------------------------------------------------------------------------
# Pass 2a: Build lookup table from 2018-2025 consensus lines
# ---------------------------------------------------------------------------

def build_lookup(conn):
    """
    Returns a dict: (s1, s2, round) -> (avg_team1_ml, avg_team2_ml)
    where s1/s2 are the seeds of team1/team2 as stored in mm_games.

    Averages across all seasons with consensus lines. Both (s1,s2) and
    (s2,s1) entries are stored (with moneylines swapped) for easy lookup.
    """
    rows = conn.execute(
        """
        SELECT
            te1.seed              AS s1,
            te2.seed              AS s2,
            g.round,
            COUNT(*)              AS n,
            AVG(bl.team1_moneyline) AS avg_ml1,
            AVG(bl.team2_moneyline) AS avg_ml2
        FROM mm_games g
        JOIN mm_betting_lines bl
             ON bl.game_id = g.id AND bl.provider = 'consensus'
        JOIN mm_tournament_entries te1
             ON te1.team_id = g.team1_id AND te1.season = g.season
        JOIN mm_tournament_entries te2
             ON te2.team_id = g.team2_id AND te2.season = g.season
        WHERE g.season BETWEEN 2018 AND 2025
          AND g.round >= 8 AND g.round <= 64
        GROUP BY te1.seed, te2.seed, g.round
        """
    ).fetchall()

    lookup = {}
    for r in rows:
        s1, s2, rnd = r["s1"], r["s2"], r["round"]
        ml1, ml2 = r["avg_ml1"], r["avg_ml2"]
        lookup[(s1, s2, rnd)] = (ml1, ml2)
        # Store reverse as well (swap seeds, swap moneylines)
        lookup[(s2, s1, rnd)] = (ml2, ml1)

    print(f"  Built lookup: {len(rows)} base entries "
          f"({len(lookup)} including reverse pairs)")
    return lookup


def find_ml(lookup, s1, s2, rnd):
    """
    Look up avg moneylines for a (s1, s2, round) matchup.
    Falls back to nearest-neighbor by seed distance in the same round,
    then the nearest round if same-round has nothing.
    Returns (ml1, ml2) or (None, None) if lookup is empty.
    """
    key = (s1, s2, rnd)
    if key in lookup:
        return lookup[key]

    # Nearest neighbor — same round first
    def dist(k):
        return abs(k[0] - s1) + abs(k[1] - s2)

    same_rnd = [(k, v) for k, v in lookup.items() if k[2] == rnd]
    if same_rnd:
        best_key, best_val = min(same_rnd, key=lambda x: dist(x[0]))
        return best_val

    # Fall back to any round (closest round + closest seeds)
    def overall_dist(k):
        return abs(k[0] - s1) + abs(k[1] - s2) + abs(k[2] - rnd) * 2

    if lookup:
        best_key, best_val = min(lookup.items(), key=lambda x: overall_dist(x[0]))
        return best_val

    return (None, None)


# ---------------------------------------------------------------------------
# Pass 2b: Insert synthetic lines for 2014-2017 unlined games
# ---------------------------------------------------------------------------

def backfill_lines(conn, lookup):
    """
    For each unlined game in BACKFILL_SEASONS (rounds 8-64), compute avg
    moneylines from the lookup and insert a 'historical_avg' provider record.
    """
    # Remove any previous historical_avg records for target seasons
    deleted = conn.execute(
        """
        DELETE FROM mm_betting_lines
        WHERE provider = 'historical_avg'
          AND game_id IN (
              SELECT id FROM mm_games WHERE season IN ({})
          )
        """.format(",".join("?" * len(BACKFILL_SEASONS))),
        BACKFILL_SEASONS,
    ).rowcount
    if deleted:
        print(f"  Removed {deleted} stale 'historical_avg' records.")

    # Find all unlined games in the target seasons
    unlined = conn.execute(
        """
        SELECT g.id AS game_id, g.season, g.round,
               te1.seed AS s1, te2.seed AS s2
        FROM mm_games g
        JOIN mm_tournament_entries te1
             ON te1.team_id = g.team1_id AND te1.season = g.season
        JOIN mm_tournament_entries te2
             ON te2.team_id = g.team2_id AND te2.season = g.season
        WHERE g.season IN ({})
          AND g.round >= 8 AND g.round <= 64
          AND NOT EXISTS (
              SELECT 1 FROM mm_betting_lines bl WHERE bl.game_id = g.id
          )
        ORDER BY g.season, g.round DESC, g.id
        """.format(",".join("?" * len(BACKFILL_SEASONS))),
        BACKFILL_SEASONS,
    ).fetchall()

    inserted = 0
    fallback_count = 0
    for row in unlined:
        game_id = row["game_id"]
        s1, s2, rnd = row["s1"], row["s2"], row["round"]
        ml1, ml2 = find_ml(lookup, s1, s2, rnd)

        if ml1 is None:
            print(f"  [WARN] No lookup match for game_id={game_id} "
                  f"season={row['season']} round={rnd} seeds={s1}v{s2}")
            continue

        # Round to integer (moneylines are integers)
        ml1_int = int(round(ml1))
        ml2_int = int(round(ml2))

        # Check if this was an exact match or fallback
        if (s1, s2, rnd) not in lookup:
            fallback_count += 1

        conn.execute(
            """
            INSERT INTO mm_betting_lines
                (game_id, provider, team1_moneyline, team2_moneyline)
            VALUES (?, 'historical_avg', ?, ?)
            """,
            (game_id, ml1_int, ml2_int),
        )
        inserted += 1

    conn.commit()
    print(f"  Inserted {inserted} synthetic 'historical_avg' lines "
          f"({fallback_count} used nearest-neighbor fallback).")
    print()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(conn):
    print("Verification: coverage after backfill")
    print("-" * 60)
    for r in conn.execute(
        """
        SELECT g.season,
               COUNT(DISTINCT g.id) AS total_games,
               COUNT(DISTINCT CASE WHEN bl.id IS NULL THEN g.id END) AS no_line_games,
               COUNT(DISTINCT CASE WHEN bl.provider = 'historical_avg' THEN g.id END) AS synthetic_games
        FROM mm_games g
        LEFT JOIN mm_betting_lines bl ON bl.game_id = g.id
        WHERE g.season BETWEEN 2014 AND 2025
          AND g.round >= 8 AND g.round <= 64
        GROUP BY g.season
        ORDER BY g.season
        """
    ):
        pct_covered = (1 - r["no_line_games"] / max(r["total_games"], 1)) * 100
        print(f"  {r['season']}:  total={r['total_games']:3d}  "
              f"unlined={r['no_line_games']:2d}  synthetic={r['synthetic_games']:2d}  "
              f"coverage={pct_covered:.0f}%")

    print()
    print("Sample synthetic lines (2014 R64):")
    for r in conn.execute(
        """
        SELECT g.id, g.round, te1.seed AS s1, te2.seed AS s2,
               bl.team1_moneyline, bl.team2_moneyline, bl.provider
        FROM mm_games g
        JOIN mm_betting_lines bl ON bl.game_id = g.id AND bl.provider = 'historical_avg'
        JOIN mm_tournament_entries te1 ON te1.team_id = g.team1_id AND te1.season = g.season
        JOIN mm_tournament_entries te2 ON te2.team_id = g.team2_id AND te2.season = g.season
        WHERE g.season = 2014 AND g.round = 64
        ORDER BY s1
        LIMIT 8
        """
    ):
        print(f"  R{r['round']} {r['s1']}v{r['s2']:2d}  "
              f"ml1={r['team1_moneyline']:+5d}  ml2={r['team2_moneyline']:+5d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SYNTHETIC BETTING LINES BACKFILL (2014-2017)")
    print("=" * 70)
    print()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # --- Pass 1: Fix round encoding for 2014-2015 ---
    fix_round_encoding(conn)

    # --- Pass 2: Build lookup and backfill lines ---
    print("Pass 2: Building moneyline lookup from 2018-2025 consensus lines...")
    lookup = build_lookup(conn)
    print()
    print(f"Pass 3: Inserting synthetic lines for seasons {BACKFILL_SEASONS}...")
    backfill_lines(conn, lookup)

    verify(conn)
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
