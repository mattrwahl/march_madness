"""
Database helpers for march_madness.
Thin wrapper around shared.sqlite_helpers.
"""
import sys
import logging
from pathlib import Path

# Add monorepo root to path so shared/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.sqlite_helpers import get_db, get_connection  # noqa: F401
from db.models import ALL_TABLES

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

logger = logging.getLogger(__name__)


def init_db():
    """Create all tables if they don't exist, and run any pending migrations."""
    with get_db(DB_PATH) as conn:
        for ddl in ALL_TABLES:
            conn.execute(ddl)
        # Migration: w_threshold column added for variable-N model
        try:
            conn.execute("ALTER TABLE mm_model_weights ADD COLUMN w_threshold REAL")
        except Exception:
            pass  # column already exists
        # Migration: V2 conference tournament feature columns
        for col_def in ["conf_tourney_wins INTEGER", "conf_tourney_avg_margin REAL"]:
            try:
                conn.execute(f"ALTER TABLE mm_team_metrics ADD COLUMN {col_def}")
            except Exception:
                pass
        for col_def in ["w_conf_tourney_wins REAL", "w_conf_tourney_avg_margin REAL"]:
            try:
                conn.execute(f"ALTER TABLE mm_model_weights ADD COLUMN {col_def}")
            except Exception:
                pass
    logger.info(f"Database initialized at {DB_PATH}")


def upsert_team(conn, name: str, conference: str | None = None) -> int:
    """Insert or update a team. Returns team id."""
    existing = conn.execute(
        "SELECT id FROM mm_teams WHERE name = ?", (name,)
    ).fetchone()

    if existing:
        team_id = existing["id"]
        if conference:
            conn.execute(
                "UPDATE mm_teams SET conference = ? WHERE id = ?", (conference, team_id)
            )
    else:
        conn.execute(
            "INSERT INTO mm_teams (name, conference) VALUES (?, ?)", (name, conference)
        )
        team_id = conn.execute(
            "SELECT id FROM mm_teams WHERE name = ?", (name,)
        ).fetchone()["id"]

    return team_id


def get_team_id(conn, name: str) -> int | None:
    """Look up team id by name."""
    row = conn.execute("SELECT id FROM mm_teams WHERE name = ?", (name,)).fetchone()
    return row["id"] if row else None
