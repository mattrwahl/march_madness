"""
Full DDL for the march_madness database.
All tables prefixed mm_ to avoid collision if DBs are ever merged.
"""

CREATE_MM_TEAMS = """
CREATE TABLE IF NOT EXISTS mm_teams (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,   -- CBBD canonical name
    conference  TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);
"""

CREATE_MM_TOURNAMENT_ENTRIES = """
CREATE TABLE IF NOT EXISTS mm_tournament_entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    season      INTEGER NOT NULL,
    team_id     INTEGER NOT NULL REFERENCES mm_teams(id),
    seed        INTEGER NOT NULL,       -- 1–16
    region      TEXT,                  -- East / West / South / Midwest
    cbbd_team   TEXT,                  -- CBBD raw team name (for reference)
    UNIQUE(season, team_id)
);
"""

CREATE_MM_GAMES = """
CREATE TABLE IF NOT EXISTS mm_games (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cbbd_game_id    TEXT UNIQUE NOT NULL,   -- CBBD integer game id stored as text
    season          INTEGER NOT NULL,
    game_date       TEXT,
    round           INTEGER,        -- teams remaining: 64/32/16/8/4/2/1
    region          TEXT,
    tournament      TEXT,           -- "NCAA" from CBBD
    team1_id        INTEGER REFERENCES mm_teams(id),
    team2_id        INTEGER REFERENCES mm_teams(id),
    team1_seed      INTEGER,
    team2_seed      INTEGER,
    team1_score     INTEGER,
    team2_score     INTEGER,
    winner_id       INTEGER REFERENCES mm_teams(id),
    neutral_site    INTEGER DEFAULT 1,  -- tournament games are always neutral
    created_at      TEXT DEFAULT (datetime('now'))
);
"""

CREATE_MM_TEAM_METRICS = """
CREATE TABLE IF NOT EXISTS mm_team_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    season          INTEGER NOT NULL,
    team_id         INTEGER NOT NULL REFERENCES mm_teams(id),
    -- Adjusted efficiency (from CBBD RatingsApi)
    adj_off_rating  REAL,
    adj_def_rating  REAL,
    net_rating      REAL,
    net_rank        INTEGER,
    def_rank        INTEGER,
    -- Four factors (from CBBD StatsApi)
    efg_pct         REAL,       -- effective FG% (offense)
    opp_efg_pct     REAL,       -- opponent effective FG% (defense)
    tov_ratio       REAL,       -- turnover ratio (lower = better)
    oreb_pct        REAL,       -- offensive rebound %
    ft_rate         REAL,       -- free throw rate
    -- Team stats
    ft_pct          REAL,       -- free throw %
    pace            REAL,
    -- Derived
    seed_rank_gap   INTEGER,    -- net_rank - (seed * 10); negative = under-seeded
    UNIQUE(season, team_id)
);
"""

CREATE_MM_BETTING_LINES = """
CREATE TABLE IF NOT EXISTS mm_betting_lines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         INTEGER NOT NULL REFERENCES mm_games(id),
    provider        TEXT,               -- bookmaker name
    team1_moneyline INTEGER,
    team2_moneyline INTEGER,
    team1_novig_prob REAL,
    team2_novig_prob REAL,
    fetched_at      TEXT DEFAULT (datetime('now'))
);
"""

CREATE_MM_MODEL_PICKS = """
CREATE TABLE IF NOT EXISTS mm_model_picks (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    season              INTEGER NOT NULL,
    team_id             INTEGER NOT NULL REFERENCES mm_teams(id),
    seed                INTEGER NOT NULL,
    pick_rank           INTEGER NOT NULL,   -- 1–8 (1 = highest score)
    model_score         REAL,
    initial_bet_dollars REAL DEFAULT 25.0,
    round_exit          INTEGER,            -- round where team exited (NULL until known)
    payout_dollars      REAL,               -- cumulative rolling payout at exit
    units_won           REAL,               -- (payout_dollars - initial_bet_dollars) / 100
    status              TEXT DEFAULT 'pending',  -- pending | active | cashed_out | eliminated
    created_at          TEXT DEFAULT (datetime('now')),
    UNIQUE(season, team_id)
);
"""

CREATE_MM_MODEL_WEIGHTS = """
CREATE TABLE IF NOT EXISTS mm_model_weights (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at      TEXT DEFAULT (datetime('now')),
    train_seasons   TEXT,       -- e.g. "2021,2022,2023,2024"
    val_seasons     TEXT,       -- e.g. "2025"
    -- Feature weights (z-score normalized; negative features already flipped)
    w_seed_rank_gap REAL,
    w_def_rank      REAL,
    w_opp_efg_pct   REAL,
    w_net_rating    REAL,
    w_tov_ratio     REAL,
    w_ft_pct        REAL,
    w_oreb_pct      REAL,
    w_pace          REAL,
    -- Performance metrics
    train_units_won REAL,
    val_units_won   REAL,
    notes           TEXT
);
"""

ALL_TABLES = [
    CREATE_MM_TEAMS,
    CREATE_MM_TOURNAMENT_ENTRIES,
    CREATE_MM_GAMES,
    CREATE_MM_TEAM_METRICS,
    CREATE_MM_BETTING_LINES,
    CREATE_MM_MODEL_PICKS,
    CREATE_MM_MODEL_WEIGHTS,
]
