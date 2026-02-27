import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR  = BASE_DIR / "logs"
DB_PATH  = DATA_DIR / "march_madness.db"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

CBBD_API_KEY = os.environ["CBBD_API_KEY"]

# All seasons available for backfill (skip 2020 — no tournament due to COVID)
ALL_SEASONS = list(range(2008, 2020)) + list(range(2021, 2026))  # 2008–2019, 2021–2025

# Split for training / validation / out-of-sample test
# Note: 2005–2007 are reserved as a never-seen holdout (run manually via --test)
TRAIN_SEASONS = [s for s in range(2008, 2025) if s != 2020 and s <= 2024]  # 2008–2024 excl. 2020
VAL_SEASONS   = [2025]
TEST_SEASONS  = [2005, 2006, 2007]   # holdout — never used in training

# Kept for backward compat
HISTORICAL_SEASONS = ALL_SEASONS
CURRENT_SEASON = 2026

# Betting parameters
INITIAL_BET_DOLLARS = 25.0   # per pick
NUM_PICKS = 8
ELIGIBLE_SEEDS = list(range(5, 13))  # seeds 5–12

# Cash-out round: teams-remaining notation (8 = Elite Eight, 16 = Sweet Sixteen)
# The team cashes out after WINNING a game in this round.
CASH_OUT_ROUND_E8 = 8   # default — cash out after winning S16 (entering E8)
CASH_OUT_ROUND_S16 = 16  # alternate — cash out after winning R32 (entering S16)
DEFAULT_CASH_OUT_ROUND = CASH_OUT_ROUND_E8

LOG_LEVEL = "INFO"
