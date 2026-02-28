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
# Modern era (2014+) aligns with analytics-driven seeding and full betting line coverage.
TRAIN_SEASONS = [s for s in range(2014, 2023) if s != 2020]  # 2014–2022 excl. 2020 (7 seasons)
VAL_SEASONS   = [2023, 2024]                                  # 2 modern validation seasons
TEST_SEASONS  = [2025]                                        # true out-of-sample holdout

# Kept for backward compat
HISTORICAL_SEASONS = ALL_SEASONS
CURRENT_SEASON = 2026

# Betting parameters
# flat:     fixed $25 per round per team; profits locked in each round independently
# rollover: $25 initial stake compounds forward through wins (parlay-style)
INITIAL_BET_DOLLARS = 25.0   # per pick per round (flat) or initial stake (rollover)
NUM_PICKS = 8
ELIGIBLE_SEEDS = list(range(5, 13))  # seeds 5–12

# Cash-out round: round encoding = teams remaining at that stage.
# Team cashes out after WINNING a game in this round.
#   S16 (16): win R32  → enter E8   (2 rounds max)
#   E8  ( 8): win S16  → enter F4   (3 rounds max)
#   F4  ( 4): win E8   → enter Champ (4 rounds max)
CASH_OUT_ROUND_S16 = 16
CASH_OUT_ROUND_E8  = 8
CASH_OUT_ROUND_F4  = 4
DEFAULT_CASH_OUT_ROUND = CASH_OUT_ROUND_E8

# Bet styles
BET_STYLE_FLAT     = "flat"      # fixed $25 per round, lock in profits each round
BET_STYLE_ROLLOVER = "rollover"  # compound stake rolls forward through each win
DEFAULT_BET_STYLE  = BET_STYLE_FLAT

# Variable-N model parameters
# Threshold is a z-score cutoff applied to composite scores within the eligible field.
# Teams above it are selected; if fewer than MIN_PICKS clear the bar the top MIN_PICKS
# are taken anyway. MAX_PICKS is a hard ceiling.
MIN_PICKS_PER_YEAR = 4
MAX_PICKS_PER_YEAR = 12
DEFAULT_THRESHOLD  = 0.5   # starting guess; optimizer will tune this

# Tiered conviction betting (V2)
# Top half of picks bet TIER1_BET per round; bottom half bet TIER2_BET.
# Total outlay = (NUM_PICKS/2)*TIER1_BET + (NUM_PICKS/2)*TIER2_BET == INITIAL_BET_DOLLARS*NUM_PICKS
TIER1_BET = 37.50  # top-4 picks — higher conviction
TIER2_BET = 12.50  # bottom-4 picks — lower conviction

# L2 regularization penalty on feature weights (prevents overfitting)
L2_LAMBDA = 0.01

LOG_LEVEL = "INFO"
