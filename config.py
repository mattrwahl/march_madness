import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR  = BASE_DIR / "logs"
DB_PATH  = DATA_DIR / "march_madness.db"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Load .env from the project root (no-op if file doesn't exist)
load_dotenv(BASE_DIR / ".env")

CBBD_API_KEY = os.environ["CBBD_API_KEY"]

# Seasons for historical training (backfill)
HISTORICAL_SEASONS = list(range(2021, 2026))   # 2021–2025 inclusive
CURRENT_SEASON = 2026

# Betting parameters
INITIAL_BET_DOLLARS = 25.0   # per pick
NUM_PICKS = 8
ELIGIBLE_SEEDS = list(range(5, 13))  # seeds 5–12

# Logging
LOG_LEVEL = "INFO"
