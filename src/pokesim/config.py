"""Project-wide settings, overridable via environment variables (see .env.example)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DB_URL = os.environ.get("POKESIM_DB_URL", "postgresql+psycopg://pokesim:pokesim@127.0.0.1:5433/pokesim")
SHOWDOWN_DIR = Path(os.environ.get("POKESIM_SHOWDOWN_DIR", ROOT / "external/pokemon-showdown"))
FORMAT = os.environ.get("POKESIM_FORMAT", "gen9championsbssregmc")
RAW_DIR = ROOT / "data/raw"
PROCESSED_DIR = ROOT / "data/processed"
