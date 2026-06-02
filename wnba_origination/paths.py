"""Central path config — override RAPM_DIR env var if data lives elsewhere."""
import os
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DATA = HERE / "data"
BDB_DIR = DATA / "bigdataball"

# Default: the `wnba_rapm` git submodule at the repo root. Override with
# WNBA_RAPM_DIR env var to point at a different copy of the wnba_data tree.
RAPM_DIR = Path(
    os.getenv("WNBA_RAPM_DIR", str(REPO_ROOT / "wnba_rapm" / "wnba_data"))
).resolve()

# Raw PBP JSON is excluded from the submodule (~400 MB). Default to a sibling
# cache directory populated by scripts/fetch_pbp.py; fall back to the
# submodule's raw_pbp/ if a user opted to place JSON there.
_RAW_PBP_IN_SUBMODULE = RAPM_DIR / "raw_pbp"
_RAW_PBP_IN_CACHE = Path(
    os.getenv("WNBA_RAW_PBP_DIR", str(REPO_ROOT / "wnba_rapm_cache" / "raw_pbp"))
).resolve()
RAW_PBP_DIR = _RAW_PBP_IN_SUBMODULE if _RAW_PBP_IN_SUBMODULE.exists() else _RAW_PBP_IN_CACHE

EC_ALL_SEASONS = Path(r"C:/Users/shank.subramani_betf/Desktop/wnba_ec_all_seasons.csv")
EC_SCRAPER = Path(r"C:/Users/shank.subramani_betf/Desktop/wnba_ec_all.py")
PYTHON = r"C:/Users/shank.subramani_betf/AppData/Local/Programs/Python/Python313/python.exe"

# Derived RAPM paths
def rapm_season(year: int, window: str = "RS") -> Path:
    return RAPM_DIR / f"rapm_{year}_{window}.csv"

def rapm_8factor(year: int = 2025, window: str = "RS", span: str = "5yr") -> Path:
    return RAPM_DIR / f"rapm_8factor_{year}_{window}_{span}.csv"

def stints(year: int, window: str = "RS") -> Path:
    return RAPM_DIR / "stints" / f"stints_{year}_{window}.csv"

def stints_rich(year: int, window: str = "RS") -> Path:
    return RAPM_DIR / "stints_rich" / f"stints_rich_{year}_{window}.csv"

PLAYER_MINUTES = RAPM_DIR / "player_minutes.csv"
PLAYER_NAMES = RAPM_DIR / "player_names.csv"

PLAYER_STORE = DATA / "player_store.csv"
EC_2026 = DATA / "ec_2026.csv"
RAPM_2026 = DATA / "rapm_2026_RS.csv"
