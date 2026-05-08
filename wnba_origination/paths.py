"""Central path config — override RAPM_DIR env var if data lives elsewhere."""
import os
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
BDB_DIR = DATA / "bigdataball"

RAPM_DIR = Path(
    os.getenv("WNBA_RAPM_DIR",
              r"C:/Users/shank.subramani_betf/Desktop/ShotsDashboard/WNBA_RAPM/wnba_data")
)

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
