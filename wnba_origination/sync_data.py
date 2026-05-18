"""
Sync analysis CSVs from WNBA_RAPM/analysis (and wnba_data) into
wnba_origination/data/. Called from the sidebar Refresh button.

Source of truth for everything analysis-related is WNBA_RAPM. This script
mirrors the files this app actually reads into a local copy so the app has
a clean single data folder and doesn't reach across the repo.
"""

import shutil
from pathlib import Path

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

WNBA_RAPM = APP_DIR.parent / "WNBA_RAPM"
ANALYSIS = WNBA_RAPM / "analysis"
WNBA_DATA = WNBA_RAPM / "wnba_data"

# (source, destination_filename)
FILES = [
    (ANALYSIS / "pace_stats.csv",             "pace_stats.csv"),
    (ANALYSIS / "ft_decomp.csv",              "ft_decomp.csv"),
    (ANALYSIS / "bonus_by_quarter.csv",       "bonus_by_quarter.csv"),
    (ANALYSIS / "foul_violation_rates.csv",   "foul_violation_rates.csv"),
    (ANALYSIS / "player_stats.csv",           "analysis_player_stats.csv"),
    (ANALYSIS / "ft_shooting_foul_locs.csv",  "ft_shooting_foul_locs.csv"),
    (WNBA_DATA / "games_2025_Regular_Season.csv", "games_2025_RS.csv"),
    (WNBA_DATA / "games_2026_Regular_Season.csv", "games_2026_RS.csv"),
    (WNBA_DATA / "stints_rich" / "stints_rich_2025_RS.csv", "stints_rich_2025.csv"),
    (WNBA_DATA / "stints_rich" / "stints_rich_2026_RS.csv", "stints_rich_2026.csv"),
    (WNBA_DATA / "rapm_2025_RS.csv",          "rapm_2025_RS.csv"),
    (WNBA_DATA / "rapm_2025_RS_3yr.csv",      "rapm_2025_RS_3yr.csv"),
]


def sync(verbose: bool = True) -> dict:
    """Copy all source files to data/. Returns dict of {dest: status}."""
    results = {}
    for src, dest_name in FILES:
        dest = DATA_DIR / dest_name
        if not src.exists():
            results[dest_name] = "source missing"
            if verbose:
                print(f"  ! {dest_name}: source not found ({src})")
            continue
        try:
            shutil.copy2(src, dest)
            results[dest_name] = f"ok ({src.stat().st_size:,} bytes)"
            if verbose:
                print(f"  + {dest_name}: copied")
        except Exception as exc:
            results[dest_name] = f"error: {exc}"
            if verbose:
                print(f"  ! {dest_name}: {exc}")
    return results


if __name__ == "__main__":
    print(f"Syncing data into {DATA_DIR} ...")
    out = sync()
    ok = sum(1 for v in out.values() if v.startswith("ok"))
    print(f"\nDone: {ok}/{len(out)} files synced.")
