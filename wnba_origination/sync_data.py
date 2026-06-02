"""
Sync data files from the wnba_rapm submodule into wnba_origination/data/.

The wnba_rapm submodule (at <repo>/wnba_rapm/wnba_data) is the source of truth
for stints, games, and RAPM coefficients. This script mirrors the subset of
files the streamlit app actually reads from the local data/ folder.

Analysis CSVs (pace_stats.csv, bonus_by_quarter.csv, ft_decomp.csv,
foul_violation_rates.csv) are generated in place by scripts/regen_analysis.py
and live alongside the synced files — they are NOT mirrored here.
"""

import shutil
from pathlib import Path

from paths import RAPM_DIR

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# (source, destination_filename)
FILES = [
    (RAPM_DIR / "games_2025_Regular_Season.csv", "games_2025_RS.csv"),
    (RAPM_DIR / "games_2026_Regular_Season.csv", "games_2026_RS.csv"),
    (RAPM_DIR / "stints_rich" / "stints_rich_2025_RS.csv", "stints_rich_2025.csv"),
    (RAPM_DIR / "stints_rich" / "stints_rich_2026_RS.csv", "stints_rich_2026.csv"),
    (RAPM_DIR / "rapm_2025_RS.csv",          "rapm_2025_RS.csv"),
    (RAPM_DIR / "rapm_2025_RS_3yr.csv",      "rapm_2025_RS_3yr.csv"),
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
