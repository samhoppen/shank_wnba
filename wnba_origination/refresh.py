"""
Orchestrator for the full WNBA pipeline.

Runs, in order:
    1. wnba_rapm/update_stints.py  — append new games to stints CSVs
    2. wnba_rapm/rapm_reproducible.ipynb — refit RAPM (via nbconvert)
    3. scripts/fetch_pbp.py        — download raw PBP JSON for new games
    4. scripts/regen_analysis.py   — rebuild pace_stats/bonus/ft/fouls
    5. sync_data.py                — mirror submodule CSVs to data/
    6. player_store.py             — rebuild unified player table
    7. game_log.py                 — rebuild per-game four/five factors

Usage:
    python refresh.py              # rebuild everything for current year (2026)
    python refresh.py --year 2025  # explicit year
    python refresh.py --skip-notebook   # skip the RAPM refit (fast path)
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
SUBMODULE = REPO_ROOT / "wnba_rapm"
SCRIPTS = HERE / "scripts"


def run(cmd: list[str], cwd: Path = HERE, check: bool = True) -> None:
    print(f"\n>>> {' '.join(str(c) for c in cmd)}  (cwd={cwd.name})")
    result = subprocess.run(cmd, cwd=str(cwd))
    if check and result.returncode != 0:
        print(f"ERROR: command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--skip-notebook", action="store_true",
                        help="Skip RAPM refit (use existing rapm CSVs)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip stints + PBP downloads")
    args = parser.parse_args()

    py = sys.executable
    year = str(args.year)

    if not args.skip_fetch:
        print("\n=== Step 1: Update stints (wnba_rapm submodule) ===")
        run([py, "update_stints.py", "--season", year], cwd=SUBMODULE, check=False)

    if not args.skip_notebook:
        print("\n=== Step 2: Refit RAPM (notebook) ===")
        run([py, "-m", "jupyter", "nbconvert",
             "--to", "notebook", "--execute",
             "--inplace", "rapm_reproducible.ipynb"],
            cwd=SUBMODULE, check=False)

    if not args.skip_fetch:
        print("\n=== Step 3: Fetch raw PBP JSON ===")
        run([py, str(SCRIPTS / "fetch_pbp.py"), "--year", year], check=False)

    print("\n=== Step 4: Regenerate analysis CSVs ===")
    extra = ["--append-2025"] if args.year != 2025 else []
    run([py, str(SCRIPTS / "regen_analysis.py"), "--year", year, *extra], check=False)

    print("\n=== Step 5: Sync submodule CSVs into data/ ===")
    run([py, "sync_data.py"], check=False)

    print("\n=== Step 6: Rebuild player store ===")
    run([py, "player_store.py"], check=False)

    print("\n=== Step 7: Rebuild game log ===")
    run([py, "game_log.py", "--year", year], check=False)

    print("\n✓ Done. Restart Streamlit (or hit R in the browser) to pick up changes.")


if __name__ == "__main__":
    main()
